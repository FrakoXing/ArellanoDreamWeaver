# -*- coding: utf-8 -*-
"""
伪标签生成脚本 (半监督 v2)
=========================

用训练好的 v1 模型对无标签 token 文件推理, 生成伪标签供 v2 半监督微调。

流程:
  1. 加载 v1 模型 (PyTorch 或 ONNX)
  2. 遍历 labels.json 中的 unlabeled 列表
  3. 推理 → 归一化定数
  4. 置信度过滤 (预测值与桶中心接近的才保留)
  5. 输出 labels_pseudo.json

用法:
  # 用 PyTorch checkpoint 生成伪标签
  python scripts/pseudo_label.py --checkpoint logs/rating_best.pth

  # 用 ONNX 模型生成伪标签 (更快, 无需 PyTorch)
  python scripts/pseudo_label.py --onnx exports/rating/rating_ort_opset17.onnx

  # 限制数量 (调试用)
  python scripts/pseudo_label.py --checkpoint logs/rating_best.pth --limit 100
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rating_common import (
    compute_features,
    denormalize_level,
    game_token_to_name,
    FEATURE_DIM,
    GAME_TOKEN_PHIRA,
    GAME_TOKEN_ARELLANO,
)


# ============================================================
# PyTorch 推理
# ============================================================

class TorchPredictor:
    """用 PyTorch checkpoint 推理"""

    def __init__(self, checkpoint_path: str):
        import torch
        from src.models.rating_model import RatingModel, RatingConfig

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_cfg = ckpt.get("config", {}).get("model", {})
        valid_fields = {f for f in RatingConfig.__dataclass_fields__}
        filtered = {k: v for k, v in model_cfg.items() if k in valid_fields}

        self.model = RatingModel(RatingConfig(**filtered))
        sd = ckpt.get("model_state_dict", {})
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()

        self.max_seq_len = filtered.get("max_seq_len", 2048)

        # 特征标准化
        self.feat_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.feat_std = np.ones(FEATURE_DIM, dtype=np.float32)
        stats_path = PROJECT_ROOT / "data" / "rating" / "feature_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            self.feat_mean = np.array(stats["mean"], dtype=np.float32)
            self.feat_std = np.array(stats["std"], dtype=np.float32) + 1e-8

        print(f"📂 PyTorch 模型已加载: {checkpoint_path}")

    def predict(self, tokens: List[int]) -> float:
        import torch

        game_token_id = int(tokens[0]) if tokens else GAME_TOKEN_PHIRA

        # 截断/填充
        seq_len = self.max_seq_len
        if len(tokens) >= seq_len:
            t = tokens[:seq_len]
            m = [1] * seq_len
        else:
            t = tokens + [0] * (seq_len - len(tokens))
            m = [1] * len(tokens) + [0] * (seq_len - len(tokens))

        # 特征
        feat = compute_features(tokens, game_token_id)
        feat = (feat - self.feat_mean) / self.feat_std

        with torch.no_grad():
            out = self.model(
                torch.tensor([t], dtype=torch.long),
                torch.tensor([m], dtype=torch.long),
                torch.tensor([feat], dtype=torch.float32),
                torch.tensor([game_token_id], dtype=torch.long),
            )
        return float(out.item())


# ============================================================
# ONNX 推理
# ============================================================

class ONNXPredictor:
    """用 ONNX 模型推理 (更快)"""

    def __init__(self, onnx_path: str, meta_path: Optional[str] = None):
        from infer_rating_onnx import RatingONNXInferer
        self.inferer = RatingONNXInferer(onnx_path, meta_path)

    def predict(self, tokens: List[int]) -> float:
        rating_norm, _ = self.inferer.infer(tokens)
        return rating_norm


# ============================================================
# 伪标签生成
# ============================================================

def generate_pseudo_labels(
    predictor,
    tokens_dir: Path,
    labels_file: str,
    output_file: str,
    confidence_threshold: float = 0.05,
    min_samples: int = 500,
    limit: int = 0,
) -> Dict[str, Any]:
    """
    生成伪标签。

    Args:
        predictor:        TorchPredictor 或 ONNXPredictor
        tokens_dir:       Token 文件目录
        labels_file:      labels.json (含 unlabeled 列表)
        output_file:      输出 labels_pseudo.json
        confidence_threshold: 置信度阈值 (预测值与桶中心差 < 此值才保留)
        min_samples:      最少保留样本数 (太少则跳过)
        limit:            最多处理 N 个 (0=全部)
    """
    import torch

    # 加载无标签列表
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    unlabeled = labels_data.get("unlabeled", [])
    if not unlabeled:
        print("❌ labels.json 中无 unlabeled 列表")
        return {}

    if limit > 0:
        unlabeled = unlabeled[:limit]

    print(f"\n📥 无标签文件: {len(unlabeled)}")
    print(f"   置信度阈值: {confidence_threshold}")
    print(f"{'='*60}")

    # 分桶中心 (8 个桶)
    n_buckets = 8
    bucket_centers = np.array([(i + 0.5) / n_buckets for i in range(n_buckets)])

    pseudo_labels = {}
    errors = []

    for i, fname in enumerate(unlabeled):
        fpath = tokens_dir / fname
        if not fpath.exists():
            continue

        try:
            tokens = torch.load(fpath, map_location='cpu', weights_only=True).long().tolist()
        except Exception:
            try:
                tokens = torch.load(fpath, map_location='cpu').long().tolist()
            except Exception as e:
                errors.append((fname, str(e)))
                continue

        game_token_id = int(tokens[0]) if tokens else GAME_TOKEN_PHIRA
        game = game_token_to_name(game_token_id)

        try:
            pred_norm = predictor.predict(tokens)
        except Exception as e:
            errors.append((fname, str(e)))
            continue

        # 置信度过滤: 预测值与最近桶中心的距离
        nearest_center = bucket_centers[np.argmin(np.abs(bucket_centers - pred_norm))]
        confidence = abs(pred_norm - nearest_center)

        if confidence < confidence_threshold:
            pred_raw = denormalize_level(pred_norm, game)
            pseudo_labels[fname] = {
                "game": game,
                "level": round(pred_raw, 1),
                "normalized": pred_norm,
                "osu_keys": None,
                "source": "pseudo",
                "confidence": float(confidence),
            }

        if (i + 1) % 200 == 0:
            print(f"   进度: {i+1}/{len(unlabeled)} | 已保留: {len(pseudo_labels)}")

    print(f"\n{'='*60}")
    print(f"📊 伪标签生成结果:")
    print(f"   处理: {len(unlabeled)}")
    print(f"   保留: {len(pseudo_labels)} ({len(pseudo_labels)/len(unlabeled)*100:.1f}%)")
    print(f"   错误: {len(errors)}")

    if len(pseudo_labels) < min_samples:
        print(f"\n   ⚠️ 保留样本不足 ({len(pseudo_labels)} < {min_samples})")
        print(f"   建议: 增大 confidence_threshold 或检查模型质量")
        print(f"   仍将保存, 但 v2 训练可能效果不佳")

    # 保存
    output_data = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "model_source": "v1",
        "stats": {
            "total_unlabeled": len(unlabeled),
            "pseudo_labeled": len(pseudo_labels),
            "errors": len(errors),
            "confidence_threshold": confidence_threshold,
        },
        "labels": pseudo_labels,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 保存: {output_path}")

    # 按游戏统计
    by_game = {}
    for label in pseudo_labels.values():
        g = label["game"]
        by_game[g] = by_game.get(g, 0) + 1
    print(f"   按游戏: {by_game}")

    return output_data


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="伪标签生成 (半监督 v2)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch checkpoint 路径")
    parser.add_argument("--onnx", type=str, default=None,
                        help="ONNX 模型路径 (替代 checkpoint, 更快)")
    parser.add_argument("--meta", type=str, default=None,
                        help="model_meta.json (ONNX 模式用)")
    parser.add_argument("--tokens-dir", type=str, default="data/tokens")
    parser.add_argument("--labels", type=str, default="data/rating/labels.json")
    parser.add_argument("--output", type=str, default="data/rating/labels_pseudo.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.05,
                        help="置信度阈值 (默认 0.05)")
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 个 (0=全部)")
    args = parser.parse_args()

    if not args.checkpoint and not args.onnx:
        print("❌ 请指定 --checkpoint 或 --onnx")
        sys.exit(1)

    # 创建推理器
    if args.onnx:
        predictor = ONNXPredictor(args.onnx, args.meta)
    else:
        predictor = TorchPredictor(args.checkpoint)

    tokens_dir = PROJECT_ROOT / args.tokens_dir
    labels_file = str(PROJECT_ROOT / args.labels)
    output_file = str(PROJECT_ROOT / args.output)

    generate_pseudo_labels(
        predictor=predictor,
        tokens_dir=tokens_dir,
        labels_file=labels_file,
        output_file=output_file,
        confidence_threshold=args.confidence_threshold,
        min_samples=args.min_samples,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
