# -*- coding: utf-8 -*-
"""
定数评级 ONNX 推理验证脚本
==========================

加载导出的 ONNX 模型, 对单个 token 文件或 Arellano ChartData JSON 进行推理,
验证 ONNX → 反归一化 → 原始定数 的完整链路。

用法:
  # 对单个 token 文件推理
  python scripts/infer_rating_onnx.py --model exports/rating/rating_ort_opset17.onnx \\
      --token-file data/tokens/10018_IN Lv.12_xxx.pt

  # 批量推理 + 与文件名标签对比
  python scripts/infer_rating_onnx.py --model exports/rating/rating_ort_opset17.onnx \\
      --batch --limit 50

  # 指定元数据 (反归一化 + 特征标准化)
  python scripts/infer_rating_onnx.py --model rating.onnx \\
      --meta exports/rating/model_meta.json --token-file xxx.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rating_common import (
    parse_filename,
    denormalize_level,
    game_token_to_name,
    compute_features,
    FEATURE_DIM,
    GAME_TOKEN_PHIRA,
    GAME_TOKEN_OSU,
    GAME_TOKEN_ARELLANO,
)


# ============================================================
# ONNX 推理器
# ============================================================

class RatingONNXInferer:
    """ONNX 定数评级推理器"""

    def __init__(self, model_path: str, meta_path: Optional[str] = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("需要 onnxruntime: pip install onnxruntime")

        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider']
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        print(f"📂 加载 ONNX 模型: {model_path}")
        print(f"   输入: {self.input_names}")
        print(f"   输出: {self.output_name}")

        # 加载元数据
        self.meta = self._load_meta(meta_path, model_path)
        self.feat_mean = np.array(
            self.meta.get("feature_stats", {}).get("mean", [0.0]*FEATURE_DIM),
            dtype=np.float32
        )
        self.feat_std = np.array(
            self.meta.get("feature_stats", {}).get("std", [1.0]*FEATURE_DIM),
            dtype=np.float32
        ) + 1e-8
        self.max_seq_len = self.meta.get("model_config", {}).get("max_seq_len", 2048)

    @staticmethod
    def _load_meta(meta_path: Optional[str], model_path: str) -> Dict:
        """加载元数据 JSON"""
        if meta_path and Path(meta_path).exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 尝试同目录的 model_meta.json
        auto = Path(model_path).parent / "model_meta.json"
        if auto.exists():
            with open(auto, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 尝试 data/rating/normalization.json
        norm_path = PROJECT_ROOT / "data" / "rating" / "normalization.json"
        norm = {}
        if norm_path.exists():
            with open(norm_path, 'r', encoding='utf-8') as f:
                norm = json.load(f)

        return {
            "normalization": norm,
            "feature_stats": {"mean": [0.0]*FEATURE_DIM, "std": [1.0]*FEATURE_DIM},
            "model_config": {"max_seq_len": 2048},
        }

    def infer(
        self,
        tokens: List[int],
        game_token_id: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        对单个 token 序列推理。

        Args:
            tokens:          Token ID 列表
            game_token_id:   Game Token ID (默认取 tokens[0])

        Returns:
            (rating_norm, rating_raw): 归一化值, 原始定数
        """
        if game_token_id is None:
            game_token_id = int(tokens[0]) if tokens else GAME_TOKEN_PHIRA

        # 截断/填充
        seq_len = self.max_seq_len
        if len(tokens) >= seq_len:
            tokens_arr = tokens[:seq_len]
            mask = [1] * seq_len
        else:
            tokens_arr = tokens + [0] * (seq_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (seq_len - len(tokens))

        # 手工特征
        feat = compute_features(tokens, game_token_id)
        feat = (feat - self.feat_mean) / self.feat_std

        # 构造输入
        inputs = {
            "tokens": np.array([tokens_arr], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
            "handcrafted_features": np.array([feat], dtype=np.float32),
            "game_token_ids": np.array([game_token_id], dtype=np.int64),
        }

        # 推理
        output = self.session.run(None, inputs)[0]
        rating_norm = float(output[0])

        # 反归一化
        game = game_token_to_name(game_token_id)
        rating_raw = denormalize_level(rating_norm, game)

        return rating_norm, rating_raw


# ============================================================
# 批量评估
# ============================================================

def batch_evaluate(
    inferer: RatingONNXInferer,
    tokens_dir: Path,
    labels_file: str,
    limit: int = 0,
) -> Dict[str, Any]:
    """
    批量推理并与标签对比, 输出 MAE/命中率。

    Returns:
        {"mae": ..., "hit_0.5": ..., "hit_1.0": ..., "count": N, ...}
    """
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    labels = labels_data.get("labels", labels_data)

    import torch

    fnames = list(labels.keys())
    if limit > 0:
        fnames = fnames[:limit]

    print(f"\n📊 批量评估: {len(fnames)} 样本")
    print(f"{'='*60}")

    preds_raw = []
    targets_raw = []
    games = []
    errors = []

    for i, fname in enumerate(fnames):
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

        label = labels[fname]
        target_norm = label["normalized"]
        game_token_id = int(tokens[0]) if tokens else GAME_TOKEN_PHIRA
        game = game_token_to_name(game_token_id)
        target_raw = denormalize_level(target_norm, game)

        try:
            pred_norm, pred_raw = inferer.infer(tokens, game_token_id)
        except Exception as e:
            errors.append((fname, str(e)))
            continue

        preds_raw.append(pred_raw)
        targets_raw.append(target_raw)
        games.append(game)

        if (i + 1) % 100 == 0:
            print(f"   进度: {i+1}/{len(fnames)}")

    preds_raw = np.array(preds_raw)
    targets_raw = np.array(targets_raw)
    abs_err = np.abs(preds_raw - targets_raw)

    mae = float(abs_err.mean())
    results = {
        "count": len(preds_raw),
        "mae": mae,
        "hit_0.5": float((abs_err <= 0.5).mean()),
        "hit_1.0": float((abs_err <= 1.0).mean()),
        "hit_1.5": float((abs_err <= 1.5).mean()),
        "max_error": float(abs_err.max()),
    }

    # 按游戏分别统计
    games_arr = np.array(games)
    for g in ["phira", "osu", "arellano"]:
        mask = games_arr == g
        if mask.any():
            results[f"mae_{g}"] = float(abs_err[mask].mean())
            results[f"count_{g}"] = int(mask.sum())

    # 打印结果
    print(f"\n{'='*60}")
    print(f"📈 评估结果 ({results['count']} 样本)")
    print(f"{'='*60}")
    print(f"   MAE:       {results['mae']:.4f}")
    print(f"   ±0.5 命中: {results['hit_0.5']*100:.1f}%")
    print(f"   ±1.0 命中: {results['hit_1.0']*100:.1f}%")
    print(f"   ±1.5 命中: {results['hit_1.5']*100:.1f}%")
    print(f"   最大误差:  {results['max_error']:.4f}")
    for g in ["phira", "osu", "arellano"]:
        key = f"mae_{g}"
        if key in results:
            print(f"   {g}: MAE={results[key]:.4f} (n={results[f'count_{g}']})")

    if errors:
        print(f"\n   ⚠️ {len(errors)} 个错误:")
        for fname, err in errors[:5]:
            print(f"     {fname}: {err}")

    # 打印前 20 个预测 vs 标签
    print(f"\n   前 20 个预测 vs 标签:")
    for i in range(min(20, len(preds_raw))):
        game = games[i]
        pred = preds_raw[i]
        target = targets_raw[i]
        err = abs(pred - target)
        print(f"     [{game:8s}] pred={pred:6.2f}  target={target:6.2f}  err={err:.2f}")

    return results


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="定数评级 ONNX 推理验证")
    parser.add_argument("--model", type=str, required=True,
                        help="ONNX 模型路径")
    parser.add_argument("--meta", type=str, default=None,
                        help="model_meta.json 路径 (默认自动查找)")
    parser.add_argument("--token-file", type=str, default=None,
                        help="单个 token 文件路径")
    parser.add_argument("--batch", action="store_true",
                        help="批量评估模式")
    parser.add_argument("--tokens-dir", type=str, default="data/tokens")
    parser.add_argument("--labels", type=str, default="data/rating/calibration_set.json",
                        help="标签文件 (批量模式用, 默认校准集)")
    parser.add_argument("--limit", type=int, default=0,
                        help="批量模式最多评估 N 个 (0=全部)")
    args = parser.parse_args()

    if not args.batch and not args.token_file:
        print("❌ 请指定 --token-file 或 --batch")
        sys.exit(1)

    inferer = RatingONNXInferer(args.model, args.meta)

    if args.batch:
        tokens_dir = PROJECT_ROOT / args.tokens_dir
        labels_file = str(PROJECT_ROOT / args.labels)
        if not Path(labels_file).exists():
            print(f"❌ 标签文件不存在: {labels_file}")
            sys.exit(1)
        batch_evaluate(inferer, tokens_dir, labels_file, args.limit)

    else:
        # 单文件推理
        import torch
        token_path = Path(args.token_file)
        if not token_path.exists():
            print(f"❌ 文件不存在: {token_path}")
            sys.exit(1)

        try:
            tokens = torch.load(token_path, map_location='cpu', weights_only=True).long().tolist()
        except Exception:
            tokens = torch.load(token_path, map_location='cpu').long().tolist()

        game_token_id = int(tokens[0]) if tokens else GAME_TOKEN_PHIRA
        game = game_token_to_name(game_token_id)

        rating_norm, rating_raw = inferer.infer(tokens, game_token_id)

        # 尝试从文件名解析标签对比
        label = parse_filename(token_path.name)

        print(f"\n{'='*60}")
        print(f"📋 推理结果")
        print(f"{'='*60}")
        print(f"   文件:     {token_path.name}")
        print(f"   游戏:     {game}")
        print(f"   归一化值: {rating_norm:.6f}")
        print(f"   原始定数: {rating_raw:.2f}")

        if label:
            print(f"   文件名标签: {label['level']:.2f} ({label['game']})")
            err = abs(rating_raw - label['level'])
            print(f"   误差:     {err:.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
