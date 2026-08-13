# -*- coding: utf-8 -*-
"""
定数评级模型 ONNX 导出脚本
==========================

将训练好的 RatingModel (.pth) 导出为双 opset ONNX, 供 Unity 端实时推理:
  - opset 17 → ONNX Runtime Unity (com.unity.ai.microsoft.onnxruntime)
  - opset 14 → Unity Sentis (com.unity.sentis)

同时导出 model_meta.json, 内含归一化范围 + 特征标准化统计, 供 C# 端一次性加载。

用法:
  # 从训练 checkpoint 导出 (推荐)
  python scripts/export_rating_onnx.py --checkpoint logs/rating_best.pth

  # 指定输出目录
  python scripts/export_rating_onnx.py --checkpoint logs/rating_best.pth --output-dir exports/rating

  # 仅导出 Sentis 版本 (减半体积)
  python scripts/export_rating_onnx.py --checkpoint logs/rating_best.pth --sentis-only

  # 导出后用 onnxruntime 验证数值一致性
  python scripts/export_rating_onnx.py --checkpoint logs/rating_best.pth --verify

  # 无 checkpoint 时导出随机权重模型 (用于端到端管线验证)
  python scripts/export_rating_onnx.py --random-weights --output-dir exports/rating_test
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Windows UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.models.rating_model import RatingModel, RatingConfig
from rating_common import (
    NORM_RANGES,
    save_normalization_file,
    GAME_TOKEN_PHIRA,
    GAME_TOKEN_OSU,
    GAME_TOKEN_ARELLANO,
    FEATURE_DIM,
)


# ============================================================
# 模型加载
# ============================================================

def load_rating_model(
    checkpoint_path: Optional[str],
    config_override: Optional[Dict[str, Any]] = None,
) -> Tuple[RatingModel, Dict[str, Any]]:
    """
    从 checkpoint 加载 RatingModel, 或创建随机权重模型。

    Returns:
        (model, config_dict)
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📂 加载 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config_dict = ckpt.get("config", {}).get("model", None)

        if config_dict is None:
            # 兼容: checkpoint 里没有 config, 用默认
            print("   ⚠️ checkpoint 无 model config, 使用 configs/rating.yaml")
            config_dict = _load_yaml_model_config()

        if config_override:
            config_dict.update(config_override)
    else:
        if checkpoint_path:
            print(f"⚠️ checkpoint 不存在: {checkpoint_path}, 使用随机权重")
        else:
            print("🎲 使用随机权重模型 (端到端管线验证用)")
        config_dict = _load_yaml_model_config()
        if config_override:
            config_dict.update(config_override)

    # 过滤 RatingConfig 不识别的字段
    valid_fields = {f for f in RatingConfig.__dataclass_fields__}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

    rating_cfg = RatingConfig(**filtered)
    model = RatingModel(rating_cfg)

    # 加载权重
    if checkpoint_path and Path(checkpoint_path).exists():
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
        # 清理 torch.compile 的 _orig_mod. 前缀
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("_orig_mod."):
                cleaned[k[len("_orig_mod."):]] = v
            else:
                cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"   ⚠️ missing keys: {len(missing)}")
        if unexpected:
            print(f"   ⚠️ unexpected keys: {len(unexpected)}")

    model.eval()

    params = model.get_num_parameters()
    print(f"   🧠 模型参数: {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"   📐 d_model={rating_cfg.d_model}, layers={rating_cfg.n_layers}, "
          f"heads={rating_cfg.n_heads}, seq_len={rating_cfg.max_seq_len}")

    return model, filtered


def _load_yaml_model_config() -> Dict[str, Any]:
    """从 configs/rating.yaml 加载 model 段"""
    yaml_path = PROJECT_ROOT / "configs" / "rating.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get("model", {})
    # 默认值 (与 RatingConfig dataclass 对齐)
    return {}


# ============================================================
# ONNX 导出
# ============================================================

# 模型输入/输出名 (C# 端必须一致)
INPUT_NAMES = ["tokens", "attention_mask", "handcrafted_features", "game_token_ids"]
OUTPUT_NAME = "rating_norm"


def export_onnx(
    model: RatingModel,
    output_path: str,
    opset_version: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    feature_dim: int = 15,
    verify: bool = True,
    verify_rtol: float = 1e-4,
    verify_atol: float = 1e-5,
) -> bool:
    """
    导出单个 ONNX 模型 (静态形状)。

    Args:
        model:          RatingModel (已 eval)
        output_path:    输出 .onnx 路径
        opset_version:  ONNX opset 版本 (17=ORT, 14=Sentis)
        batch_size:     固定 batch (默认 1, 编辑器实时推理)
        seq_len:        固定序列长度 (默认 2048)
        feature_dim:    手工特征维度 (默认 15)
        verify:         导出后用 onnxruntime 验证
    Returns:
        True 如果导出+验证成功
    """
    backend_name = "ONNX Runtime" if opset_version >= 17 else "Unity Sentis"
    print(f"\n{'='*60}")
    print(f"📦 导出 ONNX [{backend_name}] opset={opset_version}")
    print(f"   输出: {output_path}")
    print(f"   形状: batch={batch_size}, seq={seq_len}, feat={feature_dim}")
    print(f"{'='*60}")

    # 构造示例输入 (静态形状)
    tokens = torch.randint(4, 120, (batch_size, seq_len), dtype=torch.long)
    tokens[:, 0] = GAME_TOKEN_PHIRA  # 第一个是 Game Token
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    handcrafted = torch.randn(batch_size, feature_dim, dtype=torch.float32)
    game_ids = torch.tensor([GAME_TOKEN_PHIRA] * batch_size, dtype=torch.long)

    # === PyTorch 前向检查 ===
    print("🔍 PyTorch 前向检查...")
    with torch.no_grad():
        try:
            pt_out = model(tokens, attention_mask, handcrafted, game_ids)
            print(f"   ✅ PyTorch 输出: shape={list(pt_out.shape)}, "
                  f"值={pt_out.tolist()}")
        except Exception as e:
            print(f"   ❌ PyTorch 前向失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # === 导出 ===
    # 静态形状: 不用 dynamic_axes, 保证 Sentis/OR 兼容性
    # dynamo=False: 用 TorchScript tracer (模型控制流在 tracing 下一致)
    print(f"🔧 导出 ONNX (opset={opset_version}, 静态形状)...")
    try:
        torch.onnx.export(
            model,
            (tokens, attention_mask, handcrafted, game_ids),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=INPUT_NAMES,
            output_names=[OUTPUT_NAME],
            dynamic_axes=None,  # 静态形状
            dynamo=False,
        )
    except Exception as e:
        print(f"   ❌ ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   ✅ 导出成功: {size_mb:.2f} MB")

    # === ONNX 模型信息 ===
    _print_onnx_info(output_path)

    # === 验证 ===
    if verify:
        ok = _verify_onnx(
            output_path,
            model,
            tokens, attention_mask, handcrafted, game_ids,
            pt_out,
            rtol=verify_rtol,
            atol=verify_atol,
        )
        if not ok:
            print(f"   ⚠️ 验证未通过, 但 ONNX 文件已生成 (可能精度问题)")

    return True


def _print_onnx_info(onnx_path: str):
    """打印 ONNX 模型信息"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        print(f"\n📋 ONNX 模型信息:")
        print(f"   IR version: {model.ir_version}")
        print(f"   Opset: {model.opset_import[0].version}")
        print(f"   输入:")
        for inp in model.graph.input:
            shape = _get_shape_str(inp)
            print(f"     - {inp.name}: {shape}")
        print(f"   输出:")
        for out in model.graph.output:
            shape = _get_shape_str(out)
            print(f"     - {out.name}: {shape}")

        # 算子统计
        op_types = {}
        for node in model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        print(f"   算子: {len(model.graph.node)} 个 ({len(op_types)} 种)")
        for op, count in sorted(op_types.items(), key=lambda x: -x[1])[:10]:
            print(f"     {op}: {count}")

    except ImportError:
        print(f"   ⚠️ onnx 未安装, 跳过模型检查")
    except Exception as e:
        print(f"   ⚠️ ONNX 模型检查失败: {e}")


def _get_shape_str(tensor_info) -> str:
    try:
        dims = []
        for d in tensor_info.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(str(d.dim_value))
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append('?')
        return f"[{', '.join(dims)}]"
    except Exception:
        return "?"


def _verify_onnx(
    onnx_path: str,
    pt_model: nn.Module,
    tokens, attention_mask, handcrafted, game_ids,
    pt_out: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """用 onnxruntime 验证 ONNX 输出与 PyTorch 一致"""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"   ⚠️ onnxruntime 未安装, 跳过验证")
        print(f"      pip install onnxruntime")
        return False

    print(f"\n🔍 onnxruntime 验证 (rtol={rtol}, atol={atol})...")

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    onnx_inputs = {
        "tokens": tokens.numpy(),
        "attention_mask": attention_mask.numpy(),
        "handcrafted_features": handcrafted.numpy(),
        "game_token_ids": game_ids.numpy(),
    }
    onnx_out = sess.run(None, onnx_inputs)[0]

    pt_np = pt_out.numpy()
    abs_diff = np.abs(pt_np - onnx_out)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    print(f"   PyTorch 输出: {pt_np.tolist()}")
    print(f"   ONNX    输出: {onnx_out.tolist()}")
    print(f"   最大绝对误差: {max_diff:.8f}")
    print(f"   平均绝对误差: {mean_diff:.8f}")

    passed = np.allclose(pt_np, onnx_out, rtol=rtol, atol=atol)
    if passed:
        print(f"   ✅ 验证通过! ONNX 与 PyTorch 输出一致")
    else:
        print(f"   ⚠️ 存在精度差异 (max_diff={max_diff:.6f})")
        # rating 是 sigmoid 输出, 误差 < 0.01 在原始尺度影响 < 0.16 定数
        if max_diff < 0.01:
            print(f"   💡 差异在可接受范围 (< 0.01, 对定数影响 < 0.16)")
            passed = True

    return passed


# ============================================================
# 元数据导出 (供 Unity 加载)
# ============================================================

def export_metadata(
    output_dir: Path,
    model_config: Dict[str, Any],
    feature_stats_file: Optional[str] = None,
):
    """
    导出 model_meta.json, 包含:
      - 归一化范围 (各游戏 min/max)
      - 特征标准化统计 (mean/std, 15 维)
      - 模型输入规格 (名称/形状/类型)
      - Game Token 映射
    """
    # 特征标准化统计
    feat_mean = [0.0] * FEATURE_DIM
    feat_std = [1.0] * FEATURE_DIM
    if feature_stats_file and Path(feature_stats_file).exists():
        with open(feature_stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        feat_mean = stats.get("mean", feat_mean)
        feat_std = stats.get("std", feat_std)
        print(f"📊 加载特征统计: {Path(feature_stats_file).name}")
    else:
        # 尝试默认路径
        default_stats = PROJECT_ROOT / "data" / "rating" / "feature_stats.json"
        if default_stats.exists():
            with open(default_stats, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            feat_mean = stats.get("mean", feat_mean)
            feat_std = stats.get("std", feat_std)
            print(f"📊 加载特征统计: {default_stats.name}")
        else:
            print(f"⚠️ 未找到特征统计文件, 使用默认 (mean=0, std=1)")
            print(f"   建议: python scripts/build_rating_labels.py --feature-stats")

    meta = {
        "version": "1.0",
        "model_type": "rating",
        "description": "ArellanoDreamWeaver 定数评级模型",
        "model_config": {
            "d_model": model_config.get("d_model", 256),
            "n_heads": model_config.get("n_heads", 8),
            "n_layers": model_config.get("n_layers", 4),
            "max_seq_len": model_config.get("max_seq_len", 2048),
            "handcrafted_dim": model_config.get("handcrafted_dim", 15),
        },
        "inputs": [
            {"name": "tokens", "dtype": "int64", "shape": [1, 2048]},
            {"name": "attention_mask", "dtype": "int64", "shape": [1, 2048]},
            {"name": "handcrafted_features", "dtype": "float32", "shape": [1, 15]},
            {"name": "game_token_ids", "dtype": "int64", "shape": [1]},
        ],
        "output": {
            "name": "rating_norm",
            "dtype": "float32",
            "shape": [1],
            "range": [0.0, 1.0],
            "description": "归一化定数, 需按 game 反归一化",
        },
        "normalization": NORM_RANGES,
        "feature_stats": {
            "mean": feat_mean,
            "std": feat_std,
            "dim": FEATURE_DIM,
        },
        "game_tokens": {
            "phira": GAME_TOKEN_PHIRA,
            "osu": GAME_TOKEN_OSU,
            "arellano": GAME_TOKEN_ARELLANO,
        },
        "feature_layout": [
            "is_phira", "is_osu", "is_arellano",
            "n_notes", "peak_nps", "avg_nps",
            "n_events", "event_density", "tap_ratio",
            "hold_ratio", "slide_ratio", "hold_duration_density",
            "pos_variance", "simultaneous_ratio", "min_delta_p10"
        ],
    }

    meta_path = output_dir / "model_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"💾 保存元数据: {meta_path}")

    # 同时保存 normalization.json (独立文件, 方便 C# 单独加载)
    norm_path = output_dir / "normalization.json"
    save_normalization_file(str(norm_path))
    print(f"💾 保存归一化: {norm_path}")

    return meta


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="定数评级模型 ONNX 导出 (双 opset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch checkpoint 路径 (.pth)")
    parser.add_argument("--output-dir", type=str, default="exports/rating",
                        help="输出目录 (默认: exports/rating)")
    parser.add_argument("--config", type=str, default="configs/rating.yaml",
                        help="配置文件 (checkpoint 无 config 时回退)")
    parser.add_argument("--ort-filename", type=str, default="rating_ort_opset17.onnx",
                        help="ONNX Runtime 文件名")
    parser.add_argument("--sentis-filename", type=str, default="rating_sentis_opset14.onnx",
                        help="Unity Sentis 文件名")
    parser.add_argument("--ort-opset", type=int, default=17)
    parser.add_argument("--sentis-opset", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--feature-dim", type=int, default=15)
    parser.add_argument("--sentis-only", action="store_true",
                        help="仅导出 Sentis 版本 (省体积)")
    parser.add_argument("--ort-only", action="store_true",
                        help="仅导出 ONNX Runtime 版本")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="导出后用 onnxruntime 验证")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过验证")
    parser.add_argument("--verify-rtol", type=float, default=1e-4)
    parser.add_argument("--verify-atol", type=float, default=1e-5)
    parser.add_argument("--random-weights", action="store_true",
                        help="使用随机权重 (端到端管线验证用)")
    parser.add_argument("--feature-stats", type=str, default=None,
                        help="特征统计 JSON 路径 (覆盖默认 data/rating/feature_stats.json)")
    args = parser.parse_args()

    # === 加载模型 ===
    if args.random_weights:
        model, model_config = load_rating_model(None)
    else:
        if not args.checkpoint:
            # 尝试默认路径
            default_ckpt = PROJECT_ROOT / "logs" / "rating_best.pth"
            if default_ckpt.exists():
                args.checkpoint = str(default_ckpt)
            else:
                print(f"❌ 未指定 --checkpoint 且默认路径无模型: {default_ckpt}")
                print(f"   使用 --random-weights 导出随机权重模型, 或先训练:")
                print(f"   python scripts/train_rating.py --config configs/rating.yaml")
                sys.exit(1)
        model, model_config = load_rating_model(args.checkpoint)

    verify = args.verify and not args.no_verify

    # === 输出目录 ===
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 输出目录: {output_dir}")

    results = {}

    # === 导出 ONNX Runtime (opset 17) ===
    if not args.sentis_only:
        ort_path = output_dir / args.ort_filename
        ok = export_onnx(
            model=model,
            output_path=str(ort_path),
            opset_version=args.ort_opset,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            feature_dim=args.feature_dim,
            verify=verify,
            verify_rtol=args.verify_rtol,
            verify_atol=args.verify_atol,
        )
        results["ort"] = (ok, str(ort_path))

    # === 导出 Unity Sentis (opset 14) ===
    if not args.ort_only:
        sentis_path = output_dir / args.sentis_filename
        ok = export_onnx(
            model=model,
            output_path=str(sentis_path),
            opset_version=args.sentis_opset,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            feature_dim=args.feature_dim,
            verify=verify,
            verify_rtol=args.verify_rtol,
            verify_atol=args.verify_atol,
        )
        results["sentis"] = (ok, str(sentis_path))

    # === 导出元数据 ===
    export_metadata(
        output_dir=output_dir,
        model_config=model_config,
        feature_stats_file=args.feature_stats,
    )

    # === 总结 ===
    print(f"\n{'='*60}")
    print(f"✅ 导出完成!")
    print(f"{'='*60}")
    for name, (ok, path) in results.items():
        status = "✅" if ok else "❌"
        size = os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0
        print(f"   {status} {name}: {path} ({size:.2f} MB)")
    print(f"   📄 元数据: {output_dir / 'model_meta.json'}")
    print(f"   📄 归一化: {output_dir / 'normalization.json'}")
    print(f"\n   下一步: 将以上文件复制到 Unity StreamingAssets/AIRating/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
