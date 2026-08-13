# -*- coding: utf-8 -*-
"""
定数评级 AI 单元测试
====================

测试范围:
  1. 标签解析 (文件名 → 定数 + 归一化)
  2. 手工特征提取 (Token → 15 维特征)
  3. RatingModel 前向传播
  4. ONNX 导出 + 推理一致性
  5. Token 编码/解码往返

运行:
  python tests/test_rating.py
  python -m pytest tests/test_rating.py -v
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch

from rating_common import (
    parse_filename,
    normalize_level,
    denormalize_level,
    compute_features,
    FEATURE_DIM,
    GAME_TOKEN_PHIRA,
    GAME_TOKEN_OSU,
    GAME_TOKEN_ARELLANO,
    NORM_RANGES,
)


# ============================================================
# 测试工具
# ============================================================

PASS = 0
FAIL = 0

def assert_eq(name, actual, expected):
    global PASS, FAIL
    if actual == expected:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}: 期望 {expected}, 实际 {actual}")

def assert_close(name, actual, expected, tol=1e-6):
    global PASS, FAIL
    if abs(actual - expected) < tol:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}: 期望 {expected}, 实际 {actual} (误差 {abs(actual-expected):.6f})")

def assert_true(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}: {detail}")


# ============================================================
# 1. 标签解析测试
# ============================================================

def test_label_parser():
    print("\n📋 测试 1: 标签解析")
    print("-" * 50)

    # Phigros 格式
    r = parse_filename("10018_IN Lv.12_别让我担心_小鱼仔娱乐.pt")
    assert_true("Phigros IN Lv.12 解析", r is not None)
    if r:
        assert_eq("  game", r["game"], "phira")
        assert_close("  level", r["level"], 12.0)
        assert_close("  normalized", r["normalized"], 0.75)  # 12/16

    # osu!mania 格式
    r = parse_filename("10100_4K LV.25_S.A.T.E.L.L.I.T.E_小子.pt")
    assert_true("osu 4K LV.25 解析", r is not None)
    if r:
        assert_eq("  game", r["game"], "osu")
        assert_eq("  osu_keys", r["osu_keys"], "4K")
        assert_close("  level", r["level"], 25.0)
        assert_close("  normalized", r["normalized"], 0.5)  # 25/50

    # 脏数据过滤
    r = parse_filename("12391_SP Lv.114514_xxx.pt")
    assert_true("脏数据 114514 过滤", r is None)

    r = parse_filename("10046_超人_音弾超人_sJDeviko.pt")
    assert_true("无档位信息过滤", r is None)

    # 边界值
    r = parse_filename("10061_AT  Lv.16_Poppy_Fyato.pt")
    assert_true("AT Lv.16 (上限) 解析", r is not None)
    if r:
        assert_close("  normalized (上限)", r["normalized"], 1.0)

    # 反归一化
    assert_close("phira 反归一化 0.5", denormalize_level(0.5, "phira"), 8.0)
    assert_close("osu 反归一化 0.5", denormalize_level(0.5, "osu"), 25.0)
    assert_close("arellano 反归一化 0.75", denormalize_level(0.75, "arellano"), 12.0)


# ============================================================
# 2. 手工特征测试
# ============================================================

def test_handcrafted_features():
    print("\n📋 测试 2: 手工特征提取")
    print("-" * 50)

    # 构造简单 token 序列 (Arellano, 2 个 TAP)
    from unified_tokenizer import (
        GAME_ARELLANO, SOS, EOS, TYPE_TAP, NOTE_ABOVE,
        TIME_OFFSET_START, POS_X_START,
    )

    tokens = [
        GAME_ARELLANO,                          # 118
        SOS,                                     # 1
        TYPE_TAP,                                # 4
        TIME_OFFSET_START + 1,                  # 8 = 0.25 拍
        POS_X_START + 5,                        # 76 = x=0.25
        NOTE_ABOVE,                             # 112
        TYPE_TAP,                                # 4
        TIME_OFFSET_START + 1,                  # 0.25 拍
        POS_X_START + 6,                        # 77 = x=0.5
        NOTE_ABOVE,                             # 112
        EOS,                                    # 2
    ]

    feat = compute_features(tokens, GAME_TOKEN_ARELLANO)

    assert_eq("特征维度", len(feat), FEATURE_DIM)
    assert_close("is_phira", feat[0], 0.0)
    assert_close("is_osu", feat[1], 0.0)
    assert_close("is_arellano", feat[2], 1.0)
    assert_close("n_notes", feat[3], 2.0)
    assert_close("n_events", feat[6], 0.0)  # Arellano 无事件
    assert_close("tap_ratio", feat[8], 1.0)
    assert_close("hold_ratio", feat[9], 0.0)
    assert_close("slide_ratio", feat[10], 0.0)

    # 空序列
    empty_tokens = [GAME_ARELLANO, SOS, EOS]
    feat_empty = compute_features(empty_tokens, GAME_ARELLANO)
    assert_close("空序列 n_notes", feat_empty[3], 0.0)


# ============================================================
# 3. 模型前向测试
# ============================================================

def test_model_forward():
    print("\n📋 测试 3: RatingModel 前向传播")
    print("-" * 50)

    from src.models.rating_model import RatingModel, RatingConfig

    config = RatingConfig(
        vocab_size=256, d_model=128, n_heads=4, n_layers=2,
        d_ff=512, max_seq_len=256, dropout=0.0, handcrafted_dim=15,
    )
    model = RatingModel(config)
    model.eval()

    params = model.get_num_parameters()
    assert_true("参数量 > 0", params["total"] > 0, f"total={params['total']}")

    B, T = 2, 256
    tokens = torch.randint(4, 120, (B, T), dtype=torch.long)
    tokens[:, 0] = GAME_TOKEN_ARELLANO
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, 128:] = 0  # 第二个样本后半部分 pad
    feat = torch.randn(B, 15, dtype=torch.float32)
    game_ids = torch.tensor([GAME_TOKEN_ARELLANO, GAME_TOKEN_PHIRA], dtype=torch.long)

    with torch.no_grad():
        out = model(tokens, mask, feat, game_ids)

    assert_eq("输出 shape", tuple(out.shape), (B,))
    assert_true("输出在 (0,1) 范围", (out > 0).all() and (out < 1).all(),
                f"range=[{out.min():.4f}, {out.max():.4f}]")

    # 不同输入产生不同输出
    # 注: 常数偏移会被 LayerNorm 消除, 必须用随机扰动改变特征相对模式
    with torch.no_grad():
        out2 = model(tokens, mask, torch.randn_like(feat), game_ids)
    assert_true("不同特征 → 不同输出 (特征通路连通)",
                not torch.equal(out, out2),
                f"out={out.tolist()}, out2={out2.tolist()}")


# ============================================================
# 4. ONNX 导出 + 推理一致性测试
# ============================================================

def test_onnx_export():
    print("\n📋 测试 4: ONNX 导出 + 推理一致性")
    print("-" * 50)

    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠️ onnxruntime 未安装, 跳过")
        return

    from src.models.rating_model import RatingModel, RatingConfig

    config = RatingConfig(
        vocab_size=256, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, max_seq_len=128, dropout=0.0, handcrafted_dim=15,
    )
    model = RatingModel(config)
    model.eval()

    # 固定输入
    B, T = 1, 128
    torch.manual_seed(42)
    tokens = torch.randint(4, 120, (B, T), dtype=torch.long)
    tokens[:, 0] = GAME_TOKEN_PHIRA
    mask = torch.ones(B, T, dtype=torch.long)
    feat = torch.randn(B, 15, dtype=torch.float32)
    game_ids = torch.tensor([GAME_TOKEN_PHIRA], dtype=torch.long)

    with torch.no_grad():
        pt_out = model(tokens, mask, feat, game_ids)

    # 导出
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        onnx_path = tmp.name

    try:
        torch.onnx.export(
            model,
            (tokens, mask, feat, game_ids),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["tokens", "attention_mask", "handcrafted_features", "game_token_ids"],
            output_names=["rating_norm"],
            dynamic_axes=None,
            dynamo=False,
        )

        # ONNX 推理
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        onnx_out = sess.run(None, {
            "tokens": tokens.numpy(),
            "attention_mask": mask.numpy(),
            "handcrafted_features": feat.numpy(),
            "game_token_ids": game_ids.numpy(),
        })[0]

        max_diff = abs(float(pt_out[0]) - float(onnx_out[0]))
        assert_true("ONNX 与 PyTorch 一致", max_diff < 1e-4,
                    f"max_diff={max_diff:.6f}")
    finally:
        os.unlink(onnx_path)


# ============================================================
# 5. Token 编码/解码往返测试
# ============================================================

def test_token_roundtrip():
    print("\n📋 测试 5: Token 编码/解码往返")
    print("-" * 50)

    from unified_tokenizer import ChartTokenizer, UniNote

    tk = ChartTokenizer(game="arellano")

    # 构造测试音符
    notes = [
        UniNote(type=0, time=0.0, hold_time=0.0, position_x=0.0, is_above=True),
        UniNote(type=1, time=1.0, hold_time=2.0, position_x=0.5, is_above=True),
        UniNote(type=2, time=3.0, hold_time=1.0, position_x=-0.5, is_above=False),
    ]

    # 编码
    tokens = tk.encode(notes)
    assert_true("编码包含 GAME_ARELLANO", tokens[0] == GAME_TOKEN_ARELLANO)
    assert_true("编码包含 SOS", tokens[1] == 1)
    assert_true("编码包含 EOS", tokens[-1] == 2)

    # 解码
    decoded = tk.decode(tokens)
    assert_eq("解码音符数", len(decoded), len(notes))

    if len(decoded) == len(notes):
        assert_close("音符 0 time", decoded[0].time, 0.0)
        assert_close("音符 1 time", decoded[1].time, 1.0)
        assert_close("音符 2 time", decoded[2].time, 3.0)
        assert_eq("音符 0 type", decoded[0].type, 0)
        assert_eq("音符 1 type", decoded[1].type, 1)
        assert_eq("音符 2 type", decoded[2].type, 2)


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("🧪 定数评级 AI 单元测试")
    print("=" * 60)

    test_label_parser()
    test_handcrafted_features()
    test_model_forward()
    test_onnx_export()
    test_token_roundtrip()

    print(f"\n{'='*60}")
    print(f"📊 测试结果: {PASS} 通过, {FAIL} 失败")
    print(f"{'='*60}")

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
