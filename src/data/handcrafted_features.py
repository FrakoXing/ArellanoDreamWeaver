# -*- coding: utf-8 -*-
"""
手工特征提取器
==============

从 Token 序列实时计算 15 维手工特征，作为评级模型的辅助输入通道。

设计约束:
  1. 完全不依赖 ChartData / 音频 — 只用 Token 序列
  2. Python 与 Unity C# (TokenFeatureExtractor.cs) 必须逐元素数值一致
  3. 复用 unified_tokenizer.py 的 ChartTokenizer.decode_with_events 解析 token

15 维特征向量布局:
  [0]  is_phira        Game one-hot
  [1]  is_osu
  [2]  is_arellano
  [3]  n_notes         总音符数
  [4]  peak_nps        1 秒滑窗峰值 NPS
  [5]  avg_nps         平均 NPS
  [6]  n_events        总事件数
  [7]  event_density   事件/拍
  [8]  tap_ratio       Tap 占比
  [9]  hold_ratio      Hold 占比
  [10] slide_ratio     Slide 占比
  [11] hold_duration_density  Hold 总时长/拍
  [12] pos_variance    X 位置方差
  [13] simultaneous_ratio     同拍多音符比例
  [14] min_delta_p10   时间间隔 10 分位 (瞬时高密度)
"""

import sys
from pathlib import Path
from typing import List

import numpy as np

# 添加项目根目录 + scripts 目录到路径 (scripts 无 __init__.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from unified_tokenizer import (
    ChartTokenizer,
    GAME_PHIRA,
    GAME_OSU,
    GAME_ARELLANO,
)

FEATURE_DIM = 15

# 复用单个 ChartTokenizer 实例 (无状态, 线程安全用于推理)
_TOKENIZER = ChartTokenizer()

# BPM 估算: Token 内无 BPM 信息, 按 150 估算 NPS
# (Phigros/osu 主流谱面 BPM 集中在 120-200, 150 误差可接受;
#  模型主要靠 Token 注意力学真实密度, 手工特征仅辅助先验)
_ESTIMATED_BPM = 150.0
_BEATS_PER_SEC = _ESTIMATED_BPM / 60.0  # 2.5 beats/sec


def compute_features(tokens: List[int], game_token_id: int) -> np.ndarray:
    """
    从 Token 序列计算 15 维手工特征。

    Args:
        tokens:          Token ID 列表 (含 GAME_TOKEN + SOS + ... + EOS)
        game_token_id:   Game Token ID (116=phira, 117=osu, 118=arellano)
                         注意: 应与 tokens[0] 一致, 此处单独传入是为了
                         ONNX 推理时显式输入 (避免模型内部依赖 tokens[0])

    Returns:
        np.ndarray, shape=[15], dtype=float32
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # === Game one-hot ===
    feat[0] = 1.0 if game_token_id == GAME_PHIRA else 0.0
    feat[1] = 1.0 if game_token_id == GAME_OSU else 0.0
    feat[2] = 1.0 if game_token_id == GAME_ARELLANO else 0.0

    # === Token → notes/events (复用现有解码器) ===
    try:
        notes, events = _TOKENIZER.decode_with_events(list(tokens))
    except Exception:
        return feat  # 解码失败返回零向量

    n_notes = len(notes)
    n_events = len(events)

    feat[3] = float(n_notes)
    feat[6] = float(n_events)

    if n_notes == 0:
        return feat

    # 提取数组
    times = np.array([n.time for n in notes], dtype=np.float64)       # 拍
    types = np.array([n.type for n in notes], dtype=np.int64)         # 0/1/2
    xs = np.array([n.position_x for n in notes], dtype=np.float64)    # -1~1
    holds = np.array([n.hold_time for n in notes], dtype=np.float64)  # 拍

    # 时间跨度 (拍)
    if len(times) >= 2:
        total_beats = float(times.max() - times.min())
    else:
        total_beats = 0.0
    total_beats = max(total_beats, 1.0)  # 避免除零

    total_seconds = total_beats / _BEATS_PER_SEC

    # === 密度特征 ===
    sorted_times = np.sort(times)

    # 峰值 NPS: 1 秒滑窗内最多音符数
    peak_nps = 0.0
    if len(sorted_times) >= 2:
        window_beats = _BEATS_PER_SEC  # 1 秒 = 2.5 拍
        j = 0
        for i in range(len(sorted_times)):
            while sorted_times[i] - sorted_times[j] > window_beats:
                j += 1
            count = i - j + 1
            if count > peak_nps:
                peak_nps = float(count)
    else:
        peak_nps = 1.0

    feat[4] = peak_nps

    # 平均 NPS
    avg_nps = n_notes / total_seconds if total_seconds > 0 else 0.0
    feat[5] = float(avg_nps)

    # 事件密度 (事件/拍)
    event_density = n_events / total_beats if total_beats > 0 else 0.0
    feat[7] = float(event_density)

    # === 类型比例 ===
    tap_count = int((types == 0).sum())
    hold_count = int((types == 1).sum())
    slide_count = int((types == 2).sum())

    feat[8] = tap_count / n_notes
    feat[9] = hold_count / n_notes
    feat[10] = slide_count / n_notes

    # Hold 总时长密度 (拍/拍)
    hold_total = float(holds[types == 1].sum()) if hold_count > 0 else 0.0
    feat[11] = hold_total / total_beats if total_beats > 0 else 0.0

    # === 时空多样性 ===
    # 位置方差
    pos_variance = float(xs.var()) if n_notes > 1 else 0.0
    feat[12] = pos_variance

    # 同拍多音符比例 (1/4 拍精度内视为同时)
    quantized = np.round(times * 4) / 4  # 量化到 1/4 拍
    unique_times, counts = np.unique(quantized, return_counts=True)
    if len(unique_times) > 0:
        simultaneous_ratio = float((counts > 1).sum()) / len(unique_times)
    else:
        simultaneous_ratio = 0.0
    feat[13] = simultaneous_ratio

    # 时间间隔 10 分位 (反映瞬时高密度, 值越小密度越高)
    if len(sorted_times) >= 2:
        deltas = np.diff(sorted_times)
        # deltas 可能有 0 (同时音符), 过滤掉 0 再算分位
        nonzero_deltas = deltas[deltas > 1e-9]
        if len(nonzero_deltas) > 0:
            min_delta_p10 = float(np.percentile(nonzero_deltas, 10))
        else:
            min_delta_p10 = 0.0
    else:
        min_delta_p10 = 0.0
    feat[14] = min_delta_p10

    return feat


def compute_features_torch(tokens_tensor, game_token_id: int) -> np.ndarray:
    """
    便捷重载: 接受 torch.LongTensor, 内部转 list。
    """
    return compute_features(tokens_tensor.tolist(), game_token_id)


# ============================================================
# 特征标准化统计 (训练集均值/方差, 供 Python/C# 共享)
# ============================================================

def compute_feature_stats(token_files, tokens_dir) -> dict:
    """
    遍历有标签 token 文件, 计算 15 维特征的均值/方差。

    Args:
        token_files: 文件名列表
        tokens_dir: token 文件目录

    Returns:
        {"mean": [...15], "std": [...15], "count": N}
    """
    import torch
    feats = []
    tokens_dir = Path(tokens_dir)
    for fname in token_files:
        fpath = tokens_dir / fname
        if not fpath.exists():
            continue
        try:
            tokens = torch.load(fpath, map_location='cpu').long().tolist()
            game_id = int(tokens[0]) if len(tokens) > 0 else GAME_PHIRA
            f = compute_features(tokens, game_id)
            feats.append(f)
        except Exception:
            continue

    if not feats:
        return {"mean": [0.0] * FEATURE_DIM, "std": [1.0] * FEATURE_DIM, "count": 0}

    arr = np.stack(feats)  # [N, 15]
    mean = arr.mean(axis=0).tolist()
    std = arr.std(axis=0).tolist()
    # 防止 std=0
    std = [s if s > 1e-6 else 1.0 for s in std]
    return {"mean": mean, "std": std, "count": len(feats)}


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("handcrafted_features 自测")
    print("=" * 60)

    # 构造一个简单 token 序列: GAME_PHIRA + SOS + 2个TAP + EOS
    # TAP@0/x0/above, TAP@0.25拍/x0.5/above
    from unified_tokenizer import (
        TYPE_TAP, NOTE_ABOVE, SOS, EOS,
        TIME_OFFSET_START, POS_X_START,
    )
    tokens = [
        GAME_PHIRA,                    # 116
        SOS,                           # 1
        TYPE_TAP,                      # 4
        TIME_OFFSET_START + 1,         # 8 = 0.25拍
        POS_X_START + 5,               # 76 = x=0.25... 实际 (5/8)*2-1=0.25
        NOTE_ABOVE,                    # 112
        TYPE_TAP,                      # 4
        TIME_OFFSET_START + 1,         # 0.25拍
        POS_X_START + 6,               # 77 = x=0.5
        NOTE_ABOVE,                    # 112
        EOS,                           # 2
    ]

    feat = compute_features(tokens, GAME_PHIRA)
    print(f"  shape: {feat.shape}")
    print(f"  values: {feat}")
    print(f"  is_phira: {feat[0]} (期望 1.0)")
    print(f"  n_notes:  {feat[3]} (期望 2.0)")
    print(f"  n_events: {feat[6]} (期望 0.0)")
    print(f"  tap_ratio: {feat[8]} (期望 1.0)")
    print(f"  hold_ratio: {feat[9]} (期望 0.0)")

    assert feat.shape == (15,), f"shape 错误: {feat.shape}"
    assert feat[0] == 1.0, "is_phira 应为 1.0"
    assert feat[3] == 2.0, "n_notes 应为 2.0"
    assert feat[8] == 1.0, "tap_ratio 应为 1.0"
    print("\n  ✓ 基本特征正确")
