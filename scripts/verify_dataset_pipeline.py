# -*- coding: utf-8 -*-
"""数据管线验证脚本: 过滤 + 窗口化 + 音频对齐 (验证后删除)"""
import sys, json, random
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.token_dataset import TokenDataset
from src.data.audio_processor import AudioProcessor

TOKENS_DIR = "data/tokens"
MAX_SEQ = 4096
BOUNDARY = (4, 5, 6, 120)
OK = 0
FAIL = 0

def check(name, cond, detail=""):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")

# ============================================================
# 1. 过滤统计 (不加载音频, 快)
# ============================================================
print("=" * 60)
print("[1] 超长谱面过滤 (use_real_audio=False)")
ds = TokenDataset(
    tokens_dir=TOKENS_DIR,
    max_seq_len=MAX_SEQ,
    use_real_audio=False,
    window_mode="random",
    max_chart_tokens=200_000,
)
check("过滤后样本数 == 4530-151 = 4379", len(ds) == 4379, f"got {len(ds)}")
# 过滤阈值正确性: 剔除的正好是 151 张超长谱面, 保留的 <= 200k
over = sum(1 for v in ds._token_lengths.values() if v > 200_000)
check("剔除 151 张超长谱面", over == 151, f"dropped: {over}")
kept_over = sum(1 for p in ds.token_files if ds._token_lengths[p.name] > 200_000)
check("保留谱面无超长", kept_over == 0, f"kept over-threshold: {kept_over}")

# ============================================================
# 2. 随机窗口正确性
# ============================================================
print("=" * 60)
print("[2] 随机窗口: 连续性 + 边界吸附")
random.seed(42)
torch.manual_seed(42)

# 找一个长谱面 (> 4096) 反复取样验证
long_idx = [i for i, p in enumerate(ds.token_files) if ds._token_lengths[p.name] > MAX_SEQ]
check("存在长谱面", len(long_idx) > 0, f"{len(long_idx)} 张")
long_file = ds.token_files[long_idx[0]]
full = json.load(open(long_file, encoding='utf-8'))['tokens']
T = len(full)
print(f"  长谱面: {long_file.name} (T={T})")

windows = [long_idx[0]] * 8  # 同一文件多次取样
for k in range(8):
    s = ds[long_idx[0]]
    win = s['input_ids'].tolist()
    # 窗口内容必须等于源序列的连续切片
    start = None
    for i in range(T - MAX_SEQ + 1):
        if win == full[i:i + MAX_SEQ]:
            start = i
            break
    check(f"窗口{k}是源序列连续切片", start is not None,
          f"start={start}")
    if start is not None and start > 0:
        check(f"窗口{k}起点是边界token", full[start] in BOUNDARY,
              f"tok={full[start]}")
    # 短窗口全 PAD 的注意力掩码应为全 1 (中间窗口无 PAD)
    check(f"窗口{k}注意力掩码全1", bool(s['attention_mask'].all()),
          f"mask_sum={s['attention_mask'].sum().item()}")

# 短谱面: 补 PAD + 起点 0
short_idx = [i for i, p in enumerate(ds.token_files) if ds._token_lengths[p.name] < 500]
if short_idx:
    s = ds[short_idx[0]]
    full_short = json.load(open(ds.token_files[short_idx[0]], encoding='utf-8'))['tokens']
    check("短谱面头部与源一致", s['input_ids'][:len(full_short)].tolist() == full_short)
    check("短谱面尾部为 PAD", int(s['input_ids'][len(full_short)]) == 0)
    npad = int((s['attention_mask'] == 0).sum())
    check("短谱面掩码对齐 PAD", npad == MAX_SEQ - len(full_short), f"npad={npad}")

# ============================================================
# 3. 音频窗口对齐 (真实音频, 只测 1 个样本)
# ============================================================
print("=" * 60)
print("[3] 音频窗口对齐")
ds_real = TokenDataset(
    tokens_dir=TOKENS_DIR,
    max_seq_len=MAX_SEQ,
    use_real_audio=True,
    window_mode="head",  # head: 窗口固定为开头, frac=(0, 4096/T)
    max_chart_tokens=200_000,
)
# head 模式下 frac_start=0, 验证音频 == 整首特征的前 4096/T 段插值
sample_idx = long_idx[0]
s = ds_real[sample_idx]
tok_path = ds_real.token_files[sample_idx]
audio_path = ds_real._find_audio_file(tok_path)
check("找到音频", audio_path is not None and audio_path.exists(), str(audio_path))

proc = AudioProcessor(sample_rate=22050, n_mels=128)
full_feat = proc.process(str(audio_path), duration=180.0, target_steps=None)  # [F, 128] 归一化
print(f"  全曲特征: {tuple(full_feat.shape)}")
T = ds_real._token_lengths[tok_path.name]
frac_end = min(1.0, MAX_SEQ / T)
# 管线语义: 整首特征先插值到 max_seq_len, 再按窗口比例切片后插值回
resized_full = ds_real._resize_audio(full_feat, MAX_SEQ)
f1 = min(MAX_SEQ, max(1, int(round(frac_end * MAX_SEQ))))
expected = ds_real._resize_audio(resized_full[:f1], MAX_SEQ)
got = s['audio_features']
diff = (expected - got).abs().max().item()
check("head窗口音频 == 整首插值后按比例切片", diff < 1e-5, f"max_diff={diff:.2e}")
check("音频形状", tuple(got.shape) == (MAX_SEQ, 128), str(tuple(got.shape)))

# ============================================================
# 4. sliding 模式窗口统计
# ============================================================
print("=" * 60)
print("[4] sliding 模式")
ds_slide = TokenDataset(
    tokens_dir=TOKENS_DIR,
    max_seq_len=MAX_SEQ,
    use_real_audio=False,
    window_mode="sliding",
    window_stride=1024,
    max_chart_tokens=200_000,
)
n_windows = len(ds_slide)
print(f"  滑动窗口样本数: {n_windows:,}")
# 抽查一个长谱面的窗口序列
li = long_idx[0]
path = ds_slide.token_files[li]
T = ds_slide._token_lengths[path.name]
expected_starts = list(range(0, T - MAX_SEQ, 1024))
if (T - MAX_SEQ) % 1024 != 0:
    expected_starts.append(T - MAX_SEQ)
actual = [s for (i, s) in ds_slide._window_index if i == li]
check("窗口起点序列正确", actual == expected_starts,
      f"{len(actual)} windows, last={actual[-1] if actual else None}, T-MAX={T - MAX_SEQ}")

# 每张谱面窗口数 >= 1, 总和 == n_windows
per_chart = {}
for i, s in ds_slide._window_index:
    per_chart[i] = per_chart.get(i, 0) + 1
check("窗口总数 == 索引长度", sum(per_chart.values()) == n_windows)

# ============================================================
# 5. 汇总
# ============================================================
print("=" * 60)
orig_total = sum(ds._token_lengths.values())
kept_total = sum(v for v in ds._token_lengths.values() if v <= 200_000)
print(f"[STATS] 原始数据: 4530 张 / {orig_total:,} tokens")
print(f"[STATS] 过滤后:   {len(ds)} 张 / {kept_total:,} tokens "
      f"(剔除 {orig_total - kept_total:,} tokens = {(orig_total - kept_total)/orig_total*100:.1f}%)")
print(f"[STATS] random 模式: 每 epoch {len(ds)} 个窗口样本 (窗口位置每 epoch 随机变化)")
print(f"[STATS] sliding 模式: {n_windows:,} 个窗口样本 (stride=1024)")
print(f"[STATS] 理论信息量: 过滤前模型每次只能看到每谱开头 4096 token; "
      f"现在可覆盖全部 {kept_total:,} tokens")
print()
print(f"RESULT: {OK} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
