# -*- coding: utf-8 -*-
"""
定数标签构建脚本
================

扫描 data/tokens/*.pt, 从文件名解析定数标签, 生成:
  - data/rating/labels.json          有标签样本字典 (~6300 条)
  - data/rating/normalization.json   各游戏归一化范围
  - data/rating/calibration_set.json 校准集 (~200 条, 不参与训练)
  - data/rating/feature_stats.json   手工特征均值/方差 (可选, --feature-stats)

用法:
  python scripts/build_rating_labels.py
  python scripts/build_rating_labels.py --tokens-dir data/tokens --output-dir data/rating
  python scripts/build_rating_labels.py --feature-stats   # 同时算特征统计 (较慢)
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Windows 控制台 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 绕过 src/data/__init__.py (避免 librosa 依赖), 统一用 rating_common
from rating_common import parse_filename, NORM_RANGES, save_normalization_file


def build_labels(tokens_dir: Path, output_dir: Path,
                 calibration_ratio: float = 0.03,
                 compute_feature_stats: bool = False,
                 seed: int = 42):
    """
    扫描 token 文件, 解析标签, 输出全部产物。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 1. 扫描 token 文件 ===
    token_files = sorted(tokens_dir.glob("*.pt"))
    # 兼容 .json 格式
    token_files.extend(sorted(tokens_dir.glob("*.json")))
    token_files = sorted(set(token_files))

    print(f"📥 扫描 token 目录: {tokens_dir}")
    print(f"   共 {len(token_files)} 个文件")
    print(f"   输出目录: {output_dir}")
    print("=" * 60)

    # === 2. 解析每个文件名 ===
    labels = {}
    unlabeled = []
    parse_errors = []

    for fpath in token_files:
        fname = fpath.name
        try:
            result = parse_filename(fname)
        except Exception as e:
            parse_errors.append((fname, str(e)))
            result = None

        if result is not None:
            labels[fname] = result
        else:
            unlabeled.append(fname)

    total = len(token_files)
    labeled_count = len(labels)
    unlabeled_count = len(unlabeled)

    # === 3. 统计 ===
    by_game = defaultdict(int)
    by_bucket = defaultdict(lambda: defaultdict(int))
    for label in labels.values():
        game = label["game"]
        by_game[game] += 1
        level = label["level"]
        # 按定数分桶统计
        if game == "phira":
            bucket = int(level // 2) * 2  # 0-1, 2-3, 4-5, ...
            bucket_key = f"{bucket}-{bucket+1}"
        else:
            bucket = int(level // 10) * 10
            bucket_key = f"{bucket}-{bucket+9}"
        by_bucket[game][bucket_key] += 1

    print(f"📊 标签解析结果:")
    print(f"   有标签:   {labeled_count} ({labeled_count/total*100:.1f}%)")
    print(f"   无标签:   {unlabeled_count} ({unlabeled_count/total*100:.1f}%)")
    if parse_errors:
        print(f"   解析错误: {len(parse_errors)}")
    print(f"\n   按游戏分布:")
    for game, count in sorted(by_game.items(), key=lambda x: -x[1]):
        print(f"     {game:10s}: {count}")

    print(f"\n   按定数分桶 (Phigros):")
    for bucket_key in sorted(by_bucket.get("phira", {}).keys(), key=lambda x: int(x.split('-')[0])):
        count = by_bucket["phira"][bucket_key]
        bar = "█" * min(count // 30, 40)
        print(f"     Lv.{bucket_key:6s}: {count:5d} {bar}")

    if by_bucket.get("osu"):
        print(f"\n   按定数分桶 (osu!mania):")
        for bucket_key in sorted(by_bucket["osu"].keys(), key=lambda x: int(x.split('-')[0])):
            count = by_bucket["osu"][bucket_key]
            bar = "█" * min(count // 5, 40)
            print(f"     Lv.{bucket_key:6s}: {count:5d} {bar}")

    # === 4. 抽样校准集 (按游戏+难度分桶, 不参与训练) ===
    import random
    random.seed(seed)

    # 按 (game, bucket) 分组
    bucket_groups = defaultdict(list)
    for fname, label in labels.items():
        game = label["game"]
        level = label["level"]
        if game == "phira":
            bucket = int(level // 2)
        else:
            bucket = int(level // 10)
        bucket_groups[(game, bucket)].append(fname)

    calibration_files = set()
    for key, fnames in bucket_groups.items():
        n_sample = max(1, int(len(fnames) * calibration_ratio))
        sampled = random.sample(fnames, min(n_sample, len(fnames)))
        calibration_files.update(sampled)

    # 校准集从训练标签中移除
    calibration_labels = {f: labels[f] for f in calibration_files if f in labels}
    train_labels = {f: l for f, l in labels.items() if f not in calibration_files}

    print(f"\n🎯 校准集抽样: {len(calibration_labels)} 条 (不参与训练)")
    print(f"   训练标签:   {len(train_labels)} 条")

    # === 5. 保存 labels.json (训练用, 已排除校准集) ===
    labels_data = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "total_files": total,
            "labeled_files": labeled_count,
            "train_files": len(train_labels),
            "calibration_files": len(calibration_labels),
            "unlabeled_files": unlabeled_count,
            "by_game": dict(by_game),
            "by_bucket": {g: dict(b) for g, b in by_bucket.items()},
        },
        "labels": train_labels,
        "unlabeled": unlabeled,
    }
    labels_path = output_dir / "labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 保存: {labels_path}")

    # === 6. 保存 calibration_set.json ===
    cal_path = output_dir / "calibration_set.json"
    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "1.0",
            "count": len(calibration_labels),
            "labels": calibration_labels,
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 保存: {cal_path}")

    # === 7. 保存 normalization.json ===
    norm_path = output_dir / "normalization.json"
    save_normalization_file(str(norm_path))
    print(f"💾 保存: {norm_path}")

    # === 8. 可选: 计算特征统计 ===
    if compute_feature_stats:
        print(f"\n⚙️  计算手工特征统计 (遍历 {len(train_labels)} 文件, 较慢)...")
        # compute_features 已从 rating_common 导入
        import torch
        import numpy as np

        feats = []
        for i, fname in enumerate(train_labels.keys()):
            if i % 500 == 0:
                print(f"   进度: {i}/{len(train_labels)}")
            fpath = tokens_dir / fname
            if not fpath.exists():
                continue
            try:
                tokens = torch.load(fpath, map_location='cpu', weights_only=True).long().tolist()
                game_id = int(tokens[0]) if tokens else 116
                f = compute_features(tokens, game_id)
                feats.append(f)
            except Exception:
                continue

        if feats:
            arr = np.stack(feats)
            mean = arr.mean(axis=0).tolist()
            std = arr.std(axis=0).tolist()
            std = [s if s > 1e-6 else 1.0 for s in std]
            stats = {"mean": mean, "std": std, "count": len(feats)}
            stats_path = output_dir / "feature_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"💾 保存: {stats_path} (count={len(feats)})")

    print(f"\n{'='*60}")
    print(f"✅ 标签构建完成!")
    print(f"   训练标签: {len(train_labels)}")
    print(f"   校准集:   {len(calibration_labels)}")
    print(f"   无标签:   {len(unlabeled)} (供伪标签迭代用)")
    print(f"{'='*60}")

    return labels_data


def main():
    parser = argparse.ArgumentParser(
        description="从 token 文件名构建定数标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tokens-dir", type=str, default="data/tokens",
                        help="Token 文件目录 (默认: data/tokens)")
    parser.add_argument("--output-dir", type=str, default="data/rating",
                        help="输出目录 (默认: data/rating)")
    parser.add_argument("--calibration-ratio", type=float, default=0.03,
                        help="校准集抽样比例 (默认: 0.03 ≈ 200 条)")
    parser.add_argument("--feature-stats", action="store_true",
                        help="同时计算手工特征均值/方差 (较慢)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokens_dir = (PROJECT_ROOT / args.tokens_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    if not tokens_dir.exists():
        print(f"❌ Token 目录不存在: {tokens_dir}")
        sys.exit(1)

    build_labels(
        tokens_dir=tokens_dir,
        output_dir=output_dir,
        calibration_ratio=args.calibration_ratio,
        compute_feature_stats=args.feature_stats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
