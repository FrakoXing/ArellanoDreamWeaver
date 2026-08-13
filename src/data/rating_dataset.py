# -*- coding: utf-8 -*-
"""
定数评级数据集 (RatingDataset)
==============================

加载 Token + 定数标签 + 实时计算手工特征。

与 TokenDataset 的区别:
  - 不依赖音频 (评级只需谱面 Token)
  - 加载定数标签 (归一化值)
  - 实时计算 15 维手工特征并标准化
  - 支持伪标签降权 (v2 半监督训练)

label_source:
  - "labels": 用 labels.json 真标签
  - "pseudo": 用 labels_pseudo.json 伪标签
  - "any":    优先真标签, 缺失则用伪标签 (v2 训练用)
"""

import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 直接从文件路径加载同级模块, 绕过 src/data/__init__.py
# (后者导入了依赖 librosa 的 dataset.py, 评级模型不需要该依赖)
_HERE = Path(__file__).parent


def _load_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_label_parser = _load_sibling("label_parser", _HERE / "label_parser.py")
_handcrafted_features = _load_sibling("handcrafted_features", _HERE / "handcrafted_features.py")

game_token_to_name = _label_parser.game_token_to_name
compute_features = _handcrafted_features.compute_features
FEATURE_DIM = _handcrafted_features.FEATURE_DIM


class RatingDataset(Dataset):
    """
    Token + 标签 + 实时手工特征数据集
    """

    def __init__(
        self,
        tokens_dir: str,
        labels_file: str,
        max_seq_len: int = 2048,
        feature_stats_file: Optional[str] = None,
        labels_pseudo_file: Optional[str] = None,
        label_source: str = "labels",
        pseudo_weight: float = 0.5,
        exclude_files: Optional[List[str]] = None,
    ):
        """
        Args:
            tokens_dir:           Token 文件目录
            labels_file:          真标签 JSON
            max_seq_len:          截断/填充长度
            feature_stats_file:   特征标准化统计 JSON (含 mean/std)
            labels_pseudo_file:   伪标签 JSON (label_source="any" 时用)
            label_source:         "labels" / "pseudo" / "any"
            pseudo_weight:        伪标签样本 loss 降权系数
            exclude_files:        排除的文件名列表 (校准集用, 避免训练/评估泄漏)
        """
        self.tokens_dir = Path(tokens_dir)
        self.max_seq_len = max_seq_len
        self.pseudo_weight = pseudo_weight
        self.label_source = label_source

        # 加载标签
        self.true_labels = self._load_labels(labels_file)
        self.pseudo_labels = {}
        if labels_pseudo_file and Path(labels_pseudo_file).exists():
            self.pseudo_labels = self._load_labels(labels_pseudo_file)

        # 构建样本列表
        self.samples = self._build_samples(exclude_files)

        # 特征标准化统计
        self.feat_mean = None
        self.feat_std = None
        if feature_stats_file and Path(feature_stats_file).exists():
            with open(feature_stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            self.feat_mean = np.array(stats["mean"], dtype=np.float32)
            self.feat_std = np.array(stats["std"], dtype=np.float32) + 1e-8

        print(f"[RatingDataset] {len(self.samples)} 样本 "
              f"(label_source={label_source}, "
              f"true={len(self.true_labels)}, pseudo={len(self.pseudo_labels)})")

    @staticmethod
    def _load_labels(path: str) -> Dict[str, Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("labels", data)

    def _build_samples(
        self, exclude_files: Optional[List[str]]
    ) -> List[tuple]:
        """构建 (filename, label_dict) 列表"""
        exclude_set = set(exclude_files) if exclude_files else set()
        samples = []
        seen = set()

        # 真标签优先
        if self.label_source in ("labels", "any"):
            for fname, label in self.true_labels.items():
                if fname in exclude_set or fname in seen:
                    continue
                if (self.tokens_dir / fname).exists():
                    samples.append((fname, label))
                    seen.add(fname)

        # 伪标签补充 (label_source="any" 时填补真标签缺失的)
        if self.label_source in ("pseudo", "any"):
            for fname, label in self.pseudo_labels.items():
                if fname in exclude_set or fname in seen:
                    continue
                if (self.tokens_dir / fname).exists():
                    samples.append((fname, label))
                    seen.add(fname)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname, label = self.samples[idx]
        fpath = self.tokens_dir / fname

        # 加载 token
        try:
            tokens = torch.load(fpath, map_location='cpu', weights_only=True).long()
        except Exception:
            # 旧格式兼容
            tokens = torch.load(fpath, map_location='cpu').long()

        tokens_list = tokens.tolist()
        effective_len = len(tokens_list)

        # 截断/填充到 max_seq_len
        if effective_len >= self.max_seq_len:
            # 头部截断 (评级看开头就有信号)
            tokens_list = tokens_list[:self.max_seq_len]
            effective_len = self.max_seq_len
        else:
            # 尾部填充 PAD=0
            tokens_list = tokens_list + [0] * (self.max_seq_len - effective_len)

        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:effective_len] = 1

        # Game Token ID (序列第一个 token)
        game_token_id = int(tokens_list[0]) if effective_len > 0 else 116

        # 手工特征
        feat = compute_features(tokens_list, game_token_id)
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / self.feat_std
        feat_tensor = torch.from_numpy(feat).float()

        # 标签
        rating = torch.tensor(label["normalized"], dtype=torch.float32)

        # 样本权重 (伪标签降权)
        weight = self.pseudo_weight if label.get("source") == "pseudo" else 1.0
        weight_tensor = torch.tensor(weight, dtype=torch.float32)

        return {
            "tokens": tokens_tensor,                              # [T] int64
            "attention_mask": attention_mask,                     # [T] int64
            "handcrafted_features": feat_tensor,                  # [15] float32
            "game_token_ids": torch.tensor([game_token_id], dtype=torch.long),
            "rating": rating,                                     # scalar float32
            "weight": weight_tensor,                              # scalar float32
            "filename": fname,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """DataLoader collate"""
        out = {}
        for k in ["tokens", "attention_mask", "handcrafted_features",
                  "rating", "weight"]:
            out[k] = torch.stack([b[k] for b in batch])
        # game_token_ids: [1] → stack 成 [B, 1] → squeeze
        out["game_token_ids"] = torch.stack([b["game_token_ids"] for b in batch]).squeeze(-1)
        out["filename"] = [b["filename"] for b in batch]
        return out


# ============================================================
# 分桶采样器 (解决类别不均衡)
# ============================================================

def build_bucket_sampler(dataset: RatingDataset, n_buckets: int = 8):
    """
    按 normalized rating 分桶, 每桶等概率采样 (小桶权重大)。

    Returns:
        WeightedRandomSampler
    """
    from torch.utils.data import WeightedRandomSampler

    ratings = []
    for i in range(len(dataset)):
        _, label = dataset.samples[i]
        ratings.append(label["normalized"])

    ratings = np.array(ratings)
    # 分桶边界 (n_buckets 个桶, n_buckets-1 个边界)
    bounds = np.linspace(0, 1, n_buckets + 1)[1:-1]
    bucket_ids = np.digitize(ratings, bounds)  # 0 ~ n_buckets-1

    bucket_counts = np.bincount(bucket_ids, minlength=n_buckets).astype(np.float64)
    bucket_counts = np.maximum(bucket_counts, 1.0)  # 避免除零

    # 权重 = 1 / bucket_count (小桶权重大)
    weights = 1.0 / bucket_counts[bucket_ids]
    weights = weights / weights.sum()  # 归一化

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(dataset),
        replacement=True,
    )


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("RatingDataset 自测")
    print("=" * 60)

    # 构造假标签文件
    tokens_dir = PROJECT_ROOT / "data" / "tokens"
    if not tokens_dir.exists():
        print("  [SKIP] data/tokens 不存在, 跳过实测")
    else:
        # 找前 5 个真实 token 文件
        token_files = sorted(tokens_dir.glob("*.pt"))[:5]
        if not token_files:
            print("  [SKIP] 无 .pt 文件")
        else:
            fake_labels = {
                f.name: {
                    "game": "phira",
                    "level": 12.0 + i,
                    "normalized": 0.75 + i * 0.05,
                    "osu_keys": None,
                    "source": "filename",
                }
                for i, f in enumerate(token_files)
            }
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False, encoding='utf-8'
            ) as tmp:
                json.dump({"labels": fake_labels}, tmp, ensure_ascii=False)
                labels_path = tmp.name

            try:
                ds = RatingDataset(
                    tokens_dir=str(tokens_dir),
                    labels_file=labels_path,
                    max_seq_len=2048,
                )
                print(f"  样本数: {len(ds)}")
                if len(ds) > 0:
                    sample = ds[0]
                    for k, v in sample.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape} dtype={v.dtype}")
                        else:
                            print(f"    {k}: {v}")

                    # 验证 shape
                    assert sample["tokens"].shape == (2048,)
                    assert sample["attention_mask"].shape == (2048,)
                    assert sample["handcrafted_features"].shape == (15,)
                    assert sample["game_token_ids"].shape == (1,)
                    print("\n  ✓ shape 全部正确")

                    # 测试 collate
                    from torch.utils.data import DataLoader
                    loader = DataLoader(
                        ds, batch_size=2, shuffle=False,
                        collate_fn=RatingDataset.collate_fn,
                    )
                    batch = next(iter(loader))
                    assert batch["tokens"].shape == (2, 2048)
                    assert batch["handcrafted_features"].shape == (2, 15)
                    assert batch["game_token_ids"].shape == (2,)  # squeeze 后
                    print("  ✓ collate 正确")

                    # 测试分桶采样器
                    sampler = build_bucket_sampler(ds, n_buckets=4)
                    print(f"  ✓ 分桶采样器构建成功")
            finally:
                Path(labels_path).unlink(missing_ok=True)

    print("\n  ✓ RatingDataset 自测通过")
