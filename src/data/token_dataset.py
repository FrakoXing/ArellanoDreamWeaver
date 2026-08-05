"""
Token 序列数据集
================

加载 unified_tokenizer.py 生成的 Token 序列文件，
与音频文件自动匹配。

数据组织:
  data/tokens/   ← Token 序列 JSON/PT 文件
  data/audio/    ← 对应的音频文件

文件名匹配规则:
  song.json ↔ song.mp3/wav/flac/ogg
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 使用绝对导入，支持直接运行脚本
from src.data.audio_processor import AudioProcessor


class TokenDataset(Dataset):
    """
    Token 序列数据集
    
    用于加载 unified_tokenizer.py 生成的 Token 序列，
    配合音频特征进行训练。
    """
    
    def __init__(
        self,
        tokens_dir: str,
        audio_dir: Optional[str] = None,
        max_seq_len: int = 4096,
        audio_feature_dim: int = 128,
        sample_rate: int = 22050,
        use_real_audio: bool = True,
        cache_audio_features: bool = True,
        audio_cache_dir: Optional[str] = None,
        default_duration: float = 180.0,  # 默认 3 分钟
    ):
        """
        Initialize dataset.

        Args:
            tokens_dir: Token files directory (data/tokens/)
            audio_dir: Audio files directory, None uses tokens_dir/../audio/
            max_seq_len: Max sequence length (tokens are padded/truncated)
            audio_feature_dim: Audio feature dimension (Mel bands)
            sample_rate: Audio sample rate
            use_real_audio: Whether to use real audio (False = random noise)
            cache_audio_features: Whether to cache audio features in RAM
            audio_cache_dir: Pre-processed audio cache directory (data/audio_cache/).
                             If provided, .pt cache files are loaded instead of
                             processing raw audio with librosa. Much faster!
            default_duration: Default chart duration (seconds)
        """
        self.tokens_dir = Path(tokens_dir)
        self.audio_dir = Path(audio_dir) if audio_dir else self.tokens_dir.parent / "audio"
        self.max_seq_len = max_seq_len
        self.audio_feature_dim = audio_feature_dim
        self.sample_rate = sample_rate
        self.use_real_audio = use_real_audio
        self.cache_audio_features = cache_audio_features
        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir else None
        self.default_duration = default_duration

        # Audio processor (lazy init, only if disk cache is missing)
        self._audio_processor = None
        
        # 加载 Token 文件列表
        self.token_files = self._collect_token_files()
        
        if len(self.token_files) == 0:
            print(f"[WARN] 在 {tokens_dir} 中未找到 Token 文件")
            print("请先运行: python scripts/unified_tokenizer.py --dir data/raw/ --outdir data/tokens/")
        
        # 音频特征缓存
        self._audio_cache: Dict[str, torch.Tensor] = {}
        
        # 统计
        self._stats = {
            "total_tokens": len(self.token_files),
            "matched_audio": 0,
            "missing_audio": 0,
        }
        
        # 检查音频匹配
        if self.use_real_audio:
            self._check_audio_files()
    
    def _collect_token_files(self) -> List[Path]:
        """收集所有 Token 文件 (支持 .json 和 .pt)"""
        files = []
        files.extend(self.tokens_dir.glob("*.json"))
        files.extend(self.tokens_dir.glob("*.pt"))
        files.sort()
        return files
    
    def _check_audio_files(self):
        """检查音频文件匹配情况"""
        for token_path in self.token_files:
            audio_path = self._find_audio_file(token_path)
            if audio_path and audio_path.exists():
                self._stats["matched_audio"] += 1
            else:
                self._stats["missing_audio"] += 1
        
        print(f"\n[DATA] Token Dataset Stats:")
        print(f"  Token 文件数: {self._stats['total_tokens']}")
        print(f"  匹配音频数:   {self._stats['matched_audio']}")
        print(f"  缺失音频数:   {self._stats['missing_audio']}")
        
        if self._stats["missing_audio"] > 0:
            print(f"\n[WARN] 有 {self._stats['missing_audio']} 个 Token 文件缺少对应音频")
            print(f"   音频文件应放在: {self.audio_dir}")
            print(f"   命名规则: song.json ↔ song.mp3/wav/flac")
    
    def _find_audio_file(self, token_path: Path) -> Optional[Path]:
        """
        查找与 Token 文件对应的音频文件
        
        匹配规则: song.json ↔ song.mp3/wav/flac/ogg/m4a
        """
        base_name = token_path.stem
        
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
        
        for ext in audio_extensions:
            audio_path = self.audio_dir / f"{base_name}{ext}"
            if audio_path.exists():
                return audio_path
        
        return None
    
    def _load_tokens(self, token_path: Path) -> torch.Tensor:
        """
        加载 Token 序列
        
        Returns:
            LongTensor [seq_len]
        """
        ext = token_path.suffix.lower()
        
        if ext == '.json':
            with open(token_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tokens = data.get('tokens', [])
            return torch.tensor(tokens, dtype=torch.long)
        
        elif ext == '.pt':
            return torch.load(token_path, map_location='cpu')
        
        else:
            raise ValueError(f"不支持的 Token 文件格式: {ext}")
    
    def _pad_or_truncate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Padding 或截断 Token 序列到固定长度"""
        if len(tokens) >= self.max_seq_len:
            return tokens[:self.max_seq_len]
        else:
            # Padding with 0 (PAD token)
            padding = torch.zeros(self.max_seq_len - len(tokens), dtype=torch.long)
            return torch.cat([tokens, padding])
    
    def _load_audio_features(self, token_path: Path, duration: float) -> torch.Tensor:
        """
        Load or generate audio features.

        Priority:
          1. RAM cache (fastest)
          2. Disk cache (.pt files from prepare_audio_cache.py)
          3. Real-time librosa processing (slowest)
          4. Random noise (fallback)

        Returns:
            FloatTensor [max_seq_len, audio_feature_dim]
        """
        cache_key = token_path.stem

        # === Layer 1: RAM cache ===
        if self.cache_audio_features and cache_key in self._audio_cache:
            return self._audio_cache[cache_key]

        # === Layer 2: Disk cache (.pt files) ===
        if self.audio_cache_dir is not None:
            disk_cache_path = self.audio_cache_dir / f"{cache_key}.pt"
            if disk_cache_path.exists():
                try:
                    features = torch.load(disk_cache_path, map_location='cpu', weights_only=True)

                    # Validate shape
                    if features.shape[0] != self.max_seq_len:
                        features = self._resize_audio(features, self.max_seq_len)

                    if self.cache_audio_features:
                        self._audio_cache[cache_key] = features

                    return features
                except Exception as e:
                    pass  # Fall through to real-time processing on cache miss

        # === Layer 3: Real-time librosa processing ===
        audio_path = self._find_audio_file(token_path)

        if audio_path and audio_path.exists() and self.use_real_audio:
            try:
                if self._audio_processor is None:
                    self._audio_processor = AudioProcessor(
                        sample_rate=self.sample_rate,
                        n_mels=self.audio_feature_dim,
                    )
                features = self._audio_processor.align_to_chart(
                    audio_path,
                    chart_duration=duration,
                    target_steps=self.max_seq_len,
                )
            except Exception as e:
                features = torch.randn(self.max_seq_len, self.audio_feature_dim)
        else:
            # === Layer 4: Random noise fallback ===
            features = torch.randn(self.max_seq_len, self.audio_feature_dim)

        # Save to RAM cache
        if self.cache_audio_features:
            self._audio_cache[cache_key] = features

        return features

    @staticmethod
    def _resize_audio(features: torch.Tensor, target_steps: int) -> torch.Tensor:
        """Resize audio features to target_steps via linear interpolation."""
        if features.shape[0] == target_steps:
            return features

        src_len = features.shape[0]
        src_indices = torch.linspace(0, src_len - 1, src_len)
        tgt_indices = torch.linspace(0, src_len - 1, target_steps)

        result = torch.zeros(target_steps, features.shape[1])
        for i in range(features.shape[1]):
            result[:, i] = torch.from_numpy(
                np.interp(tgt_indices.numpy(), src_indices.numpy(), features[:, i].numpy())
            ).float()

        return result
    
    def __len__(self) -> int:
        return len(self.token_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        token_path = self.token_files[idx]
        
        # 加载 Token 序列
        try:
            tokens = self._load_tokens(token_path)
        except Exception as e:
            print(f"加载 {token_path} 失败: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Padding/截断
        tokens = self._pad_or_truncate(tokens)
        
        # 估算时长 (从 Token 数量粗略估计，或从文件名解析)
        # TODO: 可以在 Token JSON 中存储 duration 字段
        duration = self.default_duration
        
        # 加载音频特征
        audio_features = self._load_audio_features(token_path, duration)
        
        # 生成 attention mask (非 padding 部分为 1)
        # 找到第一个 PAD token 的位置
        pad_positions = (tokens == 0).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            first_pad = pad_positions[0].item()
        else:
            first_pad = len(tokens)
        
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:first_pad] = 1
        
        sample = {
            "input_ids": tokens,              # Token 序列 [max_seq_len]
            "attention_mask": attention_mask, # Attention mask [max_seq_len]
            "audio_features": audio_features, # 音频特征 [max_seq_len, audio_dim]
            "filename": token_path.name,
        }
        
        return sample
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """自定义 batch 整理函数"""
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key == "filename":
                collated[key] = [item[key] for item in batch]
            else:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = values
        
        return collated
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return self._stats


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys
    
    tokens_dir = sys.argv[1] if len(sys.argv) > 1 else "data/tokens"
    
    print(f"测试 Token 数据集: {tokens_dir}")
    
    dataset = TokenDataset(
        tokens_dir=tokens_dir,
        use_real_audio=True,
        cache_audio_features=False,
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n样本内容:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # 测试 DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=TokenDataset.collate_fn,
        )
        
        batch = next(iter(loader))
        print(f"\nBatch 内容:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}")
            else:
                print(f"  {key}: {value}")
