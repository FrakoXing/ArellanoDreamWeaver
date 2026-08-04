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
        default_duration: float = 180.0,  # 默认 3 分钟
    ):
        """
        初始化数据集
        
        Args:
            tokens_dir: Token 序列文件目录 (data/tokens/)
            audio_dir: 音频文件目录，None 则使用 tokens_dir 同级 "audio/"
            max_seq_len: 最大序列长度 (Token 序列会被 padding/truncating)
            audio_feature_dim: 音频特征维度 (Mel 频谱通道数)
            sample_rate: 音频采样率
            use_real_audio: 是否使用真实音频 (False 用随机噪声)
            cache_audio_features: 是否缓存音频特征到内存
            default_duration: 默认谱面时长 (秒)，用于没有时长的情况
        """
        self.tokens_dir = Path(tokens_dir)
        self.audio_dir = Path(audio_dir) if audio_dir else self.tokens_dir.parent / "audio"
        self.max_seq_len = max_seq_len
        self.audio_feature_dim = audio_feature_dim
        self.sample_rate = sample_rate
        self.use_real_audio = use_real_audio
        self.cache_audio_features = cache_audio_features
        self.default_duration = default_duration
        
        # 音频处理器
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=audio_feature_dim
        )
        
        # 加载 Token 文件列表
        self.token_files = self._collect_token_files()
        
        if len(self.token_files) == 0:
            print(f"⚠️  在 {tokens_dir} 中未找到 Token 文件")
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
        
        print(f"\n📊 Token 数据集统计:")
        print(f"  Token 文件数: {self._stats['total_tokens']}")
        print(f"  匹配音频数:   {self._stats['matched_audio']}")
        print(f"  缺失音频数:   {self._stats['missing_audio']}")
        
        if self._stats["missing_audio"] > 0:
            print(f"\n⚠️  有 {self._stats['missing_audio']} 个 Token 文件缺少对应音频")
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
        加载或生成音频特征
        
        Returns:
            FloatTensor [max_seq_len, audio_feature_dim]
        """
        cache_key = token_path.stem
        
        # 检查缓存
        if self.cache_audio_features and cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        
        # 查找音频文件
        audio_path = self._find_audio_file(token_path)
        
        if audio_path and audio_path.exists() and self.use_real_audio:
            try:
                features = self.audio_processor.align_to_chart(
                    audio_path,
                    chart_duration=duration,
                    target_steps=self.max_seq_len
                )
            except Exception as e:
                print(f"⚠️  处理音频失败 ({audio_path.name}): {e}")
                print("   使用随机噪声作为后备")
                features = torch.randn(self.max_seq_len, self.audio_feature_dim)
        else:
            # 使用随机噪声
            features = torch.randn(self.max_seq_len, self.audio_feature_dim)
        
        # 缓存
        if self.cache_audio_features:
            self._audio_cache[cache_key] = features
        
        return features
    
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
