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

# 窗口起点吸附边界: 音符类型 token (TAP/HOLD/SLIDE) 或事件块开始 (EVENT_BEGIN)
# 避免随机窗口从音符属性 (time/x/hold/direction) 中间开始
_BOUNDARY_TOKENS = (4, 5, 6, 120)


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
        max_chart_tokens: Optional[int] = 200_000,  # 超长谱面过滤阈值 (None = 不过滤)
        window_mode: str = "random",       # "random" / "sliding" / "head"
        window_stride: int = 1024,         # sliding 模式窗口步长
        snap_window_to_note: bool = True,  # 随机窗口起点吸附到音符/事件边界
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
            max_chart_tokens: 超长谱面过滤阈值。token 数超过该值的谱面被剔除
                             (避免少数巨型谱面主导数据集), None = 不过滤。
            window_mode: 长谱面截断方式:
                         "random"  - 每个样本随机取一个 max_seq_len 窗口
                                     (每 epoch 覆盖不同位置, 音频按窗口比例对齐)
                         "sliding" - 按 window_stride 滑窗, 每谱面多个样本,
                                     完整覆盖全部内容
                         "head"    - 旧行为, 只取开头 (不推荐)
            window_stride: sliding 模式的窗口步长 (token)
            snap_window_to_note: 随机窗口起点吸附到最近的音符/事件边界,
                                 避免窗口从音符属性中间开始
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

        # 窗口化参数
        self.max_chart_tokens = max_chart_tokens if max_chart_tokens and max_chart_tokens > 0 else None
        if window_mode not in ("random", "sliding", "head"):
            print(f"[WARN] 未知 window_mode='{window_mode}', 回退到 'random'")
            window_mode = "random"
        self.window_mode = window_mode
        self.window_stride = max(64, window_stride)
        self.snap_window_to_note = snap_window_to_note

        # Audio processor (lazy init, only if disk cache is missing)
        self._audio_processor = None

        # Token 长度索引 (init 时一次性扫描, 供过滤和窗口计算复用)
        self._token_lengths: Dict[str, int] = {}
        # sliding 模式: 扁平索引 -> (文件索引, 窗口起点)
        self._window_index: List[Tuple[int, int]] = []

        # 加载 Token 文件列表 (含超长谱面过滤)
        self.token_files = self._collect_token_files()
        self.token_files = self._filter_long_charts(self.token_files)

        if self.window_mode == "sliding":
            self._build_window_index()

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

    @staticmethod
    def _token_file_length(path: Path) -> int:
        """读取 token 文件长度"""
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return len(json.load(f).get('tokens', []))
        else:
            return len(torch.load(path, map_location='cpu'))

    def _filter_long_charts(self, files: List[Path]) -> List[Path]:
        """
        过滤超长谱面 (> max_chart_tokens), 并预扫描全部 token 长度。

        长尾谱面 (单张 >20 万 token) 占数据集 token 总量过半,
        会让模型被少数巨型谱面主导, 直接剔除。
        """
        lengths = {}
        for f in files:
            try:
                lengths[f.name] = self._token_file_length(f)
            except Exception:
                lengths[f.name] = 0
        self._token_lengths = lengths

        if self.max_chart_tokens is None:
            return files

        kept, dropped, dropped_tokens = [], [], 0
        total_tokens = sum(lengths.values())
        for f in files:
            if lengths[f.name] > self.max_chart_tokens:
                dropped.append(f)
                dropped_tokens += lengths[f.name]
            else:
                kept.append(f)

        if dropped:
            pct = dropped_tokens / total_tokens * 100 if total_tokens > 0 else 0
            print(f"[DATA] 超长谱面过滤 (> {self.max_chart_tokens:,} tokens):")
            print(f"  剔除: {len(dropped)} 张 ({pct:.1f}% 的数据量)")
            print(f"  保留: {len(kept)} 张")
        return kept

    def _build_window_index(self):
        """sliding 模式: 每个长谱面按 window_stride 生成多个窗口样本"""
        index = []
        for file_idx, path in enumerate(self.token_files):
            total_len = self._token_lengths.get(path.name, 0)
            if total_len <= self.max_seq_len:
                index.append((file_idx, 0))
            else:
                last_start = total_len - self.max_seq_len
                for s in range(0, last_start, self.window_stride):
                    index.append((file_idx, s))
                # 保证覆盖到序列末尾
                if last_start % self.window_stride != 0:
                    index.append((file_idx, last_start))
        self._window_index = index
        if index:
            print(f"[DATA] sliding 窗口: {len(index):,} 个样本 "
                  f"(来自 {len(self.token_files)} 张谱面)")
    
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
    
    def _extract_window(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        随机/head 模式: 从完整 token 序列取一个 max_seq_len 窗口。

        Args:
            tokens: 完整 token 序列 [T]

        Returns:
            (window, window_start):
                window 为 [max_seq_len] (不足补 PAD),
                window_start 为窗口起点 (用于音频对齐, 短谱为 0)
        """
        total_len = len(tokens)

        if total_len <= self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - total_len, dtype=torch.long)
            return torch.cat([tokens, padding]), 0

        if self.window_mode == "head":
            start = 0
        else:
            start = int(torch.randint(0, total_len - self.max_seq_len + 1, (1,)).item())
            if self.snap_window_to_note:
                start = self._snap_to_boundary(tokens, start, total_len - self.max_seq_len)

        return tokens[start:start + self.max_seq_len], start

    @staticmethod
    def _snap_to_boundary(tokens: torch.Tensor, start: int, max_start: int) -> int:
        """
        把窗口起点吸附到最近的音符/事件边界 token。

        避免窗口从音符属性 (time_offset/x/hold/direction) 中间开始,
        保证窗口开头的 token 是完整的音符或事件块。
        """
        for i in range(start, max_start + 1):
            if int(tokens[i]) in _BOUNDARY_TOKENS:
                return i
        return max_start
    
    def _load_audio_features(
        self,
        token_path: Path,
        duration: float,
        frac_start: float = 0.0,
        frac_end: float = 1.0,
    ) -> torch.Tensor:
        """
        Load or generate audio features.

        Priority:
          1. RAM cache (fastest)
          2. Disk cache (.pt files from prepare_audio_cache.py)
          3. Real-time librosa processing (slowest)
          4. Random noise (fallback)

        Args:
            token_path: Token 文件路径
            duration: 谱面时长 (秒)
            frac_start / frac_end: token 窗口在整首谱面中的时间比例 [0,1]。
                音频特征按该比例切出对应时段, 再插值回 max_seq_len —
                保证"音频片段 ↔ token 窗口"对齐, 而不是整首歌压缩后
                只对应谱面开头一小段。

        Returns:
            FloatTensor [max_seq_len, audio_feature_dim]

        Note:
            DataLoader worker(Linux fork)中只读 RAM 缓存、不写入。
            fork 后每个 worker 内存独立,若各自写入缓存会复制整份
            特征到每个 worker,4 个 worker 就是 4 份 → 内存膨胀,
            训练到中途被内核 OOM killer 杀掉(表现为:
            "DataLoader worker ... is killed by signal: Killed")。
        """
        cache_key = token_path.stem

        # 是否在 DataLoader worker 进程中
        in_worker = torch.utils.data.get_worker_info() is not None

        # === Layer 1: RAM cache ===
        if self.cache_audio_features and cache_key in self._audio_cache:
            return self._window_slice(self._audio_cache[cache_key], frac_start, frac_end)

        # === Layer 2: Disk cache (.pt files) ===
        if self.audio_cache_dir is not None:
            disk_cache_path = self.audio_cache_dir / f"{cache_key}.pt"
            if disk_cache_path.exists():
                try:
                    features = torch.load(disk_cache_path, map_location='cpu', weights_only=True)

                    # Validate shape
                    if features.shape[0] != self.max_seq_len:
                        features = self._resize_audio(features, self.max_seq_len)

                    if self.cache_audio_features and not in_worker:
                        self._audio_cache[cache_key] = features

                    return self._window_slice(features, frac_start, frac_end)
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
                # target_steps=None: 提取全曲特征再按窗口切分
                # (对齐到整首的归一化统计, 各窗口尺度一致)
                features = self._audio_processor.align_to_chart(
                    audio_path,
                    chart_duration=duration,
                    target_steps=None,
                )
                # 整首特征插值到 max_seq_len 后缓存 (仅缓存窗口无关的整首表示)
                features = self._resize_audio(features, self.max_seq_len)
            except Exception as e:
                features = torch.randn(self.max_seq_len, self.audio_feature_dim)
        else:
            # === Layer 4: Random noise fallback ===
            features = torch.randn(self.max_seq_len, self.audio_feature_dim)

        # Save to RAM cache (worker 中不写,避免每 worker 复制一份)
        if self.cache_audio_features and not in_worker:
            self._audio_cache[cache_key] = features

        return self._window_slice(features, frac_start, frac_end)

    def _window_slice(
        self,
        features: torch.Tensor,
        frac_start: float,
        frac_end: float,
    ) -> torch.Tensor:
        """
        按窗口时间比例切分音频特征, 并插值回 max_seq_len。

        features 是整首谱面 (归一化后) 的特征 [F, dim] 或 [max_seq_len, dim],
        切片 [frac_start*F, frac_end*F) 对应 token 窗口覆盖的时段,
        再插值到 max_seq_len 帧, 与 token 窗口逐帧对齐。
        """
        src_len = features.shape[0]
        f0 = int(round(frac_start * src_len))
        f1 = max(f0 + 1, int(round(frac_end * src_len)))
        f1 = min(f1, src_len)
        segment = features[f0:f1]
        return self._resize_audio(segment, self.max_seq_len)

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
        # sliding 模式: 每个窗口是一个样本
        if self.window_mode == "sliding":
            return len(self._window_index)
        return len(self.token_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        # sliding 模式: idx 是窗口索引
        if self.window_mode == "sliding":
            file_idx, window_start = self._window_index[idx]
            token_path = self.token_files[file_idx]
        else:
            token_path = self.token_files[idx]
            window_start = None

        # 加载 Token 序列
        try:
            tokens = self._load_tokens(token_path)
        except Exception as e:
            print(f"加载 {token_path} 失败: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # 窗口截断/填充: 长谱面取窗口, 短谱面补 PAD
        if window_start is None:
            tokens, window_start = self._extract_window(tokens)
        else:
            window = tokens[window_start:window_start + self.max_seq_len]
            if len(window) < self.max_seq_len:
                padding = torch.zeros(self.max_seq_len - len(window), dtype=torch.long)
                window = torch.cat([window, padding])
            tokens = window

        # 估算时长 (从 Token 数量粗略估计，或从文件名解析)
        # TODO: 可以在 Token JSON 中存储 duration 字段
        duration = self.default_duration

        # 音频特征: 按 token 窗口在整首谱面中的比例切对应时段
        # (token 数量近似时间长度; 谱面内音符密度不均时是近似对齐)
        total_len = self._token_lengths.get(token_path.name, 0) or len(tokens)
        frac_start = window_start / max(1, total_len)
        frac_end = min(1.0, (window_start + self.max_seq_len) / max(1, total_len))
        audio_features = self._load_audio_features(
            token_path, duration,
            frac_start=frac_start, frac_end=frac_end,
        )
        
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
