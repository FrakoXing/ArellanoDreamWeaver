"""
谱面数据集模块
==============
支持加载真实的音频-谱面对数据进行训练
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.chart.structures import ChartData, chart_to_tensor
from .audio_processor import AudioProcessor


class ChartDataset(Dataset):
    """
    谱面数据集 (支持真实音频)
    
    数据组织方式:
    - data/processed/ 下放置谱面 JSON 文件
    - data/audio/ 下放置对应的音频文件
    - 谱面和音频通过文件名匹配 (如: Calm.json ↔ Calm.mp3)
    
    输入: 音频特征 + 谱面张量 (干净)
    输出: 用于 Diffusion 的谱面 + 多任务标签
    """
    
    def __init__(
        self,
        data_dir: str,
        audio_dir: Optional[str] = None,
        max_seq_len: int = 4096,
        feature_dim: int = 8,
        audio_feature_dim: int = 128,
        sample_rate: int = 22050,
        use_real_audio: bool = True,
        transform=None,
        cache_audio_features: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 谱面 JSON 文件目录
            audio_dir: 音频文件目录，None 则使用 data_dir 同级目录 "audio/"
            max_seq_len: 最大序列长度
            feature_dim: 谱面特征维度
            audio_feature_dim: 音频特征维度
            sample_rate: 音频采样率
            use_real_audio: 是否使用真实音频 (False 则使用随机噪声)
            transform: 数据增强变换
            cache_audio_features: 是否缓存音频特征到内存
        """
        self.data_dir = Path(data_dir)
        self.audio_dir = Path(audio_dir) if audio_dir else self.data_dir.parent / "audio"
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.sample_rate = sample_rate
        self.use_real_audio = use_real_audio
        self.transform = transform
        self.cache_audio_features = cache_audio_features
        
        # 音频处理器
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=audio_feature_dim
        )
        
        # 加载数据文件列表
        self.chart_files = list(self.data_dir.glob("*.json"))
        
        if len(self.chart_files) == 0:
            print(f"警告: 在 {data_dir} 中未找到 JSON 文件")
            print("请将谱面数据放在 data/processed/ 目录下")
        
        # 音频特征缓存
        self._audio_cache: Dict[str, torch.Tensor] = {}
        
        # 统计信息
        self._stats = {
            "total_charts": len(self.chart_files),
            "matched_audio": 0,
            "missing_audio": 0
        }
        
        # 检查音频文件匹配情况
        if self.use_real_audio:
            self._check_audio_files()
    
    def _check_audio_files(self):
        """检查音频文件匹配情况"""
        for chart_path in self.chart_files:
            audio_path = self._find_audio_file(chart_path)
            if audio_path and audio_path.exists():
                self._stats["matched_audio"] += 1
            else:
                self._stats["missing_audio"] += 1
        
        print(f"\n📊 数据集统计:")
        print(f"  谱面文件数: {self._stats['total_charts']}")
        print(f"  匹配音频数: {self._stats['matched_audio']}")
        print(f"  缺失音频数: {self._stats['missing_audio']}")
        
        if self._stats["missing_audio"] > 0:
            print(f"\n⚠️  有 {self._stats['missing_audio']} 个谱面缺少对应音频文件")
            print(f"   音频文件应放在: {self.audio_dir}")
            print(f"   命名规则: 谱面名.json ↔ 谱面名.mp3/wav/flac")
    
    def _find_audio_file(self, chart_path: Path) -> Optional[Path]:
        """
        查找与谱面对应的音频文件
        
        支持格式: mp3, wav, flac, ogg, m4a
        """
        base_name = chart_path.stem
        
        # 支持的音频格式
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
        
        for ext in audio_extensions:
            audio_path = self.audio_dir / f"{base_name}{ext}"
            if audio_path.exists():
                return audio_path
        
        return None
    
    def _load_audio_features(self, chart_path: Path, duration: float) -> torch.Tensor:
        """
        加载或生成音频特征
        
        Args:
            chart_path: 谱面文件路径
            duration: 谱面时长 (秒)
            
        Returns:
            音频特征张量 [max_seq_len, audio_feature_dim]
        """
        cache_key = chart_path.stem
        
        # 检查缓存
        if self.cache_audio_features and cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        
        # 查找音频文件
        audio_path = self._find_audio_file(chart_path)
        
        if audio_path and audio_path.exists() and self.use_real_audio:
            # 使用真实音频
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
            # 使用随机噪声 (后备)
            features = torch.randn(self.max_seq_len, self.audio_feature_dim)
        
        # 缓存
        if self.cache_audio_features:
            self._audio_cache[cache_key] = features
        
        return features
    
    def __len__(self) -> int:
        return len(self.chart_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        chart_path = self.chart_files[idx]
        
        # 加载谱面
        try:
            chart = ChartData.load_from_file(str(chart_path))
        except Exception as e:
            print(f"加载 {chart_path} 失败: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # 转换为张量表示
        chart_tensor = chart_to_tensor(chart, self.max_seq_len, self.feature_dim)
        
        # 获取谱面时长
        duration = chart.musicLength if chart.musicLength > 0 else 180.0  # 默认3分钟
        
        # 加载音频特征
        audio_features = self._load_audio_features(chart_path, duration)
        
        # 模拟文本条件 (实际应从 LLM 解析提示词)
        text_condition = torch.randn(512)
        
        # 提取多任务标签
        bpm = chart.default_bpm / 300.0  # 归一化到 0-1
        difficulty = self._estimate_difficulty(chart)
        
        sample = {
            "audio_features": audio_features,
            "chart_clean": torch.from_numpy(chart_tensor).float(),
            "text_condition": text_condition,
            
            # 多任务标签
            "bpm": torch.tensor([bpm], dtype=torch.float32),
            "difficulty": torch.tensor([difficulty], dtype=torch.long),
            
            # 元数据
            "filename": str(chart_path.name),
            "note_count": chart.total_notes,
            "duration": duration
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _estimate_difficulty(self, chart: ChartData) -> int:
        """
        估计谱面难度 (0-4)
        
        基于:
        - 音符密度
        - 音符类型分布
        - BPM
        """
        notes = chart.get_all_notes()
        if not notes:
            return 0
        
        # 计算音符密度 (音符/秒)
        duration = max(chart.musicLength, 1.0)
        density = len(notes) / duration
        
        # 简单难度评估
        if density < 2.0:
            return 0  # 简单
        elif density < 4.0:
            return 1  # 中等
        elif density < 6.0:
            return 2  # 困难
        elif density < 8.0:
            return 3  # 专家
        else:
            return 4  # 大师
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """自定义 batch 整理函数"""
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key in ["filename", "duration"]:
                # 字符串和浮点数保持列表
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


class ChartDataCollator:
    """
    数据整理器 (与 Dataset.collate_fn 兼容)
    
    用于 DataLoader 的 collate_fn 参数
    """
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        return ChartDataset.collate_fn(batch)


# ============================================================
# 演示数据集 (用于测试)
# ============================================================

class DemoChartDataset(Dataset):
    """
    演示数据集 (使用合成数据)
    
    用于在没有真实数据时测试训练流程
    """
    
    def __init__(self, num_samples: int = 32, max_seq_len: int = 4096):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        
        # 预生成数据
        self.audio = torch.randn(num_samples, max_seq_len, 128)
        self.charts = torch.randn(num_samples, max_seq_len, 8)
        self.conditions = torch.randn(num_samples, 512)
        self.bpms = (torch.rand(num_samples) * 0.5 + 0.3).unsqueeze(1)
        self.difficulties = torch.randint(0, 5, (num_samples,))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "audio_features": self.audio[idx],
            "chart_clean": self.charts[idx],
            "text_condition": self.conditions[idx],
            "bpm": self.bpms[idx],
            "difficulty": self.difficulties[idx],
            "filename": f"demo_{idx}.json",
            "note_count": torch.randint(100, 1000, (1,)).item()
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        return ChartDataset.collate_fn(batch)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    
    print(f"测试数据集: {data_dir}")
    
    dataset = ChartDataset(
        data_dir=data_dir,
        use_real_audio=True,
        cache_audio_features=False
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
