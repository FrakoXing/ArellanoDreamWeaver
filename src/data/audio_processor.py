"""
音频特征提取模块
================
将音频文件转换为模型可用的特征张量
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, Union
import torch


class AudioProcessor:
    """
    音频处理器
    
    功能:
    - 加载音频文件 (mp3, wav, flac, ogg 等)
    - 提取梅尔频谱特征
    - 对齐到谱面时间长度
    - 返回 PyTorch 张量
    
    用法:
        processor = AudioProcessor()
        features = processor.process("song.mp3", duration=180.0)
        # features shape: [time_steps, feature_dim]
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
        fmin: int = 20,
        fmax: Optional[int] = None,
        normalize: bool = True,
        device: str = "cpu"
    ):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率 (Hz)
            n_mels: 梅尔滤波器数量
            hop_length: 帧移 (样本数)
            n_fft: FFT 窗口大小
            fmin: 最低频率 (Hz)
            fmax: 最高频率 (Hz)，None 表示使用 sample_rate/2
            normalize: 是否对特征进行归一化
            device: 设备 ("cpu" 或 "cuda")
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.normalize = normalize
        self.device = device
        
    def load_audio(
        self, 
        audio_path: Union[str, Path],
        target_sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率，None 使用默认
            mono: 是否转为单声道
            
        Returns:
            (audio_data, sample_rate)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        target_sr = target_sr or self.sample_rate
        
        # 使用 librosa 加载
        audio, sr = librosa.load(
            str(audio_path),
            sr=target_sr,
            mono=mono
        )
        
        return audio, sr
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        提取梅尔频谱特征
        
        Args:
            audio: 音频数据 (1D numpy array)
            sr: 采样率，None 使用默认
            
        Returns:
            梅尔频谱 (time_steps, n_mels)
        """
        sr = sr or self.sample_rate
        
        # 计算梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # 转换为对数尺度 (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 转置为 (time_steps, n_mels)
        log_mel = log_mel.T
        
        return log_mel
    
    def extract_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        提取完整特征集 (梅尔频谱 + 额外特征)
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            特征矩阵 (time_steps, feature_dim)
        """
        sr = sr or self.sample_rate
        
        # 1. 梅尔频谱 (n_mels)
        mel_spec = self.extract_mel_spectrogram(audio, sr)
        
        # 2. 色度特征 (chroma, 12维)
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        ).T
        
        # 3. 频谱对比度 (spectral_contrast, 7维)
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        ).T
        
        # 4. 频谱质心 (spectral_centroid, 1维)
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        ).T
        
        # 5. 过零率 (zero_crossing_rate, 1维)
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        ).T
        
        # 对齐长度 (取最小值)
        min_len = min(len(mel_spec), len(chroma), len(contrast), len(centroid), len(zcr))
        
        # 拼接所有特征
        features = np.concatenate([
            mel_spec[:min_len],
            chroma[:min_len],
            contrast[:min_len],
            centroid[:min_len],
            zcr[:min_len]
        ], axis=-1)
        
        return features
    
    def process(
        self,
        audio_path: Union[str, Path],
        duration: Optional[float] = None,
        target_steps: Optional[int] = None,
        feature_type: str = "mel"
    ) -> torch.Tensor:
        """
        处理音频文件，返回特征张量
        
        Args:
            audio_path: 音频文件路径
            duration: 目标时长 (秒)，None 表示使用完整音频
            target_steps: 目标时间步数，None 表示不插值
            feature_type: 特征类型 ("mel" 或 "full")
            
        Returns:
            特征张量 [time_steps, feature_dim]
        """
        # 加载音频
        audio, sr = self.load_audio(audio_path)
        
        # 截断到目标时长
        if duration is not None:
            max_samples = int(duration * sr)
            audio = audio[:max_samples]
        
        # 提取特征
        if feature_type == "mel":
            features = self.extract_mel_spectrogram(audio, sr)
        elif feature_type == "full":
            features = self.extract_features(audio, sr)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
        
        # 插值到目标步数
        if target_steps is not None and len(features) != target_steps:
            features = self._interpolate(features, target_steps)
        
        # 归一化
        if self.normalize:
            features = self._normalize(features)
        
        # 转为张量
        tensor = torch.from_numpy(features).float()
        
        return tensor
    
    def _interpolate(self, features: np.ndarray, target_steps: int) -> np.ndarray:
        """
        时间维度插值
        
        Args:
            features: 原始特征 (time_steps, feature_dim)
            target_steps: 目标时间步数
            
        Returns:
            插值后的特征
        """
        from scipy import interpolate
        
        time_steps = np.arange(len(features))
        target_time = np.linspace(0, len(features) - 1, target_steps)
        
        interpolated = np.zeros((target_steps, features.shape[1]), dtype=np.float32)
        
        for i in range(features.shape[1]):
            f = interpolate.interp1d(time_steps, features[:, i], kind='linear')
            interpolated[:, i] = f(target_time)
        
        return interpolated
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        特征归一化 (标准化到均值0，方差1)
        
        Args:
            features: 原始特征
            
        Returns:
            归一化后的特征
        """
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        
        # 避免除零
        std = np.maximum(std, 1e-8)
        
        normalized = (features - mean) / std
        
        return normalized
    
    def align_to_chart(
        self,
        audio_path: Union[str, Path],
        chart_duration: float,
        target_steps: int = 4096
    ) -> torch.Tensor:
        """
        处理音频并对齐到谱面时长
        
        Args:
            audio_path: 音频文件路径
            chart_duration: 谱面时长 (秒)
            target_steps: 目标时间步数 (与谱面 max_seq_len 一致)
            
        Returns:
            对齐后的特征张量 [target_steps, feature_dim]
        """
        return self.process(
            audio_path,
            duration=chart_duration,
            target_steps=target_steps,
            feature_type="mel"
        )


# ============================================================
# 便捷函数
# ============================================================

def process_audio_file(
    audio_path: str,
    duration: Optional[float] = None,
    target_steps: int = 4096,
    n_mels: int = 128
) -> torch.Tensor:
    """
    快速处理单个音频文件
    
    Args:
        audio_path: 音频文件路径
        duration: 目标时长 (秒)
        target_steps: 目标时间步数
        n_mels: 梅尔滤波器数量
        
    Returns:
        特征张量
    """
    processor = AudioProcessor(n_mels=n_mels)
    return processor.process(
        audio_path,
        duration=duration,
        target_steps=target_steps,
        feature_type="mel"
    )


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python audio_processor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    processor = AudioProcessor()
    
    print(f"处理音频: {audio_file}")
    features = processor.process(audio_file, target_steps=4096)
    
    print(f"特征形状: {features.shape}")
    print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
    print(f"特征均值: {features.mean():.3f}, 标准差: {features.std():.3f}")
