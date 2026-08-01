"""
共享音频编码器: CNN特征提取 + Transformer时序建模

处理原始音频波形，提取高层语义特征供下游任务使用。
"""

import math
import torch
import torch.nn as nn
import torchaudio.transforms as T


class PositionalEncoding(nn.Module):
    """正弦位置编码 (Vaswani et al., 2017)"""
    
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        Returns:
            [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AudioEncoder(nn.Module):
    """
    共享音频编码器
    
    架构流程:
        Waveform → MelSpectrogram → CNN下采样 → 展平 → PosEncoding → Transformer Encoder
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        cnn_channels: list = None,
        dropout: float = 0.1,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # ===== 音频预处理 (Mel频谱图) =====
        # 注意: 训练时在DataLoader中预处理，这里提供可选的在线处理
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        
        # 归一化
        self.amplitude_to_db = T.AmplitudeToDB(st_ref=1.0, top_db=80.0)
        
        # ===== CNN特征提取骨干 =====
        if cnn_channels is None:
            cnn_channels = [64, 128, hidden_dim]
        
        cnn_layers = []
        in_ch = 1  # 单通道 (mel频谱)
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 时间和频率维度各减半
            ])
            in_ch = out_ch
        
        self.cnn_backbone = nn.Sequential(*cnn_layers)
        
        # ===== Transformer编码器 =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定训练)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True,
        )
        
        # ===== 位置编码 =====
        self.pos_encoding = PositionalEncoding(
            d_model=hidden_dim,
            max_len=10000,
            dropout=dropout
        )
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def extract_features(
        self,
        waveform: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor | dict:
        """
        从波形提取特征
        
        Args:
            waveform: [batch_size, audio_length] 原始波形
            return_dict: 是否返回详细中间结果
            
        Returns:
            features: [batch_size, seq_len, hidden_dim] 编码后的特征序列
        """
        # Step 1: 提取Mel频谱
        mel_spec = self.mel_transform(waveform)  # [B, 1, T_mel, n_mels]
        mel_db = self.amplitude_to_db(mel_spec)   # 转换为dB刻度
        
        # 标准化 (per-sample z-score)
        mean = mel_db.mean(dim=[1, 2, 3], keepdim=True)
        std = mel_db.std(dim=[1, 2, 3], keepdim=True) + 1e-6
        mel_normalized = (mel_db - mean) / std
        
        # Step 2: CNN特征提取
        cnn_features = self.cnn_backbone(mel_normalized)  # [B, D, T', F']
        
        # Step 3: 展平为序列
        B, C, T, F = cnn_features.shape
        seq = cnn_features.view(B, C, -1).permute(0, 2, 1)  # [B, T*F, D]
        
        # Step 4: Transformer需要 [seq_len, B, D] 格式
        seq_t = seq.permute(1, 0, 2)  # [T*F, B, D]
        seq_encoded = self.pos_encoding(seq_t)
        encoded = self.transformer_encoder(seq_encoded)  # [T*F, B, D]
        
        # 转回 [B, T, D]
        output = encoded.permute(1, 0, 2)  # [B, T*F, D]
        output = self.output_norm(output)
        
        if return_dict:
            return {
                'features': output,
                'mel_spec': mel_db,
                'cnn_features': cnn_features,
            }
        
        return output
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """前向传播 (兼容接口)"""
        return self.extract_features(waveform)
    
    def get_pooled_output(self, waveform: torch.Tensor) -> torch.Tensor:
        """获取全局池化的固定长度向量 (用于非时序任务)"""
        features = self.extract_features(waveform)  # [B, T, D]
        pooled = features.mean(dim=1)  # [B, D]
        return pooled


class MultiScaleAudioEncoder(nn.Module):
    """
    多尺度音频编码器 (增强版)
    在不同时间分辨率上提取特征并融合
    """
    
    def __init__(
        self,
        base_encoder: AudioEncoder,
        num_scales: int = 3,
        fusion_method: str = "concat",  # concat, attention, weighted
    ):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.num_scales = num_scales
        self.fusion_method = fusion_method
        hidden_dim = base_encoder.hidden_dim
        
        # 不同尺度的池化/采样层
        self.scale_poolers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2**i, stride=2**i) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])
        
        # 融合层
        if fusion_method == "concat":
            self.fusion_proj = nn.Linear(hidden_dim * num_scales, hidden_dim)
        elif fusion_method == "attention":
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_dim * num_scales, num_scales),
                nn.Softmax(dim=-1),
            )
            self.fusion_proj = nn.Linear(hidden_dim * num_scales, hidden_dim)
        else:  # weighted
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.fusion_proj = nn.Identity()
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """多尺度特征融合"""
        base_features = self.base_encoder.extract_features(waveform)
        base_t = base_features.permute(0, 2, 1)  # [B, D, T]
        
        scale_features = [base_t]
        for pooler in self.scale_poolers[1:]:
            scaled = pooler(base_t)
            # 上采样回原长度 (或padding)
            scaled = nn.functional.interpolate(scaled, size=base_t.shape[-1], mode='nearest')
            scale_features.append(scaled)
        
        # 拼接所有尺度
        multi_scale = torch.cat(scale_features, dim=1)  # [B, D*num_scales, T]
        
        # 融合
        fused = self.fusion_proj(multi_scale.permute(0, 2, 1))  # [B, T, D]
        
        return fused
