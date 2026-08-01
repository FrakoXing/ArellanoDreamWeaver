"""
ArellanoDreamWeaver - 多任务 Transformer 编码器
=============================================
核心架构: 音乐特征 + 文本语义 + 谱面模式的联合建模
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Transformer 配置数据类"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 4096
    activation: str = "gelu"
    
    # 条件注入
    conditioning_dim: int = 512
    
    # 多任务头配置
    num_note_types: int = 4       # tap, hold, slide, none
    num_difficulty_levels: int = 6  # EZ/NM/HD/IN/AT/LEGACY
    num_styles: int = 10


class PositionalEncoding(nn.Module):
    """学习式位置编码 (支持扩展到更长序列)"""
    
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 使用可学习的位置嵌入 (比固定的 sinusoidal 更灵活)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # 缩放因子
        self.scale = math.sqrt(d_model)
        
        # 初始化
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            加上位置编码的 tensor
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = x * self.scale + self.pos_embedding(positions)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制 (带可选的 Flash Attention 支持)"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # QKV 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: 可选的 attention mask
            causal_mask: 是否使用因果掩码 (自回归)
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性投影并重塑为多头形式
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if causal_mask:
            seq_len = query.size(1)
            causal = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到值
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """前馈网络 (带 GELU 激活)"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveLayerNorm(nn.Module):
    """自适应层归一化 (用于条件 Diffusion 的条件注入)"""
    
    def __init__(self, d_model: int, conditioning_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        
        # 根据条件生成 scale 和 shift
        self.condition_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, d_model * 2)
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, seq_len, d_model]
            condition: 条件 [batch, conditioning_dim] 或 [batch, seq_len, conditioning_dim]
        """
        normed = self.norm(x)
        
        # 从条件生成调制参数
        params = self.condition_proj(condition)
        if params.dim() == 3:
            scale, shift = params.chunk(2, dim=-1)
        else:
            # 广播: condition 是全局的
            scale, shift = params.chunk(2, dim=-1)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return normed * (1 + scale) + shift


class TransformerEncoderLayer(nn.Module):
    """单层 Transformer 编码器 (Pre-LN 架构，训练更稳定)"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        # 自注意力 (带自适应 LN)
        self.norm1 = AdaptiveLayerNorm(config.d_model, config.conditioning_dim)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        
        # 前馈网络 (带自适应 LN)
        self.norm2 = AdaptiveLayerNorm(config.d_model, config.conditioning_dim)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 自注意力的残差连接
        residual = x
        x = self.norm1(x, condition if condition is not None else torch.zeros_like(x[:, :1, :]))
        attn_out, attn_weights = self.attn(x, x, x, mask=mask, **kwargs)
        x = residual + self.dropout(attn_out)
        
        # 前馈网络的残差连接
        residual = x
        x = self.norm2(x, condition if condition is not None else torch.zeros_like(x[:, :1, :]))
        ff_out = self.ff(x)
        x = residual + self.dropout(ff_out)
        
        info = {"attention_weights": attn_weights}
        return x, info


class CrossModalAttentionLayer(nn.Module):
    """跨模态交叉注意力层 (融合音乐和文本信息)"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,   # 主模态 (如谱面)
        kv: torch.Tensor,      # 辅助模态 (如文本条件)
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将辅助模态的信息融合到主模态中
        """
        residual = query
        q = self.norm_q(query)
        kv_normed = self.norm_kv(kv)
        
        cross_out, _ = self.cross_attn(q, kv_normed, kv_normed, mask=mask)
        
        return residual + self.dropout(self.norm_out(cross_out))


class MusicFeatureEncoder(nn.Module):
    """音乐特征编码器 (从音频/MIDI 提取的特征 → Transformer 表示)"""
    
    def __init__(
        self,
        input_dim: int = 128,      # mel频带数或MIDI特征维度
        d_model: int = 512,
        n_conv_layers: int = 3
    ):
        super().__init__()
        
        # 1D CNN 提取局部模式
        layers = []
        in_ch = input_dim
        out_channels = [256, 384, d_model]
        
        for i, out_ch in enumerate(out_channels[:n_conv_layers]):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            if i < n_conv_layers - 1:
                layers.append(nn.MaxPool1d(2))  # 下采样
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*layers)
        
        # 投影到 d_model
        self.projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim, seq_len]
        Returns:
            [batch_size, seq_len', d_model]
        """
        if x.dim() == 3 and x.size(-1) != self.cnn[0].in_channels:
            # 假设输入是 [B, T, C]，需要转置
            x = x.transpose(1, 2)
        
        x = self.cnn(x)
        x = x.transpose(1, 2)  # 回到 [B, T', C]
        x = self.projection(x)
        return self.norm(x)


class TextConditionEncoder(nn.Module):
    """文本条件编码器 (LLM 理解后的提示词表示)"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        max_length: int = 256
    ):
        super().__init__()
        
        # Token 嵌入
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_length, d_model)
        
        # 小型 Transformer 编码器文本
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 全局池化
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 初始化
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            全局文本表示: [batch_size, d_model] 和序列表示: [batch_size, seq_len, d_model]
        """
        B, T = input_ids.shape
        
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        # 创建 padding mask
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 序列级别表示
        seq_repr = x
        
        # 全局表示 (带 mask 的平均池化)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x_sum = (x * mask_expanded).sum(dim=1)
            x_len = mask_expanded.sum(dim=1).clamp(min=1)
            global_repr = x_sum / x_len
        else:
            global_repr = x.mean(dim=1)
        
        global_repr = self.output_proj(global_repr)
        
        return global_repr, seq_repr


class BPMFeatureEncoder(nn.Module):
    """BPM 特征编码器"""
    
    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = [128, 256, 512], output_dim: int = 512):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, bpm_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bpm_features: [batch_size, seq_len, 4] 
                          包含: [bpm_value, beat_position, time_signature_numerator, confidence]
        Returns:
            [batch_size, seq_len, output_dim]
        """
        return self.norm(self.mlp(bpm_features))


class MultiTaskTransformer(nn.Module):
    """
    🎯 多任务 Transformer - ArellanoDreamWeaver 的核心模型
    
    功能:
    1. 编码音乐音频特征
    2. 编码 LLM 理解后的文本提示词
    3. 编码 BPM/节拍信息
    4. 跨模态融合所有特征
    5. 为 Diffusion 模型提供条件表示
    6. 直接预测多个任务输出 (多任务学习正则化)
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # === 输入编码器 ===
        self.music_encoder = MusicFeatureEncoder(
            input_dim=128,  # mel频带
            d_model=config.d_model
        )
        
        self.text_encoder = TextConditionEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=4
        )
        
        self.bpm_encoder = BPMFeatureEncoder(
            input_dim=4,  # bpm, position, time_sig, confidence
            output_dim=config.d_model
        )
        
        # === 共享位置编码 ===
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # === Transformer 主干 (多层堆叠) ===
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # === 跨模态融合层 ===
        self.music_text_fusion = CrossModalAttentionLayer(
            d_model=config.d_model,
            n_heads=config.n_heads
        )
        
        # === 最终层归一化 ===
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # === 多任务输出头 ===
        self._build_output_heads()
    
    def _build_output_heads(self):
        """构建多任务输出头"""
        cfg = self.config
        d = cfg.d_model
        
        # 音符类型预测 (分类)
        self.note_head = nn.Sequential(
            nn.Linear(d, cfg.d_model // 2),
            nn.LayerNorm(cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_note_types)
        )
        
        # 音符位置预测 (回归)
        self.position_head = nn.Sequential(
            nn.Linear(d, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 2)  # [x_position, lane_position]
        )
        
        # 时序属性预测 (回归)
        self.timing_head = nn.Sequential(
            nn.Linear(d, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 2)  # [time, hold_duration]
        )
        
        # 难度评估 (分类)
        self.difficulty_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局池化
            nn.Flatten(),
            nn.Linear(d, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.num_difficulty_levels)
        )
        
        # 风格分类
        self.style_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.num_styles)
        )
        
        # Diffusion 条件输出 (关键! 用于指导扩散过程)
        self.diffusion_condition_head = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU()
        )
    
    def encode_music(self, audio_features: torch.Tensor) -> torch.Tensor:
        """编码音频特征"""
        music_emb = self.music_encoder(audio_features)
        return self.pos_encoding(music_emb)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码文本提示词，返回全局和序列表示"""
        global_repr, seq_repr = self.text_encoder(input_ids, attention_mask)
        return global_repr, seq_repr
    
    def encode_bpm(self, bpm_features: torch.Tensor) -> torch.Tensor:
        """编码 BPM 信息"""
        return self.bpm_encoder(bpm_features)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        bpm_features: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            audio_features: [B, T_audio, 128] mel频谱或其他音频特征
            text_input_ids: [B, T_text] token IDs
            text_attention_mask: [B, T_text]
            bpm_features: [B, T_bpm, 4] BPM 相关特征 (可选)
            src_mask: 可选的源序列掩码
            
        Returns:
            包含以下键的字典:
            - 'diffusion_condition': 给 Diffusion 模型的条件
            - 'note_logits': 音符类型 logits
            - 'position_pred': 位置预测
            - 'timing_pred': 时序预测
            - 'difficulty_logits': 难度分类
            - 'style_logits': 风格分类
            - 'attention_weights' (可选): 注意力权重
        """
        # === 各模态编码 ===
        music_repr = self.encode_music(audio_features)  # [B, T_m, D]
        text_global, text_seq = self.encode_text(text_input_ids, text_attention_mask)
        # text_global: [B, D], text_seq: [B, T_t, D]
        
        # === 组合主要输入 (以音乐为主) ===
        x = music_repr
        
        # 如果有 BPM 特征，融合进来
        if bpm_features is not None:
            bpm_emb = self.encode_bpm(bpm_features)
            x = x + bpm_emb  # 加性融合
        
        # === 扩展文本为全局条件 (广播到每个时间步) ===
        # text_global: [B, D] → [B, 1, D] 用于 AdaLN 的条件
        condition = text_global.unsqueeze(1)  # [B, 1, D]
        
        # 同时保留文本序列用于跨模态注意力
        text_condition = text_seq  # [B, T_t, D]
        
        # === Transformer 层处理 ===
        all_attentions = []
        for layer in self.layers:
            x, layer_info = layer(x, condition=condition, mask=src_mask)
            if return_attention:
                all_attentions.append(layer_info["attention_weights"])
            
            # 在中间层进行一次跨模态融合 (增强交互)
            # (这里可以设计更复杂的融合策略)
        
        # 最终跨模态融合 (将文本信息注入音乐表示)
        x = self.music_text_fusion(x, text_condition)
        
        # 最终归一化
        x = self.final_norm(x)
        
        # === 多任务输出 ===
        outputs = {}
        
        # Diffusion 条件 (最重要的输出!)
        outputs["diffusion_condition"] = self.diffusion_condition_head(x)  # [B, T, D]
        
        # 音符相关预测 (每个时间步)
        outputs["note_logits"] = self.note_head(x)           # [B, T, num_note_types]
        outputs["position_pred"] = self.position_head(x)     # [B, T, 2]
        outputs["timing_pred"] = self.timing_head(x)         # [B, T, 2]
        
        # 全局属性预测 (整个谱面级别)
        # 转换维度: [B, T, D] → [B, D, T] 以便 AdaptiveAvgPool1d
        x_for_global = x.transpose(1, 2)
        outputs["difficulty_logits"] = self.difficulty_head(x_for_global)   # [B, num_difficulties]
        outputs["style_logits"] = self.style_head(x_for_global)             # [B, num_styles]
        
        if return_attention:
            outputs["attention_weights"] = all_attentions
        
        return outputs
    
    def get_num_parameters(self) -> Dict[str, int]:
        """获取各组件参数量统计"""
        stats = {}
        total = 0
        
        for name, module in [
            ("music_encoder", self.music_encoder),
            ("text_encoder", self.text_encoder),
            ("bpm_encoder", self.bpm_encoder),
            ("transformer_layers", self.layers),
            ("output_heads", self.note_head),
        ]:
            n_params = sum(p.numel() for p in module.parameters())
            stats[name] = n_params
            total += n_params
        
        stats["total"] = total
        return stats


# ============================================================
# 工厂函数 & 测试代码
# ============================================================

def create_default_model() -> MultiTaskTransformer:
    """创建默认配置的模型实例"""
    config = TransformerConfig()
    return MultiTaskTransformer(config)


if __name__ == "__main__":
    # 快速测试
    print("=" * 60)
    print("ArellanoDreamWeaver - MultiTaskTransformer Test")
    print("=" * 60)
    
    # 创建模型
    model = create_default_model()
    
    # 打印参数量
    params = model.get_num_parameters()
    for name, count in params.items():
        print(f"  {name}: {count:,} parameters")
    print(f"\n  Total: {params['total']:,} parameters")
    print(f"  (~{params['total']/1e6:.1f}M)")
    
    # 测试前向传播
    B, T_audio, T_text = 2, 256, 32
    
    dummy_audio = torch.randn(B, T_audio, 128)  # mel频谱
    dummy_text_ids = torch.randint(0, 50257, (B, T_text))
    dummy_text_mask = torch.ones(B, T_text)
    dummy_bpm = torch.randn(B, T_audio, 4)  # BPM特征
    
    print(f"\nInput shapes:")
    print(f"  Audio: {dummy_audio.shape}")
    print(f"  Text IDs: {dummy_text_ids.shape}")
    print(f"  BPM: {dummy_bpm.shape}")
    
    # 前向
    model.eval()
    with torch.no_grad():
        outputs = model(
            audio_features=dummy_audio,
            text_input_ids=dummy_text_ids,
            text_attention_mask=dummy_text_mask,
            bpm_features=dummy_bpm,
            return_attention=True
        )
    
    print(f"\nOutput shapes:")
    for name, tensor in outputs.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {name}: {tensor.shape}")
        else:
            print(f"  {name}: {type(tensor)}")
    
    print("\n✓ MultiTaskTransformer test passed!")
