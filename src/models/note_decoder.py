"""
Note预测头 - 自回归Transformer解码器

将音频特征解码为Note Token序列，支持Teacher Forcing训练和自回归推理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NoteDecoder(nn.Module):
    """
    自回归Transformer解码器用于Note序列生成
    
    Token词汇表定义:
        特殊Token:
            PAD = 0   (填充)
            SOS = 1   (序列开始)
            EOS = 2   (序列结束)
            MASK = 3  (掩码，可选)
            
        Note类型Token (4-6):
            TYPE_TAP = 4
            TYPE_HOLD = 5  
            TYPE_SLIDE = 6
            
        相对时间偏移Token (7-70): 
            量化到0.25拍精度，范围0-15拍
            
        X位置Token (71-79):
            离散化为9档 [-1.0, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.0]
            
        Hold时长Token (80-111): 
            仅Hold/Slide使用，量化到0.25拍精度
            
        方向Token (112-113):
            ABOVE = 112, BELOW = 113
    """
    
    # Token常量
    PAD = 0; SOS = 1; EOS = 2; MASK = 3
    TYPE_TAP = 4; TYPE_HOLD = 5; TYPE_SLIDE = 6
    TIME_OFFSET_START = 7; TIME_OFFSET_END = 70
    POS_X_START = 71; POS_X_END = 79
    HOLD_TIME_START = 80; HOLD_TIME_END = 111
    NOTE_ABOVE = 114; NOTE_BELOW = 115
    
    # 默认词表大小
    DEFAULT_VOCAB_SIZE = 128
    
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = None,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        share_embeddings: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        
        # ===== Token & 位置嵌入 =====
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # 可学习的缩放因子 (类似GPT-2)
        self.embed_scale = nn.Parameter(torch.tensor(1.0))
        
        # Dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # ===== Transformer Decoder层 =====
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        
        # ===== 输出投影 =====
        if share_embeddings:
            # 权重绑定 (减少参数量)
            self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.output_projection.weight = self.token_embedding.weight
        else:
            self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # ===== 因果掩码 (预计算以提高效率) =====
        self.register_buffer(
            'causal_mask',
            self._generate_causal_mask(max_seq_len),
            persistent=False
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    @staticmethod
    def _generate_causal_mask(size: int) -> torch.Tensor:
        """生成因果注意力掩码 (上三角为True表示被遮挡)"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def create_attention_mask(self, target_tokens: torch.Tensor, pad_id: int = PAD) -> torch.Tensor:
        """
        创建因果注意力掩码 (2D, 用于 batch_first=True)
        
        Args:
            target_tokens: [batch_size, seq_len]
            pad_id: padding token id (未使用，保留兼容性)
            
        Returns:
            mask: [seq_len, seq_len] bool tensor (因果掩码，上三角为True)
        """
        seq_len = target_tokens.shape[-1]
        
        # 因果掩码 (上三角为 True，表示被遮挡)
        causal_mask = self.causal_mask[:seq_len, :seq_len]  # [T, T]
        
        return causal_mask
    
    def embed_input(self, tokens: torch.Tensor) -> torch.Tensor:
        """嵌入输入tokens"""
        B, T = tokens.shape
        
        token_emb = self.token_embedding(tokens) * self.embed_scale
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        embedded = self.embed_dropout(token_emb + pos_emb)
        return embedded
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        前向传播 (训练模式: Teacher Forcing)
        
        Args:
            encoder_output: [batch_size, src_len, hidden_dim] 来自音频编码器的输出
            target_tokens: [batch_size, tgt_len] 目标Note token序列 (训练时提供)
            target_padding_mask: [batch_size, tgt_len] 目标端padding掩码
            memory_key_padding_mask: [batch_size, src_len] 源端memory padding掩码
            return_logits: 是否返回logits (False则返回隐藏状态)
            
        Returns:
            logits: [batch_size, tgt_len, vocab_size] 或 hidden states
        """
        assert target_tokens is not None, "Training requires target_tokens"
        
        B, T = target_tokens.shape
        
        # 输入嵌入
        decoder_input = self.embed_input(target_tokens)
        
        # 创建因果+Padding掩码
        tgt_mask = self.create_attention_mask(target_tokens)
        
        # Transformer解码
        decoded = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask[:T, :T],
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        decoded = self.final_layer_norm(decoded)
        
        if return_logits:
            logits = self.output_projection(decoded)
            return logits
        
        return decoded
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        sos_token_id: int = SOS,
        eos_token_id: int = EOS,
        pad_token_id: int = PAD,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,       # 0表示不使用
        top_p: float = 1.0,    # 1.0表示不使用
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        condition_fn=None,     # 可选的条件函数
    ) -> torch.Tensor:
        """
        自回归生成 (推理模式)
        
        Args:
            encoder_output: [batch_size, src_len, hidden_dim] 音频特征
            sos_token_id: 开始token
            eos_token_id: 结束token
            max_length: 最大生成长度
            temperature: 采样温度 (>0随机, <0贪婪)
            top_k: Top-k采样参数
            top_p: Nucleus sampling参数
            repetition_penalty: 重复惩罚
            condition_fn: 条件约束函数 (可选)
            
        Returns:
            generated: [batch_size, generated_seq_len] 生成的token序列
        """
        B = encoder_output.shape[0]
        device = encoder_output.device
        
        # 初始状态: 只有SOS token
        generated = torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)
        
        # 缓存 (优化: 避免重复计算已生成的部分)
        past_key_values = None if use_cache else None
        
        for step in range(max_length):
            # 只取最后的部分进行解码 (配合cache使用)
            current_input = generated if not use_cache else generated[:, -1:]
            
            # 嵌入当前输入
            if use_cache and step > 0:
                # 使用缓存时只处理最新token
                current_embedded = self.embed_input(current_input)
            else:
                # 第一步或无缓存时处理全部
                current_embedded = self.embed_input(current_input)
                
                # 扩展维度以匹配Transformer Decoder期望的 [T, B, D] 格式
                current_embedded = current_embedded.permute(1, 0, 2)
                encoder_for_decode = encoder_output.permute(1, 0, 2)
            
            # 解码 (简化版，实际应实现KV-Cache)
            if step == 0:
                # 第一次: 处理完整序列
                logits = self.forward(encoder_output, generated)
            else:
                # 后续步骤: 只处理新token (此处为简化实现，完整版本需修改forward)
                logits = self.forward(encoder_output, generated)
            
            next_token_logits = logits[:, -1, :]  # [B, V]
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for prev_token in set(generated[0].tolist()):
                    if next_token_logits[0, prev_token] < 0:
                        next_token_logits[0, prev_token] *= repetition_penalty
                    else:
                        next_token_logits[0, prev_token] /= repetition_penalty
            
            # 温度缩放
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (Nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除超过阈值的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            else:
                # 贪婪解码 (温度<=0)
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # 条件约束 (可选)
            if condition_fn is not None:
                next_token = condition_fn(generated, next_token, step)
            
            # 追加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 终止检查
            if (next_token.squeeze(-1) == eos_token_id).all():
                break
        
        return generated
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = PAD,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        计算交叉熵损失
        
        Args:
            logits: [B, T, V]
            targets: [B, T]
            ignore_index: 忽略的token (通常是PAD)
            label_smoothing: 标签平滑系数
            
        Returns:
            loss: scalar tensor
        """
        # Reshape for cross entropy
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        
        if label_smoothing > 0:
            # Label smoothing
            confidence = 1.0 - label_smoothing
            smooth_value = label_smoothing / V
            one_hot = torch.zeros_like(logits_flat).scatter(1, targets_flat.unsqueeze(1), 1.0)
            smooth_targets = one_hot * confidence + (1 - one_hot) * smooth_value
            loss = (-smooth_targets * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
        
        return loss.mean()


class NoteDecoderWithCrossAttention(NoteDecoder):
    """
    带显式条件注入的Note解码器
    
    支持通过Cross Attention注入LLM生成的风格条件向量。
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get('hidden_dim', 512)
        
        # 条件Cross-Attention层
        self.condition_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=kwargs.get('num_heads', 8),
            dropout=kwargs.get('dropout', 0.1),
            batch_first=True,
        )
        
        # 条件门控
        self.condition_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # 条件投影
        self.condition_projection = nn.Sequential(
            nn.Linear(kwargs.get('condition_dim', 384), hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward_with_condition(
        self,
        encoder_output: torch.Tensor,
        target_tokens: torch.Tensor,
        condition_embedding: torch.Tensor,  # [B, cond_dim]
        **kwargs
    ) -> torch.Tensor:
        """
        带条件的解码
        
        Args:
            condition_embedding: LLM生成的风格/参数嵌入
        """
        # 基础解码
        hidden_states = super().forward(
            encoder_output, target_tokens, return_logits=False, **kwargs
        )
        
        # 投影条件
        projected_cond = self.condition_projection(condition_embedding).unsqueeze(1)
        
        # Cross attention注入
        conditioned, attn_weights = self.condition_cross_attn(
            query=hidden_states,
            key=projected_cond,
            value=projected_cond,
        )
        
        # 门控融合
        gate = self.condition_gate(torch.cat([hidden_states, conditioned], dim=-1))
        output = hidden_states + gate * conditioned
        
        # 投影到logits
        logits = self.output_projection(output)
        return logits
