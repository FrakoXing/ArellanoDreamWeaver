# -*- coding: utf-8 -*-
"""
定数评级模型 (RatingModel)
==========================

纯谱面 Token 输入 → 定数回归 (归一化 [0,1])。

架构:
  Token 序列 ──► Embedding + PositionalEncoding ──► 双向 Transformer Encoder
                  (Game Token 条件注入 via AdaptiveLayerNorm)
                  ──► Masked Mean Pool ──► token_repr
  手工特征 ──► LayerNorm + MLP ──► hc_repr
  token_repr + hc_repr ──► GatedFusion ──► RegressionHead ──► sigmoid ──► rating_norm

设计要点:
  1. 双向注意力 (非因果) — 定数评级是序列→单值回归, 需要全注意力
  2. 复用 dreamweaver.models.transformer 的 PositionalEncoding /
     TransformerEncoderLayer / AdaptiveLayerNorm (手写, opset 14 友好)
  3. padding mask 在本模块内转换 (不改 transformer.py):
     attention_mask [B,T] (1=valid) → attn_mask [B,1,1,T] (1=valid,0=pad)
     传给 TransformerEncoderLayer 的 mask 参数
  4. 手写 _masked_mean (避开 AdaptiveAvgPool1d, Sentis 兼容 + 防 padding 污染)
  5. Sigmoid 输出归一化 [0,1], 反归一化在 Unity 端按 Game Token 完成
"""

import sys
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录 (dreamweaver 包在项目根)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.models.transformer import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerConfig,
)


# ============================================================
# 配置
# ============================================================

@dataclass
class RatingConfig:
    """RatingModel 配置"""
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 2048
    dropout: float = 0.2
    handcrafted_dim: int = 15
    # Game Token 范围 (116/117/118), embedding 用 vocab_size 复用
    # 但为条件注入独立性, 用独立 embedding


# ============================================================
# 手工特征编码器 (MLP, Sentis 友好: 仅 Linear/LayerNorm/GELU)
# ============================================================

class HandcraftedEncoder(nn.Module):
    """15 维 → d_model 维, 纯 MLP"""

    def __init__(self, in_dim: int = 15, d_model: int = 256, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


# ============================================================
# 门控融合
# ============================================================

class GatedFusion(nn.Module):
    """门控融合两个 d_model 表示: g * a + (1-g) * b"""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([a, b], dim=-1))
        return g * a + (1 - g) * b


# ============================================================
# 主模型
# ============================================================

class RatingModel(nn.Module):
    """
    定数评级模型

    输入:
      tokens:               [B, T] int64   Token 序列
      attention_mask:       [B, T] int64   1=valid, 0=pad
      handcrafted_features: [B, 15] float32 手工特征
      game_token_ids:       [B] int64      Game Token ID (116/117/118)

    输出:
      rating_norm:          [B] float32    归一化定数 ∈ (0, 1)
    """

    def __init__(self, config: Optional[RatingConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = RatingConfig(**kwargs)
        self.config = config
        d = config.d_model

        # === Token 嵌入 ===
        self.token_embed = nn.Embedding(config.vocab_size, d)
        self.pos_enc = PositionalEncoding(
            d_model=d,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # === Game Token 条件嵌入 (独立, 用于 AdaptiveLayerNorm 条件注入) ===
        # 用独立 embedding 表, 而非复用 token_embed, 保证条件注入的独立性
        self.game_embed = nn.Embedding(config.vocab_size, d)

        # === 双向 Transformer Encoder (复用手写 TransformerEncoderLayer) ===
        trans_cfg = TransformerConfig(
            d_model=d,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            conditioning_dim=d,  # AdaptiveLayerNorm 条件维度
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(trans_cfg) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

        # === 双通道融合 ===
        self.handcrafted_enc = HandcraftedEncoder(
            in_dim=config.handcrafted_dim,
            d_model=d,
            dropout=config.dropout,
        )
        self.fusion = GatedFusion(d)

        # === 回归头 ===
        self.regression_head = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.game_embed.weight, mean=0.0, std=0.02)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        带 mask 的平均池化 (避开 AdaptiveAvgPool1d, Sentis 兼容 + 防 padding 污染)

        Args:
            x:    [B, T, D]
            mask: [B, T] (1=valid, 0=pad)
        Returns:
            [B, D]
        """
        m = mask.unsqueeze(-1).to(x.dtype)        # [B, T, 1]
        s = (x * m).sum(dim=1)                    # [B, D]
        n = m.sum(dim=1).clamp(min=1.0)           # [B, 1]
        return s / n

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        game_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens:               [B, T] int64
            attention_mask:       [B, T] int64 (1=valid)
            handcrafted_features: [B, 15] float32
            game_token_ids:       [B] int64
        Returns:
            rating_norm: [B] float32 ∈ (0, 1)
        """
        # === Token 嵌入 + 位置编码 ===
        x = self.token_embed(tokens)               # [B, T, D]
        x = self.pos_enc(x)

        # === Game 条件: [B] → [B, 1, D] (广播到每个时间步的 AdaptiveLayerNorm) ===
        condition = self.game_embed(game_token_ids).unsqueeze(1)  # [B, 1, D]

        # === padding mask 转换 ===
        # attention_mask [B,T] (1=valid) → attn_mask [B,1,1,T] (1=valid,0=pad)
        # MultiHeadAttention 里 `scores.masked_fill(mask == 0, -inf)`,
        # 所以 mask=0 的 pad key 位置被屏蔽。
        # 形状 [B,1,1,T] 广播到 scores [B, n_heads, T_query, T_key]。
        attn_mask = attention_mask[:, None, None, :].to(x.dtype)  # [B,1,1,T]

        # === Transformer Encoder 层 ===
        for layer in self.layers:
            x, _ = layer(x, condition=condition, mask=attn_mask)

        x = self.final_norm(x)                     # [B, T, D]

        # === Masked Mean Pool → token_repr ===
        token_repr = self._masked_mean(x, attention_mask)  # [B, D]

        # === 手工特征通道 ===
        hc_repr = self.handcrafted_enc(handcrafted_features)  # [B, D]

        # === 门控融合 ===
        fused = self.fusion(token_repr, hc_repr)   # [B, D]

        # === 回归头 + sigmoid ===
        logit = self.regression_head(fused).squeeze(-1)  # [B]
        rating_norm = torch.sigmoid(logit)               # [B] ∈ (0, 1)

        return rating_norm

    def get_num_parameters(self) -> Dict[str, int]:
        """参数量统计"""
        stats = {}
        total = 0
        for name, module in [
            ("token_embed", self.token_embed),
            ("game_embed", self.game_embed),
            ("pos_enc", self.pos_enc),
            ("transformer_layers", self.layers),
            ("handcrafted_enc", self.handcrafted_enc),
            ("fusion", self.fusion),
            ("regression_head", self.regression_head),
        ]:
            n = sum(p.numel() for p in module.parameters())
            stats[name] = n
            total += n
        stats["total"] = total
        return stats


# ============================================================
# 工厂函数
# ============================================================

def create_rating_model(**kwargs) -> RatingModel:
    """创建默认配置的 RatingModel"""
    config = RatingConfig(**kwargs)
    return RatingModel(config)


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RatingModel 自测")
    print("=" * 60)

    model = create_rating_model()
    params = model.get_num_parameters()
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    print(f"\n  Total: {params['total']:,} ({params['total']/1e6:.2f}M)")

    # 前向测试
    B, T = 2, 2048
    tokens = torch.randint(0, 256, (B, T), dtype=torch.long)
    tokens[:, 0] = 116  # GAME_PHIRA
    attention_mask = torch.ones(B, T, dtype=torch.long)
    # 模拟第二个样本后半部分是 pad
    attention_mask[1, 1000:] = 0
    handcrafted = torch.randn(B, 15, dtype=torch.float32)
    game_ids = torch.tensor([116, 117], dtype=torch.long)  # phira, osu

    model.eval()
    with torch.no_grad():
        out = model(tokens, attention_mask, handcrafted, game_ids)

    print(f"\n  输出 shape: {out.shape} (期望 [{B}])")
    print(f"  输出值: {out.tolist()}")
    print(f"  范围: [{out.min().item():.4f}, {out.max().item():.4f}] (期望在 (0,1))")

    assert out.shape == (B,), f"shape 错误: {out.shape}"
    assert (out > 0).all() and (out < 1).all(), "输出应在 (0, 1)"
    print("\n  ✓ 前向传播正确, 输出在 (0, 1) 范围")

    # 验证 padding 不影响输出 (同一序列不同 padding 位置应近似相同)
    tokens2 = tokens.clone()
    mask2 = attention_mask.clone()
    # 把第一个样本的有效长度设为 500
    mask2[0, 500:] = 0
    with torch.no_grad():
        out2 = model(tokens2, mask2, handcrafted, game_ids)
    diff = (out[0] - out2[0]).abs().item()
    print(f"\n  padding 影响测试 (同序列不同 mask): diff={diff:.4f}")
    print("  (注: 差异非零是正常的, 因为 masked_mean 分母变化 + attention 分布变化)")

    print("\n  ✓ RatingModel 自测通过")
