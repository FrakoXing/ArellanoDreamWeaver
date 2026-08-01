"""
ArellanoDreamWeaver - 条件 Diffusion U-Net
=========================================
用于从噪声逐步生成高质量谱面的去噪网络
支持: 分类器-free 引导、时间步条件注入、跨注意力条件
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Diffusion 模型配置"""
    # 网络结构
    in_channels: int = 64        # 输入通道 (谱面特征维度)
    out_channels: int = 64       # 输出通道
    base_channels: int = 128     # 基础通道数
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    n_res_blocks: int = 2        # 每个 resolution 的残差块数量
    attention_resolutions: Tuple[int, ...] = (16, 8)  # 使用注意力的分辨率
    
    # 条件
    condition_dim: int = 512     # 来自 Transformer 的条件维度
    time_embed_dim: int = 256    # 时间步嵌入维度
    
    # 条件注入方式
    condition_type: str = "crossattn"  # crossattn, adagn, film
    
    # Dropout (用于 classifier-free training)
    dropout: float = 0.1


class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码风格的时间步嵌入"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size] 时间步张量 (0 到 T-1 的整数)
        Returns:
            [batch_size, embed_dim]
        """
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb_exp = torch.arange(half_dim, device=t.device).float() * (-emb_scale)
        
        emb = t.float().unsqueeze(1) * torch.exp(emb_exp).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """带时间步和条件注入的残差块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        condition_dim: int,
        dropout: float = 0.1,
        use_condition: bool = True,
        condition_type: str = "adagn"
    ):
        super().__init__()
        
        # GroupNorm + Conv1d (谱面是1D时序数据)
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 时间步投影
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        # 条件注入 (根据配置选择方式)
        self.use_condition = use_condition
        if use_condition:
            if condition_type == "film":
                # FiLM: Feature-wise Linear Modulation
                self.condition_proj = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(condition_dim, out_channels * 2)
                )
            elif condition_type == "adagn":
                # AdaGN: Adaptive Group Normalization
                self.condition_norm = nn.GroupNorm(min(32, out_channels), out_channels)
                self.condition_proj = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(condition_dim, out_channels * 2)
                )
            
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 跳跃连接
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                                if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.condition_type = condition_type
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T]
            time_emb: [B, time_embed_dim]
            condition: [B, T', condition_dim] 或 [B, condition_dim]
        Returns:
            [B, C_out, T]
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        
        # 注入时间步信息
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        
        # 注入条件信息
        if self.use_condition and condition is not None:
            if self.condition_type == "film":
                params = self.condition_proj(
                    condition.mean(dim=1) if condition.dim() == 3 else condition
                )
                scale, shift = params.chunk(2, dim=-1)
                h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
                
            elif self.condition_type == "adagn":
                params = self.condition_proj(
                    condition.mean(dim=1) if condition.dim() == 3 else condition
                )
                scale, shift = params.chunk(2, dim=-1)
                
                h = self.norm2(h)
                h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
                h = self.act(h)
                h = self.dropout(self.conv2(h))
                return self.skip_connection(x) + h
        
        h = self.act(self.norm2(h))
        h = self.dropout(self.conv2(h))
        
        return self.skip_connection(x) + h


class CrossAttentionBlock(nn.Module):
    """跨注意力块 (用于注入 Transformer 提取的条件)"""
    
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        
        self.heads = heads
        d_k = query_dim // heads
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.scale = d_k ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,       # [B, T_q, D]
        context: torch.Tensor  # [B, T_ctx, D_ctx]
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        q = self.to_q(x).view(B, T, self.heads, -1).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.heads, -1).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.heads, -1).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        
        return self.to_out(out)


class Downsample(nn.Module):
    """下采样 (使用带零填充的卷积避免棋盘伪影)"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样 (转置卷积)"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet1D(nn.Module):
    """
    🎨 条件 Diffusion U-Net for 谱面生成
    
    架构特点:
    1. 接收带噪谱面 x_t 和时间步 t 作为输入
    2. 通过跳跃连接保留细节
    3. 在多个分辨率层级注入来自 MultiTaskTransformer 的条件
    4. 预测噪声 ε (或直接预测 x_0)
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        ch = config.base_channels
        mults = config.channel_mults
        n_res = config.n_res_blocks
        cond_dim = config.condition_dim
        time_dim = config.time_embed_dim
        cond_type = config.condition_type
        
        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # 初始卷积
        self.init_conv = nn.Conv1d(config.in_channels, ch, kernel_size=3, padding=1)
        
        # === 下采样路径 ===
        self.down_blocks = nn.ModuleList([])
        self.down_samples = nn.ModuleList([])
        self.down_cross_attns = nn.ModuleList([])
        
        ch_in = ch
        for i, mult in enumerate(mults):
            ch_out = ch * mult
            
            res_blocks = nn.ModuleList([
                ResidualBlock(
                    ch_in if j == 0 else ch_out,
                    ch_out,
                    time_dim,
                    cond_dim,
                    config.dropout,
                    use_condition=(cond_type != "crossattn"),
                    condition_type=cond_type if cond_type != "crossattn" else "adagn"
                )
                for j in range(n_res)
            ])
            self.down_blocks.append(res_blocks)
            
            # 在指定分辨率添加交叉注意力
            resolution = 4096 // (2 ** i)  # 假设初始分辨率
            if resolution in config.attention_resolutions or cond_type == "crossattn":
                self.down_cross_attns.append(CrossAttentionBlock(ch_out, cond_dim))
            else:
                self.down_cross_attns.append(None)
            
            # 不是最后一层则下采样
            if i < len(mults) - 1:
                self.down_samples.append(Downsample(ch_out))
            
            ch_in = ch_out
        
        # === 中间层 ===
        self.mid_block1 = ResidualBlock(ch_in, ch_in, time_dim, cond_dim, config.dropout)
        self.mid_cross_attn = CrossAttentionBlock(ch_in, cond_dim)
        self.mid_block2 = ResidualBlock(ch_in, ch_in, time_dim, cond_dim, config.dropout)
        
        # === 上采样路径 ===
        self.up_blocks = nn.ModuleList([])
        self.up_samples = nn.ModuleList([])
        self.up_cross_attns = nn.ModuleList([])
        
        for i, mult in enumerate(reversed(mults)):
            ch_out = ch * mult
            
            res_blocks = nn.ModuleList([
                ResidualBlock(
                    ch_in if j == 0 else ch_out + ch_out,  # 有 skip connection
                    ch_out,
                    time_dim,
                    cond_dim,
                    config.dropout,
                    use_condition=(cond_type != "crossattn"),
                    condition_type=cond_type if cond_type != "crossattn" else "adagn"
                )
                for j in range(n_res + 1)  # +1 因为要处理 skip connection
            ])
            self.up_blocks.append(res_blocks)
            
            resolution = 4096 // (2 ** (len(mults) - 1 - i))
            if resolution in config.attention_resolutions or cond_type == "crossattn":
                self.up_cross_attns.append(CrossAttentionBlock(ch_out, cond_dim))
            else:
                self.up_cross_attns.append(None)
            
            if i < len(mults) - 1:
                self.up_samples.append(Upsample(ch_out))
            
            ch_in = ch_out
        
        # 最终输出层
        self.final_norm = nn.GroupNorm(min(32, ch), ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv1d(ch, config.out_channels, kernel_size=3, padding=1)
        
        # 条件的全局表示 (用于非 cross-attn 方式)
        self.condition_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torchTensor, List[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: [B, C, T] 带噪谱面
            timestep: [B] 时间步 (0~T-1)
            condition: [B, T_cond, D_cond] 来自 Transformer 的条件
                       如果为 None，用于 classifier-free 训练的无条件情况
            
        Returns:
            noise_pred: [B, C, T] 预测的噪声
        """
        B, _, T = x.shape
        
        # 时间嵌入
        t_emb = self.time_embed(timestep)  # [B, time_dim]
        
        # 处理条件
        if condition is not None:
            global_cond = self.condition_encoder(
                condition.mean(dim=1) if condition.dim() == 3 else condition
            )
        else:
            # 无条件的零向量 (classifier-free)
            global_cond = torch.zeros(B, self.config.condition_dim, device=x.device)
            condition = torch.zeros(B, 1, self.config.condition_dim, device=x.device)
        
        # === 初始卷积 ===
        h = self.init_conv(x)
        
        # === 下采样 (Encoder) ===
        skips = []
        for i, (res_blocks, cross_attn) in enumerate(zip(self.down_blocks, self.down_cross_attns)):
            for block in res_blocks:
                h = block(h, t_emb, condition if self.config.condition_type != "crossattn" else None)
            
            if cross_attn is not None:
                # Cross-attention: h 是 [B,C,T], 需要 transpose 成 [B,T,C]
                h_t = h.transpose(1, 2)
                h_t = cross_attn(h_t, condition)
                h = h_t.transpose(1, 2)
            
            skips.append(h)
            
            if i < len(self.down_samples):
                h = self.down_samples[i](h)
        
        # === 中间层 ===
        h = self.mid_block1(h, t_emb, global_cond)
        h_mid = h.transpose(1, 2)
        h_mid = self.mid_cross_attn(h_mid, condition)
        h = h_mid.transpose(1, 2)
        h = self.mid_block2(h, t_emb, global_cond)
        
        # === 上采样 (Decoder) ===
        for i, (res_blocks, cross_attn) in enumerate(zip(self.up_blocks, self.up_cross_attns)):
            # Skip connection
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for j, block in enumerate(res_blocks):
                h = block(h, t_emb, global_cond if self.config.condition_type != "crossattn" else None)
                
                if j == 0 and cross_attn is not None:
                    h_t = h.transpose(1, 2)
                    h_t = cross_attn(h_t, condition)
                    h = h_t.transpose(1, 2)
            
            if i < len(self.up_samples):
                h = self.up_samples[i](h)
        
        # === 最终输出 ===
        h = self.final_act(self.final_norm(h))
        output = self.final_conv(h)
        
        if return_features:
            return output, []
        return output


# ============================================================
# 完整的 Diffusion 模型封装
# ============================================================

class ChartDiffusionModel(nn.Module):
    """
    完整的谱面扩散模型，包含 U-Net + 噪声调度
    """
    
    def __init__(
        self,
        unet_config: Optional[DiffusionConfig] = None,
        n_timesteps: int = 1000,
        beta_schedule: str = "cosine"
    ):
        super().__init__()
        
        if unet_config is None:
            unet_config = DiffusionConfig()
        
        self.unet = ConditionalUNet1D(unet_config)
        self.n_timesteps = n_timesteps
        self.beta_schedule = beta_schedule
        
        # 注册噪声调度缓冲区
        self._register_noise_schedule()
    
    def _register_noise_schedule(self):
        """注册噪声调度参数为 buffer"""
        T = self.n_timesteps
        
        if self.beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, T)
        elif self.beta_schedule == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = torch.clamp(betas, 0.0001, 0.02)
        elif self.beta_schedule == "sqrt_linear_cosine":
            # 更复杂的调度 (可学习版本的基础)
            steps = torch.linspace(0, 1, T)
            betas = torch.sqrt(steps) * 0.02 + 0.0001
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        # 计算需要的量
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为 buffer (不参与梯度计算但会随模型移动设备)
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        
        # 用于采样的预计算值
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas).float())
        
        # 后验方差
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped', 
                             torch.log(torch.clamp(posterior_variance, min=1e-20)).float())
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod).float())
        self.register_buffer('posterior_mean_coef2',
                             (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod).float())
    
    @torch.no_grad()
    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        前向扩散过程: q(x_t | x_0)
        
        Args:
            x0: [B, C, T] 原始干净谱面
            noise: [B, C, T] 标准高斯噪声
            timesteps: [B] 每个样本的时间步
        
        Returns:
            x_t: 加噪后的谱面
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(-1).unsqueeze(-1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def predict_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测噪声 ε_θ(x_t, t, c)
        """
        return self.unet(x_t, timesteps, condition)
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        unconditional_condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        单步去噪采样: p_θ(x_{t-1} | x_t)
        支持分类器-free 引导
        """
        B = x_t.size(0)
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        
        if guidance_scale > 1.0 and unconditional_condition is not None:
            # 分类器-free 引导
            noise_cond = self.predict_noise(x_t, t_tensor, condition)
            noise_uncond = self.predict_noise(x_t, t_tensor, unconditional_condition)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.predict_noise(x_t, t_tensor, condition)
        
        # 计算均值
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        mean = coef1 * x_t + coef2 * noise_pred
        
        # 添加噪声 (除了最后一步)
        if t > 0:
            variance = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean


if __name__ == "__main__":
    print("=" * 60)
    print("ArellanoDreamWeaver - Conditional UNet Test")
    print("=" * 60)
    
    # 创建模型
    config = DiffusionConfig()
    model = ConditionalUNet1D(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params/1e6:.1f}M")
    
    # 测试前向
    B, C, T = 2, 64, 256
    dummy_x = torch.randn(B, C, T)
    dummy_t = torch.randint(0, 1000, (B,))
    dummy_cond = torch.randn(B, 128, 512)  # Transformer 条件
    
    print(f"\nInput shapes:")
    print(f"  Noisy chart: {dummy_x.shape}")
    print(f"  Timestep: {dummy_t.shape}")
    print(f"  Condition: {dummy_cond.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_x, dummy_t, dummy_cond)
    
    print(f"\nOutput shape (predicted noise): {output.shape}")
    
    # 测试完整 Diffusion Model
    diffusion_model = ChartDiffusionModel(n_timesteps=1000)
    
    # 测试加噪过程
    clean_chart = torch.randn(B, C, T)
    noise = torch.randn_like(clean_chart)
    timesteps = torch.tensor([500, 300])
    
    noisy_chart = diffusion_model.add_noise(clean_chart, noise, timesteps)
    print(f"\nNoisy chart at t={timesteps.tolist()}: {noisy_chart.shape}")
    
    # 测试噪声预测
    pred_noise = diffusion_model.predict_noise(noisy_chart, timesteps, dummy_cond)
    print(f"Predicted noise: {pred_noise.shape}")
    
    print("\n✓ ConditionalUNet & Diffusion test passed!")
