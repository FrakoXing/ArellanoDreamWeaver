"""
ArellanoDreamWeaver - 噪声调度器 & 采样器
======================================
管理 Diffusion 过程的噪声调度和采样算法
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from abc import ABC, abstractmethod


class NoiseSchedule(ABC):
    """噪声调度基类"""
    
    @abstractmethod
    def get_betas(self, n_timesteps: int) -> torch.Tensor:
        """返回 beta 序列 [T]"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """调度名称"""
        pass


class LinearNoiseSchedule(NoiseSchedule):
    """线性噪声调度 (原始 DDPM)"""
    
    def __init__(self, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def get_betas(self, n_timesteps: int) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, n_timesteps)
    
    def get_name(self) -> str:
        return "linear"


class CosineNoiseSchedule(NoiseSchedule):
    """余弦噪声调度 ( Improved DDPM，更平滑的采样)"""
    
    def __init__(self, s: float = 0.008):
        """
        Args:
            s: 小偏移量，避免 t=0 时 beta=0
        """
        self.s = s
    
    def get_betas(self, n_timesteps: int) -> torch.Tensor:
        steps = torch.arange(n_timesteps + 1, dtype=torch.float64) / n_timesteps
        alphas_bar = torch.cos((steps + self.s) / (1 + self.s) * math.pi / 2).pow(2)
        alphas_bar = alphas_bar / alphas_bar[0]  # 归一化使 α_0_bar = 1
        
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        return torch.clamp(betas, 0.0001, 0.02).float()
    
    def get_name(self) -> str:
        return "cosine"


class SqrtLinearCosineSchedule(NoiseSchedule):
    """sqrt(linear) * cosine 混合调度 (更灵活)"""
    
    def __init__(self, beta_end: float = 0.02, s: float = 0.008):
        self.beta_end = beta_end
        self.s = s
    
    def get_betas(self, n_timesteps: int) -> torch.Tensor:
        # sqrt 线性部分
        steps_norm = torch.linspace(0, 1, n_timesteps)
        sqrt_linear = torch.sqrt(steps_norm) * self.beta_end
        
        # 余弦调制
        cos_modulation = torch.cos(torch.linspace(0, math.pi/2, n_timesteps))
        
        betas = sqrt_linear * cos_modulation
        betas = torch.clamp(betas, 0.0001, 0.02)
        
        return betas.float()
    
    def get_name(self) -> str:
        return "sqrt_linear_cosine"


class NoiseScheduler:
    """
    完整的噪声调度器
    
    管理:
    - 前向扩散 q(x_t | x_{t-1})
    - 后验分布 q(x_{t-1} | x_t, x_0)
    - 预计算的各种系数 (用于高效采样)
    """
    
    def __init__(
        self,
        schedule: Union[NoiseSchedule, str] = "cosine",
        n_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.n_timesteps = n_timesteps
        
        # 创建或选择调度
        if isinstance(schedule, str):
            schedule_map = {
                "linear": LinearNoiseSchedule(beta_start, beta_end),
                "cosine": CosineNoiseSchedule(),
                "sqrt_linear_cosine": SqrtLinearCosineSchedule(beta_end)
            }
            schedule_obj = schedule_map.get(schedule, CosineNoiseSchedule())
        else:
            schedule_obj = schedule
        
        self.schedule_type = schedule_obj.get_name()
        
        # 获取 betas 并预计算所有需要的量
        betas = schedule_obj.get_betas(n_timesteps)
        self._setup_buffers(betas)
    
    def _setup_buffers(self, betas: torch.Tensor):
        """预计算并注册所有缓冲区"""
        T = len(betas)
        
        # 基础量
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为 buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 用于采样的计算
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recipm1_alphas', torch.sqrt(1.0 / alphas - 1))
        
        # 后验分布参数 q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散过程: q(x_t | x_0) = √(ᾱₜ)·x₀ + √(1-ᾱₑ)·ε
        
        Args:
            x0: 原始数据 [B, C, T] 或 [B, T, C]
            noise: 噪声 ε ~ N(0, I)，形状同 x0
            timesteps: 时间步 [B]，每个元素在 [0, T-1]
            
        Returns:
            x_t: 加噪后的数据
            sqrt_alpha_prod, sqrt_one_minus_alpha_prod (可选，用于损失加权)
        """
        # 获取对应时间步的系数
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()     # [B]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()  # [B]
        
        # 调整维度以便广播: [B] → [B, 1, 1] 或 [B, 1]
        while sqrt_alpha_prod.dim() < x0.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        x_t = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
        
        return x_t, sqrt_alpha_prod, sqrt_one_minus_alpha_prod
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        单步去噪 (DDIM 或 DDPM)
        
        预测 x_0 并从后验中采样 x_{t-1}
        """
        # 获取当前时间步的值
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        beta_t = self.betas[timestep]
        
        # 预测 x_0 (假设模型输出是噪声 ε)
        pred_x0 = (sample - beta_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # 可选的 clip
        
        # 计算方向指向 xt 的噪声
        pred_dir = (1 - alpha_prod_t_prev - sigma**2).sqrt() * model_output
        
        # 计算 sigma (DDIM 随机性控制)
        sigma = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t).sqrt()
        sigma = 0.0 if timestep == 0 else sigma
        
        # 计算均值
        prev_sample_mean = alpha_prod_t_prev.sqrt() * pred_x0 + pred_dir
        
        # 如果 sigma > 0，添加噪声
        if sigma > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample_mean + sigma * noise
        else:
            prev_sample = prev_sample_mean
        
        return prev_sample
    
    def get_variance(self, timestep: int, predicted_variance=None) -> tuple:
        """获取后验方差"""
        variance = self.posterior_variance[timestep]
        log_variance = self.posterior_log_variance_clipped[timestep]
        
        return variance, log_variance
    
    @property
    def device(self) -> torch.device:
        """获取当前设备"""
        return self.betas.device
    
    def to(self, device: torch.device):
        """移动到指定设备"""
        for name, buf in self.__dict__.items():
            if isinstance(buf, torch.Tensor):
                setattr(self, name, buf.to(device))
        return self
    
    def __repr__(self) -> str:
        return f"NoiseScheduler(schedule={self.schedule_type}, timesteps={self.n_timesteps})"


# ============================================================
# 采样器
# ============================================================

class DiffusionSampler(ABC):
    """采样器基类"""
    
    @abstractmethod
    def sample(
        self,
        model,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """执行完整采样过程"""
        pass


class DDPMSampler(DiffusionSampler):
    """标准 DDPM 采样器 (马尔可夫链，需要 T 步)"""
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        guidance_scale: float = 1.0,
        unconditional_condition: Optional[torch.Tensor] = None
    ):
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.unconditional_condition = unconditional_condition
    
    @torch.no_grad()
    def sample(
        self,
        model,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        执行 DDPM 采样 (逐步去噪)
        
        Args:
            model: 噪声预测模型 (接受 x_t, t, condition)
            shape: 输出张量形状，如 (B, C, T)
            condition: 条件张量
            
        Returns:
            生成的样本
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.scheduler.n_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            if self.guidance_scale > 1 and self.unconditional_condition is not None:
                # 分类器-free 引导
                noise_cond = model(x, t_batch, condition)
                noise_uncond = model(x, t_batch, self.unconditional_condition)
                noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(x, t_batch, condition)
            
            # 计算均值
            coef1 = self.scheduler.posterior_mean_coef1[t].view(-1, 1, 1)
            coef2 = self.scheduler.posterior_mean_coef2[t].view(-1, 1, 1)
            mean = coef1 * x + coef2 * noise_pred
            
            # 添加噪声 (最后一步除外)
            if t > 0:
                var = self.scheduler.posterior_variance[t].view(-1, 1, 1)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        
        return x


class DDIMSampler(DiffusionSampler):
    """
    DDIM 采样器 (非马尔可夫，可加速)
    
    特点:
    - 可以用远少于训练步数的步数生成 (如 50 步 vs 1000 步)
    - 通过 eta 参数控制随机性 (eta=0 为确定性生成)
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        n_steps: int = 50,          # DDIM 采样步数
        eta: float = 0.0,           # 随机性 (0=确定性, 1=随机)
        discretization: str = "uniform",  # 时间步离散化方式
        guidance_scale: float = 1.0,
        unconditional_condition: Optional[torch.Tensor] = None
    ):
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.eta = eta
        self.discretization = discretization
        self.guidance_scale = guidance_scale
        self.unconditional_condition = unconditional_condition
        
        # 生成采样时间表
        self._build_timestep_sequence()
    
    def _build_timestep_sequence(self):
        """构建 DDIM 采样时间序列 (子序列)"""
        T = self.scheduler.n_timesteps
        
        if self.discretization == "uniform":
            # 均匀选取
            indices = list(range(0, T, T // self.n_steps))
            while len(indices) < self.n_steps + 1:
                indices.append(T - 1)
            indices = sorted(set(indices))[-(self.n_steps + 1):]
        elif self.discretization == "quad":
            # 二次采样 (两端更密)
            import numpy as np
            frac = np.linspace(0, 1, self.n_steps + 1) ** 2
            indices = (frac * (T - 1)).astype(int).tolist()
        else:
            raise ValueError(f"Unknown discretization: {self.discretization}")
        
        self.timestep_seq = indices
    
    @torch.no_grad()
    def sample(
        self,
        model,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        progress_callback=None,
        **kwargs
    ) -> torch.Tensor:
        """
        DDIM 采样
        
        Args:
            model: 噪声预测模型
            shape: 输出张量形状
            condition: 条件
            progress_callback: 进度回调函数 callback(step, total)
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 遍历时间步序列
        for i in range(len(self.timestep_seq) - 1):
            t_curr = self.timestep_seq[i]
            t_prev = self.timestep_seq[i + 1]
            
            t_batch = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
            
            # 预测噪声 (支持 classifier-free)
            if self.guidance_scale > 1 and self.unconditional_condition is not None:
                noise_cond = model(x, t_batch, condition)
                noise_uncond = model(x, t_batch, self.unconditional_condition)
                noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(x, t_batch, condition)
            
            # 使用 scheduler 的 step 方法计算前一步
            x = self.scheduler.step(noise_pred, t_curr, x, eta=self.eta)
            
            if progress_callback is not None:
                progress_callback(i + 1, len(self.timestep_seq) - 1)
        
        return x


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ArellanoDreamWeaver - NoiseScheduler Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试不同调度器
    schedules_to_test = ["linear", "cosine", "sqrt_linear_cosine"]
    
    for sched_name in schedules_to_test:
        print(f"\n--- Testing {sched_name} schedule ---")
        
        scheduler = NoiseScheduler(schedule=sched_name, n_timesteps=1000)
        scheduler = scheduler.to(device)
        
        print(f"  Scheduler: {scheduler}")
        print(f"  Betas range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
        print(f"  Alphas_cumprod range: [{scheduler.alphas_cumprod.min():.6f}, {scheduler.alphas_cumprod.max():.6f}]")
        
        # 测试加噪
        B, C, T = 2, 64, 256
        x0 = torch.randn(B, C, T, device=device)
        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, 1000, (B,), device=device)
        
        x_t, sqrt_a, sqrt_1ma = scheduler.add_noise(x0, noise, timesteps)
        
        print(f"  Input shape: {x0.shape}")
        print(f"  Noisy output shape: {x_t.shape}")
        print(f"  At t={timesteps.tolist()}:")
        print(f"    sqrt(ᾱ): {sqrt_a.flatten()[0]:.4f}, sqrt(1-ᾱ): {sqrt_1ma.flatten()[0]:.4f}")
        
        # 验证加噪公式
        expected = sqrt_a.flatten()[0] * x0[0] + sqrt_1ma.flatten()[0] * noise[0]
        actual = x_t[0]
        diff = (expected - actual).abs().max().item()
        print(f"    Formula verification error: {diff:.8f}")
        assert diff < 1e-5, "加噪公式验证失败!"
    
    print("\n✅ All noise schedule tests passed!")
    
    # 测试 DDIM 采样流程 (使用 dummy model)
    print("\n--- Testing DDIM Sampler ---")
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(64, 64, 3, padding=1)
        
        def forward(self, x, t, cond=None):
            return self.conv(x)  # 返回与输入同形的输出
    
    dummy_model = DummyModel().to(device)
    
    scheduler = NoiseScheduler("cosine", 1000).to(device)
    sampler = DDIMSampler(scheduler, n_steps=10, eta=0.0)
    
    print(f"  Sampling with {sampler.n_steps} DDIM steps...")
    print(f"  Timestep sequence length: {len(sampler.timestep_seq)}")
    print(f"  First few steps: {sampler.timestep_seq[:5]}")
    print(f"  Last few steps: {sampler.timestep_seq[-5:]}")
    
    result = sampler.sample(dummy_model, shape=(1, 64, 128), condition=None)
    print(f"  Generated sample shape: {result.shape}")
    print(f"  Sample range: [{result.min():.4f}, {result.max():.4f}]")
    
    print("\n✅ DDIM sampler test passed!")

import torch.nn as nn
