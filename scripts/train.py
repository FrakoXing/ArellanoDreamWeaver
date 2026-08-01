"""
ArellanoDreamWeaver - 训练入口脚本
=================================
训练多任务 Transformer + Diffusion 制谱模型
"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.models.transformer import (
    MultiTaskTransformer,
    TransformerConfig
)
from dreamweaver.models.diffusion import (
    ConditionalUNet1D,
    DiffusionConfig,
    ChartDiffusionModel
)
from dreamweaver.diffusion.noise_scheduler import NoiseScheduler, DDIMSampler
from dreamweaver.chart.structures import ChartData, chart_to_tensor


# ============================================================
# 数据集定义
# ============================================================

class ChartDataset(Dataset):
    """
    谱面数据集
    
    输入: 音频特征 + 谱面张量 (干净)
    输出: 用于 Diffusion 的谱面 + 多任务标签
    """
    
    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 4096,
        feature_dim: int = 8,
        audio_feature_dim: int = 128,
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.transform = transform
        
        # 加载数据文件列表
        self.chart_files = list(self.data_dir.glob("*.json"))
        
        if len(self.chart_files) == 0:
            print(f"警告: 在 {data_dir} 中未找到 JSON 文件")
            print("请将谱面数据放在 data/processed/ 目录下")
    
    def __len__(self) -> int:
        return len(self.chart_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 加载谱面
        chart_path = self.chart_files[idx]
        try:
            chart = ChartData.load_from_file(str(chart_path))
        except Exception as e:
            print(f"加载 {chart_path} 失败: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # 转换为张量表示
        chart_tensor = chart_to_tensor(chart, self.max_seq_len, self.feature_dim)
        
        # 模拟音频特征 (实际应从音频文件提取)
        audio_features = torch.randn(self.max_seq_len, self.audio_feature_dim)
        
        # 模拟文本条件 (实际来自 LLM 解析提示词)
        text_condition = torch.randn(512)  # 全局条件
        
        sample = {
            "audio_features": audio_features,
            "chart_clean": torch.from_numpy(chart_tensor).float(),
            "text_condition": text_condition,
            
            # 多任务标签 (可选)
            "bpm": torch.tensor([chart.default_bpm / 300.0]),
            "difficulty": torch.tensor([2]),  # 假设中等难度
            
            # 元数据
            "filename": str(chart_path.name),
            "note_count": chart.total_notes
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        """自定义 batch 整理函数"""
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key in ["filename"]:
                collated[key] = [item[key] for item in batch]
            else:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = values
        
        return collated


# ============================================================
# 损失函数
# ============================================================

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    包含:
    - Diffusion MSE loss (预测噪声)
    - Note classification cross-entropy
    - Position/MSE regression
    - Difficulty classification
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        num_note_types: int = 4
    ):
        super().__init__()
        
        # 默认权重
        default_weights = {
            "diffusion": 2.0,
            "note_type": 1.0,
            "position": 0.5,
            "timing": 0.5,
            "difficulty": 0.3,
            "style": 0.2
        }
        
        self.weights = weights or default_weights
        
        # 损失函数
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            model_outputs: 模型输出的字典
            targets: 目标值的字典
            noise: 真实噪声 (用于 diffusion loss)
            
        Returns:
            各项损失和总损失
        """
        losses = {}
        
        # === Diffusion Loss (核心) ===
        if "diffusion_pred" in model_outputs and noise is not None:
            losses["diffusion"] = self.mse(model_outputs["diffusion_pred"], noise)
        
        # === Note Type Classification ===
        if "note_logits" in model_outputs and "note_types" in targets:
            losses["note_type"] = self.ce(
                model_outputs["note_logits"].transpose(1, 2),
                targets["note_types"].long()
            )
        
        # === Position Regression ===
        if "position_pred" in model_outputs and "positions" in targets:
            losses["position"] = self.l1(model_outputs["position_pred"], targets["positions"])
        
        # === Timing Regression ===
        if "timing_pred" in model_outputs and "timings" in targets:
            losses["timing"] = self.l1(model_outputs["timing_pred"], targets["timings"])
        
        # === Difficulty Classification ===
        if "difficulty_logits" in model_outputs and "difficulty" in targets:
            losses["difficulty"] = self.ce(
                model_outputs["difficulty_logits"],
                targets["difficulty"].long().squeeze()
            )
        
        # === 加权总损失 ===
        total_loss = sum(
            self.weights.get(k, 1.0) * v 
            for k, v in losses.items() if v.requires_grad
        )
        losses["total"] = total_loss
        
        return losses


# ============================================================
# 训练器
# ============================================================

class DreamWeaverTrainer:
    """
    ArellanoDreamWeaver 训练器
    
    管理:
    - 模型创建 & 初始化
    - 数据加载
    - 训练循环
    - 验证
    - 检查点保存/加载
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n🖥️  使用设备: {self.device}")
        
        # 创建模型
        self._build_models()
        
        # 创建优化器和调度器
        self._setup_optimizer()
        
        # 损失函数
        self.criterion = MultiTaskLoss(config.get("training", {}).get("loss_weights", {}))
        
        # 噪声调度器
        diffusion_cfg = config.get("model", {}).get("diffusion", {})
        self.noise_scheduler = NoiseScheduler(
            schedule=diffusion_cfg.get("schedule", "cosine"),
            n_timesteps=diffusion_cfg.get("n_timesteps", 1000)
        ).to(self.device)
        
        # 统计
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 日志目录
        log_dir = Path(config.get("logging", {}).get("log_dir", "logs"))
        self.log_dir = PROJECT_ROOT / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard (可选)
        self.writer = None
        if config.get("logging", {}).get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                pass
    
    def _build_models(self):
        """构建模型组件"""
        cfg_model = self.config.get("model", {})
        tfm_cfg = cfg_model.get("transformer", {})
        
        # Transformer 配置
        transformer_config = TransformerConfig(
            d_model=tfm_cfg.get("d_model", 512),
            n_heads=tfm_cfg.get("n_heads", 8),
            n_layers=tfm_cfg.get("n_layers", 6),
            d_ff=tfm_cfg.get("d_ff", 2048),
            dropout=tfm_cfg.get("dropout", 0.1)
        )
        
        # 创建多任务 Transformer
        self.transformer = MultiTaskTransformer(transformer_config).to(self.device)
        
        # Diffusion U-Net 配置
        diff_config = DiffusionConfig(
            in_channels=64,   # 谱面特征维度
            out_channels=64,
            condition_dim=512  # 来自 Transformer 的条件维度
        )
        
        self.diffusion_unet = ConditionalUNet1D(diff_config).to(self.device)
        
        # 打印模型信息
        tfm_params = sum(p.numel() for p in self.transformer.parameters())
        unet_params = sum(p.numel() for p in self.diffusion_unet.parameters())
        total_params = tfm_params + unet_params
        
        print(f"\n📊 模型参数统计:")
        print(f"  MultiTask Transformer: {tfm_params:,} (~{tfm_params/1e6:.1f}M)")
        print(f"  Diffusion U-Net:       {unet_params:,} (~{unet_params/1e6:.1f}M)")
        print(f"  总计:                   {total_params:,} (~{total_params/1e6:.1f}M)")
    
    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        train_cfg = self.config.get("training", {})
        lr = train_cfg.get("learning_rate", 1e-4)
        wd = train_cfg.get("weight_decay", 0.01)
        
        # 合并所有参数
        all_params = [
            {"params": self.transformer.parameters()},
            {"params": self.diffusion_unet.parameters()}
        ]
        
        optimizer_name = train_cfg.get("optimizer", "adamw").lower()
        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(all_params, lr=lr, weight_decay=wd)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(all_params, lr=lr, weight_decay=wd)
        else:
            self.optimizer = optim.AdamW(all_params, lr=lr, weight_decay=wd)
        
        # 学习率调度器
        scheduler_name = train_cfg.get("scheduler", "cosine").lower()
        epochs = train_cfg.get("epochs", 100)
        
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler_name == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=lr * 10,
                total_steps=epochs,
                anneal_strategy='cos'
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.transformer.train()
        self.diffusion_unet.train()
        
        epoch_losses = {}
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            audio_features = batch["audio_features"].to(self.device)
            chart_clean = batch["chart_clean"].to(self.device)
            text_condition = batch["text_condition"].to(self.device)
            
            B = audio_features.size(0)
            
            # === Step 1: Transformer 编码 ===
            # 使用 dummy text input IDs (实际应从 LLM 得到)
            dummy_text_ids = torch.randint(0, 50257, (B, 32), device=self.device)
            dummy_text_mask = torch.ones(B, 32, device=self.device)
            
            # Dummy BPM features
            bpm_features = torch.cat([
                batch["bpm"].unsqueeze(1).to(self.device),  # bpm_value
                torch.zeros(B, 1, device=self.device),      # beat_position
                torch.ones(B, 1, device=self.device),        # time_sig_num
                torch.ones(B, 1, device=self.device)         # confidence
            ], dim=-1)
            bpm_features = bpm_features.unsqueeze(1).expand(-1, audio_features.size(1), -1)
            
            # Forward through transformer
            tfm_outputs = self.transformer(
                audio_features=audio_features,
                text_input_ids=dummy_text_ids,
                text_attention_mask=dummy_text_mask,
                bpm_features=bpm_features
            )
            
            # 获取 Diffusion 条件
            condition = tfm_outputs["diffusion_condition"]
            
            # === Step 2: Diffusion Training ===
            # 随机采样时间步
            timesteps = torch.randint(
                0, self.noise_scheduler.n_timesteps, (B,), device=self.device
            )
            
            # 生成随机噪声
            noise = torch.randn_like(chart_clean)
            
            # 前向扩散: 添加噪声到干净谱面
            x_noisy, _, _ = self.noise_scheduler.add_noise(chart_clean, noise, timesteps)
            
            # U-Net 预测噪声
            # 注意: U-Net 期望 [B, C, T], 需要转置
            x_noisy_1d = x_noisy.transpose(1, 2)
            noise_pred = self.diffusion_unet(x_noisy_1d, timesteps, condition)
            
            # === 计算损失 ===
            targets = {
                "bpm": batch["bpm"].to(self.device),
                "difficulty": batch["difficulty"].to(self.device)
            }
            
            # 准备 Diffusion 输出用于损失计算
            model_outputs_for_loss = {
                "diffusion_pred": noise_pred,
                "note_logits": tfm_outputs["note_logits"],
                "position_pred": tfm_outputs["position_pred"],
                "timing_pred": tfm_outputs["timing_pred"],
                "difficulty_logits": tfm_outputs["difficulty_logits"]
            }
            
            losses = self.criterion(model_outputs_for_loss, targets, noise)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses["total"].backward()
            
            # 梯度裁剪
            grad_clip = self.config.get("training", {}).get("gradient_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(
                list(self.transformer.parameters()) + 
                list(self.diffusion_unet.parameters()),
                grad_clip
            )
            
            self.optimizer.step()
            
            # 更新统计
            self.global_step += 1
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v.item())
            
            # 更新进度条
            avg_total = sum(epoch_losses["total"][-100:]) / min(len(epoch_losses["total"]), 100)
            pbar.set_postfix({"loss": f"{avg_total:.4f}"})
            
            # TensorBoard 日志
            if self.writer and self.global_step % 50 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}_step", v.item(), self.global_step)
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", current_lr, self.global_step)
        
        # Epoch 结束后更新学习率
        self.scheduler.step()
        
        # 返回平均损失
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, filepath: str, extra_info: Optional[Dict] = None):
        """保存检查点"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "transformer_state_dict": self.transformer.state_dict(),
            "diffusion_state_dict": self.diffusion_unet.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, filepath)
        print(f"💾 已保存检查点到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.transformer.load_state_dict(checkpoint["transformer_state_dict"])
        self.diffusion_unet.load_state_dict(checkpoint["diffusion_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        
        print(f"📂 已从 {filepath} 恢复训练 (epoch={self.epoch}, step={self.global_step})")
    
    def train(self):
        """完整训练流程"""
        train_cfg = self.config.get("training", {})
        epochs = train_cfg.get("epochs", 100)
        batch_size = train_cfg.get("batch_size", 32)
        save_every = train_cfg.get("save_every", 10)
        
        print(f"\n{'='*60}")
        print(f"🚀 开始训练 ArellanoDreamWeaver")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"{'='*60}\n")
        
        # 创建数据集和数据加载器
        data_dir = PROJECT_ROOT / self.config.get("data", {}).get("processed_dir", "data/processed")
        
        dataset = ChartDataset(str(data_dir))
        
        if len(dataset) == 0:
            print("\n⚠️  未找到训练数据!")
            print("使用合成数据进行演示...")
            
            # 创建演示用的假数据集
            from torch.utils.data import TensorDataset
            
            demo_audio = torch.randn(32, 4096, 128)
            demo_chart = torch.randn(32, 4096, 8)
            demo_cond = torch.randn(32, 512)
            demo_bpm = torch.rand(32) * 0.5 + 0.3  # 90~180 BPM 归一化
            demo_diff = torch.randint(0, 5, (32,))
            
            dataset = TensorDataset(demo_audio, demo_chart, demo_cond, demo_bpm, demo_diff)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows 兼容性
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=getattr(dataset, 'collate_fn', None)
        )
        
        print(f"数据集大小: {len(dataset)} samples\n")
        
        # 训练循环
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_start_time = datetime.now()
            
            # 训练一个 epoch
            avg_losses = self.train_epoch(dataloader)
            
            # 打印日志
            elapsed = (datetime.now() - epoch_start_time).total_seconds()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            print(f"\n[Epoch {epoch+1}/{epochs}] "
                  f"Loss={avg_losses['total']:.4f} | "
                  f"Diff={avg_losses.get('diffusion', 0):.4f} | "
                  f"Time={elapsed:.1f}s | "
                  f"LR={current_lr:.6f}")
            
            # TensorBoard 日志
            if self.writer:
                for k, v in avg_losses.items():
                    self.writer.add_scalar(f"train/{k}_epoch", v, epoch)
                self.writer.add_scalar("train/lr", current_lr, epoch)
            
            # 保存最佳模型
            if avg_losses["total"] < self.best_loss:
                self.best_loss = avg_losses["total"]
                best_path = self.log_dir / "best_model.pth"
                self.save_checkpoint(str(best_path))
            
            # 定期保存
            if (epoch + 1) % save_every == 0 or epoch == 0:
                ckpt_path = self.log_dir / f"checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(str(ckpt_path))
        
        # 最终保存
        final_path = self.log_dir / "final_model.pth"
        self.save_checkpoint(str(final_path))
        
        print(f"\n{'='*60}")
        print(f"✅ 训练完成! 最佳 Loss: {self.best_loss:.4f}")
        print(f"模型保存在: {self.log_dir}")
        print(f"{'='*60}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train ArellanoDreamWeaver Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式 (小规模快速测试)")
    parser.add_argument("--device", type=str, default=None,
                        help="强制指定设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        torch.manual_seed(42)
        if args.device.startswith("cuda"):
            assert torch.cuda.is_available(), "CUDA 不可用!"
    
    # 加载配置
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print("运行 init_dreamweaver.py 生成默认配置")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # 调试模式覆盖
    if args.debug:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 4
        config["training"]["save_every"] = 1
        print("\n⚡ Debug模式: 快速测试 (2 epochs)")
    
    # 创建训练器并开始训练
    trainer = DreamWeaverTrainer(config)
    
    # 可选: 从检查点恢复
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == "__main__":
    main()
