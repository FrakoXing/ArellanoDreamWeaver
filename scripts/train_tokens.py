"""
多格式谱面训练脚本 (完整版)
===========================

支持 PEC / osu! / Arellano 多种谱面格式混合训练。
使用 Token 序列格式，配合 NoteDecoder 自回归模型。

数据流程:
  1. 谱面文件 (.pec/.osu/.json) → unified_tokenizer.py → Token 序列 (.json/.pt)
  2. Token 序列 + 音频文件 → TokenDataset → 训练

用法:
  # 使用默认配置
  python scripts/train_tokens.py

  # 指定数据目录
  python scripts/train_tokens.py --tokens-dir data/tokens --audio-dir data/audio

  # 调试模式 (快速测试)
  python scripts/train_tokens.py --debug
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入模型
from src.models.note_decoder import NoteDecoder

# 导入数据集
from src.data.token_dataset import TokenDataset


# ============================================================
# 音频投影层
# ============================================================

class AudioProjection(nn.Module):
    """
    将预处理的音频特征投影到模型隐藏维度
    
    输入: [B, T, D_audio] (Mel 频谱特征)
    输出: [B, T, hidden_dim]
    """
    
    def __init__(self, audio_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [B, T, D_audio]
        Returns:
            projected: [B, T, hidden_dim]
        """
        return self.projection(audio_features)


# ============================================================
# 训练器
# ============================================================

class TokenTrainer:
    """
    Token 序列训练器
    
    训练 NoteDecoder 自回归模型：
    - 输入: 音频特征 (经过 AudioProjection 投影)
    - 输出: Token 序列 (谱面)
    - Loss: CrossEntropy (预测下一个 Token)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n🖥️  使用设备: {self.device}")
        
        # 创建模型
        self._build_models()
        
        # 创建优化器
        self._setup_optimizer()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 PAD token
        
        # 统计
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 日志目录
        log_dir = Path(config.get("logging", {}).get("log_dir", "logs"))
        self.log_dir = PROJECT_ROOT / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = None
        if config.get("logging", {}).get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                pass
    
    def _build_models(self):
        """构建模型"""
        cfg = self.config.get("model", {})
        
        # 音频投影层配置
        audio_cfg = cfg.get("audio_projection", {})
        audio_dim = audio_cfg.get("audio_dim", 128)
        hidden_dim = audio_cfg.get("hidden_dim", 512)
        
        self.audio_projection = AudioProjection(
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        # NoteDecoder 配置
        decoder_cfg = cfg.get("note_decoder", {})
        self.note_decoder = NoteDecoder(
            vocab_size=decoder_cfg.get("vocab_size", 128),
            hidden_dim=decoder_cfg.get("hidden_dim", 512),
            num_layers=decoder_cfg.get("num_layers", 6),
            num_heads=decoder_cfg.get("num_heads", 8),
            max_seq_len=decoder_cfg.get("max_seq_len", 4096),
            dropout=decoder_cfg.get("dropout", 0.1),
        ).to(self.device)
        
        # 打印参数
        proj_params = sum(p.numel() for p in self.audio_projection.parameters())
        dec_params = sum(p.numel() for p in self.note_decoder.parameters())
        total = proj_params + dec_params
        
        print(f"\n📊 模型参数:")
        print(f"  AudioProjection: {proj_params:,} ({proj_params/1e6:.2f}M)")
        print(f"  NoteDecoder:     {dec_params:,} ({dec_params/1e6:.2f}M)")
        print(f"  总计:            {total:,} ({total/1e6:.2f}M)")
    
    def _setup_optimizer(self):
        """设置优化器"""
        cfg = self.config.get("training", {})
        lr = float(cfg.get("learning_rate", 1e-4))
        wd = float(cfg.get("weight_decay", 0.01))
        
        params = list(self.audio_projection.parameters()) + list(self.note_decoder.parameters())
        
        opt_name = cfg.get("optimizer", "adamw").lower()
        if opt_name == "adamw":
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.98))
        elif opt_name == "adam":
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
        else:
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.98))
        
        # 学习率调度器
        epochs = cfg.get("epochs", 100)
        warmup_epochs = cfg.get("warmup_epochs", 5)
        
        # 使用带 warmup 的调度器
        self.scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_epochs,
            num_training_steps=epochs
        )
    
    def _get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """带 warmup 的余弦退火调度器"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.audio_projection.train()
        self.note_decoder.train()
        
        epoch_losses = []
        
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}", leave=False)
        
        for batch in pbar:
            # 数据
            input_ids = batch["input_ids"].to(self.device)           # [B, T] Token 序列
            attention_mask = batch["attention_mask"].to(self.device) # [B, T]
            audio_features = batch["audio_features"].to(self.device) # [B, T, D]
            
            B, T = input_ids.shape
            
            # === Step 1: 音频特征投影 ===
            encoder_output = self.audio_projection(audio_features)  # [B, T, hidden_dim]
            
            # === Step 2: NoteDecoder 训练 (Teacher Forcing) ===
            # 输入序列: [SOS, t1, t2, ..., t_{n-1}]
            target_input = input_ids[:, :-1]  # [B, T-1]
            # 目标序列: [t1, t2, ..., t_n]
            target_output = input_ids[:, 1:]  # [B, T-1]
            
            # 创建 padding mask (目标序列中哪些是 padding)
            target_padding_mask = (target_output == 0)  # [B, T-1]
            
            # Forward
            logits = self.note_decoder(
                encoder_output=encoder_output,
                target_tokens=target_input,
                target_padding_mask=target_padding_mask,
            )  # [B, T-1, V]
            
            # === Step 3: 计算 Loss ===
            # CrossEntropy: 预测下一个 token
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),  # [B*(T-1), V]
                target_output.reshape(-1),             # [B*(T-1)]
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_clip = self.config.get("training", {}).get("gradient_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(
                list(self.audio_projection.parameters()) + list(self.note_decoder.parameters()),
                grad_clip
            )
            
            self.optimizer.step()
            
            # 统计
            self.global_step += 1
            epoch_losses.append(loss.item())
            
            # 进度条
            avg_loss = sum(epoch_losses[-100:]) / min(len(epoch_losses), 100)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # TensorBoard
            if self.writer and self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        
        self.scheduler.step()
        
        return {"loss": sum(epoch_losses) / len(epoch_losses)}
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "audio_projection_state_dict": self.audio_projection.state_dict(),
            "note_decoder_state_dict": self.note_decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"💾 检查点: {filepath} ({size_mb:.1f}MB)")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.audio_projection.load_state_dict(checkpoint["audio_projection_state_dict"])
        self.note_decoder.load_state_dict(checkpoint["note_decoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        
        print(f"📂 恢复训练: {filepath} (epoch={self.epoch}, step={self.global_step})")
    
    def train(self, dataloader: DataLoader):
        """完整训练流程"""
        cfg = self.config.get("training", {})
        epochs = cfg.get("epochs", 100)
        save_every = cfg.get("save_every", 10)
        
        print(f"\n{'='*60}")
        print(f"🚀 开始训练")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"   Learning rate: {self.config['training']['learning_rate']}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.epoch = epoch
            start_time = datetime.now()
            
            # 训练
            avg_losses = self.train_epoch(dataloader)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            lr = self.optimizer.param_groups[0]["lr"]
            
            print(f"\n[Epoch {epoch+1}/{epochs}] "
                  f"Loss={avg_losses['loss']:.4f} | "
                  f"Time={elapsed:.1f}s | LR={lr:.6f}")
            
            # TensorBoard
            if self.writer:
                self.writer.add_scalar("train/loss_epoch", avg_losses["loss"], epoch)
                self.writer.add_scalar("train/lr", lr, epoch)
            
            # 保存最佳
            if avg_losses["loss"] < self.best_loss:
                self.best_loss = avg_losses["loss"]
                self.save_checkpoint(str(self.log_dir / "best_model.pth"))
            
            # 定期保存
            if (epoch + 1) % save_every == 0 or epoch == 0:
                self.save_checkpoint(str(self.log_dir / f"checkpoint_epoch_{epoch+1}.pth"))
        
        # 最终保存
        self.save_checkpoint(str(self.log_dir / "final_model.pth"))
        
        print(f"\n{'='*60}")
        print(f"✅ 训练完成! 最佳 Loss: {self.best_loss:.4f}")
        print(f"{'='*60}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多格式谱面训练脚本 (Token 序列)")
    
    parser.add_argument("--tokens-dir", default="data/tokens",
                        help="Token 序列目录")
    parser.add_argument("--audio-dir", default="data/audio",
                        help="音频文件目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式 (2 epochs)")
    parser.add_argument("--log-dir", default="logs")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        "model": {
            "audio_projection": {
                "audio_dim": 128,      # 输入音频特征维度 (Mel频谱通道数)
                "hidden_dim": args.hidden_dim,
            },
            "note_decoder": {
                "vocab_size": 128,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "max_seq_len": args.max_seq_len,
                "dropout": 0.1,
            },
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "save_every": 10,
            "warmup_epochs": 5,
            "optimizer": "adamw",
        },
        "logging": {
            "log_dir": args.log_dir,
            "tensorboard": True,
        },
    }
    
    if args.debug:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 4
        config["training"]["save_every"] = 1
        print("\n⚡ Debug 模式: 2 epochs")
    
    # 创建数据集
    tokens_dir = PROJECT_ROOT / args.tokens_dir
    audio_dir = PROJECT_ROOT / args.audio_dir
    
    print(f"\n📂 数据目录:")
    print(f"   Tokens: {tokens_dir}")
    print(f"   Audio:  {audio_dir}")
    
    dataset = TokenDataset(
        tokens_dir=str(tokens_dir),
        audio_dir=str(audio_dir),
        max_seq_len=args.max_seq_len,
    )
    
    if len(dataset) == 0:
        print("\n❌ 没有找到 Token 数据!")
        print("请先运行: python scripts/unified_tokenizer.py --dir data/raw/ --outdir data/tokens/")
        sys.exit(1)
    
    print(f"\n📊 数据集大小: {len(dataset)} samples")
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TokenDataset.collate_fn,
    )
    
    # 训练
    trainer = TokenTrainer(config)
    
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)
    
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
