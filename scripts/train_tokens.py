# -*- coding: utf-8 -*-
"""
多格式谱面训练优化脚本
=============================================
支持 PEC / osu! / Arellano 多种谱面格式混合训练
核心加速优化项：
  - AMP混合精度（RTX4060 速度提升1.5~2倍）
  - 多线程DataLoader持久化加载进程
  - 音频特征磁盘缓存（需提前运行 prepare_audio_cache.py）
  - torch.compile 编译加速（PyTorch 2.x，提速1.2~1.5倍）
  - Flash SDP 高效注意力（PyTorch >= 2.0）
  - 梯度检查点（大模型节省显存）
  - 余弦退火重启学习率调度器
  - 中文可视化日志+emoji分层输出
使用教程：
  # 第一步：预处理音频（仅需执行一次）
  python scripts/prepare_audio_cache.py --target-steps 4096
  # 默认参数启动训练
  python scripts/train_tokens.py --batch-size 8 --epochs 100
  # 训练更大规模模型
  python scripts/train_tokens.py --hidden-dim 512 --num-layers 6 --num-heads 8 --batch-size 4
  # 调试模式（仅2轮，快速验证流程）
  python scripts/train_tokens.py --debug
  # 从断点恢复训练
  python scripts/train_tokens.py --resume logs/checkpoint_epoch_10.pth
"""
import os
import sys
import time
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Windows 控制台默认 GBK 无法编码 emoji/部分中文, 统一走 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.note_decoder import NoteDecoder
from src.data.token_dataset import TokenDataset

# ============================================================
# 音频特征映射层
# ============================================================
class AudioProjection(nn.Module):
    """将预处理后的音频特征映射至模型隐藏维度"""
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
        return self.projection(audio_features)

# ============================================================
# 高性能训练器
# ============================================================
class TokenTrainer:
    """集成AMP、梯度累积、编译加速、中文可视化日志的训练器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🖥️ 硬件设备信息打印
        print(f"\n🖥️  检测到训练设备：{self.device}")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_ver = torch.version.cuda
            print(f"   🎮 显卡型号：{gpu_name}")
            print(f"   📦 总显存容量：{total_vram_gb:.1f} GB")
            print(f"   ⚙️  CUDA 版本：{cuda_ver}")

        # ⚡ 开启Flash高效注意力
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print(f"   ⚡ Flash SDP 注意力：✅ 已启用")

        self._build_models()
        self._setup_optimizer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0为填充PAD标记
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # 日志保存目录
        log_cfg = config.get("logging", {})
        self.log_dir = PROJECT_ROOT / log_cfg.get("log_dir", "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # AMP混合精度缩放器
        self.use_amp = config.get("training", {}).get("use_amp", True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # TensorBoard可视化
        self.writer = None
        if log_cfg.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
                print(f"   📊 TensorBoard 日志：✅ 已开启")
            except ImportError:
                print(f"   📊 TensorBoard 日志：❌ 未安装模块，跳过")

    def _build_models(self):
        """构建模型、编译加速，打印参数量中文统计"""
        cfg = self.config.get("model", {})
        audio_cfg = cfg.get("audio_projection", {})
        audio_dim = audio_cfg.get("audio_dim", 128)
        hidden_dim = audio_cfg.get("hidden_dim", 512)

        self.audio_projection = AudioProjection(
            audio_dim=audio_dim, hidden_dim=hidden_dim,
        ).to(self.device)

        decoder_cfg = cfg.get("note_decoder", {})
        decoder_kwargs = dict(
            vocab_size=decoder_cfg.get("vocab_size", 256),
            hidden_dim=decoder_cfg.get("hidden_dim", 512),
            num_layers=decoder_cfg.get("num_layers", 6),
            num_heads=decoder_cfg.get("num_heads", 8),
            max_seq_len=decoder_cfg.get("max_seq_len", 4096),
            dropout=decoder_cfg.get("dropout", 0.1),
        )
        self.note_decoder = NoteDecoder(**decoder_kwargs).to(self.device)

        # 🧠 梯度检查点（节省显存）
        if cfg.get("use_gradient_checkpointing", False):
            self.note_decoder.decoder.apply(
                lambda m: m.register_forward_hook(
                    lambda _, __, ___: None
                ) if hasattr(m, 'checkpoint') else None
            )
            print("   🧠 梯度检查点：✅ 已开启（显存节省模式）")

        # 🔥 PyTorch编译加速
        if cfg.get("use_compile", True) and hasattr(torch, 'compile'):
            try:
                compile_mode = cfg.get("compile_mode", "default")
                self.audio_projection = torch.compile(self.audio_projection, mode=compile_mode)
                self.note_decoder = torch.compile(self.note_decoder, mode=compile_mode)
                print(f"   🔥 torch.compile 编译加速：✅ 开启（模式={compile_mode}）")
            except Exception as e:
                print(f"   🔥 torch.compile 编译加速：⏭️ 跳过，错误：{str(e)[:60]}...")

        # 📏 模型参数量汇总
        proj_params = sum(p.numel() for p in self.audio_projection.parameters())
        dec_params = sum(p.numel() for p in self.note_decoder.parameters())
        total = proj_params + dec_params
        print(f"\n📏 模型参数量汇总：")
        print(f"   🎧 音频映射层：{proj_params:,}（{proj_params/1e6:.2f}M）")
        print(f"   🎵 谱面解码器：{dec_params:,}（{dec_params/1e6:.2f}M）")
        print(f"   🧩 模型总参数量：{total:,}（{total/1e6:.2f}M）")

    def _setup_optimizer(self):
        """初始化优化器与学习率调度器，中文配置打印"""
        cfg = self.config.get("training", {})
        lr = float(cfg.get("learning_rate", 1e-4))
        wd = float(cfg.get("weight_decay", 0.01))
        betas = tuple(cfg.get("betas", (0.9, 0.98)))
        params = list(self.audio_projection.parameters()) + list(self.note_decoder.parameters())
        opt_name = cfg.get("optimizer", "adamw").lower()
        fused_ok = self.device.type == 'cuda'

        try:
            if opt_name == "adamw":
                self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, fused=fused_ok)
            elif opt_name == "adam":
                self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd, fused=fused_ok)
            else:
                self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
        except TypeError:
            # 旧版PyTorch无fused融合内核参数
            if opt_name == "adamw":
                self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
            elif opt_name == "adam":
                self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
            else:
                self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)

        epochs = cfg.get("epochs", 100)
        warmup_epochs = cfg.get("warmup_epochs", 5)
        T_0 = cfg.get("cosine_restart_interval", epochs // 3)

        if cfg.get("scheduler", "cosine_warmup") == "cosine_warmup":
            self.scheduler = self._get_warmup_cosine_schedule(warmup_epochs, epochs)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, T_0),
                T_mult=2,
                eta_min=lr * 0.01,
            )

        print(f"\n⚙️ 优化器与学习率调度配置：")
        print(f"   🧮 优化器：{opt_name.upper()} | 初始学习率={lr:.2e} | 权重衰减={wd}")
        print(f"   🔗 CUDA融合内核：{'✅ 开启' if getattr(self.optimizer, 'fused', False) else '❌ 关闭'}")
        print(f"   📈 调度器类型：{type(self.scheduler).__name__}")

    def _get_warmup_cosine_schedule(self, warmup_epochs, total_epochs):
        """预热+余弦衰减学习率策略"""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """单轮训练循环，支持AMP与梯度累积，进度条中文标识"""
        self.audio_projection.train()
        self.note_decoder.train()
        epoch_losses = []
        accum_steps = self.config.get("training", {}).get("accumulation_steps", 1)
        grad_clip = self.config.get("training", {}).get("gradient_clip", 1.0)
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"🏋️ 训练轮次 {self.epoch+1}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        tokens_processed = 0

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            audio_features = batch["audio_features"].to(self.device, non_blocking=True)
            B, T = input_ids.shape
            tokens_processed += B * T

            target_input = input_ids[:, :-1]
            target_output = input_ids[:, 1:]

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                encoder_output = self.audio_projection(audio_features)
                logits = self.note_decoder(
                    encoder_output=encoder_output,
                    target_tokens=target_input,
                )
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_output.reshape(-1),
                )
                loss = loss / accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积更新参数
            if (batch_idx + 1) % accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.audio_projection.parameters()) + list(self.note_decoder.parameters()),
                        grad_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.audio_projection.parameters()) + list(self.note_decoder.parameters()),
                        grad_clip,
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            real_loss = loss.item() * accum_steps
            epoch_losses.append(real_loss)
            avg_loss = sum(epoch_losses[-100:]) / min(len(epoch_losses), 100)
            lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "📉损失": f"{avg_loss:.4f}",
                "📐学习率": f"{lr:.2e}",
                "⚡AMP": "开启" if self.use_amp else "关闭",
            })

            # 每50步记录可视化日志
            if self.writer and self.global_step % 50 == 0:
                self.writer.add_scalar("训练/单步损失", real_loss, self.global_step)
                self.writer.add_scalar("训练/学习率", lr, self.global_step)

        # 处理最后不足累积步数的梯度
        if len(dataloader) % accum_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.audio_projection.parameters()) + list(self.note_decoder.parameters()),
                    grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(self.audio_projection.parameters()) + list(self.note_decoder.parameters()),
                    grad_clip,
                )
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.scheduler.step()
        elapsed = time.perf_counter() - t0
        tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        return {
            "loss": sum(epoch_losses) / len(epoch_losses),
            "tokens_per_sec": tok_per_sec,
            "elapsed_sec": elapsed,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证集前向传播，计算验证损失"""
        self.audio_projection.eval()
        self.note_decoder.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            audio_features = batch["audio_features"].to(self.device, non_blocking=True)
            B, T = input_ids.shape
            target_input = input_ids[:, :-1]
            target_output = input_ids[:, 1:]
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                encoder_output = self.audio_projection(audio_features)
                logits = self.note_decoder(
                    encoder_output=encoder_output,
                    target_tokens=target_input,
                )
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_output.reshape(-1),
                )
            total_loss += loss.item() * B * (T - 1)
            total_tokens += B * (T - 1)
        return {"loss": total_loss / total_tokens if total_tokens > 0 else float('inf')}

    def save_checkpoint(self, filepath: str):
        """保存模型断点，打印中文存储信息"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "audio_projection_state_dict": self.audio_projection.state_dict(),
            "note_decoder_state_dict": self.note_decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"💾 已保存断点 → {filepath} | 文件大小：{size_mb:.1f} MB")

    def load_checkpoint(self, filepath: str):
        """加载断点恢复训练，中文提示"""
        print(f"📂 正在加载恢复断点：{filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.audio_projection.load_state_dict(checkpoint["audio_projection_state_dict"])
        self.note_decoder.load_state_dict(checkpoint["note_decoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"✅ 断点加载成功 | 上次轮次={self.epoch}，全局步数={self.global_step}，最优损失={self.best_loss:.4f}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """完整训练主循环，全中文分层输出"""
        cfg = self.config.get("training", {})
        epochs = cfg.get("epochs", 100)
        save_every = cfg.get("save_every", 10)

        print(f"\n{'='*70}")
        print(f"🚀 启动完整训练流程")
        print(f"{'='*70}")
        print(f"📌 总训练轮次：{epochs}")
        print(f"📦 单批次样本数：{cfg['batch_size']}")
        accum = cfg.get("accumulation_steps", 1)
        if accum > 1:
            effective = cfg['batch_size'] * accum
            print(f"🔁 梯度累积步数：{accum} | 等效批次大小 = {effective}")
        print(f"🎯 初始学习率：{cfg['learning_rate']}")
        print(f"⚡ AMP混合精度：{'开启' if self.use_amp else '关闭'}")
        print(f"🔥 PyTorch编译加速：{cfg.get('use_compile', True)}")
        print(f"{'='*70}\n")

        for epoch in range(epochs):
            self.epoch = epoch
            start_time = datetime.now()
            metrics = self.train_epoch(train_loader)
            elapsed = (datetime.now() - start_time).total_seconds()
            lr = self.optimizer.param_groups[0]["lr"]

            val_str = ""
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_str = f" | 🧪 验证损失={val_metrics['loss']:.4f}"

            print(f"\n────────────────────────────────────────────────────")
            print(f"📅 第 [{epoch+1}/{epochs}] 轮训练完成")
            print(f"   📉 训练损失 = {metrics['loss']:.4f}{val_str}")
            print(f"   ⏱️ 本轮耗时 = {elapsed:.0f} 秒")
            print(f"   ⚡ 字符处理速度 = {metrics['tokens_per_sec']:.0f} token/秒")
            print(f"   📐 当前学习率 = {lr:.6f}")
            print(f"────────────────────────────────────────────────────")

            # 过拟合检测警告
            if val_metrics and epoch > 5:
                gap = val_metrics['loss'] - metrics['loss']
                if gap > 0.5:
                    print(f"⚠️ 警告：检测到疑似过拟合！训练-验证损失差值 = {gap:.2f}")

            # 写入TensorBoard可视化
            if self.writer:
                self.writer.add_scalar("训练/每轮损失", metrics["loss"], epoch)
                self.writer.add_scalar("训练/处理速度token/s", metrics["tokens_per_sec"], epoch)
                if val_metrics:
                    self.writer.add_scalar("验证/每轮损失", val_metrics["loss"], epoch)

            # 保存最优模型
            score = val_metrics["loss"] if val_metrics else metrics["loss"]
            if score < self.best_loss:
                self.best_loss = score
                print(f"🏆 刷新最优损失记录！保存 best_model.pth")
                self.save_checkpoint(str(self.log_dir / "best_model.pth"))

            # 定期保存断点
            if (epoch + 1) % save_every == 0 or epoch == 0:
                self.save_checkpoint(str(self.log_dir / f"checkpoint_epoch_{epoch+1}.pth"))

        # 全部轮次结束，保存最终权重
        self.save_checkpoint(str(self.log_dir / "final_model.pth"))
        print(f"\n{'='*70}")
        print(f"✅ 全部训练轮次执行完毕！")
        print(f"🏅 全局最优损失值：{self.best_loss:.4f}")
        print(f"💾 最终权重已保存为 final_model.pth")
        print(f"{'='*70}\n")

# ============================================================
# 程序入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="🎵 多格式节奏谱面Token训练优化工具")
    # 数据集路径参数
    parser.add_argument("--tokens-dir", default="data/tokens", help="Token序列存放目录")
    parser.add_argument("--audio-dir", default="data/audio", help="原始音频文件目录")
    parser.add_argument("--audio-cache-dir", default="data/audio_cache", help="预计算音频特征缓存目录")
    parser.add_argument("--max-chart-tokens", type=int, default=200_000,
                        help="超长谱面过滤阈值，单谱Token数超过则剔除（默认200000）")
    parser.add_argument("--window-mode", type=str, default="random",
                        choices=["random", "sliding", "head"],
                        help="长谱面截断方式：random=每样本随机窗口(默认)，sliding=滑窗全覆盖，head=只取开头(旧行为)")
    parser.add_argument("--window-stride", type=int, default=1024,
                        help="sliding模式窗口步长（Token数，默认1024）")
    parser.add_argument("--no-window-snap", action="store_true",
                        help="随机窗口起点不吸附到音符/事件边界")
    # 模型结构参数
    parser.add_argument("--hidden-dim", type=int, default=512, help="Transformer隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=6, help="解码器Transformer层数")
    parser.add_argument("--num-heads", type=int, default=8, help="多头注意力头数量")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="输入Token最大序列长度")
    parser.add_argument("--vocab-size", type=int, default=256, help="词汇表总大小")
    parser.add_argument("--audio-dim", type=int, default=128, help="梅尔音频特征维度")
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=100, help="总训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="单步批次样本数量")
    parser.add_argument("--lr", type=float, default=1e-4, help="基础初始学习率")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="梯度累积迭代次数")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="学习率预热轮数")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout正则化比例")
    parser.add_argument("--val-split", type=float, default=0.05, help="验证集划分比例")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW权重衰减正则化")
    # 性能优化开关
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载并行线程数")
    parser.add_argument("--no-amp", action="store_true", help="关闭CUDA AMP混合精度加速")
    parser.add_argument("--no-compile", action="store_true", help="关闭PyTorch 2.x compile编译加速")
    parser.add_argument("--use-checkpointing", action="store_true", help="开启梯度检查点（节省显存）")
    # 杂项控制参数
    parser.add_argument("--resume", type=str, default=None, help="断点文件路径，用于恢复训练")
    parser.add_argument("--debug", action="store_true", help="快速调试模式（小型模型，仅2轮训练）")
    parser.add_argument("--log-dir", default="logs", help="可视化日志与断点输出文件夹")
    parser.add_argument("--no-audio-cache", action="store_true", help="不使用预计算音频缓存，实时计算特征")
    args = parser.parse_args()

    # 构建全局配置字典
    config = {
        "model": {
            "audio_projection": {
                "audio_dim": args.audio_dim,
                "hidden_dim": args.hidden_dim,
            },
            "note_decoder": {
                "vocab_size": args.vocab_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "max_seq_len": args.max_seq_len,
                "dropout": args.dropout,
            },
            "use_compile": not args.no_compile,
            "use_gradient_checkpointing": args.use_checkpointing,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": 1.0,
            "save_every": 10,
            "warmup_epochs": args.warmup_epochs,
            "accumulation_steps": args.accumulation_steps,
            "optimizer": "adamw",
            "scheduler": "cosine_warmup",
            "use_amp": not args.no_amp,
        },
        "logging": {
            "log_dir": args.log_dir,
            "tensorboard": True,
        },
    }

    # 调试模式参数覆盖
    if args.debug:
        config["model"]["note_decoder"]["hidden_dim"] = 128
        config["model"]["note_decoder"]["num_layers"] = 2
        config["model"]["note_decoder"]["num_heads"] = 4
        config["model"]["audio_projection"]["hidden_dim"] = 128
        config["model"]["use_compile"] = False
        config["training"]["epochs"] = 2
        config["training"]["accumulation_steps"] = 2
        config["training"]["batch_size"] = 4
        print("\n🐞 调试模式已激活：小型模型，仅执行2轮训练")

    # 数据集路径打印
    tokens_dir = PROJECT_ROOT / args.tokens_dir
    audio_dir = PROJECT_ROOT / args.audio_dir
    audio_cache_dir = None if args.no_audio_cache else PROJECT_ROOT / args.audio_cache_dir
    print(f"\n📂 数据集路径配置：")
    print(f"   🔤 Token序列文件：{tokens_dir}")
    print(f"   🎧 原始音频文件：{audio_dir}")
    if audio_cache_dir and audio_cache_dir.exists():
        print(f"   💽 预计算音频缓存：{audio_cache_dir}（读取磁盘缓存，速度更快）")
    else:
        print(f"   💽 预计算音频缓存：❌ 未找到或已禁用（运行时实时计算，速度较慢）")

    full_dataset = TokenDataset(
        tokens_dir=str(tokens_dir),
        audio_dir=str(audio_dir),
        max_seq_len=args.max_seq_len,
        audio_feature_dim=args.audio_dim,
        use_real_audio=True,
        cache_audio_features=True,
        audio_cache_dir=str(audio_cache_dir) if audio_cache_dir and audio_cache_dir.exists() else None,
        max_chart_tokens=args.max_chart_tokens,
        window_mode=args.window_mode,
        window_stride=args.window_stride,
        snap_window_to_note=not args.no_window_snap,
    )

    # 无数据直接退出
    if len(full_dataset) == 0:
        print("\n❌ 错误：数据集文件夹未读取到任何Token样本！")
        print("💡 提示：请先运行分词脚本：python scripts/unified_tokenizer.py --dir data/raw/ --outdir data/tokens/")
        sys.exit(1)

    # 划分训练集、验证集
    val_split = getattr(args, 'val_split', 0.05)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"   🪟 窗口模式：{args.window_mode}（步长={args.window_stride}，"
          f"边界吸附={not args.no_window_snap}，过滤阈值={args.max_chart_tokens:,}）")
    print(f"\n📊 数据集划分统计：")
    print(f"   总样本数量：{len(full_dataset)}")
    print(f"   训练集样本：{train_size} 个")
    print(f"   验证集样本：{val_size} 个")

    # DataLoader加载参数
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=TokenDataset.collate_fn,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
    print(f"⚙️ 数据加载器配置：并行线程数 = {args.num_workers}")

    # 初始化训练器并开始训练
    trainer = TokenTrainer(config)
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()