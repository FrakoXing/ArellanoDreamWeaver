# -*- coding: utf-8 -*-
"""
定数评级模型训练脚本
====================

用法:
  # v1 训练 (默认配置)
  python scripts/train_rating.py --config configs/rating.yaml

  # debug 模式 (2 epoch, 小 batch, 快速验证流程)
  python scripts/train_rating.py --config configs/rating.yaml --debug

  # v2 半监督微调 (真标签 + 伪标签)
  python scripts/train_rating.py --config configs/rating.yaml \
      --label-source any --lr 3e-5 --output logs/rating_v2.pth

  # 从断点恢复
  python scripts/train_rating.py --resume logs/rating_v1.pth
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Windows UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import importlib.util
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.models.rating_model import RatingModel, RatingConfig
from rating_common import denormalize_level, game_token_to_name

# 用 importlib 加载 rating_dataset, 绕过 src/data/__init__.py (避免 librosa 依赖)
def _load_rating_dataset():
    spec = importlib.util.spec_from_file_location(
        "rating_dataset", PROJECT_ROOT / "src" / "data" / "rating_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_rating_ds_mod = _load_rating_dataset()
RatingDataset = _rating_ds_mod.RatingDataset
build_bucket_sampler = _rating_ds_mod.build_bucket_sampler


# ============================================================
# 损失函数
# ============================================================

class RatingLoss(nn.Module):
    """Huber Loss + 伪标签降权"""

    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred:   [B]
            target: [B]
            weight: [B] (伪标签降权)
        """
        loss = self.huber(pred, target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


# ============================================================
# 验证指标
# ============================================================

def compute_metrics(preds: List[float], targets: List[float],
                    game_token_ids: List[int],
                    hit_thresholds: List[float] = (0.5, 1.0, 1.5)) -> Dict[str, float]:
    """
    计算验证指标 (反归一化到原始尺度后计算)。

    Returns:
        {"mae": ..., "hit_0.5": ..., "hit_1.0": ..., "hit_1.5": ...,
         "mae_phira": ..., "mae_osu": ...}
    """
    preds = np.array(preds)
    targets = np.array(targets)
    games = [game_token_to_name(gid) for gid in game_token_ids]

    # 反归一化到原始尺度
    preds_raw = np.array([denormalize_level(p, g) for p, g in zip(preds, games)])
    targets_raw = np.array([denormalize_level(t, g) for t, g in zip(targets, games)])

    abs_err = np.abs(preds_raw - targets_raw)
    mae = float(abs_err.mean())

    metrics = {"mae": mae}
    for thr in hit_thresholds:
        metrics[f"hit_{thr}"] = float((abs_err <= thr).mean())

    # 按游戏分别统计
    phira_mask = np.array([g == "phira" for g in games])
    osu_mask = np.array([g == "osu" for g in games])

    metrics["mae_phira"] = float(abs_err[phira_mask].mean()) if phira_mask.any() else float('nan')
    metrics["mae_osu"] = float(abs_err[osu_mask].mean()) if osu_mask.any() else float('nan')
    metrics["count_phira"] = int(phira_mask.sum())
    metrics["count_osu"] = int(osu_mask.sum())

    return metrics


# ============================================================
# 学习率调度 (cosine warmup)
# ============================================================

def get_cosine_warmup_schedule(optimizer, warmup_epochs: int, total_epochs: int,
                                min_lr: float, steps_per_epoch: int):
    """余弦退火 + warmup"""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        base_lr = optimizer.param_groups[0]['lr']
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        target = min_lr + (base_lr - min_lr) * cosine
        return target / base_lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 训练器
# ============================================================

class RatingTrainer:
    """定数评级训练器 (复用 train_tokens.py 的范式: AMP/cosine/中文日志/断点)"""

    def __init__(self, config: Dict[str, Any], args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n🖥️  训练设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"   🎮 {torch.cuda.get_device_name(0)}")
            print(f"   📦 显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        # === 构建模型 ===
        model_cfg = config["model"]
        rating_cfg = RatingConfig(**model_cfg)
        self.model = RatingModel(rating_cfg).to(self.device)
        params = self.model.get_num_parameters()
        print(f"   🧠 模型参数: {params['total']:,} ({params['total']/1e6:.2f}M)")

        # === 优化器 ===
        train_cfg = config["training"]
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr if args.lr else train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.criterion = RatingLoss(delta=train_cfg["huber_delta"])

        # === AMP (仅 CUDA) ===
        self.use_amp = train_cfg.get("use_amp", True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print(f"   ⚡ AMP 混合精度: ✅")

        # === 训练状态 ===
        # epochs 已在 main() 中根据 --debug/--epochs 写入 config["training"]["epochs"]
        self.epochs = train_cfg["epochs"]

        # === 数据 ===
        self._build_dataloaders()

        # === 学习率调度 ===
        self.scheduler = get_cosine_warmup_schedule(
            self.optimizer,
            warmup_epochs=train_cfg["warmup_epochs"],
            total_epochs=self.epochs,
            min_lr=train_cfg["min_lr"],
            steps_per_epoch=len(self.train_loader),
        )

        # === 状态 ===
        self.global_step = 0
        self.epoch = 0
        self.best_mae = float('inf')
        self.patience_counter = 0

        # === 日志 ===
        log_cfg = config.get("logging", {})
        self.log_dir = PROJECT_ROOT / log_cfg.get("log_dir", "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_cfg.get("log_interval", 20)
        self.save_top_k = log_cfg.get("save_top_k", 3)

        self.writer = None
        if log_cfg.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir / "rating"))
                print(f"   📊 TensorBoard: ✅")
            except ImportError:
                print(f"   📊 TensorBoard: ❌ 未安装")

        # 恢复
        if args.resume:
            self._resume(args.resume)

    def _build_dataloaders(self):
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]
        batch_size = self.args.batch_size if self.args.batch_size else train_cfg["batch_size"]

        # 特征标准化文件 (若存在则加载)
        feat_stats = data_cfg.get("feature_stats_file")

        # 校准集排除
        calibration_file = PROJECT_ROOT / data_cfg.get("calibration_file", "")
        exclude_files = []
        if calibration_file.exists():
            with open(calibration_file, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
            exclude_files = list(cal_data.get("labels", {}).keys())
            print(f"   🎯 排除校准集: {len(exclude_files)} 条")

        # 训练集
        labels_file = self.args.labels if self.args.labels else \
                      str(PROJECT_ROOT / data_cfg["labels_file"])
        pseudo_file = str(PROJECT_ROOT / data_cfg.get("labels_pseudo_file", ""))

        label_source = self.args.label_source if self.args.label_source else \
                       data_cfg.get("label_source", "labels")

        self.train_dataset = RatingDataset(
            tokens_dir=str(PROJECT_ROOT / data_cfg["tokens_dir"]),
            labels_file=labels_file,
            max_seq_len=data_cfg["max_seq_len"],
            feature_stats_file=feat_stats,
            labels_pseudo_file=pseudo_file if label_source == "any" else None,
            label_source=label_source,
            pseudo_weight=train_cfg["pseudo_weight"],
            exclude_files=exclude_files,
        )

        sampler = build_bucket_sampler(
            self.train_dataset,
            n_buckets=train_cfg["num_buckets"],
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=data_cfg.get("num_workers", 0),
            collate_fn=RatingDataset.collate_fn,
            pin_memory=self.device.type == 'cuda',
        )

        # 验证集 = 校准集
        if calibration_file.exists():
            self.val_dataset = RatingDataset(
                tokens_dir=str(PROJECT_ROOT / data_cfg["tokens_dir"]),
                labels_file=str(calibration_file),
                max_seq_len=data_cfg["max_seq_len"],
                feature_stats_file=feat_stats,
                label_source="labels",
                pseudo_weight=1.0,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=data_cfg.get("num_workers", 0),
                collate_fn=RatingDataset.collate_fn,
            )
        else:
            self.val_loader = None

        print(f"   📚 训练样本: {len(self.train_dataset)}")
        print(f"   📚 验证样本: {len(self.val_dataset) if self.val_loader else 0}")
        print(f"   📦 batch_size: {batch_size}, buckets: {train_cfg['num_buckets']}")

    def _resume(self, checkpoint_path: str):
        print(f"   📂 恢复断点: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_mae = ckpt.get("best_mae", float('inf'))
        print(f"   ✅ 从 epoch {self.epoch} 恢复, best_mae={self.best_mae:.4f}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            tokens = batch["tokens"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            feat = batch["handcrafted_features"].to(self.device)
            game_ids = batch["game_token_ids"].to(self.device)
            target = batch["rating"].to(self.device)
            weight = batch["weight"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    pred = self.model(tokens, mask, feat, game_ids)
                    loss = self.criterion(pred, target, weight)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config["training"]["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(tokens, mask, feat, game_ids)
                loss = self.criterion(pred, target, weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config["training"]["grad_clip"])
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()
            n_batches += 1

            if self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0
                print(f"  [step {self.global_step}] loss={loss.item():.4f} lr={lr:.2e} "
                      f"({elapsed:.1f}s)")

        avg_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0
        return {"train_loss": avg_loss, "epoch_time": elapsed}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        preds, targets, game_ids = [], [], []

        for batch in self.val_loader:
            tokens = batch["tokens"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            feat = batch["handcrafted_features"].to(self.device)
            game_ids_batch = batch["game_token_ids"].to(self.device)
            target = batch["rating"]

            pred = self.model(tokens, mask, feat, game_ids_batch)

            preds.extend(pred.cpu().numpy().tolist())
            targets.extend(target.numpy().tolist())
            game_ids.extend(game_ids_batch.cpu().numpy().tolist())

        metrics = compute_metrics(
            preds, targets, game_ids,
            hit_thresholds=self.config["eval"]["hit_thresholds"],
        )
        return metrics

    def save_checkpoint(self, path: str, is_best: bool = False):
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_mae": self.best_mae,
            "config": self.config,
            "is_best": is_best,
        }
        torch.save(ckpt, path)
        tag = " 🏆best" if is_best else ""
        print(f"  💾 保存: {path}{tag}")

    def train(self):
        train_cfg = self.config["training"]
        patience = train_cfg["early_stop_patience"]
        output = self.args.output or "logs/rating.pth"

        print(f"\n🚀 开始训练: {self.epochs} epoch")
        print(f"   早停耐心: {patience}, 输出: {output}")
        print("=" * 60)

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            print(f"\n📋 Epoch {epoch+1}/{self.epochs}")

            # 训练
            train_info = self.train_epoch()
            print(f"  ✅ train_loss={train_info['train_loss']:.4f} "
                  f"({train_info['epoch_time']:.1f}s)")

            # 验证
            val_metrics = self.validate()
            if val_metrics:
                print(f"  📊 val: mae={val_metrics['mae']:.4f} "
                      f"±0.5={val_metrics['hit_0.5']*100:.1f}% "
                      f"±1.0={val_metrics['hit_1.0']*100:.1f}%")
                print(f"     phira(mae={val_metrics['mae_phira']:.4f}, "
                      f"n={val_metrics['count_phira']}) "
                      f"osu(mae={val_metrics['mae_osu']:.4f}, "
                      f"n={val_metrics['count_osu']})")

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar("val/mae", val_metrics["mae"], epoch)
                    self.writer.add_scalar("val/hit_0.5", val_metrics["hit_0.5"], epoch)
                    self.writer.add_scalar("train/loss", train_info["train_loss"], epoch)

                # 早停 + 保存
                current_mae = val_metrics["mae"]
                is_best = current_mae < self.best_mae
                if is_best:
                    self.best_mae = current_mae
                    self.patience_counter = 0
                    self.save_checkpoint(
                        str(self.log_dir / "rating_best.pth"), is_best=True)
                else:
                    self.patience_counter += 1
                    print(f"  ⏳ 早停计数: {self.patience_counter}/{patience}")
                    if self.patience_counter >= patience:
                        print(f"\n⏹️  早停触发 (val_mae 连续 {patience} epoch 未下降)")
                        break

            # 定期保存
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                self.save_checkpoint(
                    str(self.log_dir / f"rating_epoch_{epoch+1}.pth"))

        # 最终保存
        self.save_checkpoint(str(PROJECT_ROOT / output))
        print(f"\n{'='*60}")
        print(f"✅ 训练完成! best_mae={self.best_mae:.4f}")
        print(f"   最终模型: {PROJECT_ROOT / output}")
        print(f"   最优模型: {self.log_dir / 'rating_best.pth'}")
        print(f"{'='*60}")


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """加载 yaml 配置"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="定数评级模型训练")
    parser.add_argument("--config", type=str, default="configs/rating.yaml")
    parser.add_argument("--output", type=str, default=None, help="输出 checkpoint 路径")
    parser.add_argument("--labels", type=str, default=None, help="覆盖标签文件")
    parser.add_argument("--label-source", type=str, default=None,
                        choices=["labels", "pseudo", "any"])
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="debug 模式 (2 epoch)")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # debug 模式覆盖
    if args.debug:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 8
        config["data"]["num_workers"] = 0
        config["logging"]["log_interval"] = 5
        if not args.output:
            args.output = "logs/rating_debug.pth"
        print("🐛 DEBUG 模式: 2 epoch, batch=8")

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    # 注入 epochs 到 trainer
    trainer = RatingTrainer(config, args)
    trainer.epochs = config["training"]["epochs"]
    trainer.train()


if __name__ == "__main__":
    main()
