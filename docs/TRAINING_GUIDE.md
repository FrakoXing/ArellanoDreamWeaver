# ArellanoDreamWeaver 训练指南

## 🎯 快速开始

### 1. 准备数据

将谱面和音频文件放入对应目录：

```
data/
├── processed/    # 谱面 JSON 文件
│   └── Calm.json
└── audio/        # 对应的音频文件
    └── Calm.mp3
```

**重要**: 谱面和音频通过文件名匹配（如 `Calm.json` ↔ `Calm.mp3`）

### 2. 检查数据

```bash
# 检查配对情况
python scripts/prepare_data.py --check

# 验证数据完整性
python scripts/prepare_data.py --validate

# 处理音频特征
python scripts/prepare_data.py --process

# 生成统计报告
python scripts/prepare_data.py --stats
```

### 3. 开始训练

```bash
# Debug 模式（快速测试）
python scripts/train.py --config configs/default_new.yaml --debug

# 完整训练
python scripts/train.py --config configs/default_new.yaml
```

## 📊 数据要求

### 最小数据集

| 规模 | 谱面数量 | 用途 |
|------|---------|------|
| 测试 | 1-5 | 验证流程 |
| 开发 | 10-50 | 调参 |
| 训练 | 100+ | 有意义结果 |
| 生产 | 500+ | 完整训练 |

### 文件格式

**谱面**: JSON 格式，包含 `judgeSegments`、`bpmList`、`musicLength` 等字段

**音频**: MP3/WAV/FLAC/OGG/M4A，采样率 22050Hz（自动重采样）

## 🔧 配置说明

在 `configs/default_new.yaml` 中：

```yaml
training:
  use_real_audio: true          # True=使用真实音频，False=使用随机噪声
  cache_audio_features: true    # 缓存音频特征到内存（加速训练）

data:
  processed_dir: "data/processed"
  audio_dir: "data/audio"
  audio:
    sample_rate: 22050
    feature_dim: 128
```

## 📈 训练监控

训练日志保存在 `logs/` 目录：

```bash
# 查看 TensorBoard
tensorboard --logdir logs/
```

## ❓ 常见问题

### Q: Loss 不下降？

A: 检查是否使用了真实音频（`use_real_audio: true`），以及数据量是否足够。

### Q: 内存不足？

A: 设置 `cache_audio_features: false`，或减小 `batch_size`。

### Q: 音频文件缺失？

A: 训练会使用随机噪声作为后备，但模型无法学到有意义的模式。建议补全音频。

## 📚 详细文档

- [数据准备指南](docs/DATA_PREPARATION.md)
- [模型架构说明](docs/MODEL_ARCHITECTURE.md)（待编写）

## 🚀 下一步

1. 准备更多谱面-音频配对数据
2. 调整超参数
3. 训练并评估模型
4. 使用 `generate.py` 生成新谱面
