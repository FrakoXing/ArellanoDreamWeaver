# 训练系统优化总结

## 📋 更改概述

将 ArellanoDreamWeaver 从"使用随机噪声模拟音频"升级为"支持真实音频特征提取"的完整训练系统。

## 🆕 新增文件

### 1. 核心模块

| 文件 | 说明 |
|------|------|
| `src/data/audio_processor.py` | 音频特征提取模块，支持梅尔频谱、色度特征等 |
| `src/data/dataset.py` | 数据集模块，支持真实音频-谱面对加载 |
| `scripts/prepare_data.py` | 数据预处理工具，检查配对、处理特征、生成统计 |

### 2. 配置文件

| 文件 | 说明 |
|------|------|
| `configs/default_new.yaml` | 更新后的训练配置，包含音频相关参数 |

### 3. 文档

| 文件 | 说明 |
|------|------|
| `docs/DATA_PREPARATION.md` | 详细的数据准备指南 |
| `docs/TRAINING_GUIDE.md` | 快速训练指南 |

### 4. 目录结构

```
data/
├── processed/    # 谱面 JSON 文件
├── audio/        # 音频文件 (新增)
├── cache/        # 音频特征缓存 (新增)
└── raw/          # 原始数据
```

## 🔧 修改的文件

### `scripts/train.py`

- 添加了从 `src.data.dataset` 导入新数据集模块
- 更新了数据加载逻辑，支持配置项：
  - `use_real_audio`: 是否使用真实音频
  - `cache_audio_features`: 是否缓存音频特征
  - `audio_dir`: 音频文件目录

## 🚀 使用方法

### 步骤 1: 准备数据

将谱面和音频文件放入对应目录：

```bash
# 谱面文件
data/processed/Calm.json
data/processed/Song1.json

# 对应的音频文件（文件名匹配）
data/audio/Calm.mp3
data/audio/Song1.mp3
```

### 步骤 2: 检查数据

```bash
# 检查配对情况
python scripts/prepare_data.py --check

# 验证数据完整性
python scripts/prepare_data.py --validate

# 处理音频特征（提取并缓存）
python scripts/prepare_data.py --process

# 生成统计报告
python scripts/prepare_data.py --stats
```

### 步骤 3: 开始训练

```bash
# Debug 模式（快速测试）
python scripts/train.py --config configs/default_new.yaml --debug

# 完整训练
python scripts/train.py --config configs/default_new.yaml
```

## 📊 关键特性

### 音频特征提取

- **梅尔频谱**: 128维，对数尺度
- **采样率**: 22050 Hz（可配置）
- **自动对齐**: 根据谱面 `musicLength` 自动截断/插值
- **特征缓存**: 首次处理后缓存到 `data/cache/`，后续训练直接加载

### 数据配对

- 通过文件名自动匹配：`Calm.json` ↔ `Calm.mp3`
- 支持多种音频格式：MP3, WAV, FLAC, OGG, M4A, AAC
- 缺失音频时使用随机噪声作为后备（但会影响训练效果）

### 难度估计

自动根据音符密度估计难度（0-4级）：
- 简单: < 2 音符/秒
- 中等: 2-4 音符/秒
- 困难: 4-6 音符/秒
- 专家: 6-8 音符/秒
- 大师: > 8 音符/秒

## ⚙️ 配置说明

在 `configs/default_new.yaml` 中：

```yaml
training:
  use_real_audio: true          # True=真实音频，False=随机噪声
  cache_audio_features: true    # 缓存音频特征（加速训练）

data:
  processed_dir: "data/processed"
  audio_dir: "data/audio"
  cache_dir: "data/cache"
  audio:
    sample_rate: 22050
    feature_dim: 128
```

## 📈 预期效果

### 优化前

- ❌ 音频特征为随机噪声
- ❌ 模型无法学习音乐-谱面映射
- ❌ Loss 2.677 无实际意义

### 优化后

- ✅ 使用真实音频特征（梅尔频谱）
- ✅ 模型可以学习音乐与谱面的对应关系
- ✅ Loss 下降反映真实的训练进度
- ✅ 生成的谱面与音乐节奏对齐

## 🔍 故障排查

### 问题 1: 音频文件缺失

**症状**: 警告 "缺少音频的谱面: X 个"

**解决**: 将音频文件放入 `data/audio/`，确保文件名匹配

### 问题 2: 内存不足

**症状**: CUDA out of memory 或 RAM 不足

**解决**: 
- 设置 `cache_audio_features: false`
- 减小 `batch_size`
- 减小 `max_seq_len`

### 问题 3: Loss 不下降

**症状**: Loss 长时间保持在 2.5+ 不下降

**解决**:
- 检查是否使用了真实音频
- 增加数据量（至少 10+ 谱面）
- 检查学习率设置
- 查看 TensorBoard 监控训练曲线

## 📝 下一步计划

1. **数据增强**: 添加音频时间拉伸、音调变换等
2. **多GPU训练**: 支持分布式训练
3. **推理脚本**: 使用训练好的模型生成谱面
4. **评估指标**: 添加谱面质量评估指标

## 📚 相关文档

- [数据准备指南](DATA_PREPARATION.md)
- [训练指南](TRAINING_GUIDE.md)
- [原 README](../README.md)
