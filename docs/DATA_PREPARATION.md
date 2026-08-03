# 数据准备指南

本文档说明如何为 ArellanoDreamWeaver 准备训练数据。

## 目录结构

```
data/
├── processed/          # 谱面 JSON 文件
│   ├── Calm.json
│   ├── Song1.json
│   └── Song2.json
├── audio/              # 对应的音频文件
│   ├── Calm.mp3
│   ├── Song1.mp3
│   └── Song2.mp3
├── cache/              # 音频特征缓存 (自动生成)
│   ├── Calm.npy
│   └── ...
└── raw/                # 原始数据 (可选)
```

## 数据配对规则

谱面文件和音频文件通过**文件名**匹配：

- `Calm.json` ↔ `Calm.mp3`
- `Song1.json` ↔ `Song1.wav`
- `MyChart.json` ↔ `MyChart.flac`

支持的音频格式：`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`

## 快速开始

### 1. 检查数据配对情况

```bash
python scripts/prepare_data.py --check
```

输出示例：
```
============================================================
📋 检查谱面-音频配对情况
============================================================

找到谱面文件: 10 个
找到音频文件: 8 个

✅ 已匹配: 8 对
   Calm.json ↔ Calm.mp3
   Song1.json ↔ Song1.wav
   ...

⚠️  缺少音频的谱面: 2 个
   ❌ Song9.json
   ❌ Song10.json
```

### 2. 验证数据完整性

```bash
python scripts/prepare_data.py --validate
```

### 3. 处理音频特征

```bash
python scripts/prepare_data.py --process
```

这会提取音频特征并缓存到 `data/cache/` 目录。

### 4. 生成统计报告

```bash
python scripts/prepare_data.py --stats
```

输出 `dataset_stats.json`，包含：
- 谱面总数
- 音符类型分布
- BPM 分布
- 难度分布
- 音符密度统计

### 5. 执行所有步骤

```bash
python scripts/prepare_data.py --all
```

## 谱面文件格式

谱面 JSON 文件应遵循以下结构：

```json
{
  "judgeSegments": [
    {
      "segmentEvents": {
        "MoveX": [{"beat": 0.0, "value": 0.0}],
        "MoveY": [{"beat": 0.0, "value": 0.0}],
        "Rotate": [],
        "Alpha": [],
        "Length": []
      },
      "notesAbove": [
        {
          "type": 0,
          "time": 0.5,
          "holdTime": 0.0,
          "positionX": 0.0,
          "isFakeNote": false,
          "offset": 0.0,
          "isAbove": true,
          "hasOther": false
        }
      ],
      "notesBelow": [],
      "far": null
    }
  ],
  "BoundaryList": [],
  "bpmList": [
    {"num": 120000, "den": 1000}
  ],
  "offset": 0.0,
  "musicLength": 180.0,
  "beatSubdivision": 4
}
```

关键字段：
- `musicLength`: 音乐时长（秒），用于对齐音频特征
- `bpmList`: BPM 信息
- `notesAbove` / `notesBelow`: 音符列表

## 音频文件要求

- **采样率**: 22050 Hz（默认，会自动重采样）
- **声道**: 单声道或立体声（自动转换）
- **格式**: MP3, WAV, FLAC, OGG, M4A, AAC
- **时长**: 应与谱面的 `musicLength` 一致

## 训练配置

在 `configs/default.yaml` 中添加：

```yaml
data:
  processed_dir: "data/processed"
  audio_dir: "data/audio"
  cache_dir: "data/cache"
  max_seq_len: 4096
  chart:
    feature_dim: 8
  audio:
    feature_dim: 128
    sample_rate: 22050

training:
  use_real_audio: true          # 使用真实音频（False 则使用随机噪声）
  cache_audio_features: true    # 缓存音频特征到内存
  # ... 其他训练参数
```

## 数据集统计

建议的最小数据集规模：

| 规模 | 谱面数量 | 说明 |
|------|---------|------|
| 测试 | 1-5 | 验证流程是否跑通 |
| 开发 | 10-50 | 调参和模型选择 |
| 训练 | 100+ | 获得有意义的结果 |
| 生产 | 500+ | 完整的模型训练 |

## 常见问题

### Q: 音频文件缺失怎么办？

A: 训练时会使用随机噪声作为后备，但模型无法学到有意义的音乐-谱面映射。建议补全音频文件。

### Q: 谱面和音频时长不一致？

A: 音频会被截断或插值到谱面的 `musicLength`。建议确保两者一致。

### Q: 如何增加训练数据？

A: 
1. 从 ArellanoDreamEdit 导出更多谱面
2. 确保每个谱面有对应的音频文件
3. 运行 `prepare_data.py --all` 处理新数据

### Q: 内存不足？

A: 设置 `cache_audio_features: false`，不缓存音频特征到内存。

## 数据增强（未来计划）

目前不支持数据增强。未来计划添加：
- 音频时间拉伸
- 音调变换
- 谱面时间偏移
- 随机噪声注入

## 下一步

数据准备完成后，运行训练：

```bash
# Debug 模式（快速测试）
python scripts/train.py --config configs/default.yaml --debug

# 完整训练
python scripts/train.py --config configs/default.yaml
```
