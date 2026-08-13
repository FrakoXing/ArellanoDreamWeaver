# 定数评级 AI 使用指南

## 概述

定数评级 AI 是 ArellanoDreamWeaver 的子模块，用于对谱面进行连续难度定数评级。
它接受纯谱面 Token 序列作为输入，输出归一化定数 [0, 1]，再按游戏反归一化为原始定数（如 Phigros 12.4）。

### 架构

```
Token 序列 ──► Embedding + PosEnc ──► 双向 Transformer Encoder
                 (Game Token 条件注入 via AdaptiveLayerNorm)
                 ──► Masked Mean Pool ──► token_repr
手工特征(15维) ──► LayerNorm + MLP ──► hc_repr
token_repr + hc_repr ──► GatedFusion ──► RegressionHead ──► sigmoid ──► rating_norm
```

- **模型参数**: ~5.13M（轻量，编辑器友好）
- **输入**: tokens[1,2048] + attention_mask[1,2048] + handcrafted_features[1,15] + game_token_ids[1]
- **输出**: rating_norm[1] ∈ (0, 1)
- **推理耗时**: ~10-30ms（CPU，单谱面）

---

## 文件结构

```
ArellanoDreamWeaver/
├── configs/rating.yaml              # 配置文件
├── src/
│   ├── models/rating_model.py       # RatingModel 主模型
│   └── data/
│       ├── label_parser.py          # 文件名解析 + 归一化
│       ├── handcrafted_features.py  # 15 维手工特征
│       └── rating_dataset.py        # RatingDataset + 分桶采样
├── scripts/
│   ├── rating_common.py             # 共享加载器 (绕过 librosa 依赖)
│   ├── build_rating_labels.py       # 标签构建
│   ├── train_rating.py              # 训练器
│   ├── export_rating_onnx.py        # ONNX 双 opset 导出
│   ├── infer_rating_onnx.py         # ONNX 推理验证
│   ├── pseudo_label.py              # 伪标签生成 (v2 半监督)
│   └── verify_cs_alignment.py       # C# 数值对齐验证
├── tests/test_rating.py             # 单元测试
└── docs/RATING_GUIDE.md             # 本文档

ArellanoDreamEdit/                    # Unity 编辑器
└── Assets/Scripts/AI/
    ├── ChartTokenConstants.cs        # Token 词汇表常量
    ├── RatingMath.cs                 # 数学工具 (银行家舍入 + 百分位 + 归一化)
    ├── ChartTokenizer.cs             # ChartData → Token 编码
    ├── TokenDecoder.cs               # Token → 音符解码
    ├── TokenFeatureExtractor.cs      # Token → 15 维特征
    ├── AIRatingRunner.cs             # ONNX 推理后端 (Sentis + ORT)
    ├── AIRatingConfig.cs             # ScriptableObject 配置
    ├── AIRatingService.cs            # 高层服务 API
    └── Editor/
        └── AIRatingEditorWindow.cs   # 编辑器窗口
```

---

## 训练管线

### 1. 构建标签

```bash
# 扫描 data/tokens/*.pt，从文件名解析定数标签
python scripts/build_rating_labels.py

# 同时计算特征标准化统计（推荐，较慢）
python scripts/build_rating_labels.py --feature-stats
```

产物：
- `data/rating/labels.json` — 训练标签（~6300 条）
- `data/rating/calibration_set.json` — 校准集（~200 条，不参与训练）
- `data/rating/normalization.json` — 归一化范围
- `data/rating/feature_stats.json` — 特征均值/方差（`--feature-stats` 时生成）

### 2. 训练 v1 模型

```bash
# 默认配置训练
python scripts/train_rating.py --config configs/rating.yaml

# Debug 模式（2 epoch，快速验证）
python scripts/train_rating.py --config configs/rating.yaml --debug
```

产物：
- `logs/rating_best.pth` — 最优模型
- `logs/rating.pth` — 最终模型

### 3. 生成伪标签（v2 半监督）

```bash
# 用 v1 模型对无标签数据推理
python scripts/pseudo_label.py --checkpoint logs/rating_best.pth

# 或用 ONNX 模型（更快）
python scripts/pseudo_label.py --onnx exports/rating/rating_ort_opset17.onnx
```

产物：`data/rating/labels_pseudo.json`

### 4. 训练 v2 模型（真标签 + 伪标签）

```bash
python scripts/train_rating.py --config configs/rating.yaml \
    --label-source any --lr 3e-5 --output logs/rating_v2.pth
```

---

## ONNX 导出

```bash
# 从训练好的 checkpoint 导出（双 opset）
python scripts/export_rating_onnx.py --checkpoint logs/rating_best.pth

# 仅导出 Sentis 版本（省体积）
python scripts/export_rating_onnx.py --checkpoint logs/rating_best.pth --sentis-only

# 导出后自动验证
python scripts/export_rating_onnx.py --checkpoint logs/rating_best.pth --verify
```

产物（`exports/rating/`）：
| 文件 | 用途 | opset |
|------|------|-------|
| `rating_ort_opset17.onnx` | ONNX Runtime Unity | 17 |
| `rating_sentis_opset14.onnx` | Unity Sentis | 14 |
| `model_meta.json` | 元数据（归一化范围 + 特征统计） | — |
| `normalization.json` | 归一化范围（独立文件） | — |

### 推理验证

```bash
# 单文件推理
python scripts/infer_rating_onnx.py --model exports/rating/rating_ort_opset17.onnx \
    --token-file data/tokens/10018_IN\ Lv.12_xxx.pt

# 批量评估（校准集）
python scripts/infer_rating_onnx.py --model exports/rating/rating_ort_opset17.onnx \
    --batch --limit 50
```

---

## Unity 集成

### 1. 安装 AI 包

在 Unity Package Manager 中安装以下之一：

- **Unity Sentis**（推荐）：`com.unity.sentis`，原生 Unity 集成，opset 14
- **ONNX Runtime**：`com.unity.ai.microsoft.onnxruntime`，opset 17

### 2. 部署模型文件

将导出的文件复制到 Unity 项目：

```
ArellanoDreamEdit/Assets/StreamingAssets/AIRating/
├── rating_sentis_opset14.onnx    # Sentis 模型
├── rating_ort_opset17.onnx       # ORT 模型（可选）
├── model_meta.json               # 元数据
└── normalization.json            # 归一化范围
```

### 3. 创建配置

1. Project 窗口右键 → Create → Arellano → AIRating Config
2. 在 Inspector 中设置 `modelFileName`（如 `rating_sentis_opset14.onnx`）
3. 点击 "从 model_meta.json 导入特征统计" 按钮

### 4. 使用编辑器窗口

- 菜单栏 → Arellano → 定数评级 AI（快捷键 Ctrl+R）
- 窗口自动检测场景中的 ChartEditorController
- 开启"自动更新"后，编辑谱面时每 2 秒自动重算定数

### 5. C# API

```csharp
using Arellano.AI.Rating;

// 初始化
var service = AIRatingService.Instance;
service.Initialize(config); // config 可为 null（自动加载）

// 评级
RatingResult result = service.Rate(chartData);
Debug.Log($"定数: {result.rawRating:F1} ({result.game})");
// 输出: 定数: 12.4 (arellano)

// 释放
service.Dispose();
```

---

## 数值对齐验证

Python 和 C# 必须产出逐值一致的结果。验证方法：

```bash
# 1. 生成参考数据
python scripts/verify_cs_alignment.py \
    --chart-json path/to/Chart.json \
    --output cs_verify.json

# 2. 在 C# 中加载 cs_verify.json，对比 Token 序列和特征值
#    量化示例 (C# 应产出相同值):
#      time:  0.25 → 8,  1.0 → 11,  15.75 → 70
#      pos:  -1.0 → 71, 0.0 → 75,   1.0 → 79
#      hold:  0.25 → 81, 1.0 → 84,   7.75 → 111
```

### 关键对齐点

| 项目 | Python | C# | 说明 |
|------|--------|-----|------|
| 舍入 | `round()` 银行家舍入 | `Math.Round(ToEven)` | 0.5→0, 1.5→2 |
| 百分位 | `np.percentile` linear | `RatingMath.Percentile` | linear 插值 |
| 方差 | `np.var` ddof=0 | `RatingMath.Variance` | 总体方差 |
| 归一化 | `normalize_level` | `RatingMath.NormalizeLevel` | 各游戏独立范围 |

---

## 跨游戏定数体系

| 游戏 | Token ID | 范围 | 归一化 |
|------|----------|------|--------|
| Phigros | 116 | 0–16 | level / 16 |
| osu!mania | 117 | 0–50 | level / 50 |
| Arellano | 118 | 0–16 | level / 16 |

推理时按 Game Token ID 自动选择正确的反归一化范围。

---

## 常见问题

### Q: 模型未加载（IsBackendAvailable = false）

**A**: 检查以下几点：
1. Sentis 或 ORT 包是否已安装
2. ONNX 文件是否在 `StreamingAssets/AIRating/` 下
3. `AIRatingConfig.modelFileName` 是否正确
4. 查看 Console 日志中的错误信息

### Q: C# 和 Python 特征值不一致

**A**: 检查以下几点：
1. 舍入方式是否使用银行家舍入（`MidpointRounding.ToEven`）
2. `np.round` 在 Python 中也是银行家舍入
3. 百分位计算是否使用 linear 插值
4. 特征标准化是否加载了正确的 mean/std

### Q: 训练时 librosa 报错

**A**: 评级模块通过 `rating_common.py` 绕过了 `src/data/__init__.py` 中的 librosa 依赖。如果仍然报错，确保通过 `rating_common` 而非直接 `from src.data import ...` 导入。

### Q: ONNX 导出失败

**A**: 检查以下几点：
1. PyTorch 版本 ≥ 2.0
2. 模型是否已 `eval()` 模式
3. 使用 `--random-weights` 先测试导出管线
4. 查看 DeprecationWarning（可忽略，我们用 `dynamo=False` 有意选择旧导出器）

---

## 性能指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Phigros MAE | < 0.4 | 平均绝对误差 |
| ±0.5 命中率 | > 75% | 误差 ≤ 0.5 的比例 |
| ±1.0 命中率 | > 90% | 误差 ≤ 1.0 的比例 |
| 推理延迟 | < 30ms | CPU 单谱面 |
| 模型体积 | ~20MB | ONNX 文件 |
```
