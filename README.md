# ArellanoDreamWeaver 🎵✨

> **基于多任务 Transformer + Diffusion 的 AI 辅助制谱模型**

## 项目架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户提示词 (自然语言)                       │
│   "生成一段 120 BPM 的流行风格谱面，难度中等，有丰富的滑动音符"    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM 理解模块                               │
│         提示词解析 → 结构化参数 (风格/难度/BPM/密度等)          │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              多任务 Transformer 编码器                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ 音乐特征  │ │ 风格嵌入  │ │ 节拍模式  │ │ LLM条件  │        │
│  │ Embedding│ │ Module   │ │ Encoder  │ │ Embedding│        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        │
│       └─────────────┴───────────┴─────────────┘              │
│                         ▼                                    │
│              Cross-Attention Fusion Layer                    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              条件 Diffusion 去噪网络                          │
│                                                               │
│     X_T (噪声谱面) ──→ [U-Net + Transformer] ──→ X_0 (清晰谱面) │
│            ↑                                              │
│            │ 条件注入 (来自Transformer的特征)                  │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    多任务输出头                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ 音符预测  │ │ BPM预测  │ │ 难度评估  │ │ 风格分类  │        │
│  │ Head     │ │ Head     │ │ Head     │ │ Head     │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ▼
                   🎼 生成的谱面数据 (ChartData)
```

## 核心特性

### 1. 多任务 Transformer 架构
- **音乐特征编码**: BPM、节拍、音高、时长的联合表示学习
- **跨模态融合**: 音乐音频特征 + 文本语义特征的深度融合
- **多头注意力**: 捕获长距离的谱面模式和音乐结构

### 2. 条件 Diffusion 生成
- **渐进式去噪**: 从纯噪声逐步生成高质量谱面
- **条件引导**: 以音乐特征和用户需求为条件控制生成方向
- **可控性**: 支持通过调整条件强度控制生成的创造性

### 3. LLM 智能理解
- **自然语言接口**: "帮我生成一段激昂的重金属谱面"
- **语义解析**: 理解风格、难度、节奏密度、特殊技巧等复杂需求
- **参数推荐**: 根据音乐特征自动推荐合理的制谱参数

### 4. 与 ArellanoDreamEdit 无缝对接
- **统一数据格式**: 直接输出 `ChartData` JSON 格式
- **实时预览支持**: 生成的谱面可立即导入编辑器预览

## 快速开始

```bash
# 1. 初始化项目 (安装依赖 + 配置)
python init_dreamweaver.py

# 2. 训练模型
python train.py --config configs/default.yaml

# 3. 使用 AI 制谱
python generate.py \
    --prompt "生成一段 128 BPM 的电子音乐谱面，包含大量滑动音符" \
    --audio input_music.mp3 \
    --output chart.json

# 4. 仅使用 LLM 分析提示词
python analyze_prompt.py --prompt "制作一首悲伤的钢琴曲谱面"
```

## 项目结构

```
ArellanoDreamWeaver/
├── README.md                          # 本文档
├── requirements.txt                   # Python 依赖
├── init_dreamweaver.py               # 🔮 初始化脚本 (Init Myself)
├── setup.py                           # 包安装配置
│
├── configs/                           # 配置文件目录
│   ├── default.yaml                   # 默认训练配置
│   ├── model.yaml                     # 模型架构配置
│   └── diffusion.yaml                 # Diffusion 超参数
│
├── dreamweaver/                       # 核心包
│   ├── __init__.py                    # 包初始化
│   ├── config.py                      # 配置管理器
│   │
│   ├── models/                        # 模型定义
│   │   ├── __init__.py
│   │   ├── transformer.py             # 多任务 Transformer
│   │   ├── diffusion.py               # 条件 Diffusion U-Net
│   │   ├── encoder.py                 # 音乐/文本编码器
│   │   └── heads.py                   # 多任务输出头
│   │
│   ├── diffusion/                     # Diffusion 过程
│   │   ├── __init__.py
│   │   ├── noise_scheduler.py         # 噪声调度器
│   │   ├── sampler.py                 # 采样算法 (DDPM/DDIM)
│   │   └── guidance.py                # 分类器-free 引导
│   │
│   ├── llm/                           # LLM 接口模块
│   │   ├── __init__.py
│   │   ├── prompt_parser.py           # 提示词解析器
│   │   ├── embedder.py                # 文本嵌入层
│   │   └── templates.py               # 提示词模板
│   │
│   ├── data/                          # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py                 # 谱面数据集
│   │   ├── preprocess.py              # 数据预处理
│   │   └── augment.py                 # 数据增强
│   │
│   ├── chart/                         # 谱面处理
│   │   ├── __init__.py
│   │   ├── structures.py              # 数据结构定义 (兼容 DreamEdit)
│   │   ├── serializer.py              # JSON 序列化/反序列化
│   │   └── validator.py               # 谱面验证器
│   │
│   └── utils/                         # 工具函数
│       ├── __init__.py
│       ├── audio.py                   # 音频特征提取
│       ├── metrics.py                 # 评估指标
│       └── visualization.py           # 可视化工具
│
├── scripts/                           # 运行脚本
│   ├── train.py                       # 训练入口
│   ├── generate.py                    # AI 制谱入口
│   ├── analyze_prompt.py              # 提示词分析工具
│   └── export.py                      # 模型导出
│
├── tests/                             # 测试用例
│   ├── test_models.py
│   ├── test_diffusion.py
│   ├── test_llm.py
│   └── test_chart.py
│
└── checkpoints/                       # 模权重存储 (gitignore)
    └── .gitkeep
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.0+ |
| Transformer | 自定义多任务架构 |
| Diffusion | DDPM + DDIM 采样 |
| LLM 接口 | OpenAI API / 本地模型 |
| 音频处理 | librosa + torchaudio |
| 数据格式 | JSON (兼容 ArellanoDreamEdit) |

## 许可证

MIT License
