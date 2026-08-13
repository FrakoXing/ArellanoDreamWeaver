# Scripts 使用说明

> 本文档汇总 `scripts/` 目录下所有脚本的用途、参数与使用示例。
> 所有命令均在项目根目录下执行（`python scripts/<脚本名>.py`）。

---

## 目录

1. [总览：脚本按流程分类](#一总览脚本按流程分类)
2. [数据下载](#二数据下载)
3. [格式转换](#三格式转换)
4. [Token 化与数据预处理](#四token-化与数据预处理)
5. [检查与分析工具](#五检查与分析工具)
6. [模型训练](#六模型训练)
7. [导出与推理生成](#七导出与推理生成)
8. [提示词分析](#八提示词分析)
9. [典型端到端流程](#九典型端到端流程)
10. [其他说明](#十其他说明)

---

## 一、总览：脚本按流程分类

| 阶段 | 脚本 | 作用 |
|------|------|------|
| 数据下载 | `download_phira.py` | 从 Phira 批量下载 `.pez` 谱面包 |
| 格式转换 | `process_pez.py` | 解压 `.pez` 提取谱面 / 音频 / 封面 |
| | `phira_to_chart.py` | `.pez` → ChartData JSON |
| | `process_osz.py` | `.osz` 提取 `.osu` + 音频 |
| | `osu_to_chart.py` | `.osu` → ChartData JSON |
| | `converters/osu_converter.py` | `.osu` → ChartData JSON（库式版本） |
| Token 化 | `unified_tokenizer.py` | 各种谱面格式 → Token 序列（训练数据） |
| 预处理 | `prepare_audio_cache.py` | 音频特征预计算缓存（训练提速） |
| | `prepare_data.py` | 配对检查 / 特征提取 / 统计报告 / 完整性验证 |
| 检查分析 | `check_data.py` / `check_token_stats.py` / `check_time_offset.py` / `analyze_data.py` / `analyze_events.py` / `check_progress.py` / `verify_dataset_pipeline.py` / `test_api.py` / `cleanup_raw_pez.py` | 数据检查与统计 |
| 训练 | `train.py` / `train_tokens.py` | 模型训练 |
| 导出推理 | `export_onnx.py` → `infer_onnx.py` → `export_chart_pec.py` | 部署 / 生成谱面 / 导出可玩格式 |
| 提示词分析 | `analyze_prompt.py` | 单独测试 LLM 提示词解析 |

> 注：`scripts/123.txt` 不是脚本，是手动记录的标准流程命令备忘（见[第九节](#九典型端到端流程)）。

---

## 二、数据下载

### `download_phira.py` — Phira 谱面批量下载

从 [Phira](https://phira.cn/) API 批量下载谱面包（`.pez`），支持断点续传、三层持久化防丢数据（内存集合 + 增量日志 + 原子进度文件）。

**必选参数（三选一）：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--ids N1 N2 ...` | 指定 ID 列表 | `--ids 123 456 789` |
| `--range "a-b,c-d"` | 指定范围（支持多段） | `--range "1-500,1000-1200"` |
| `--start N --end M` | 起止 ID（必须成对使用） | `--start 1 --end 1000` |

**可选参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `-o, --output` | `./phira_charts` | 输出目录 |
| `-c, --concurrent` | `16` | 并发下载数 |

**示例：**

```bash
python scripts/download_phira.py --start 1 --end 1000 -c 16
python scripts/download_phira.py --range "1-500,1000-1200"
python scripts/download_phira.py --ids 123 456 789
```

**输出：** 下载目录下生成 `.pez` 文件，以及进度文件 `.download_progress.json`（已下载 ID / 黑名单）和 `.incr_progress.log`（增量兜底日志），用于断点续传。

---

## 三、格式转换

### `process_pez.py` — PEZ 谱面包处理

解压 `.pez`（ZIP 格式）谱面包，提取谱面文件（chart.json / chart.pec）、音频（mp3/ogg/wav/flac）和封面，并可按需直接转换为 Token。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `pez_file`（位置参数） | 无 | 单个 PEZ 文件路径（可选，不填则处理整个目录） |
| `-i, --input` | `data/raw_pez` | PEZ 输入目录 |
| `-o, --output` | `data/raw` | 谱面输出目录 |
| `-a, --audio` | `data/audio` | 音频输出目录 |
| `-t, --token-dir` | `data/tokens` | Token 输出目录（配合 `--tokenize`） |
| `--delete-pez` | 关 | 处理完成后删除源 PEZ 文件 |
| `--tokenize` | 关 | 将 raw 目录中的谱面转换为 Token |
| `--delete-raw` | 关 | 转换为 Token 后删除原始 JSON/PEC |

**示例：**

```bash
# 处理所有 PEZ 文件
python scripts/process_pez.py

# 处理单个 PEZ 文件
python scripts/process_pez.py path/to/chart.pez

# 解压 → 转 Token → 清理中间文件（全自动）
python scripts/process_pez.py --delete-pez --tokenize --delete-raw

# 仅将已有 raw 谱面转为 Token
python scripts/process_pez.py --tokenize --delete-raw
```

### `phira_to_chart.py` — .pez → ChartData JSON

将 Phira `.pez` 谱面包转换为项目标准的 ChartData JSON。支持 RPE（chart.json）与 PEC（*.pec）两种内部格式，`.pez` 内元信息 `info.yml` 需要 `pyyaml`（未安装时退化为正则解析）。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `input`（位置参数） | 无 | 单个 `.pez` 文件路径 |
| `--dir` | 无 | 批量转换目录（所有 `.pez`） |
| `-o, --outdir` | `data/processed` | 输出目录 |
| `-a, --audio-dir` | 无 | 音频输出目录（指定则提取音乐） |
| `--no-meta` | 关 | 不保存元信息文件 |

**示例：**

```bash
# 批量转换并提取音频
python scripts/phira_to_chart.py --dir ./phira_charts --outdir data/processed --audio-dir data/audio

# 转换单个文件
python scripts/phira_to_chart.py input.pez
```

### `process_osz.py` — .osz 自动处理

从 osu! 谱面包（`.osz`，ZIP）中提取 `.osu` 谱面文件和音频，自动整理到训练数据目录。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i, --input` | `data/raw_osz` | 输入目录 |
| `--output-osu` | `data/raw` | `.osu` 输出目录 |
| `--output-audio` | `data/audio` | 音频输出目录 |
| `-f, --file` | 无 | 处理单个 `.osz` 文件 |
| `-q, --quiet` | 关 | 静默模式 |

**示例：**

```bash
python scripts/process_osz.py
python scripts/process_osz.py --input ./osz_files --output-osu ./osu --output-audio ./audio
python scripts/process_osz.py --file song.osz
```

### `osu_to_chart.py` — .osu → ChartData JSON

将 osu! 谱面转换为 ChartData JSON。支持：
- **osu!standard (Mode=0)**：圆圈→Tap，滑条→Slide，转盘→Slide(长 hold)
- **osu!mania (Mode=3)**：普通音符→Tap，长按→Hold

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `input` / `output`（位置参数） | 无 | 单个文件转换：输入 `.osu`、输出 `.json`（可选） |
| `--dir` | 无 | 批量转换目录（所有 `.osu`） |
| `--outdir` | 无 | 批量输出目录 |
| `--no-split` | 关 | 不分割段落（所有音符放入单个 judgeSegment） |
| `--segment-beats` | `16.0` | 每段拍数 |
| `--pretty` | 开 | 格式化 JSON 输出 |

**示例：**

```bash
python scripts/osu_to_chart.py song.osu output.json
python scripts/osu_to_chart.py --dir ./osu_maps --outdir data/processed
```

### `converters/osu_converter.py` — osu! 转换器（库式版本）

功能同 `osu_to_chart.py`，但作为独立模块组织（`scripts/converters/` 子目录），推荐用于 mania 模式。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i, --input` | 无 | 输入的 `.osu` 文件路径 |
| `-d, --dir` | 无 | 输入目录（批量转换） |
| `-o, --output` | `data/processed/` | 输出目录 |
| `--no-audio` | 关 | 不复制音频文件 |
| `--max` | `0` | 最大转换文件数（0=不限） |

**示例：**

```bash
python scripts/converters/osu_converter.py --input song.osu --output data/processed/
python scripts/converters/osu_converter.py --dir ./osu_maps --output data/processed/
```

---

## 四、Token 化与数据预处理

### `unified_tokenizer.py` — 统一谱面 → Token 序列转换器 ⭐核心

将 PEC / Phira JSON / RPE JSON / osu! / Arellano 谱面格式**直接**转换为模型可用的 Token 序列，跳过 ChartData JSON 中间步骤。每种格式的音符先统一为中间表示 `Note`，再离散化为 Token 序列。

**Token 词汇表（256 大小，与 NoteDecoder 对齐）：**

| Token ID | 含义 |
|----------|------|
| 0-3 | PAD / SOS / EOS / MASK |
| 4-6 | TAP / HOLD / SLIDE |
| 7-70 | 时间偏移（相对上一音符，0~15.75 拍，0.25 拍精度） |
| 71-79 | X 位置（9 档：-1.0 ~ 1.0） |
| 80-111 | Hold 时长（0~7.75 拍，0.25 拍精度） |
| 112 / 113 | ABOVE（判定线正面）/ BELOW（判定线背面） |
| 116-118 | 游戏来源（PHIRA / OSU / ARELLANO） |
| 120-126 | 判定线事件类型（MoveX/MoveY/Rotate/Alpha/Speed 等） |
| 128-239 | 事件值 / 缓动类型 / 缓动控制点 |
| 256 | 词汇表总大小 |

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `input`（位置参数） | 无 | 输入文件路径（chart.pec / song.osu / chart.json） |
| `-d, --dir` | 无 | 批量转换目录 |
| `-o, --outdir` | `data/tokens` | 输出目录 |
| `-f, --format` | `json` | 输出格式：`json` / `pt`（PyTorch，可直接训练）/ `txt` / `bin` |
| `-g, --game` | 自动检测 | 强制指定游戏来源：`phira` / `osu` / `arellano` |
| `-v, --verify` | 关 | 编码后解码验证（自检） |
| `-s, --stats` | 关 | 打印统计信息 |

**示例：**

```bash
# 转换单个 PEC / osu! / JSON 文件
python scripts/unified_tokenizer.py chart.pec
python scripts/unified_tokenizer.py song.osu
python scripts/unified_tokenizer.py chart.json

# 批量转换目录
python scripts/unified_tokenizer.py --dir ./charts/ --outdir data/tokens/

# 输出为 PyTorch .pt（可直接用于训练）
python scripts/unified_tokenizer.py chart.pec --format pt
```

**依赖：** numpy（`--format pt` 需要 torch）。

### `prepare_audio_cache.py` — 音频特征缓存预处理

将所有音频文件预先处理为 Mel 频谱 `.pt` 缓存文件，消除训练循环中的实时 librosa 处理，数据加载提速 50~100 倍。**训练前只需执行一次**。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--tokens-dir` | `data/tokens` | Token 文件目录 |
| `--audio-dir` | `data/audio` | 音频文件目录 |
| `--cache-dir` | `data/audio_cache` | 输出缓存目录 |
| `--target-steps` | `4096` | 目标时间步数（与 `max_seq_len` 匹配） |
| `--n-mels` | `128` | Mel 频带数 |
| `--sample-rate` | `22050` | 音频采样率 |
| `--workers` | `4` | 并行处理进程数 |
| `--start` / `--end` | `0` / 无 | 处理范围（支持并行分片） |
| `--force` | 关 | 即使缓存存在也重新处理 |

**示例：**

```bash
# 全量预处理
python scripts/prepare_audio_cache.py --target-steps 4096

# 并行分片处理
python scripts/prepare_audio_cache.py --start 0 --end 1000
python scripts/prepare_audio_cache.py --start 1000 --end 2000

# 更多 CPU 核心
python scripts/prepare_audio_cache.py --workers 8
```

### `prepare_data.py` — 数据预处理（旧管线）

检查谱面-音频配对、批量提取音频特征缓存、生成数据集统计报告、验证数据完整性。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--chart-dir` | `data/processed` | 谱面文件目录 |
| `--audio-dir` | `data/audio` | 音频文件目录 |
| `--cache-dir` | `data/cache` | 缓存目录 |
| `--check` | 关 | 检查配对情况 |
| `--process` | 关 | 处理音频特征 |
| `--stats` | 关 | 生成统计报告（输出 `data/dataset_stats.json`） |
| `--validate` | 关 | 验证数据完整性 |
| `--all` | 关 | 执行所有步骤 |
| `--force` | 关 | 强制重新处理 |

**示例：**

```bash
python scripts/prepare_data.py --check    # 检查配对
python scripts/prepare_data.py --stats    # 生成统计报告
python scripts/prepare_data.py --all      # 全部执行
```

---

## 五、检查与分析工具

### `check_data.py` — 查看训练数据详情

查看 `data/tokens/` 下前 5 个训练数据文件的详细内容：Token 总数、前 50 个 Token、音符类型统计（Tap/Hold/Slide）、time_offset 分布。

```bash
python scripts/check_data.py
```

### `check_token_stats.py` — Token 序列长度统计

统计 `data/tokens/` 下所有序列的长度分布：总数/平均值、min/max/median/mean、P10~P99 分位数、长度分桶、训练内存估算。

```bash
python scripts/check_token_stats.py
```

### `check_time_offset.py` — time_offset 分布分析

逐 Token 统计 `data/tokens/` 中 time_offset（7-70）的分布，并定位 time_offset=15.75bt（token 70）占比异常高的文件——用于排查"长空白段"导致的时间偏移污染问题。

```bash
python scripts/check_time_offset.py
```

### `analyze_data.py` — 全部 Token 分布统计

统计 `data/tokens/` 下全部 Token 的分布：特殊 Token、音符类型、上下方向、各语义区间，以及 Top 20 高频 Token。

```bash
python scripts/analyze_data.py
```

### `analyze_events.py` — RPE 事件结构分析

分析 RPE 格式谱面的 `judgeLineList` 事件结构（eventLayers 中各事件类型分布与样本）。⚠️ **路径硬编码为 `processed_pez/charts`**，使用前需确认该目录存在或手动修改。

```bash
python scripts/analyze_events.py
```

### `check_progress.py` — 下载进度检查

查看 `data/raw_pez/` 下的 `.download_progress.json`（已下载/黑名单 ID 数量）与 `.incr_progress.log`（增量日志行数），以及目录中已下载的 `.pez` 数量。

```bash
python scripts/check_progress.py
```

### `verify_dataset_pipeline.py` — 数据管线验证

验证 `TokenDataset` 管线（过滤 + 窗口化 + 音频对齐）的正确性：超长谱面过滤数量、随机窗口连续性、边界吸附等断言检查（PASS/FAIL 输出）。

```bash
python scripts/verify_dataset_pipeline.py
```

### `test_api.py` — Phira API 连通性测试

用 `aiohttp` 请求 Phira API（`https://api.phira.cn/chart/{id}`）测试几个代表性 ID（1/100/1000/2000/5000）的响应状态。需 `pip install aiohttp`。

```bash
python scripts/test_api.py
```

### `cleanup_raw_pez.py` — 清理下载目录

删除 `data/raw_pez/` 下的所有文件，**保留**进度文件（`.download_progress.json`、`.incr_progress.log`）以支持断点续传。⚠️ 破坏性操作，注意备份。

```bash
python scripts/cleanup_raw_pez.py
```

---

## 六、模型训练

### `train_tokens.py` — Token 序列训练脚本 ⭐推荐（当前主训练）

训练 `NoteDecoder`（Decoder-only Transformer），支持多格式混合训练。核心优化：AMP 混合精度（1.5~2 倍提速）、多线程 DataLoader、音频特征磁盘缓存、torch.compile、Flash SDP 注意力、梯度检查点、余弦退火重启 LR、中文 emoji 日志。

**数据参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--tokens-dir` | `data/tokens` | Token 序列存放目录 |
| `--audio-dir` | `data/audio` | 原始音频文件目录 |
| `--audio-cache-dir` | `data/audio_cache` | 预计算音频特征缓存目录（`prepare_audio_cache.py` 生成） |
| `--max-chart-tokens` | `200000` | 超长谱面过滤阈值（单谱 Token 数超过则剔除） |
| `--window-mode` | `random` | 窗口模式：`random` / `sliding` / `head` |
| `--window-stride` | `1024` | sliding 模式窗口步长（Token 数） |
| `--no-window-snap` | 关 | 随机窗口起点不吸附到音符/事件边界 |

**模型参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--hidden-dim` | `512` | Transformer 隐藏层维度 |
| `--num-layers` | `6` | 解码器 Transformer 层数 |
| `--num-heads` | `8` | 多头注意力头数 |
| `--max-seq-len` | `4096` | 输入 Token 最大序列长度 |
| `--vocab-size` | `256` | 词汇表总大小 |
| `--audio-dim` | `128` | 梅尔音频特征维度 |

**训练超参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | `100` | 总训练轮数 |
| `--batch-size` | `8` | 单步批次样本数 |
| `--lr` | `1e-4` | 基础初始学习率 |
| `--accumulation-steps` | `1` | 梯度累积步数 |
| `--warmup-epochs` | `5` | 学习率预热轮数 |
| `--dropout` | `0.3` | Dropout 比例 |
| `--val-split` | `0.05` | 验证集划分比例 |
| `--weight-decay` | `0.01` | AdamW 权重衰减 |

**性能与杂项：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num-workers` | `4` | 数据加载并行线程数 |
| `--no-amp` | 关 | 关闭 CUDA AMP 混合精度 |
| `--no-compile` | 关 | 关闭 PyTorch 2.x compile 编译加速 |
| `--use-checkpointing` | 关 | 开启梯度检查点（省显存） |
| `--resume` | 无 | 断点文件路径，恢复训练 |
| `--debug` | 关 | 快速调试模式（小型模型，仅 2 轮） |
| `--log-dir` | `logs` | 日志与断点输出目录 |
| `--no-audio-cache` | 关 | 不用预计算缓存，实时算特征 |

**示例：**

```bash
# 第一步：预处理音频（仅需一次）
python scripts/prepare_audio_cache.py --target-steps 4096

# 默认参数训练
python scripts/train_tokens.py --batch-size 8 --epochs 100

# 更大规模模型
python scripts/train_tokens.py --hidden-dim 512 --num-layers 6 --num-heads 8 --batch-size 4

# 调试模式（2 轮快速验证）
python scripts/train_tokens.py --debug

# 断点恢复
python scripts/train_tokens.py --resume logs/checkpoint_epoch_10.pth
```

### `train.py` — 旧版训练入口（多任务 Transformer + Diffusion）

早期版本训练脚本，训练 MultiTaskTransformer + ConditionalUNet1D 扩散模型，参数通过 YAML 配置文件控制。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | `configs/default.yaml` | 配置文件路径 |
| `--resume` | 无 | 恢复训练的检查点路径 |
| `--debug` | 关 | 调试模式（小规模快速测试） |
| `--device` | 自动 | 强制指定设备（cuda/cpu） |

```bash
python scripts/train.py --config configs/default.yaml
```

> 说明：新管线（Token 化 + NoteDecoder）使用 `train_tokens.py`，`train.py` 已基本被取代。

---

## 七、导出与推理生成

### `export_onnx.py` — .pth → ONNX 模型导出

将训练好的 PyTorch 检查点导出为 ONNX 格式，用于跨平台部署。支持两种模式：
- **full_pipeline**（默认）：完整管线（AudioProjection + NoteDecoder）
- **decoder_only**：仅 NoteDecoder（需要外部音频特征）

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--checkpoint` | 必填 | PyTorch 检查点路径（.pth） |
| `--output` | 必填 | 输出 ONNX 文件路径 |
| `--decoder-only` | 关 | 只导出 NoteDecoder（无 AudioProjection） |
| `--batch-size` | `1` | 导出批次大小 |
| `--seq-len` | `512` | 目标序列长度 |
| `--audio-len` | `1000` | 音频特征长度 |
| `--opset-version` | `17` | ONNX opset 版本 |
| `--fp16` | 关 | 导出 float16（体积减半，部分后端不支持） |
| `--static` | 关 | 固定形状导出（无动态轴，兼容 ONNX Runtime） |
| `--verify` | 关 | 导出后用 onnxruntime 验证一致性 |
| `--no-verify` | 关 | 跳过验证 |

**示例：**

```bash
# 导出完整模型
python scripts/export_onnx.py --checkpoint logs/best_model.pth --output model.onnx

# 仅导出解码器
python scripts/export_onnx.py --checkpoint logs/best_model.pth --output decoder.onnx --decoder-only

# 固定形状 + 验证（推荐生产使用）
python scripts/export_onnx.py --checkpoint logs/best_model.pth --output model.onnx \
    --batch-size 1 --seq-len 256 --static --verify

# float16 导出（体积减半）
python scripts/export_onnx.py --checkpoint logs/best_model.pth --output model_fp16.onnx --fp16
```

> 注意：`--seq-len` / `--audio-len` 必须与推理时 `infer_onnx.py` 的 `--audio-len` / `--max-tokens` 一致。

### `infer_onnx.py` — ONNX 模型推理生成

使用导出的 ONNX 模型从音频（或随机噪声）自回归生成谱面 Token 序列，输出 JSON（可被 `export_chart_pec.py` 转为可玩谱面）。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 必填 | ONNX 模型路径（.onnx） |
| `--audio` | 无 | 音频文件路径（mp3/wav/flac） |
| `--random` | 关 | 用随机噪声代替真实音频（快速测试） |
| `--decoder-only` | 关 | 模型为 decoder-only（音频特征视为 encoder 输出） |
| `--audio-len` | `200` | 音频特征时间步（须与导出形状一致） |
| `--audio-dim` | `128` | 音频特征维度（须与导出形状一致） |
| `--max-tokens` | `256` | 最大生成 Token 数 |
| `--temperature` | `0.8` | 采样温度（0=贪心） |
| `--top-k` | `50` | Top-K 采样（0=关闭） |
| `--top-p` | `0.95` | Nucleus 采样阈值（1.0=关闭） |
| `--num-samples` | `1` | 生成样本数 |
| `--output` | 无 | 输出 JSON 文件路径（默认 `chart_result.json`） |
| `--batch-size` | 自动 | 推理批次（默认从模型自动检测） |
| `--device` | `auto` | 设备：`auto` / `cpu` / `cuda` |
| `--verbose` | 关 | 打印每个生成的 Token |

**示例：**

```bash
# 从音频文件生成
python scripts/infer_onnx.py --model model.onnx --audio data/audio/Chart.mp3 --device cuda

# 随机噪声快速测试
python scripts/infer_onnx.py --model model.onnx --random

# 指定生成参数
python scripts/infer_onnx.py --model model.onnx --audio song.mp3 \
    --max-tokens 512 --temperature 0.8 --top-k 50

# decoder-only 模式
python scripts/infer_onnx.py --model decoder.onnx --decoder-only --audio song.mp3
```

### `export_chart_pec.py` — 生成谱面导出（chart_result.json → PEC / PEZ）

把 `infer_onnx.py` 生成的 `chart_result.json` 转换为 Phira / Phigros 可玩的谱面。

**输出格式：**
- `.pec`：PEC 文本谱面（Phigros 编辑器 / Phira 均可导入）
- `.pez`：谱面包（zip：chart.pec + 音频），可直接导入 Phira

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input` | `chart_result.json` | infer_onnx.py 的输出 JSON（含 tokens） |
| `--output` | `generated.pec` | 输出文件（.pec 或 .pez） |
| `--bpm` | `120.0` | 歌曲 BPM（决定拍→秒换算） |
| `--offset-ms` | `0.0` | 谱面偏移毫秒（对齐音频起点，可负） |
| `--line` | `1` | 音符所在判定线编号 |
| `--audio` | 无 | 音频文件路径（打包 `.pez` 时需要） |
| `--verbose` | 关 | 打印前 10 个音符详情 |

**示例：**

```bash
# 生成 PEC 谱面（BPM 需与歌曲一致）
python scripts/export_chart_pec.py --input chart_result.json --output generated.pec --bpm 120

# 生成 PEZ 谱面包（含音频，可直接导入 Phira）
python scripts/export_chart_pec.py --input chart_result.json --output generated.pez \
    --audio data/audio/Chart.mp3 --bpm 120

# 调整对齐
python scripts/export_chart_pec.py --input chart_result.json --output generated.pec \
    --bpm 140 --offset-ms -120 --line 1
```

### `generate.py` — 旧版 AI 制谱入口（Diffusion + LLM）

早期生成入口：LLM 解析提示词 → 条件 Diffusion 生成谱面 → ChartData JSON（可导入 ArellanoDreamEdit 预览）。不使用 ONNX，直接加载 PyTorch 检查点；不指定 checkpoint 时进入演示模式。

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `-p, --prompt` | `生成一段谱面` | 制谱提示词（自然语言） |
| `-a, --audio` | 无 | 音频文件路径（mp3/wav/flac/ogg），提取真实音频特征 |
| `-o, --output` | `generated_chart.json` | 输出 JSON 文件路径 |
| `-c, --checkpoint` | 无 | 模型检查点路径（可选，不指定用演示模式） |
| `-s, --seed` | 无 | 随机种子（可复现生成） |
| `-n, --samples` | `1` | 生成样本数量 |
| `--device` | 自动 | 强制设备（cuda/cpu） |
| `--no-llm` | 关 | 跳过 LLM 解析，使用纯规则匹配 |

**示例：**

```bash
python scripts/generate.py --prompt "生成一段 128 BPM 的电子音乐谱面，包含大量滑动音符"
python scripts/generate.py --prompt "重金属风格" --audio song.mp3 --checkpoint logs/best_model.pth
```

> 说明：新管线由 `infer_onnx.py` + `export_chart_pec.py` 取代，`generate.py` 保留作为旧架构入口。

---

## 八、提示词分析

### `analyze_prompt.py` — 提示词分析工具

独立使用 LLM 解析提示词并查看结构化结果（不需要训练好的模型）。三种模式互斥：

| 参数 | 说明 |
|------|------|
| `-p, --prompt` | 要分析的提示词 |
| `-i, --interactive` | 启动交互式模式 |
| `-b, --batch` | 批量文件路径（每行一个提示词） |

**示例：**

```bash
python scripts/analyze_prompt.py --prompt "生成一段激昂的重金属谱面，难度高，音符密集"
python scripts/analyze_prompt.py --interactive
python scripts/analyze_prompt.py --batch prompts.txt
```

输出：音乐属性（BPM/流派/情绪/能量/节奏）、谱面参数（难度/音符密度）、技术技巧、特殊需求等结构化解析结果。

---

## 九、典型端到端流程

### 当前推荐流程（Token 管线，对应 `scripts/123.txt` 备忘）

```bash
# 1. 重新生成 Token 数据（过滤空白段）
python scripts/unified_tokenizer.py --dir data/raw/ --outdir data/tokens/

# 2. 验证新的 time_offset 分布
python scripts/check_time_offset.py

# 3. 预处理音频缓存（仅首次）
python scripts/prepare_audio_cache.py --target-steps 4096

# 4. 重新训练模型
python scripts/train_tokens.py --batch-size 8 --epochs 100

# 5. 导出 ONNX 模型
python scripts/export_onnx.py --checkpoint logs/best_model.pth --output model.onnx \
    --batch-size 1 --seq-len 256 --static --verify

# 6. 推理测试
python scripts/infer_onnx.py --model model.onnx --audio data/audio/Chart.mp3 --device cuda

# 7. 导出可玩谱面（.pec / .pez）
python scripts/export_chart_pec.py --input chart_result.json --output generated.pez \
    --audio data/audio/Chart.mp3 --bpm 120
```

### 数据采集流程

```bash
# Phira 数据
python scripts/download_phira.py --start 1 --end 1000
python scripts/process_pez.py --delete-pez --tokenize --delete-raw

# osu! 数据
python scripts/process_osz.py
python scripts/osu_to_chart.py --dir ./osu_maps --outdir data/processed
python scripts/unified_tokenizer.py --dir data/processed --outdir data/tokens/
```

---

## 十、其他说明

1. **目录约定**：数据流经的默认目录为 `data/raw_pez`（下载）→ `data/raw`（解压）→ `data/processed`（ChartData JSON）→ `data/tokens`（训练 Token）→ `data/audio`（音频）→ `data/audio_cache`（音频特征缓存）。
2. **运行位置**：所有脚本应在项目根目录执行（脚本内部会自动添加项目根到 `sys.path`）。
3. **Windows 编码**：部分脚本（如 `train_tokens.py`、`process_pez.py`）已强制 UTF-8 输出，解决 GBK 控制台无法显示 emoji/中文的问题。
4. **依赖**：`aiohttp`（下载/API 测试）、`pyyaml`（info.yml 解析）、`librosa`（音频特征）、`onnxruntime`（ONNX 验证/推理）、`torch` + `numpy`（训练）。
5. **无参数脚本**：`check_data.py`、`check_token_stats.py`、`check_time_offset.py`、`analyze_data.py`、`check_progress.py`、`verify_dataset_pipeline.py`、`test_api.py`、`cleanup_raw_pez.py` 均为无参数脚本，直接运行即可；其中 `analyze_events.py` 与 `cleanup_raw_pez.py` 的路径为硬编码，使用前请确认。
