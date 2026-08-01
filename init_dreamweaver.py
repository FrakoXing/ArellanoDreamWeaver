"""
ArellanoDreamWeaver - 初始化脚本
Init Myself - 项目初始化与配置
"""
import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime

# 修复 Windows 终端编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ANSI 颜色代码
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def log(header: str, message: str, color=Colors.OKCYAN):
    """带颜色的日志输出"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.DIM}[{timestamp}]{Colors.ENDC} {color}{header}{Colors.ENDC}: {message}")

def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    log("Python", f"当前版本: {version.major}.{version.minor}.{version.micro}", Colors.OKBLUE)
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        log("ERROR", "需要 Python 3.9 或更高版本!", Colors.FAIL)
        sys.exit(1)
    
    if version.major == 3 and version.minor >= 10:
        log("OK", "Python 版本符合要求 ✓", Colors.OKGREEN)
        return True
    return True

def create_directory_structure():
    """创建项目目录结构"""
    log("DIR", "创建项目目录结构...", Colors.HEADER)
    
    directories = [
        "dreamweaver/models",
        "dreamweaver/diffusion",
        "dreamweaver/llm",
        "dreamweaver/data",
        "dreamweaver/chart",
        "dreamweaver/utils",
        "configs",
        "scripts",
        "tests",
        "checkpoints",
        "data/raw",
        "data/processed",
        "logs",
        "exports"
    ]
    
    base_path = Path(__file__).parent
    
    for dir_name in directories:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建 __init__.py (对于包目录)
        if dir_name.startswith("dreamweaver/"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""ArellanoDreamWeaver - {dir_name.split("/")[-1]} module"""\n')
                log("CREATE", f"  └─ {dir_name}/__init__.py", Colors.DIM)
        
        log("CREATE", f"  ├─ {dir_name}/", Colors.OKGREEN)

def create_config_files():
    """创建默认配置文件"""
    log("CONFIG", "生成默认配置文件...", Colors.HEADER)
    
    base_path = Path(__file__).parent
    
    # 默认训练配置
    default_config = """
# ArellanoDreamWeaver - 默认训练配置

# === 模型参数 ===
model:
  name: "DreamWeaver-v1"
  
  # Transformer 参数
  transformer:
    d_model: 512          # 模型维度
    n_heads: 8            # 注意力头数
    n_layers: 6           # 编码器层数
    d_ff: 2048            # 前馈网络维度
    dropout: 0.1          # Dropout 率
    max_seq_len: 4096     # 最大序列长度
    
  # Diffusion 参数
  diffusion:
    n_timesteps: 1000     # 扩散步数
    beta_start: 0.0001    # Beta 起始值
    beta_end: 0.02        # Beta 结束值
    schedule: "cosine"    # 调度类型: linear, cosine, sqrt
  
  # 多任务头
  heads:
    note_prediction:      # 音符预测
      hidden_dim: 256
      num_classes: 4     # tap, hold, slide, none
    bpm_prediction:       # BPM 回归
      hidden_dim: 128
    difficulty_assessment: # 难度分类
      hidden_dim: 128
      num_classes: 6     # EZ/NM/HD/IN/AT/LEGACY
    style_classification:  # 风格分类
      hidden_dim: 128
      num_classes: 10    # pop, rock, electronic, classical, etc.

# === 训练参数 ===
training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  epochs: 100
  warmup_epochs: 5
  gradient_clip: 1.0
  
  # 损失权重
  loss_weights:
    note_prediction: 1.0
    bpm_prediction: 0.5
    difficulty: 0.3
    style: 0.2
    diffusion: 2.0
  
  optimizer: "adamw"
  scheduler: "cosine"
  
# === LLM 参数 ===
llm:
  provider: "openai"      # openai, local
  model: "gpt-4-turbo"
  temperature: 0.7
  max_tokens: 1024
  
# === 数据参数 ===
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  
  # 音频特征
  audio:
    sample_rate: 44100
    n_mels: 128
    hop_length: 512
    n_fft: 2048
  
  # 谱面特征
  chart:
    max_notes_per_segment: 64
    time_quantization: 16   # 时间量化精度 (1/16拍)

# === 日志 ===
logging:
  log_dir: "logs"
  save_every: 10           # 每 N 个 epoch 保存
  tensorboard: true
  verbose: true
"""
    config_path = base_path / "configs" / "default.yaml"
    config_path.write_text(default_config.strip())
    log("CREATE", "  ├─ configs/default.yaml", Colors.OKGREEN)
    
    # 模型架构配置
    model_config = """
# ArellanoDreamWeaver - 模型架构详细配置

architecture:
  # === 输入编码器 ===
  encoders:
    music_feature_encoder:
      type: "CNN1D"
      layers:
        - in_channels: 128  # mel spectrogram bands
          out_channels: 256
          kernel_size: 3
          stride: 2
          padding: 1
        - in_channels: 256
          out_channels: 512
          kernel_size: 3
          stride: 2
          padding: 1
      output_dim: 512
    
    text_encoder:
      type: "Transformer"
      vocab_size: 50257    # GPT-2 token vocab
      d_model: 512
      n_heads: 8
      n_layers: 4
      max_length: 256
    
    bpm_encoder:
      type: "MLP"
      input_dim: 4         # [bpm_value, beat_position, time_signature, confidence]
      hidden_dims: [128, 256, 512]
      output_dim: 512
    
    positional_encoder:
      type: "Learned"
      max_len: 4096
      d_model: 512

  # === Transformer 主干 ===
  backbone:
    type: "MultiTaskTransformer"
    d_model: 512
    n_heads: 8
    n_encoder_layers: 6
    n_decoder_layers: 6
    d_ff: 2048
    dropout: 0.1
    activation: "gelu"
    
    # 交叉注意力 (融合不同模态)
    cross_attention:
      enabled: true
      n_heads: 8
      
    # 自适应层归一化 (用于条件注入)
    ada_ln:
      enabled: true
      conditioning_dim: 512

  # === Diffusion U-Net ===
  unet:
    type: "ConditionalUNet"
    in_channels: 64       # 谱面特征维度
    out_channels: 64
    channel_mults: [1, 2, 4, 8]
    n_res_blocks: 2
    attention_resolutions: [16, 8]
    
    # 时间步嵌入
    time_embed_dim: 256
    
    # 条件嵌入 (来自 Transformer)
    condition_embed_dim: 512
    
    # 条件注入方式: crossattn, adagn, fiLM
    condition_type: "crossattn"

  # === 输出头 ===
  output_heads:
    note_head:
      type: "ClassificationHead"
      in_features: 512
      hidden_features: 256
      n_classes: 4
      dropout: 0.2
      
    position_head:
      type: "RegressionHead"
      in_features: 512
      hidden_features: 256
      output_dim: 2       # [x_position, y_position]
      
    timing_head:
      type: "RegressionHead"
      in_features: 512
      hidden_features: 256
      output_dim: 2       # [time, hold_time]

# === 总参数量估算 ===
estimated_parameters: ~85M
"""
    model_config_path = base_path / "configs" / "model.yaml"
    model_config_path.write_text(model_config.strip())
    log("CREATE", "  ├─ configs/model.yaml", Colors.OKGREEN)
    
    # Diffusion 配置
    diffusion_config = """
# ArellanoDreamWeaver - Diffusion 过程配置

noise_schedule:
  # Beta 调度
  type: "cosine"          # linear, cosine, sigmoid, sqrt_linear_cosine
  
  linear:
    beta_start: 0.0001
    beta_end: 0.02
    n_steps: 1000
    
  cosine:
    s: 0.008             # 小偏移，避免 beta=0
    
  # 可学习的噪声调度 (可选)
  learnable: false

# 采样配置
sampling:
  method: "ddim"          # ddpm, ddim, euler, heun, dpm++
  
  ddim:
    n_steps: 50           # DDIM 采样步数 (可以远少于训练步数)
    eta: 0.0              # 随机性 (0=确定性, 1=完全随机)
    discretization: "uniform"  # uniform, quad
    
  # 分类器-free 引导
  classifier_free_guidance:
    enabled: true
    guidance_scale: 2.0   # 引导强度
    p_uncond: 0.1         # 训练时丢弃条件的概率

# 数据表示
data_representation:
  # 谱面的张量表示
  format: "piano_roll_like"
  
  dimensions:
    time: 4096            # 时间步
    lanes: 8              # 轨道数 (上下各4轨)
    features: 8           # 特征维度 [tap, hold, slide, position_x, is_fake, is_above, has_other, intensity]
  
  # 归一化
  normalization:
    method: "standard"     # standard, minmax, none
    
# 训练技巧
training_tricks:
  # EMA (指数移动平均) 用于稳定生成
  ema:
    enabled: true
    decay: 0.9999
    update_after_step: 1000
    
  # 梯度缩放 (混合精度训练)
  amp:
    enabled: true
    init_scale: 65536.0
    
  # x0 预测 vs噪声预测
  prediction_type: "epsilon"  # epsilon, x0, v_prediction
"""
    diffusion_config_path = base_path / "configs" / "diffusion.yaml"
    diffusion_config_path.write_text(diffusion_config.strip())
    log("CREATE", "  └─ configs/diffusion.yaml", Colors.OKGREEN)

def create_env_template():
    """创建环境变量模板"""
    env_template = """# ArellanoDreamWeaver - 环境变量配置
# 复制此文件为 .env 并填入实际值

# === OpenAI API (用于 LLM 提示词理解) ===
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4-turbo

# === 本地 LLM (可选,替代 OpenAI) ===
LOCAL_LLM_ENDPOINT=http://localhost:8080/v1
LOCAL_LLM_MODEL_PATH=/path/to/local/model

# === 其他服务 ===
TENSORBOARD_PORT=6006
LOG_LEVEL=INFO

# === GPU 配置 ===
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
"""
    env_path = Path(__file__).parent / ".env.template"
    env_path.write_text(env_template.strip())
    log("ENV", "  ├─ .env.template (请复制为 .env 并填写)", Colors.WARNING)

def create_gitignore():
    """创建 .gitignore 文件"""
    gitignore_content = """# === Python ===
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/

# === 虚拟环境 ===
venv/
env/
.venv/

# === IDE ===
.idea/
.vscode/
*.swp
*.swo

# === 数据 (通常很大，不纳入版本控制) ===
data/raw/*.mp3
data/raw/*.wav
data/raw/*.ogg
data/processed/*.pt
data/processed/*.npy
data/datasets/*

# === 模型权重 ===
checkpoints/*.pth
checkpoints/*.bin
checkpoints/*.safetensors
!checkpoints/.gitkeep

# === 日志 ===
logs/*
!logs/.gitkeep

# === 导出 ===
exports/*
!exports/.gitkeep

# === 环境 & 密钥 (安全!) ===
.env
.env.local
.env.production

# === Jupyter ===
.ipynb_checkpoints/

# === OS ===
.DS_Store
Thumbs.db

# === HuggingFace Cache ===
~/.cache/huggingface/
"""
    gitignore_path = Path(__file__).parent / ".gitignore"
    gitignore_path.write_text(gitignore_content.strip())
    log("GIT", "  └─ .gitignore", Colors.OKGREEN)

def install_dependencies():
    """安装 Python 依赖"""
    log("DEPS", "安装 Python 依赖包...", Colors.HEADER)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt","-i","https://pypi.tuna.tsinghua.edu.cn/simple"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            log("OK", "依赖安装成功 ✓", Colors.OKGREEN)
            
            # 显示已安装的关键包
            import importlib
            key_packages = ["torch", "numpy", "librosa", "transformers"]
            for pkg in key_packages:
                try:
                    mod = importlib.import_module(pkg)
                    ver = getattr(mod, "__version__", "unknown")
                    log("PKG", f"  {pkg}: {ver}", Colors.DIM)
                except ImportError:
                    log("WARN", f"  {pkg}: 未安装 (可选)", Colors.WARNING)
        else:
            log("WARN", "依赖安装可能有问题，请检查上面的输出", Colors.WARNING)
            if result.stderr:
                print(result.stderr[-500:])  # 只显示最后500字符
                
    except subprocess.TimeoutExpired:
        log("FAIL", "依赖安装超时 (>5分钟)，请手动运行: pip install -r requirements.txt", Colors.FAIL)
    except Exception as e:
        log("ERROR", f"安装失败: {e}", Colors.FAIL)

def verify_gpu():
    """检测 GPU/CUDA 可用性"""
    log("GPU", "检测 GPU 加速支持...", Colors.HEADER)
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
            log("OK", f"CUDA 可用 - {device_name} ({vram_gb:.1f} GB VRAM) ✓", Colors.OKGREEN)
            return True
        else:
            log("WARN", "CUDA 不可用，将使用 CPU (训练会很慢)", Colors.WARNING)
            return False
    except ImportError:
        log("WARN", "PyTorch 未安装，跳过 GPU 检测", Colors.WARNING)
        return None

def create_init_script_info():
    """创建初始化信息文件"""
    info = {
        "project": "ArellanoDreamWeaver",
        "version": "0.1.0-alpha",
        "initialized_at": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "status": "initialized"
    }
    
    info_path = Path(__file__).parent / ".dreamweaver_init.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    log("INFO", "初始化信息已保存到 .dreamweaver_init.json", Colors.DIM)

def display_banner():
    """显示启动 Banner"""
    banner = f"""
{Colors.BOLD}{Colors.HEADER}
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🎵✨ ArellanoDreamWeaver ✨🎵                          ║
║                                                           ║
║     基于 Transformer + Diffusion 的 AI 制谱模型            ║
║                                                           ║
║     Multi-Task Transformer + Conditional Diffusion        ║
║     for AI-Assisted Music Chart Generation                 ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║  {Colors.OKCYAN}Init Myself - 项目初始化{Colors.ENDC}                                ║
╚═══════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def main():
    """主初始化流程"""
    display_banner()
    
    steps = [
        ("检查 Python 版本", check_python_version),
        ("创建目录结构", create_directory_structure),
        ("生成配置文件", create_config_files),
        ("创建环境变量模板", create_env_template),
        ("创建 Git 忽略规则", create_gitignore),
        ("安装 Python 依赖", install_dependencies),
        ("检测 GPU 支持", verify_gpu),
        ("保存初始化信息", create_init_script_info),
    ]
    
    completed = 0
    failed = []
    
    for name, func in steps:
        print()
        try:
            result = func()
            if result is not False:
                completed += 1
            else:
                failed.append(name)
        except Exception as e:
            log("ERROR", f"'{name}' 失败: {e}", Colors.FAIL)
            failed.append(name)
    
    # 最终总结
    print()
    print(f"{Colors.BOLD}{'═'*59}{Colors.ENDC}")
    log("DONE", f"初始化完成! ({completed}/{len(steps)} 步成功)", Colors.OKGREEN if not failed else Colors.WARNING)
    
    if failed:
        log("WARN", f"失败的步骤: {', '.join(failed)}", Colors.WARNING)
    
    print()
    print(f"{Colors.OKCYAN}下一步操作:{Colors.ENDC}")
    print(f"  1. 复制 .env.template 为 .env 并配置 API Key:")
    print(f"     {Colors.DIM}cp .env.template .env{Colors.ENDC}")
    print()
    print(f"  2. 开始训练模型:")
    print(f"     {Colors.DIM}python scripts/train.py --config configs/default.yaml{Colors.ENDC}")
    print()
    print(f"  3. 使用 AI 制谱:")
    print(f"     {Colors.DIM}python scripts/generate.py --prompt \"你的提示词\"{Colors.ENDC}")
    print()
    print(f"{Colors.OKGREEN}🎉 ArellanoDreamWeaver 已准备就绪！{Colors.ENDC}")

if __name__ == "__main__":
    main()
