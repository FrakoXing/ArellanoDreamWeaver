"""
ArellanoDreamWeaver 🎵✨
======================
基于多任务 Transformer + Diffusion 的 AI 辅助制谱模型

主要模块:
- dreamweaver.models: 核心模型 (Transformer + Diffusion)
- dreamweaver.diffusion: 噪声调度 & 采样
- dreamweaver.llm: LLM 提示词理解
- dreamweaver.data: 数据处理
- dreamweaver.chart: 谱面数据结构
- dreamweaver.utils: 工具函数

快速开始:
    from dreamweaver import DreamWeaverGenerator
    
    generator = DreamWeaverGenerator(checkpoint="path/to/model.pth")
    chart = generator.generate("生成一段 128 BPM 的电子音乐谱面")
    
版本: 0.1.0-alpha
兼容: ArellanoDreamEdit 谱面格式
"""

__version__ = "0.1.0"
__author__ = "Arellano Team"

# 导出核心类
from dreamweaver.chart.structures import (
    ChartData, Note, NoteType, BPM,
    JudgeSegment, Event, SegmentEvents,
    chart_to_tensor, tensor_to_chart
)

from dreamweaver.models.transformer import (
    MultiTaskTransformer, 
    TransformerConfig
)

from dreamweaver.models.diffusion import (
    ConditionalUNet1D,
    DiffusionConfig,
    ChartDiffusionModel
)

from dreamweaver.diffusion.noise_scheduler import (
    NoiseScheduler,
    DDPMSampler,
    DDIMSampler
)

from dreamweaver.llm.prompt_parser import (
    PromptParser,
    ChartGenerationParams,
    MusicGenre,
    DifficultyLevel
)


def create_generator(
    checkpoint_path: str = None,
    device=None
) -> "DreamWeaverGenerator":
    """
    工厂函数: 快速创建生成器实例
    
    Args:
        checkpoint_path: 模型检查点路径 (可选)
        device: 计算设备
        
    Returns:
        DreamWeaverGenerator 实例
    
    示例:
        >>> gen = create_generator()
        >>> chart = gen.generate("制作一首流行歌曲谱面")
    """
    from scripts.generate import DreamWeaverGenerator
    return DreamWeaverGenerator(checkpoint_path=checkpoint_path, device=device)
