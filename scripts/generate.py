"""
ArellanoDreamWeaver - AI 制谱生成入口
=====================================
使用训练好的模型 + LLM 理解提示词来生成谱面
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.models.transformer import MultiTaskTransformer, TransformerConfig
from dreamweaver.models.diffusion import ConditionalUNet1D, DiffusionConfig, ChartDiffusionModel
from dreamweaver.diffusion.noise_scheduler import NoiseScheduler, DDIMSampler, DDPMSampler
from dreamweaver.llm.prompt_parser import PromptParser, ChartGenerationParams
from dreamweaver.chart.structures import (
    ChartData, Note, NoteType, BPM,
    chart_to_tensor, tensor_to_chart
)


class DreamWeaverGenerator:
    """
    ArellanoDreamWeaver 生成器
    
    流程:
    1. 接收用户提示词 (自然语言)
    2. LLM 解析为结构化参数
    3. 将参数编码为条件向量
    4. 使用 Diffusion 模型生成谱面
    5. 后处理 & 转换为标准格式输出
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4-turbo"
    ):
        # 设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LLM 提示词解析器
        self.prompt_parser = PromptParser(
            provider=llm_provider,
            model_name=llm_model
        )
        
        # 模型组件 (延迟初始化)
        self.transformer: Optional[MultiTaskTransformer] = None
        self.diffusion_model: Optional[ChartDiffusionModel] = None
        self.noise_scheduler: Optional[NoiseScheduler] = None
        
        # 配置
        self.config = {
            "transformer": {
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6,
                "max_seq_len": 4096
            },
            "diffusion": {
                "in_channels": 64,
                "out_channels": 64,
                "n_timesteps": 1000,
                "schedule": "cosine"
            },
            "sampling": {
                "method": "ddim",
                "n_steps": 50,
                "eta": 0.0,
                "guidance_scale": 2.0
            }
        }
        
        print(f"\n🎵 ArellanoDreamWeaver Generator 初始化")
        print(f"   设备: {self.device}")
        
        if checkpoint_path:
            self.load_models(checkpoint_path)
    
    def _create_models(self):
        """创建模型实例"""
        tfm_cfg = self.config["transformer"]
        diff_cfg = self.config["diffusion"]
        
        # 多任务 Transformer
        transformer_config = TransformerConfig(
            d_model=tfm_cfg["d_model"],
            n_heads=tfm_cfg["n_heads"],
            n_layers=tfm_cfg["n_layers"],
            max_seq_len=tfm_cfg["max_seq_len"]
        )
        self.transformer = MultiTaskTransformer(transformer_config).to(self.device)
        
        # Diffusion U-Net
        unet_config = DiffusionConfig(
            in_channels=diff_cfg["in_channels"],
            out_channels=diff_cfg["out_channels"],
            condition_dim=512
        )
        unet = ConditionalUNet1D(unet_config).to(self.device)
        
        # 完整 Diffusion 模型 (包含噪声调度)
        self.diffusion_model = ChartDiffusionModel(
            unet_config=unet_config,
            n_timesteps=diff_cfg["n_timesteps"],
            beta_schedule=diff_cfg["schedule"]
        ).to(self.device)
        
        # 手动设置 U-Net (因为 ChartDiffusionModel 会创建自己的)
        self.diffusion_model.unet = unet
        
        # 噪声调度器
        self.noise_scheduler = NoiseScheduler(
            schedule=diff_cfg["schedule"],
            n_timesteps=diff_cfg["n_timesteps"]
        ).to(self.device)
        
        print(f"   模型创建完成")
    
    def load_models(self, checkpoint_path: str):
        """从检查点加载训练好的权重"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        self._create_models()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "transformer_state_dict" in checkpoint:
            self.transformer.load_state_dict(checkpoint["transformer_state_dict"])
            print(f"   ✅ 已加载 Transformer 权重")
        
        if "diffusion_state_dict" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_state_dict"])
            print(f"   ✅ 已加载 Diffusion 权重")
        
        # 设置为评估模式
        self.transformer.eval()
        self.diffusion_model.eval()
    
    @torch.no_grad()
    def _encode_condition(
        self,
        params: ChartGenerationParams,
        seq_length: int = 256
    ) -> torch.Tensor:
        """
        将结构化参数编码为条件张量
        
        Args:
            params: 从 LLM 解析的参数
            seq_length: 条件序列长度
            
        Returns:
            condition tensor [B, T_cond, D]
        """
        B = 1
        
        # === 方法1: 直接参数嵌入 ===
        param_features = []
        
        # BPM 归一化
        bpm_norm = (params.bpm or 120) / 300.0
        param_features.append(bpm_norm)
        
        # 能量等级
        energy_norm = (params.energy_level or 5) / 10.0
        param_features.append(energy_norm)
        
        # 音符密度
        density = params.note_density or 0.5
        param_features.append(density)
        
        # 复杂度
        complexity = params.complexity_score or 0.5
        param_features.append(complexity)
        
        # 创造性和随机性
        param_features.append(params.creativity)
        param_features.append(params.randomness)
        
        # 难度 one-hot 编码
        difficulty_map = {
            None: 3, "easy": 1, "normal": 2, "hard": 3,
            "insane": 4, "another": 5, "legacy": 6
        }
        diff_val = difficulty_map.get(str(params.difficulty), 3) / 6.0
        param_features.append(diff_val)
        
        # 构建基础特征向量
        base_feature = torch.tensor([param_features], dtype=torch.float32, device=self.device)
        
        # 扩展到序列长度 (广播)
        condition = base_feature.unsqueeze(1).expand(-1, seq_length, -1)
        
        # 通过线性层映射到完整维度 (模拟 Transformer 的条件投影)
        condition_proj = torch.nn.Linear(len(param_features), 512).to(self.device)
        condition = condition_proj(condition)
        
        return condition
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        audio_features: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: Optional[int] = None,
        progress_callback=None
    ) -> ChartData:
        """
        主生成方法
        
        Args:
            prompt: 用户自然语言提示词
            audio_features: 可选的音频特征 (如果提供，将增强生成质量)
            num_samples: 生成样本数 (目前只返回第一个)
            seed: 随机种子 (用于可复现性)
            progress_callback: 进度回调函数
            
        Returns:
            生成的 ChartData 对象
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"🎨 开始 AI 制谱生成")
        print(f"{'='*60}")
        
        # === Step 1: LLM 解析提示词 ===
        print(f"\n📝 Step 1: 理解提示词...")
        print(f"   提示词: \"{prompt}\"")
        
        params = self.prompt_parser.parse(prompt, use_fallback=True)
        
        print(f"\n   📊 解析结果:")
        print(f"      BPM: {params.bpm}")
        print(f"      流派: {params.genre}")
        print(f"      难度: {params.difficulty}")
        print(f"      音符密度: {params.note_density}")
        print(f"      特殊模式: {params.special_patterns}")
        print(f"      描述: {params.description}")
        
        # === Step 2: 编码条件 ===
        print(f"\n🔧 Step 2: 编码条件...")
        
        condition = self._encode_condition(params, seq_length=256)
        print(f"   条件张量形状: {condition.shape}")
        
        # 如果没有预训练模型，使用演示模式
        if self.transformer is None:
            print("\n⚠️  未加载训练好的模型，使用演示生成模式...")
            return self._demo_generate(prompt, params)
        
        # === Step 3: 准备音频特征 (可选) ===
        if audio_features is None:
            # 生成基于条件的伪音频特征
            audio_seq_len = 1024
            audio_features = self._generate_mock_audio_features(
                audio_seq_len, 
                bpm=params.bpm or 120,
                genre=str(params.genre.value if params.genre else "electronic")
            )
            print(f"   生成模拟音频特征: {audio_features.shape}")
        
        # === Step 4: Transformer 提取高级特征 ===
        print(f"\n🧠 Step 3: 特征提取 (Transformer)...")
        
        dummy_text_ids = torch.randint(0, 50257, (1, 32), device=self.device)
        dummy_text_mask = torch.ones(1, 32, device=self.device)
        
        # Mock BPM features
        bpm_feat = torch.zeros(1, audio_features.size(1), 4, device=self.device)
        bpm_feat[:, :, 0] = (params.bpm or 120) / 300.0
        bpm_feat[:, :, 2] = 1.0  # time signature numerator
        
        transformer_outputs = self.transformer(
            audio_features=audio_features,
            text_input_ids=dummy_text_ids,
            text_attention_mask=dummy_text_mask,
            bpm_features=bpm_feat
        )
        
        diffusion_condition = transformer_outputs["diffusion_condition"]
        print(f"   Diffusion 条件: {diffusion_condition.shape}")
        
        # === Step 5: Diffusion 采样生成 ===
        print(f"\n🎲 Step 4: 扩散生成 (Diffusion)...")
        
        sampling_cfg = self.config["sampling"]
        
        # 创建采样器
        if sampling_cfg["method"] == "ddim":
            sampler = DDIMSampler(
                scheduler=self.noise_scheduler,
                n_steps=sampling_cfg["n_steps"],
                eta=sampling_cfg["eta"],
                guidance_scale=sampling_cfg["guidance_scale"],
                unconditional_condition=torch.zeros_like(condition)  # 用于 classifier-free
            )
        else:
            sampler = DDPMSampler(
                scheduler=self.noise_scheduler,
                guidance_scale=sampling_cfg["guidance_scale"],
                unconditional_condition=torch.zeros_like(condition)
            )
        
        # 定义生成形状: [B, channels, time_steps]
        gen_shape = (num_samples, 64, 4096)  # 通道=特征维度, 时间=谱面长度
        
        # 执行采样
        generated_raw = sampler.sample(
            model=lambda x, t, c: self.diffusion_model.predict_noise(x, t, c),
            shape=gen_shape,
            condition=diffusion_condition,
            progress_callback=progress_callback
        )
        
        print(f"   生成的原始张量: {generated_raw.shape}")
        
        # === Step 6: 后处理 & 转换为 ChartData ===
        print(f"\n✨ Step 5: 后处理 & 格式转换...")
        
        # 转置回 [B, T, C]
        generated_2d = generated_raw[0].transpose(0, 1).cpu().numpy()
        
        # 转换为 ChartData
        chart = tensor_to_chart(generated_2d, base_bpm=params.bpm or 120)
        
        # 应用解析到的元数据
        chart.offset = 0.0
        chart.musicLength = 180.0  # 默认3分钟
        
        print(f"\n✅ 生成完成!")
        print(f"{chart.get_summary()}")
        
        return chart
    
    def _generate_mock_audio_features(
        self,
        seq_length: int,
        bpm: float = 120.0,
        genre: str = "electronic"
    ) -> torch.Tensor:
        """
        生成模拟的音频特征 (当没有真实音频输入时)
        
        实际应用中应使用 librosa 提取真实的 mel频谱等特征
        """
        # 基础随机特征
        features = torch.randn(seq_length, 128, device=self.device) * 0.5
        
        # 根据 BPM 添加周期性模式
        beat_period = 60.0 / bpm  # 秒/拍
        t = torch.arange(seq_length, device=self.device).float() / 100.0  # 假设时间尺度
        
        # 低频节拍脉冲 (模拟鼓点/贝斯)
        beat_freq = 1.0 / beat_period * 10  # 缩放到特征空间
        beat_pattern = torch.sin(t * beat_freq * 2 * np.pi).unsqueeze(-1)
        features[:, :16] += beat_pattern.expand(-1, 16) * 0.8
        
        # 根据流派调整频谱分布
        if genre in ["metal", "rock"]:
            # 强调中高频
            features[:, 48:80] += 0.3
        elif genre in ["electronic", "edm"]:
            # 强调低频和高频
            features[:, :32] += 0.4
            features[:, 96:] += 0.3
        elif genre in ["classical", "piano"]:
            # 更平滑的频谱
            features *= 0.7
        
        return features.unsqueeze(0)  # [1, T, C]
    
    def _demo_generate(self, prompt: str, params: ChartGenerationParams) -> ChartData:
        """
        演示模式的生成 (不需要训练好的模型)
        
        基于规则和随机性生成一个示例谱面
        """
        print(f"\n🎪 演示模式: 基于规则的谱面生成")
        
        bpm = params.bpm or 120
        duration_bars = 32  # 默认 32 小节
        
        chart = ChartData(
            offset=0.0,
            musicLength=(duration_bars * 4 * 60 / bpm),
            beatSubdivision=4
        )
        
        # 添加 BPM
        chart.add_bpm(bpm)
        
        # 根据参数生成音符
        note_density = params.note_density or 0.5
        complexity = params.complexity_score or 0.5
        total_beats = duration_bars * 4  # 总拍数
        
        # 计算大致音符数量
        base_notes = int(total_beats * note_density * 2)
        
        # 根据难度调整
        diff_multiplier = {"easy": 0.5, "normal": 1.0, "hard": 1.5, 
                          "insane": 2.0, "another": 2.5, "legacy": 3.0}
        mult = diff_multiplier.get(str(params.difficulty), 1.0)
        total_notes = int(base_notes * mult)
        
        # 生成音符
        notes_generated = 0
        current_time = 0.0
        
        while notes_generated < total_notes and current_time < total_beats:
            # 决定是否在此位置放置音符
            if np.random.random() < note_density:
                # 选择音符类型 (根据解析的比例)
                type_dist = params.note_type_distribution
                r = np.random.random()
                
                tap_threshold = type_dist.get("tap", 0.5)
                hold_threshold = tap_threshold + type_dist.get("hold", 0.25)
                
                if r < tap_threshold:
                    note_type = NoteType.TAP
                    hold_time = 0.0
                elif r < hold_threshold:
                    note_type = NoteType.HOLD
                    hold_time = min(np.random.exponential(1.0), 4.0)  # 平均1拍长按
                else:
                    note_type = NoteType.SLIDE
                    hold_time = min(np.random.exponential(0.75), 2.5)
                
                # X 位置 (根据 lane_usage 参数调整)
                x_pos = np.random.uniform(-0.9, 0.9)
                
                # 上方或下方轨道
                is_above = np.random.random() > 0.45
                
                # 是否假音 (低概率)
                is_fake = np.random.random() < 0.05
                
                note = Note(
                    type=int(note_type),
                    time=current_time,
                    holdTime=float(hold_time),
                    positionX=float(x_pos),
                    isFakeNote=is_fake,
                    isAbove=is_above
                )
                
                chart.add_note(note)
                notes_generated += 1
            
            # 时间推进 (根据密度决定步长)
            step_size = max(0.25 / (note_density + 0.1), 0.125)  # 最小 1/8拍
            current_time += step_size
        
        # 应用特殊模式
        for pattern_name in params.special_patterns:
            self._apply_special_pattern(chart, pattern_name, complexity)
        
        print(f"\n   生成了 {chart.total_notes} 个音符 ({duration_bars} 小节)")
        
        return chart
    
    def _apply_special_pattern(self, chart: ChartData, pattern: str, intensity: float):
        """在现有谱面上添加特殊模式"""
        # 这里可以扩展更多复杂的模式生成逻辑
        pass
    
    def save_chart(self, chart: ChartData, output_path: str):
        """保存生成的谱面"""
        chart.save_to_file(output_path)
        print(f"\n💾 谱面已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ArellanoDreamWeaver - AI 辅助制谱生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python generate.py --prompt "生成一段 128 BPM 的电子音乐谱面"
  
  # 指定输出文件
  python generate.py --prompt "制作一首悲伤的钢琴曲" --output my_chart.json
  
  # 使用预训练模型
  python generate.py --prompt "重金属风格，困难难度" --checkpoint checkpoints/best_model.pth
  
  # 可复现的生成
  python generate.py --prompt "流行歌曲" --seed 42
        """
    )
    
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="制谱提示词 (自然语言)")
    parser.add_argument("--output", "-o", type=str, default="generated_chart.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="模型检查点路径 (可选，不指定则使用演示模式)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="随机种子 (用于可复现生成)")
    parser.add_argument("--samples", "-n", type=int, default=1,
                        help="生成样本数量")
    parser.add_argument("--device", type=str, default=None,
                        help="强制设备 (cuda/cpu)")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过 LLM 解析，使用纯规则匹配")
    
    args = parser.parse_args()
    
    # 设置设备
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # 创建生成器
    generator = DreamWeaverGenerator(
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # 生成谱面
    chart = generator.generate(
        prompt=args.prompt,
        num_samples=args.samples,
        seed=args.seed
    )
    
    # 保存结果
    output_path = PROJECT_ROOT / args.output
    generator.save_chart(chart, str(output_path))
    
    # 打印最终统计
    print(f"\n{'='*60}")
    print(f"🎉 制谱完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
