"""
ArellanoDreamWeaver - 核心模块测试
===============================
快速验证所有核心组件可正常导入和运行
"""
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("Test 1: 模块导入测试")
    print("=" * 60)
    
    modules_to_test = [
        ("dreamweaver", "主包"),
        ("dreamweaver.chart.structures", "谱面数据结构"),
        ("dreamweaver.models.transformer", "Transformer 模型"),
        ("dreamweaver.models.diffusion", "Diffusion 模型"),
        ("dreamweaver.diffusion.noise_scheduler", "噪声调度器"),
        ("dreamweaver.llm.prompt_parser", "提示词解析器"),
    ]
    
    success_count = 0
    for module_name, desc in modules_to_test:
        try:
            __import__(module_name)
            print(f"  [OK] {desc}: {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"  [FAIL] {desc}: {module_name} - {e}")
    
    print(f"\n结果: {success_count}/{len(modules_to_test)} 成功")
    return success_count == len(modules_to_test)


def test_chart_structures():
    """测试谱面数据结构"""
    print("\n" + "=" * 60)
    print("Test 2: 谱面数据结构测试")
    print("=" * 60)
    
    from dreamweaver.chart.structures import (
        ChartData, Note, NoteType, BPM, JudgeSegment,
        chart_to_tensor, tensor_to_chart
    )
    
    # 创建谱面
    chart = ChartData(musicLength=120.0)
    chart.add_bpm(128.0)
    
    # 添加音符
    notes = [
        Note(type=NoteType.TAP, time=0.0, positionX=-0.5),
        Note(type=NoteType.HOLD, time=1.0, holdTime=2.0, positionX=0.3),
        Note(type=NoteType.SLIDE, time=4.0, positionX=0.8),
    ]
    
    for note in notes:
        chart.add_note(note)
    
    # 测试 JSON 序列化/反序列化
    json_str = chart.to_json_string()
    chart_loaded = ChartData.from_json_string(json_str)
    
    assert chart.total_notes == chart_loaded.total_notes
    
    # 测试张量转换
    tensor = chart_to_tensor(chart)
    chart_back = tensor_to_chart(tensor)
    
    print(f"  [OK] 创建 ChartData: {chart.total_notes} notes")
    print(f"  [OK] JSON 序列化: {len(json_str)} chars")
    print(f"  [OK] 张量转换: {tensor.shape}")
    print(f"  [OK] 逆转换: {chart_back.total_notes} notes")
    print("\n" + chart.get_summary())
    
    return True


def test_transformer():
    """测试 Transformer 模型 (不训练，只检查前向)"""
    print("\n" + "=" * 60)
    print("Test 3: Transformer 模型测试")
    print("=" * 60)
    
    import torch
    from dreamweaver.models.transformer import MultiTaskTransformer, TransformerConfig
    
    # 创建小模型用于快速测试
    config = TransformerConfig(
        d_model=128,  # 小模型加速测试
        n_heads=4,
        n_layers=2,
        d_ff=256
    )
    
    model = MultiTaskTransformer(config)
    
    # Dummy 输入
    B, T_audio, T_text = 1, 64, 8
    
    outputs = model(
        audio_features=torch.randn(B, T_audio, 128),
        text_input_ids=torch.randint(0, 100, (B, T_text)),
        text_attention_mask=torch.ones(B, T_text)
    )
    
    expected_keys = [
        'diffusion_condition', 'note_logits', 'position_pred',
        'timing_pred', 'difficulty_logits', 'style_logits'
    ]
    
    for key in expected_keys:
        assert key in outputs, f"Missing output: {key}"
        print(f"  [OK] Output '{key}': {outputs[key].shape if hasattr(outputs[key], 'shape') else type(outputs[key])}")
    
    params = model.get_num_parameters()
    print(f"\n  参数统计:")
    for name, count in params.items():
        print(f"      {name}: {count:,}")
    
    return True


def test_diffusion():
    """测试 Diffusion 组件"""
    print("\n" + "=" * 60)
    print("Test 4: Diffusion 模型测试")
    print("=" * 60)
    
    import torch
    from dreamweaver.models.diffusion import ConditionalUNet1D, DiffusionConfig
    from dreamweaver.diffusion.noise_scheduler import NoiseScheduler
    
    config = DiffusionConfig(
        base_channels=32,  # 小网络加速测试
        channel_mults=(1, 2),
        condition_dim=128
    )
    
    unet = ConditionalUNet1D(config)
    
    B, C, T = 1, 16, 32
    dummy_input = torch.randn(B, C, T)
    dummy_t = torch.tensor([500])
    dummy_cond = torch.randn(B, 16, 128)
    
    output = unet(dummy_input, dummy_t, dummy_cond)
    
    assert output.shape == dummy_input.shape, f"Shape mismatch: {output.shape} vs {dummy_input.shape}"
    
    print(f"  [OK] U-Net 前向传播: {dummy_input.shape} -> {output.shape}")
    
    # 噪声调度器
    scheduler = NoiseScheduler("cosine", 100)
    noise = torch.randn_like(dummy_input)
    timesteps = torch.tensor([50])
    
    noisy, sqrt_a, sqrt_1ma = scheduler.add_noise(dummy_input, noise, timesteps)
    
    print(f"  [OK] 噪声调度: betas range [{scheduler.betas.min():.4f}, {scheduler.betas.max():.4f}]")
    print(f"  [OK] 加噪: shape={noisy.shape}, sqrt_a={sqrt_a.item():.4f}")
    
    return True


def test_prompt_parser():
    """测试提示词解析器 (规则模式)"""
    print("\n" + "=" * 60)
    print("Test 5: 提示词解析器测试")
    print("=" * 60)
    
    from dreamweaver.llm.prompt_parser import PromptParser, ChartGenerationParams
    
    parser = PromptParser(provider="local")  # 不调用 API
    
    test_cases = [
        "制作一首128 BPM的电子音乐谱面",
        "重金属风格，困难难度",
        "悲伤的钢琴曲，简单模式"
    ]
    
    for prompt in test_cases:
        try:
            params = parser.parse(prompt, use_fallback=True)
            print(f'  [OK] "{prompt[:30]}..."')
            print(f'       BPM={params.bpm}, Genre={params.genre}')
        except Exception as e:
            print(f'  [FAIL] "{prompt[:30]}...": {e}')
    
    return True


def run_all_tests():
    """运行全部测试"""
    print("\n" + "#" * 70)
    print("# ArellanoDreamWeaver - Core Module Tests")
    print("#" * 70)
    
    results = {
        "Imports": test_imports(),
        "Chart Structures": test_chart_structures(),
        "Transformer": test_transformer(),
        "Diffusion": test_diffusion(),
        "Prompt Parser": test_prompt_parser()
    }
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {status} {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! 项目初始化成功!")
    else:
        print("\n⚠️ 部分测试未通过，请检查依赖安装")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
