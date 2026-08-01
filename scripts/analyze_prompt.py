"""
ArellanoDreamWeaver - 提示词分析工具
===================================
独立使用 LLM 解析提示词，查看解析结果 (不需要训练好的模型)
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.llm.prompt_parser import PromptParser, ChartGenerationParams


def analyze_prompt(prompt: str, parser: PromptParser = None) -> ChartGenerationParams:
    """
    分析提示词并返回结构化参数
    
    Args:
        prompt: 用户自然语言提示词
        parser: 可选的解析器实例
        
    Returns:
        ChartGenerationParams 结构化参数
    """
    if parser is None:
        parser = create_default_parser()
    
    return parser.parse(prompt, use_fallback=True)


def print_analysis(params: ChartGenerationParams):
    """格式化打印分析结果"""
    print("\n" + "=" * 60)
    print("📊 提示词分析结果")
    print("=" * 60)
    
    # 基本信息
    print(f"\n🎵 音乐属性:")
    print(f"   BPM:           {params.bpm or '(未指定)'}")
    print(f"   流派:          {params.genre.value if params.genre else '(未指定)'}")
    print(f"   情绪:          {params.mood or '(未指定)'}")
    print(f"   能量等级:      {params.energy_level or '(未指定)'}/10")
    print(f"   节奏感觉:      {params.tempo_feeling or '(未指定)'}")
    
    print(f"\n📈 谱面参数:")
    print(f"   难度:          {params.difficulty.value if params.difficulty else '(未指定)'}")
    print(f"   音符密度:      {params.note_density:.2f}" if params.note_density else "   音符密度: (自动推断)")
    print(f"   复杂度:        {params.complexity_score:.2f}" if params.complexity_score is not None else "   复杂度: (自动推断)")
    
    print(f"\n🎼 音符分布:")
    dist = params.note_type_distribution
    total = sum(dist.values())
    for note_type, ratio in dist.items():
        bar_len = int(ratio * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = (ratio / total * 100) if total > 0 else 0
        print(f"   {note_type.upper():8s}  {bar} {pct:.1f}%")
    
    if params.special_patterns:
        print(f"\n✨ 特殊技巧模式:")
        for pattern in params.special_patterns:
            print(f"   • {pattern}")
    
    print(f"\n📐 空间布局:")
    print(f"   轨道使用:  {params.lane_usage or '默认'}")
    print(f"   垂直范围:  {params.vertical_range or '混合'}")
    
    print(f"\n🎲 生成控制:")
    print(f"   创造性:    {'█' * int(params.creativity * 20)}{'░' * (20 - int(params.creativity * 20))} {params.creativity:.0%}")
    print(f"   随机性:    {'█' * int(params.randomness * 20)}{'░' * (20 - int(params.randomness * 20))} {params.randomness:.0%}")
    
    print(f"\n💬 描述:")
    print(f"   {params.description}")
    
    print(f"\n🏷️  关键词:")
    if params.keywords:
        print(f"   {', '.join(params.keywords)}")
    else:
        print(f"   (无)")
    
    print(f"\n⏬ 置信度评分:")
    conf = params.confidence_scores
    if conf:
        for key, value in conf.items():
            stars = "★" * min(int(value * 5), 5) + "☆" * max(5 - int(value * 5), 0)
            print(f"   {key:15s}: [{stars}] {value:.0%}")
    
    print("\n")


def create_default_parser() -> PromptParser:
    """创建默认配置的提示词解析器"""
    return PromptParser(
        provider="openai",
        model_name="gpt-4-turbo",
        temperature=0.3
    )


def interactive_mode():
    """交互式模式 - 循环接受用户输入并分析"""
    print("\n" + "=" * 60)
    print("🔮 ArellanoDreamWeaver 提示词分析器 (交互模式)")
    print("=" * 60)
    print("\n输入制谱提示词进行解析，输入 'quit' 或 'q' 退出\n")
    
    parser = create_default_parser()
    
    while True:
        try:
            prompt = input("📝 请输入提示词: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'q', 'exit']:
                print("\n👋 再见!")
                break
            
            params = analyze_prompt(prompt, parser)
            print_analysis(params)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 分析出错: {e}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ArellanoDreamWeaver - 提示词分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个提示词
  python analyze_prompt.py --prompt "制作一首128 BPM的电子音乐谱面"

  # 交互式模式
  python analyze_prompt.py --interactive

  # 批量分析多个提示词
  python analyze_prompt.py --batch prompts.txt
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", "-p", type=str,
                       help="要分析的提示词")
    group.add_argument("--interactive", "-i", action="store_true",
                       help="启动交互式模式")
    group.add_argument("--batch", "-b", type=str,
                       help="批量文件路径 (每行一个提示词)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.prompt:
        params = analyze_prompt(args.prompt)
        print_analysis(params)
        
        # 额外: 显示转换为张量的信息
        print("=== 技术细节 ===")
        cond_tensor = params.to_condition_tensor()
        print(f"\n条件张量形状: {cond_tensor.shape}")
        print(f"条件向量内容: {cond_tensor.tolist()[0]}")
        
    elif args.batch:
        if not Path(args.batch).exists():
            print(f"错误: 文件不存在: {args.batch}")
            sys.exit(1)
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"\n📂 批量处理 {len(lines)} 个提示词...\n")
        
        prompt_parser = create_default_parser()
        results = []
        
        for i, line in enumerate(lines, 1):
            print(f"\n[{i}/{len(lines)}] \"{line[:50]}{'...' if len(line) > 50 else ''}\"")
            try:
                params = analyze_prompt(line, prompt_parser)
                results.append((line, params))
                
                # 简要输出
                genre = params.genre.value if params.genre else "?"
                diff = params.difficulty.value if params.difficulty else "?"
                bpm = params.bpm or "?"
                print(f"   → BPM={bpm}, Genre={genre}, Diff={diff}, Density={params.note_density}")
                
            except Exception as e:
                print(f"   → ❌ 错误: {e}")
        
        print(f"\n✅ 批量处理完成! 共处理 {len(results)}/{len(lines)} 条提示词")


if __name__ == "__main__":
    main()
