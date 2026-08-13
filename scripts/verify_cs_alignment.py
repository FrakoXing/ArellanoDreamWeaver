# -*- coding: utf-8 -*-
"""
C# 数值对齐验证脚本
===================

生成参考数据 (tokens + features), 供 C# 端逐值对比验证。

用法:
  # 从 Arellano ChartData JSON 生成参考数据
  python scripts/verify_cs_alignment.py --chart-json path/to/Chart.json --output cs_verify.json

  # 从已有 token 文件生成
  python scripts/verify_cs_alignment.py --token-file data/tokens/xxx.pt --output cs_verify.json

输出 JSON 格式:
  {
    "source": "chart_json" | "token_file",
    "game_token_id": 118,
    "tokens": [118, 1, 4, 8, 76, 112, ...],
    "features_raw": [0.0, 0.0, 1.0, 42.0, ...],
    "quantization": {
      "time_examples": {"0.25": 8, "0.5": 9, "15.75": 70},
      "pos_examples": {"-1.0": 71, "0.0": 75, "1.0": 79},
      "hold_examples": {"0.25": 81, "7.75": 111}
    }
  }
"""

import sys
import json
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rating_common import (
    compute_features,
    FEATURE_DIM,
    GAME_TOKEN_PHIRA,
    GAME_TOKEN_OSU,
    GAME_TOKEN_ARELLANO,
    game_token_to_name,
)


def generate_from_chart_json(chart_path: str) -> dict:
    """从 Arellano ChartData JSON 生成参考数据"""
    from unified_tokenizer import (
        ArellanoParser, ChartTokenizer, parse_chart, GAME_TOKEN_MAP,
    )

    # 解析谱面
    notes, events, game, meta = parse_chart(chart_path)
    game_token_id = GAME_TOKEN_MAP.get(game, GAME_TOKEN_ARELLANO)

    # 编码
    tokenizer = ChartTokenizer(game=game)
    tokens = tokenizer.encode(notes)

    # 计算特征
    features = compute_features(tokens, game_token_id)

    return _build_output(
        source="chart_json",
        source_path=chart_path,
        game_token_id=game_token_id,
        tokens=tokens,
        features=features,
        note_count=len(notes),
    )


def generate_from_token_file(token_path: str) -> dict:
    """从已有 token 文件生成参考数据"""
    import torch

    try:
        tokens = torch.load(token_path, map_location='cpu', weights_only=True).long().tolist()
    except Exception:
        tokens = torch.load(token_path, map_location='cpu').long().tolist()

    game_token_id = int(tokens[0]) if tokens else GAME_TOKEN_ARELLANO
    features = compute_features(tokens, game_token_id)

    return _build_output(
        source="token_file",
        source_path=token_path,
        game_token_id=game_token_id,
        tokens=tokens,
        features=features,
    )


def _build_output(source, source_path, game_token_id, tokens, features, note_count=None):
    """构建输出字典"""
    # 量化示例 (验证 C# 量化方法)
    from unified_tokenizer import ChartTokenizer
    tk = ChartTokenizer()

    quant_examples = {
        "time_examples": {
            "0.0": tk.quantize_time(0.0),
            "0.25": tk.quantize_time(0.25),
            "0.5": tk.quantize_time(0.5),
            "1.0": tk.quantize_time(1.0),
            "15.75": tk.quantize_time(15.75),
        },
        "pos_examples": {
            "-1.0": tk.quantize_pos(-1.0),
            "-0.5": tk.quantize_pos(-0.5),
            "0.0": tk.quantize_pos(0.0),
            "0.5": tk.quantize_pos(0.5),
            "1.0": tk.quantize_pos(1.0),
        },
        "hold_examples": {
            "0.0": tk.quantize_hold(0.0),
            "0.25": tk.quantize_hold(0.25),
            "1.0": tk.quantize_hold(1.0),
            "7.75": tk.quantize_hold(7.75),
        },
    }

    return {
        "source": source,
        "source_path": str(source_path),
        "game_token_id": game_token_id,
        "game_name": game_token_to_name(game_token_id),
        "token_count": len(tokens),
        "note_count": note_count,
        "tokens": tokens,
        "features_raw": [float(f) for f in features],
        "feature_dim": FEATURE_DIM,
        "feature_layout": [
            "is_phira", "is_osu", "is_arellano",
            "n_notes", "peak_nps", "avg_nps",
            "n_events", "event_density", "tap_ratio",
            "hold_ratio", "slide_ratio", "hold_duration_density",
            "pos_variance", "simultaneous_ratio", "min_delta_p10"
        ],
        "quantization_examples": quant_examples,
    }


def main():
    parser = argparse.ArgumentParser(description="C# 数值对齐验证数据生成")
    parser.add_argument("--chart-json", type=str, default=None,
                        help="Arellano ChartData JSON 路径")
    parser.add_argument("--token-file", type=str, default=None,
                        help="已有 token 文件路径")
    parser.add_argument("--output", type=str, default="cs_verify.json",
                        help="输出 JSON 路径")
    args = parser.parse_args()

    if not args.chart_json and not args.token_file:
        print("❌ 请指定 --chart-json 或 --token-file")
        sys.exit(1)

    if args.chart_json:
        result = generate_from_chart_json(args.chart_json)
    else:
        result = generate_from_token_file(args.token_file)

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 参考数据已生成: {output_path}")
    print(f"   来源: {result['source']}")
    print(f"   Game Token: {result['game_token_id']}")
    print(f"   Token 数: {result['token_count']}")
    print(f"   特征: {[f'{v:.4f}' for v in result['features_raw']]}")
    print(f"\n   量化示例 (C# 应产出相同值):")
    for cat, examples in result["quantization_examples"].items():
        print(f"     {cat}:")
        for input_val, token_id in examples.items():
            print(f"       {input_val:>6s} → {token_id}")


if __name__ == "__main__":
    main()
