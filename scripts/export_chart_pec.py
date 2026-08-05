# -*- coding: utf-8 -*-
"""
生成谱面导出工具 (chart_result.json → PEC / PEZ)
==================================================
把 infer_onnx.py 生成的 chart_result.json 转换为 Phira/Phigros 可玩的谱面。

输出格式:
  - .pec   PEC 文本谱面 (Phigros 编辑器 / Phira 均可导入)
  - .pez   谱面包 (zip, chart.pec + music 音频), 可直接导入 Phira

用法:
  # 生成 PEC 谱面 (BPM 需与歌曲一致, 默认 120)
  python scripts/export_chart_pec.py --input chart_result.json --output generated.pec --bpm 120

  # 生成并打包 PEZ (放入音频, 可直接导入 Phira)
  python scripts/export_chart_pec.py --input chart_result.json --output generated.pez \
      --audio data/audio/Chart.mp3 --bpm 120

  # 调整谱面对齐
  python scripts/export_chart_pec.py --input chart_result.json --output generated.pec \
      --bpm 140 --offset-ms -120 --line 1
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.unified_tokenizer import ChartTokenizer, UniNote, UniEvent

# PEC x 世界坐标范围 (解析时 x_raw / 1024 -> -1~1, 写出时反向)
PEC_X_SCALE = 1024.0

# PEC 音符类型: 0=Tap, 1=Hold, 2=Flick
PEC_NOTE_CMD = {0: 'n1', 1: 'n2', 2: 'n3'}


def notes_to_pec(
    notes: List[UniNote],
    bpm: float = 120.0,
    offset_ms: float = 0.0,
    line: int = 1,
) -> str:
    """
    将音符列表转换为 PEC 文本。

    PEC 格式 (参考 https://pgrfm.miraheze.org/wiki/PEC):
      第一行: offset 毫秒 (PEC 内部约定需 +175)
      bp beat bpm        - BPM 变化
      n1 类型 拍 x 方向 假音符 - Tap   (类型 0)
      n2 类型 起始拍 结束拍 x 方向 假音符 - Hold (类型 1)
      n3 类型 拍 x 方向 假音符 - Flick (类型 2)
      # speed / & width  - 音符速度/宽度 (可选)
    """
    # 第一行: 偏移毫秒 (解析器约定 offset_ms = raw - 175)
    lines = [f"{int(offset_ms) + 175}"]

    # BPM 表
    lines.append(f"bp 0.000 {bpm:.3f}")

    # 按时间排序
    sorted_notes = sorted(notes, key=lambda n: (n.time, n.position_x))

    for n in sorted_notes:
        x_world = max(-1.0, min(1.0, n.position_x)) * PEC_X_SCALE
        direction = 1 if n.is_above else 0

        if n.type == 0:  # Tap
            lines.append(f"n1 0 {n.time:.3f} {x_world:.3f} {direction} 0")
        elif n.type == 1:  # Hold
            end = n.time + max(0.0, n.hold_time)
            lines.append(f"n2 1 {n.time:.3f} {end:.3f} {x_world:.3f} {direction} 0")
        elif n.type == 2:  # Slide → Flick
            lines.append(f"n3 2 {n.time:.3f} {x_world:.3f} {direction} 0")
        else:
            continue  # 未知类型跳过

    return "\n".join(lines) + "\n"


def pack_pez(
    pez_path: str,
    pec_text: str,
    audio_path: str,
    chart_name: str = "chart.pec",
) -> None:
    """
    打包 PEZ 谱面包 (zip: chart.pec + music 音频)。

    Phira 导入 .pez 时识别 chart.json / chart.pec 和音频文件。
    """
    audio_ext = Path(audio_path).suffix or ".mp3"
    music_name = f"music{audio_ext}"

    with zipfile.ZipFile(pez_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(chart_name, pec_text)
        zf.write(audio_path, music_name)

    print(f"[OK] PEZ 打包完成: {pez_path}")
    print(f"     - {chart_name}")
    print(f"     - {music_name} ({Path(audio_path).stat().st_size / 1024 / 1024:.1f} MB)")


def summarize(notes: List[UniNote], bpm: float) -> None:
    """打印谱面摘要"""
    by_type = {"tap": 0, "hold": 0, "slide": 0}
    for n in notes:
        by_type[["tap", "hold", "slide"][n.type]] += 1

    if notes:
        duration_beats = max(n.time + n.hold_time for n in notes)
        duration_sec = duration_beats * 60.0 / bpm
        first_beat = min(n.time for n in notes)
        print(f"\n[SUMMARY] 谱面摘要 (BPM={bpm}):")
        print(f"  音符总数:  {len(notes)}  (tap={by_type['tap']}, hold={by_type['hold']}, slide={by_type['slide']})")
        print(f"  时间范围:  {first_beat:.2f} ~ {duration_beats:.2f} 拍 ≈ {duration_sec:.1f} 秒")
    else:
        print("\n[WARN] 没有解码出任何音符!")


def main():
    parser = argparse.ArgumentParser(
        description="把 infer_onnx.py 生成的 chart_result.json 转换为 PEC / PEZ 谱面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 只导出 PEC 文本谱面
  python scripts/export_chart_pec.py --input chart_result.json --output generated.pec --bpm 120

  # 导出并打包成 PEZ (可直接导入 Phira)
  python scripts/export_chart_pec.py --input chart_result.json --output generated.pez \\
      --audio data/audio/Chart.mp3 --bpm 120
        """,
    )
    parser.add_argument("--input", type=str, default="chart_result.json",
                        help="infer_onnx.py 的输出 JSON (含 tokens)")
    parser.add_argument("--output", type=str, default="generated.pec",
                        help="输出文件: .pec 文本谱面 或 .pez 谱面包")
    parser.add_argument("--bpm", type=float, default=120.0,
                        help="歌曲 BPM (决定拍→秒换算, 默认 120)")
    parser.add_argument("--offset-ms", type=float, default=0.0,
                        help="谱面偏移毫秒 (默认 0, 用于对齐音频起点)")
    parser.add_argument("--line", type=int, default=1,
                        help="音符所在判定线编号 (默认 1)")
    parser.add_argument("--audio", type=str, default=None,
                        help="音频文件路径 (打包 .pez 时需要)")
    parser.add_argument("--verbose", action="store_true",
                        help="打印前 10 个音符详情")

    args = parser.parse_args()

    # === 加载生成结果 ===
    if not os.path.exists(args.input):
        print(f"[FAIL] 找不到生成结果: {args.input}")
        sys.exit(1)

    with open(args.input, encoding='utf-8') as f:
        result = json.load(f)

    tokens = result.get("tokens")
    if not tokens:
        print(f"[FAIL] {args.input} 中没有 tokens 字段 (请用 infer_onnx.py 重新生成)")
        sys.exit(1)

    print(f"[LOAD] 读取 {len(tokens)} 个 tokens: {args.input}")

    # === 解码 (使用训练同款 tokenizer) ===
    tokenizer = ChartTokenizer()
    notes, events = tokenizer.decode_with_events(tokens)

    print(f"[DECODE] 音符 {len(notes)} 个, 事件块 {len(events)} 个")
    if events:
        print(f"  [WARN] 事件暂未支持写出, 已跳过 {len(events)} 个事件块")

    if args.verbose:
        for i, n in enumerate(notes[:10]):
            print(f"    [{i}] type={n.type} time={n.time:.2f}bt x={n.position_x:.2f} "
                  f"hold={n.hold_time:.2f} {'above' if n.is_above else 'below'}")

    summarize(notes, args.bpm)

    if not notes:
        sys.exit(1)

    # === 生成 PEC ===
    pec_text = notes_to_pec(notes, bpm=args.bpm, offset_ms=args.offset_ms, line=args.line)

    output_path = Path(args.output)
    ext = output_path.suffix.lower()

    if ext == ".pez":
        # 打包 PEZ
        if args.audio is None:
            print("[FAIL] 打包 .pez 需要 --audio 参数 (谱面包包含音频)")
            sys.exit(1)
        if not os.path.exists(args.audio):
            print(f"[FAIL] 音频文件不存在: {args.audio}")
            sys.exit(1)
        pack_pez(str(output_path), pec_text, args.audio)
    else:
        # 输出 .pec 文本
        if ext != ".pec":
            output_path = output_path.with_suffix(".pec")
            print(f"[INFO] 输出扩展名修正为: {output_path}")
        output_path.write_text(pec_text, encoding='utf-8')
        print(f"[OK] PEC 谱面已写出: {output_path} ({len(pec_text.splitlines())} 行)")

    # === 使用说明 ===
    print(f"\n[USAGE] 如何游玩:")
    print(f"  1. Phira 桌面端/手机端 → 导入谱面 → 选择生成的 .pez / .pec 文件")
    print(f"  2. Phigros 编辑器 (Pgrfm) 可直接打开 .pec 查看/微调")
    print(f"  3. 注意: 生成谱面的拍数是相对增量编码, 时长约为 "
          f"{max((n.time + n.hold_time for n in notes), default=0) * 60 / args.bpm:.1f} 秒 (BPM={args.bpm:g})")


if __name__ == "__main__":
    main()
