#!/usr/bin/env python3
"""
Phira .pez 谱面 → ArellanoDreamWeaver ChartData 转换工具
========================================================

支持格式:
  - RPE (chart.json) - Re:PhiEdit 格式
  - PEC (*.pec) - PhiEditor 格式 (基础支持)

.pez 文件实际上是 ZIP 压缩包，包含:
  - info.yml    - 元信息
  - chart.json 或 *.pec - 谱面数据
  - song.mp3    - 音乐文件
  - background.png - 插图

用法:
  # 转换单个 .pez 文件
  python phira_to_chart.py input.pez

  # 批量转换目录
  python phira_to_chart.py --dir ./phira_charts --outdir data/processed

  # 同时提取音频到 data/audio
  python phira_to_chart.py --dir ./phira_charts --outdir data/processed --audio-dir data/audio

依赖: pyyaml (pip install pyyaml)
"""

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

# YAML 解析 (可选，用于 info.yml)
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("[WARN] pyyaml 未安装，info.yml 解析将使用简单正则")


# ============================================================
# Beat 时间格式: [整数拍, 分子, 分母] → 浮点拍数
# ============================================================

def beat_to_float(beat) -> float:
    """
    将 RPE beat 格式转换为浮点拍数
    
    beat 格式: [integer_part, numerator, denominator]
    例如: [4, 1, 4] = 4 + 1/4 = 4.25 拍
          [0, 0, 1] = 0 拍
    也可能是单个数字
    """
    if isinstance(beat, (int, float)):
        return float(beat)
    if isinstance(beat, list) and len(beat) >= 3:
        integer_part = beat[0]
        numerator = beat[1]
        denominator = beat[2]
        if denominator == 0:
            return float(integer_part)
        return integer_part + numerator / denominator
    if isinstance(beat, list) and len(beat) == 1:
        return float(beat[0])
    return 0.0


# ============================================================
# YAML 简单解析 (备用)
# ============================================================

def simple_parse_yaml(text: str) -> Dict[str, Any]:
    """
    简单的 YAML 解析器 (仅支持基础 key: value 格式)
    用于在没有 pyyaml 时解析 info.yml
    """
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            
            # 去除引号
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            # 类型转换
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value == '':
                pass  # 保持空字符串
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # 保持字符串
            
            result[key] = value
    return result


def parse_info_yml(text: str) -> Dict[str, Any]:
    """解析 info.yml"""
    if HAS_YAML:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            pass
    return simple_parse_yaml(text)


# ============================================================
# RPE 格式解析
# ============================================================

def parse_rpe_chart(chart_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析 RPE 格式谱面，转换为通用音符列表
    
    RPE Note 类型:
      1 = Tap
      2 = Hold
      3 = Flick
      4 = Drag
    
    转换为 ChartData:
      Tap (1) → type=0
      Hold (2) → type=1
      Flick (3) → type=2 (Slide)
      Drag (4) → type=0 (Tap)
    """
    
    # 解析 BPM 列表
    bpm_list = []
    for bpm_event in chart_data.get("BPMList", []):
        bpm_value = bpm_event.get("bpm", 120.0)
        start_beat = beat_to_float(bpm_event.get("startTime", [0, 0, 1]))
        bpm_list.append({
            "beat": start_beat,
            "bpm": bpm_value
        })
    
    # 按拍数排序
    bpm_list.sort(key=lambda x: x["beat"])
    
    # 收集所有音符
    all_notes = []
    
    # 解析判定线
    for line_idx, judge_line in enumerate(chart_data.get("judgeLineList", [])):
        bpmfactor = judge_line.get("bpmfactor", 1.0)
        
        for note_data in judge_line.get("notes", []):
            note_type_rpe = note_data.get("type", 1)
            start_beat = beat_to_float(note_data.get("startTime", [0, 0, 1]))
            end_beat = beat_to_float(note_data.get("endTime", start_beat if isinstance(note_data.get("endTime"), list) else [0, 0, 1]))
            
            # RPE positionX 范围大约是 -675 到 675，归一化到 -1 ~ 1
            pos_x = note_data.get("positionX", 0)
            if isinstance(pos_x, (int, float)):
                pos_x_normalized = max(-1.0, min(1.0, pos_x / 675.0))
            else:
                pos_x_normalized = 0.0
            
            above = note_data.get("above", 1)
            is_fake = note_data.get("isFake", 0)
            is_above = (above == 1)
            
            # 转换音符类型
            if note_type_rpe == 1:
                # Tap
                chart_type = 0
                hold_time = 0.0
            elif note_type_rpe == 2:
                # Hold
                chart_type = 1
                hold_time = max(0.0, end_beat - start_beat)
            elif note_type_rpe == 3:
                # Flick → Slide
                chart_type = 2
                hold_time = max(0.0, end_beat - start_beat)
            elif note_type_rpe == 4:
                # Drag → Tap
                chart_type = 0
                hold_time = 0.0
            else:
                chart_type = 0
                hold_time = 0.0
            
            note = {
                "type": chart_type,
                "time": start_beat,  # 以拍为单位
                "holdTime": hold_time,
                "positionX": pos_x_normalized,
                "isFakeNote": bool(is_fake),
                "offset": 0.0,
                "isAbove": is_above,
                "hasOther": False,
                # 额外信息 (可用于后续分析)
                "_source_line": line_idx,
                "_rpe_type": note_type_rpe,
                "_speed": note_data.get("speed", 1.0),
            }
            all_notes.append(note)
    
    # 按时间排序
    all_notes.sort(key=lambda n: n["time"])
    
    # 标记 hasOther (同时间多音)
    _mark_has_other(all_notes)
    
    # 构建 BPM 列表 (ChartData 格式)
    chart_bpm_list = []
    for bpm_entry in bpm_list:
        bpm_num = int(round(bpm_entry["bpm"] * 1000))
        chart_bpm_list.append({"num": bpm_num, "den": 1000})
    
    if not chart_bpm_list:
        chart_bpm_list.append({"num": 120000, "den": 1000})
    
    # 计算音乐长度
    max_time = 0.0
    for n in all_notes:
        end = n["time"] + n["holdTime"]
        if end > max_time:
            max_time = end
    music_length = round(max_time + 4.0, 2) if max_time > 0 else 1.0
    
    # 构建 judgeSegments (按 16 拍分段)
    judge_segments = _split_into_segments(all_notes, segment_beats=16.0)
    
    # 构建 BoundaryList
    boundary_list = []
    if all_notes:
        first_time = all_notes[0]["time"]
        last_time = max(n["time"] + n["holdTime"] for n in all_notes)
        boundary_list.append({
            "BoundaryEvents": {
                "MoveX": [], "MoveY": [], "Rotate": []
            },
            "StartTime": first_time,
            "EndTime": last_time,
            "DefaultX": 0.0,
            "DefaultY": -10.0,
            "Direction": False,
            "IncludedSegments": None,
            "includedSegmentIndices": list(range(len(judge_segments)))
        })
    
    return {
        "judgeSegments": judge_segments,
        "BoundaryList": boundary_list,
        "bpmList": chart_bpm_list,
        "offset": 0.0,
        "musicLength": music_length,
        "beatSubdivision": 4,
        "_meta": {
            "format": "rpe",
            "note_count": len(all_notes),
            "bpm_changes": len(bpm_list),
        }
    }


# ============================================================
# PEC 格式解析 (基础支持)
# ============================================================

def parse_pec_chart(pec_text: str) -> Dict[str, Any]:
    """
    解析 PEC (Phigros Event Chart) 格式
    
    PEC 是文本格式，基本结构:
      BPM 行: bp 数值
      音符行: 类型 时间 位置 ...
    
    注意: PEC 格式有多种变体，这里只支持最常见的格式
    """
    
    lines = pec_text.strip().split('\n')
    
    bpm_list = [{"beat": 0.0, "bpm": 120.0}]
    all_notes = []
    current_line_idx = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if not parts:
            continue
        
        cmd = parts[0]
        
        # BPM 变化
        if cmd == 'bp' and len(parts) >= 2:
            try:
                bpm_val = float(parts[1])
                bpm_list.append({"beat": 0.0, "bpm": bpm_val})
            except ValueError:
                pass
        
        # 音符: c (click/tap), h (hold), f (flick), d (drag)
        elif cmd in ('c', 'h', 'f', 'd') and len(parts) >= 3:
            try:
                time_beat = float(parts[1])
                pos_x = float(parts[2])
                
                # 归一化位置
                pos_x_normalized = max(-1.0, min(1.0, pos_x))
                
                if cmd == 'c':
                    chart_type = 0  # Tap
                    hold_time = 0.0
                elif cmd == 'h':
                    chart_type = 1  # Hold
                    hold_time = float(parts[3]) if len(parts) > 3 else 1.0
                elif cmd == 'f':
                    chart_type = 2  # Flick → Slide
                    hold_time = 0.0
                elif cmd == 'd':
                    chart_type = 0  # Drag → Tap
                    hold_time = 0.0
                else:
                    chart_type = 0
                    hold_time = 0.0
                
                note = {
                    "type": chart_type,
                    "time": time_beat,
                    "holdTime": hold_time,
                    "positionX": pos_x_normalized,
                    "isFakeNote": False,
                    "offset": 0.0,
                    "isAbove": True,
                    "hasOther": False,
                    "_source_line": current_line_idx,
                }
                all_notes.append(note)
                
            except (ValueError, IndexError):
                pass
    
    # 排序
    all_notes.sort(key=lambda n: n["time"])
    _mark_has_other(all_notes)
    
    # 构建输出
    chart_bpm_list = []
    for bpm_entry in bpm_list:
        bpm_num = int(round(bpm_entry["bpm"] * 1000))
        chart_bpm_list.append({"num": bpm_num, "den": 1000})
    
    if not chart_bpm_list:
        chart_bpm_list.append({"num": 120000, "den": 1000})
    
    max_time = 0.0
    for n in all_notes:
        end = n["time"] + n["holdTime"]
        if end > max_time:
            max_time = end
    music_length = round(max_time + 4.0, 2) if max_time > 0 else 1.0
    
    judge_segments = _split_into_segments(all_notes, segment_beats=16.0)
    
    boundary_list = []
    if all_notes:
        first_time = all_notes[0]["time"]
        last_time = max(n["time"] + n["holdTime"] for n in all_notes)
        boundary_list.append({
            "BoundaryEvents": {"MoveX": [], "MoveY": [], "Rotate": []},
            "StartTime": first_time,
            "EndTime": last_time,
            "DefaultX": 0.0,
            "DefaultY": -10.0,
            "Direction": False,
            "IncludedSegments": None,
            "includedSegmentIndices": list(range(len(judge_segments)))
        })
    
    return {
        "judgeSegments": judge_segments,
        "BoundaryList": boundary_list,
        "bpmList": chart_bpm_list,
        "offset": 0.0,
        "musicLength": music_length,
        "beatSubdivision": 4,
        "_meta": {
            "format": "pec",
            "note_count": len(all_notes),
        }
    }


# ============================================================
# 辅助函数
# ============================================================

def _mark_has_other(notes: List[Dict], threshold: float = 0.01):
    """标记同时间有多个音符的情况"""
    if not notes:
        return
    for i, note in enumerate(notes):
        has_other = False
        if i > 0 and abs(notes[i - 1]["time"] - note["time"]) < threshold:
            has_other = True
        if i < len(notes) - 1 and abs(notes[i + 1]["time"] - note["time"]) < threshold:
            has_other = True
        note["hasOther"] = has_other


def _build_segment(notes: List[Dict]) -> Dict:
    """构建单个 judgeSegment"""
    notes_above = [n for n in notes if n.get("isAbove", True)]
    notes_below = [n for n in notes if not n.get("isAbove", True)]
    return {
        "segmentEvents": {
            "MoveX": [], "MoveY": [], "Rotate": [],
            "Alpha": [], "Length": [], "Speed": []
        },
        "notesAbove": notes_above,
        "notesBelow": notes_below,
        "far": None
    }


def _split_into_segments(notes: List[Dict], segment_beats: float) -> List[Dict]:
    """按拍数将音符分成多个段落"""
    if not notes:
        return [_build_segment([])]
    
    first_time = notes[0]["time"]
    last_time = max(n["time"] + n.get("holdTime", 0.0) for n in notes)
    
    segments = []
    seg_start = first_time
    
    while seg_start <= last_time:
        seg_end = seg_start + segment_beats
        seg_notes = [n for n in notes if seg_start <= n["time"] < seg_end]
        if seg_notes:
            segments.append(_build_segment(seg_notes))
        seg_start = seg_end
    
    if not segments:
        segments = [_build_segment(notes)]
    
    return segments


def _clean_internal_fields(notes: List[Dict]):
    """移除内部字段 (以 _ 开头的)"""
    for note in notes:
        keys_to_remove = [k for k in note.keys() if k.startswith('_')]
        for k in keys_to_remove:
            del note[k]


def clean_chart_for_output(chart: Dict) -> Dict:
    """清理谱面数据，移除内部字段"""
    # 移除 _meta
    chart.pop("_meta", None)
    
    # 清理音符中的内部字段
    for seg in chart.get("judgeSegments", []):
        for note in seg.get("notesAbove", []):
            keys_to_remove = [k for k in note.keys() if k.startswith('_')]
            for k in keys_to_remove:
                del note[k]
        for note in seg.get("notesBelow", []):
            keys_to_remove = [k for k in note.keys() if k.startswith('_')]
            for k in keys_to_remove:
                del note[k]
    
    return chart


# ============================================================
# .pez 文件处理
# ============================================================

def process_pez_file(
    pez_path: str,
    output_dir: str,
    audio_dir: Optional[str] = None,
    save_meta: bool = True,
) -> Dict[str, Any]:
    """
    处理单个 .pez 文件
    
    Returns:
        {"status": "ok"/"error", "message": str, "chart_path": str, ...}
    """
    pez_path = Path(pez_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "status": "error",
        "message": "",
        "pez_file": str(pez_path),
        "chart_path": "",
        "audio_path": "",
    }
    
    try:
        # 1. 解压 .pez (ZIP)
        with zipfile.ZipFile(str(pez_path), 'r') as zf:
            file_list = zf.namelist()
            
            # 找到 info.yml
            info_yml = None
            for f in file_list:
                if f.lower() in ('info.yml', 'info.yaml'):
                    info_yml = f
                    break
            
            # 找到谱面文件
            chart_file = None
            for f in file_list:
                if f.lower().endswith('.json') and 'info' not in f.lower():
                    chart_file = f
                    break
                elif f.lower().endswith('.pec'):
                    chart_file = f
                    break
            
            # 找到音乐文件
            music_file = None
            for f in file_list:
                if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                    music_file = f
                    break
            
            # 2. 解析 info.yml
            info = {}
            if info_yml:
                info_text = zf.read(info_yml).decode('utf-8', errors='replace')
                info = parse_info_yml(info_text)
            
            chart_name = info.get("name", pez_path.stem)
            charter = info.get("charter", "unknown")
            level = info.get("level", "unknown")
            difficulty = info.get("difficulty", 0)
            
            # 3. 解析谱面数据
            if not chart_file:
                result["message"] = "找不到谱面文件 (chart.json 或 .pec)"
                return result
            
            chart_content = zf.read(chart_file).decode('utf-8', errors='replace')
            
            if chart_file.lower().endswith('.json'):
                # RPE 格式
                chart_data = json.loads(chart_content)
                chart = parse_rpe_chart(chart_data)
            elif chart_file.lower().endswith('.pec'):
                # PEC 格式
                chart = parse_pec_chart(chart_content)
            else:
                result["message"] = f"未知谱面格式: {chart_file}"
                return result
            
            note_count = chart.get("_meta", {}).get("note_count", 0)
            if note_count == 0:
                result["message"] = "谱面中没有音符"
                return result
            
            # 4. 生成输出文件名
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', chart_name)
            safe_level = re.sub(r'[<>:"/\\|?*]', '_', str(level))
            safe_charter = re.sub(r'[<>:"/\\|?*]', '_', str(charter))
            out_name = f"phira_{pez_path.stem}_{safe_level}_{safe_name}_{safe_charter}.json"
            
            # 清理内部字段
            chart_clean = clean_chart_for_output(chart)
            
            # 5. 保存谱面 JSON
            chart_out_path = output_dir / out_name
            with open(chart_out_path, 'w', encoding='utf-8') as f:
                json.dump(chart_clean, f, indent=2, ensure_ascii=False)
            
            result["chart_path"] = str(chart_out_path)
            
            # 6. 提取音乐文件
            if music_file and audio_dir:
                audio_out_dir = Path(audio_dir)
                audio_out_dir.mkdir(parents=True, exist_ok=True)
                
                # 音乐文件名与谱面同名
                audio_ext = Path(music_file).suffix
                audio_out_name = chart_out_path.stem + audio_ext
                audio_out_path = audio_out_dir / audio_out_name
                
                with open(audio_out_path, 'wb') as f:
                    f.write(zf.read(music_file))
                
                result["audio_path"] = str(audio_out_path)
            
            # 7. 保存元信息
            if save_meta:
                meta_out_path = chart_out_path.with_suffix('.meta.json')
                meta_info = {
                    "phira_id": info.get("id"),
                    "name": chart_name,
                    "level": level,
                    "difficulty": difficulty,
                    "charter": charter,
                    "composer": info.get("composer", ""),
                    "illustrator": info.get("illustrator", ""),
                    "source": "phira",
                    "source_file": str(pez_path),
                    "format": chart.get("_meta", {}).get("format", "unknown"),
                    "note_count": note_count,
                }
                with open(meta_out_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_info, f, indent=2, ensure_ascii=False)
            
            result["status"] = "ok"
            result["message"] = f"{chart_name} [{level}] by {charter} ({note_count} notes)"
            
    except zipfile.BadZipFile:
        result["message"] = "不是有效的 ZIP/PEZ 文件"
    except json.JSONDecodeError as e:
        result["message"] = f"JSON 解析失败: {e}"
    except Exception as e:
        result["message"] = f"处理失败: {e}"
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================
# 批量处理
# ============================================================

def batch_process(
    input_dir: str,
    output_dir: str,
    audio_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """批量处理目录中的所有 .pez 文件"""
    input_path = Path(input_dir)
    pez_files = sorted(input_path.glob("*.pez"))
    
    if not pez_files:
        print(f"[ERROR] 目录中没有 .pez 文件: {input_dir}")
        return {"total": 0, "ok": 0, "error": 0}
    
    print(f"📥 Phira → ArellanoDreamWeaver 批量转换")
    print(f"{'=' * 60}")
    print(f"  输入目录: {input_path.absolute()}")
    print(f"  输出目录: {Path(output_dir).absolute()}")
    if audio_dir:
        print(f"  音频目录: {Path(audio_dir).absolute()}")
    print(f"  PEZ 文件: {len(pez_files)} 个")
    print(f"{'=' * 60}\n")
    
    stats = {"total": len(pez_files), "ok": 0, "error": 0}
    
    for i, pez_file in enumerate(pez_files):
        print(f"[{i+1}/{len(pez_files)}] 处理: {pez_file.name}")
        result = process_pez_file(
            str(pez_file),
            output_dir,
            audio_dir,
        )
        
        if result["status"] == "ok":
            stats["ok"] += 1
            print(f"  ✅ {result['message']}")
            print(f"     → {result['chart_path']}")
        else:
            stats["error"] += 1
            print(f"  ❌ {result['message']}")
    
    print(f"\n{'=' * 60}")
    print(f"📊 转换完成!")
    print(f"  总计: {stats['total']}")
    print(f"  成功: {stats['ok']}")
    print(f"  失败: {stats['error']}")
    print(f"{'=' * 60}")
    
    return stats


# ============================================================
# 命令行
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phira .pez 谱面 → ArellanoDreamWeaver ChartData 转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  python phira_to_chart.py chart.pez

  # 批量转换，同时提取音频
  python phira_to_chart.py --dir ./phira_charts --outdir data/processed --audio-dir data/audio

  # 只转换谱面，不提取音频
  python phira_to_chart.py --dir ./phira_charts --outdir data/processed
        """
    )
    
    parser.add_argument('input', nargs='?', help='单个 .pez 文件路径')
    parser.add_argument('--dir', help='批量转换目录 (所有 .pez 文件)')
    parser.add_argument('--outdir', '-o', default='data/processed',
                        help='输出目录 (默认: data/processed)')
    parser.add_argument('--audio-dir', '-a', default=None,
                        help='音频输出目录 (如指定则提取音乐)')
    parser.add_argument('--no-meta', action='store_true',
                        help='不保存元信息文件')
    
    args = parser.parse_args()
    
    if args.dir:
        # 批量模式
        batch_process(args.dir, args.outdir, args.audio_dir)
    elif args.input:
        # 单文件模式
        result = process_pez_file(
            args.input,
            args.outdir,
            args.audio_dir,
            save_meta=not args.no_meta,
        )
        if result["status"] == "ok":
            print(f"✅ {result['message']}")
            print(f"   谱面: {result['chart_path']}")
            if result["audio_path"]:
                print(f"   音频: {result['audio_path']}")
        else:
            print(f"❌ {result['message']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
