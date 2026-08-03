#!/usr/bin/env python3
"""
osu! 谱面 (.osu) → ArellanoDreamWeaver ChartData 转换工具
=========================================================

支持:
  - osu! standard (Mode=0): 圆圈→Tap, 滑条→Slide, 转盘→Slide(长hold)
  - osu!mania (Mode=3): 普通音符→Tap, 长按→Hold

用法:
  python osu_to_chart.py <input.osu> [output.json]
  python osu_to_chart.py --dir <input_dir> [--outdir <output_dir>]

依赖: 仅标准库 + numpy (项目已有)
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any


# ============================================================
# osu! 文件解析
# ============================================================

@dataclass
class OsuTimingPoint:
    """osu! 时间点"""
    time: float           # 毫秒
    beat_length: float    # 红线: 每拍毫秒数; 绿线: 负值, 滑条速度倍率
    meter: int            # 拍号
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool     # True=红线(非继承), False=绿线(继承)
    effects: int


@dataclass
class OsuHitObject:
    """osu! 打击物件"""
    x: int
    y: int
    time: float           # 毫秒
    type_flags: int       # 位标记
    hit_sound: int
    # 圆圈: 无额外参数
    # 滑条:
    curve_type: str = ""
    curve_points: List[Tuple[int, int]] = field(default_factory=list)
    slides: int = 0       # 滑动次数
    slider_length: float = 0.0
    # 转盘:
    end_time: float = 0.0
    # osu!mania 长按:
    mania_end_time: float = 0.0
    # 额外音效参数
    extra: List[str] = field(default_factory=list)

    @property
    def is_circle(self) -> bool:
        return bool(self.type_flags & 1)

    @property
    def is_slider(self) -> bool:
        return bool(self.type_flags & 2)

    @property
    def is_new_combo(self) -> bool:
        return bool(self.type_flags & 4)

    @property
    def is_spinner(self) -> bool:
        return bool(self.type_flags & 8)

    @property
    def is_mania_hold(self) -> bool:
        return bool(self.type_flags & 128)


@dataclass
class OsuBeatmap:
    """解析后的 osu! 谱面"""
    version: int = 0
    # [General]
    audio_filename: str = ""
    audio_lead_in: int = 0
    preview_time: int = -1
    countdown: int = 1
    sample_set: str = "Normal"
    stack_leniency: float = 0.7
    mode: int = 0         # 0=standard, 1=taiko, 2=catch, 3=mania
    # [Metadata]
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version_name: str = ""
    source: str = ""
    tags: str = ""
    beatmap_id: int = 0
    beatmap_set_id: int = -1
    # [Difficulty]
    hp_drain_rate: float = 5.0
    circle_size: float = 5.0    # CS: osu!mania 中 = 键数
    overall_difficulty: float = 5.0
    approach_rate: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    # [TimingPoints]
    timing_points: List[OsuTimingPoint] = field(default_factory=list)
    # [HitObjects]
    hit_objects: List[OsuHitObject] = field(default_factory=list)
    # [Events]
    break_periods: List[Tuple[float, float]] = field(default_factory=list)


def parse_osu_file(filepath: str) -> OsuBeatmap:
    """解析 .osu 文件"""
    beatmap = OsuBeatmap()
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # 规范化行尾
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    
    current_section = ""
    
    for line in lines:
        line = line.strip()
        
        # 空行或注释
        if not line or line.startswith('//'):
            continue
        
        # 文件版本
        if line.startswith('osu file format'):
            beatmap.version = int(line.split('v')[-1]) if 'v' in line else 0
            continue
        
        # 章节标题
        section_match = re.match(r'^\[(\w+)\]$', line)
        if section_match:
            current_section = section_match.group(1)
            continue
        
        # 解析各章节
        if current_section == 'General':
            _parse_general(beatmap, line)
        elif current_section == 'Metadata':
            _parse_metadata(beatmap, line)
        elif current_section == 'Difficulty':
            _parse_difficulty(beatmap, line)
        elif current_section == 'Events':
            _parse_events(beatmap, line)
        elif current_section == 'TimingPoints':
            _parse_timing_point(beatmap, line)
        elif current_section == 'HitObjects':
            _parse_hit_object(beatmap, line)
    
    # 按时间排序
    beatmap.timing_points.sort(key=lambda tp: tp.time)
    beatmap.hit_objects.sort(key=lambda ho: ho.time)
    
    return beatmap


def _parse_kv(line: str) -> Tuple[str, str]:
    """解析 Key:Value 对 (支持冒号后有空格和无空格)"""
    idx = line.find(':')
    if idx < 0:
        return line.strip(), ""
    key = line[:idx].strip()
    value = line[idx + 1:].strip()
    return key, value


def _parse_general(bm: OsuBeatmap, line: str):
    key, val = _parse_kv(line)
    if key == 'AudioFilename':
        bm.audio_filename = val
    elif key == 'AudioLeadIn':
        bm.audio_lead_in = int(val) if val else 0
    elif key == 'PreviewTime':
        bm.preview_time = int(val) if val else -1
    elif key == 'Countdown':
        bm.countdown = int(val) if val else 1
    elif key == 'SampleSet':
        bm.sample_set = val
    elif key == 'StackLeniency':
        bm.stack_leniency = float(val) if val else 0.7
    elif key == 'Mode':
        bm.mode = int(val) if val else 0


def _parse_metadata(bm: OsuBeatmap, line: str):
    key, val = _parse_kv(line)
    if key == 'Title':
        bm.title = val
    elif key == 'TitleUnicode':
        bm.title_unicode = val
    elif key == 'Artist':
        bm.artist = val
    elif key == 'ArtistUnicode':
        bm.artist_unicode = val
    elif key == 'Creator':
        bm.creator = val
    elif key == 'Version':
        bm.version_name = val
    elif key == 'Source':
        bm.source = val
    elif key == 'Tags':
        bm.tags = val
    elif key == 'BeatmapID':
        bm.beatmap_id = int(val) if val else 0
    elif key == 'BeatmapSetID':
        bm.beatmap_set_id = int(val) if val else -1


def _parse_difficulty(bm: OsuBeatmap, line: str):
    key, val = _parse_kv(line)
    if key == 'HPDrainRate':
        bm.hp_drain_rate = float(val) if val else 5.0
    elif key == 'CircleSize':
        bm.circle_size = float(val) if val else 5.0
    elif key == 'OverallDifficulty':
        bm.overall_difficulty = float(val) if val else 5.0
    elif key == 'ApproachRate':
        bm.approach_rate = float(val) if val else 5.0
    elif key == 'SliderMultiplier':
        bm.slider_multiplier = float(val) if val else 1.4
    elif key == 'SliderTickRate':
        bm.slider_tick_rate = float(val) if val else 1.0


def _parse_events(bm: OsuBeatmap, line: str):
    parts = line.split(',')
    if len(parts) < 3:
        return
    evt_type = parts[0].strip()
    # Break: "2,startTime,endTime" 或 "Break,startTime,endTime"
    if evt_type in ('2', 'Break'):
        try:
            start = float(parts[1])
            end = float(parts[2])
            bm.break_periods.append((start, end))
        except (ValueError, IndexError):
            pass


def _parse_timing_point(bm: OsuBeatmap, line: str):
    parts = line.split(',')
    if len(parts) < 2:
        return
    try:
        time = float(parts[0])
        beat_length = float(parts[1])
        meter = int(parts[2]) if len(parts) > 2 else 4
        sample_set = int(parts[3]) if len(parts) > 3 else 0
        sample_index = int(parts[4]) if len(parts) > 4 else 0
        volume = int(parts[5]) if len(parts) > 5 else 100
        uninherited = int(parts[6]) == 1 if len(parts) > 6 else True
        effects = int(parts[7]) if len(parts) > 7 else 0
        
        tp = OsuTimingPoint(
            time=time, beat_length=beat_length, meter=meter,
            sample_set=sample_set, sample_index=sample_index,
            volume=volume, uninherited=uninherited, effects=effects
        )
        bm.timing_points.append(tp)
    except (ValueError, IndexError):
        pass


def _parse_hit_object(bm: OsuBeatmap, line: str):
    parts = line.split(',')
    if len(parts) < 4:
        return
    try:
        x = int(parts[0])
        y = int(parts[1])
        time = float(parts[2])
        type_flags = int(parts[3])
        hit_sound = int(parts[4]) if len(parts) > 4 else 0
        
        obj = OsuHitObject(
            x=x, y=y, time=time,
            type_flags=type_flags, hit_sound=hit_sound
        )
        
        # 根据类型解析额外参数
        if obj.is_slider and len(parts) > 5:
            # 滑条: curveType|points, slides, length, ...
            curve_data = parts[5]
            curve_parts = curve_data.split('|')
            if curve_parts:
                obj.curve_type = curve_parts[0]
                for pt in curve_parts[1:]:
                    if ':' in pt:
                        px, py = pt.split(':')
                        obj.curve_points.append((int(px), int(py)))
            obj.slides = int(parts[6]) if len(parts) > 6 else 1
            obj.slider_length = float(parts[7]) if len(parts) > 7 else 0.0
            obj.extra = parts[8:] if len(parts) > 8 else []
        
        elif obj.is_spinner and len(parts) > 5:
            # 转盘: endTime
            obj.end_time = float(parts[5])
            obj.extra = parts[6:] if len(parts) > 6 else []
        
        elif obj.is_mania_hold and len(parts) > 5:
            # osu!mania 长按: endTime (在物件参数位置)
            obj.mania_end_time = float(parts[5])
            obj.extra = parts[6:] if len(parts) > 6 else []
        
        else:
            obj.extra = parts[5:] if len(parts) > 5 else []
        
        bm.hit_objects.append(obj)
    except (ValueError, IndexError):
        pass


# ============================================================
# 时间转换: 毫秒 → 拍数
# ============================================================

def build_beat_map(timing_points: List[OsuTimingPoint]) -> List[Tuple[float, float, float]]:
    """
    构建时间→拍数映射表
    
    返回: [(time_ms, accumulated_beats, bpm), ...]
    每个条目表示: 在 time_ms 时刻, 已累计 accumulated_beats 拍, 当前 BPM
    """
    # 只取红线(非继承时间点)
    uninherited = [tp for tp in timing_points if tp.uninherited]
    if not uninherited:
        # 默认 120 BPM
        return [(0.0, 0.0, 120.0)]
    
    beat_map = []
    accumulated_beats = 0.0
    prev_time = 0.0
    
    for i, tp in enumerate(uninherited):
        bpm = 60000.0 / tp.beat_length if tp.beat_length > 0 else 120.0
        
        if i == 0:
            # 第一个时间点之前的部分
            if tp.time > 0:
                # 假设开头也是这个 BPM (或者用第一个红线的 BPM)
                beat_map.append((0.0, 0.0, bpm))
            
            beat_map.append((tp.time, accumulated_beats, bpm))
            prev_time = tp.time
        else:
            # 计算从上一个红线到当前红线累积的拍数
            dt = tp.time - prev_time
            prev_bpm = beat_map[-1][2]
            delta_beats = dt * prev_bpm / 60000.0
            accumulated_beats += delta_beats
            
            beat_map.append((tp.time, accumulated_beats, bpm))
            prev_time = tp.time
    
    return beat_map


def ms_to_beats(time_ms: float, beat_map: List[Tuple[float, float, float]]) -> float:
    """将毫秒时间转换为拍数"""
    if not beat_map:
        return time_ms * 120.0 / 60000.0
    
    # 找到对应的时间区间
    for i in range(len(beat_map) - 1, -1, -1):
        if time_ms >= beat_map[i][0]:
            base_time, base_beats, bpm = beat_map[i]
            dt = time_ms - base_time
            return base_beats + dt * bpm / 60000.0
    
    # 在所有时间点之前
    _, base_beats, bpm = beat_map[0]
    dt = time_ms - beat_map[0][0]
    return base_beats + dt * bpm / 60000.0


def get_bpm_at_time(time_ms: float, beat_map: List[Tuple[float, float, float]]) -> float:
    """获取指定时间的 BPM"""
    if not beat_map:
        return 120.0
    
    for i in range(len(beat_map) - 1, -1, -1):
        if time_ms >= beat_map[i][0]:
            return beat_map[i][2]
    
    return beat_map[0][2]


# ============================================================
# 转换: osu! → ChartData
# ============================================================

def convert_osu_to_chart(
    beatmap: OsuBeatmap,
    split_segments: bool = True,
    segment_beats: float = 16.0,
) -> Dict[str, Any]:
    """
    将 osu! 谱面转换为 ChartData 格式
    
    Args:
        beatmap: 解析后的 osu! 谱面
        split_segments: 是否按拍数分段 (每 segment_beats 拍一个段落)
        segment_beats: 每段的拍数
    
    Returns:
        ChartData 字典
    """
    beat_map = build_beat_map(beatmap.timing_points)
    
    # 获取所有 BPM 变化点
    bpm_list = []
    uninherited = [tp for tp in beatmap.timing_points if tp.uninherited]
    if uninherited:
        for tp in uninherited:
            bpm_value = 60000.0 / tp.beat_length if tp.beat_length > 0 else 120.0
            bpm_num = int(round(bpm_value * 1000))
            bpm_list.append({"num": bpm_num, "den": 1000})
    else:
        bpm_list.append({"num": 120000, "den": 1000})
    
    # 转换音符
    all_notes = []
    
    if beatmap.mode == 0:
        # osu! standard
        all_notes = _convert_standard(beatmap, beat_map)
    elif beatmap.mode == 3:
        # osu!mania
        all_notes = _convert_mania(beatmap, beat_map)
    else:
        print(f"[WARN] 不支持的游戏模式 Mode={beatmap.mode}, 尝试按 standard 转换")
        all_notes = _convert_standard(beatmap, beat_map)
    
    # 标记 hasOther (同时间多音)
    _mark_has_other(all_notes)
    
    # 计算 musicLength (最后一个音符的拍数 + 一些余量)
    max_time = 0.0
    for n in all_notes:
        end = n["time"] + n.get("holdTime", 0.0)
        if end > max_time:
            max_time = end
    music_length = round(max_time + 4.0, 2) if max_time > 0 else 1.0
    
    # 构建 judgeSegments
    if split_segments and all_notes:
        judge_segments = _split_into_segments(all_notes, segment_beats)
    else:
        # 单段落
        judge_segments = [_build_segment(all_notes)]
    
    # 构建 BoundaryList (简单: 一个覆盖全曲的边界)
    boundary_list = []
    if all_notes:
        first_time = all_notes[0]["time"]
        last_time = max(n["time"] + n.get("holdTime", 0.0) for n in all_notes)
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
    
    chart = {
        "judgeSegments": judge_segments,
        "BoundaryList": boundary_list,
        "bpmList": bpm_list,
        "offset": 0.0,
        "musicLength": music_length,
        "beatSubdivision": 4
    }
    
    return chart


def _convert_standard(beatmap: OsuBeatmap, beat_map) -> List[Dict]:
    """osu! standard → 音符列表"""
    notes = []
    
    for obj in beatmap.hit_objects:
        # 跳过转盘 (可以映射为长 Slide, 但通常不适合训练)
        if obj.is_spinner:
            time_beats = ms_to_beats(obj.time, beat_map)
            end_beats = ms_to_beats(obj.end_time, beat_map)
            hold_beats = end_beats - time_beats
            if hold_beats > 0:
                notes.append({
                    "type": 2,  # Slide
                    "time": round(time_beats, 6),
                    "holdTime": round(hold_beats, 6),
                    "positionX": 0.0,
                    "isFakeNote": False,
                    "offset": 0.0,
                    "isAbove": True,
                    "hasOther": False
                })
            continue
        
        # 位置映射: x(0-512) → positionX(-1~1)
        pos_x = (obj.x - 256) / 256.0
        pos_x = max(-1.0, min(1.0, pos_x))
        
        # y 位置决定上下轨道: y < 192 → 上方, 否则下方
        is_above = obj.y < 192
        
        time_beats = ms_to_beats(obj.time, beat_map)
        
        if obj.is_circle and not obj.is_slider:
            # 普通圆圈 → Tap
            notes.append({
                "type": 0,  # Tap
                "time": round(time_beats, 6),
                "holdTime": 0.0,
                "positionX": round(pos_x, 6),
                "isFakeNote": False,
                "offset": 0.0,
                "isAbove": is_above,
                "hasOther": False
            })
        
        elif obj.is_slider:
            # 滑条 → Slide (type=2)
            # 计算滑条持续时间
            bpm = get_bpm_at_time(obj.time, beat_map)
            # 滑条时长 = (slider_length * slides) / (slider_multiplier * 100) * beat_length_ms
            beat_length_ms = 60000.0 / bpm if bpm > 0 else 500.0
            slider_duration_ms = (
                obj.slider_length * obj.slides / (beatmap.slider_multiplier * 100.0)
            ) * beat_length_ms if beatmap.slider_multiplier > 0 else 0
            
            hold_beats = ms_to_beats(obj.time + slider_duration_ms, beat_map) - time_beats
            
            notes.append({
                "type": 2,  # Slide
                "time": round(time_beats, 6),
                "holdTime": round(max(hold_beats, 0.0), 6),
                "positionX": round(pos_x, 6),
                "isFakeNote": False,
                "offset": 0.0,
                "isAbove": is_above,
                "hasOther": False
            })
    
    return notes


def _convert_mania(beatmap: OsuBeatmap, beat_map) -> List[Dict]:
    """osu!mania → 音符列表"""
    notes = []
    
    # 键数 (columns) = CircleSize
    num_keys = max(1, int(round(beatmap.circle_size)))
    
    for obj in beatmap.hit_objects:
        # 列号: floor(x * keys / 512), 限制在 [0, keys-1]
        column = int(obj.x * num_keys / 512)
        column = max(0, min(num_keys - 1, column))
        
        # 列号 → positionX (-1 ~ 1)
        if num_keys == 1:
            pos_x = 0.0
        else:
            pos_x = (column / (num_keys - 1)) * 2 - 1
        
        time_beats = ms_to_beats(obj.time, beat_map)
        
        if obj.is_mania_hold:
            # 长按 → Hold (type=1)
            end_beats = ms_to_beats(obj.mania_end_time, beat_map)
            hold_beats = end_beats - time_beats
            
            notes.append({
                "type": 1,  # Hold
                "time": round(time_beats, 6),
                "holdTime": round(max(hold_beats, 0.0), 6),
                "positionX": round(pos_x, 6),
                "isFakeNote": False,
                "offset": 0.0,
                "isAbove": True,   # mania 全部放上方
                "hasOther": False
            })
        elif obj.is_circle:
            # 普通音符 → Tap (type=0)
            notes.append({
                "type": 0,  # Tap
                "time": round(time_beats, 6),
                "holdTime": 0.0,
                "positionX": round(pos_x, 6),
                "isFakeNote": False,
                "offset": 0.0,
                "isAbove": True,   # mania 全部放上方
                "hasOther": False
            })
        # mania 不使用滑条和转盘
    
    return notes


def _mark_has_other(notes: List[Dict], threshold: float = 0.01):
    """标记同时间有多个音符的情况"""
    if not notes:
        return
    
    # 按时间排序
    sorted_notes = sorted(notes, key=lambda n: n["time"])
    
    for i, note in enumerate(sorted_notes):
        has_other = False
        # 检查前后音符是否在同一时间
        if i > 0 and abs(sorted_notes[i - 1]["time"] - note["time"]) < threshold:
            has_other = True
        if i < len(sorted_notes) - 1 and abs(sorted_notes[i + 1]["time"] - note["time"]) < threshold:
            has_other = True
        note["hasOther"] = has_other


def _build_segment(notes: List[Dict]) -> Dict:
    """构建单个 judgeSegment"""
    notes_above = [n for n in notes if n.get("isAbove", True)]
    notes_below = [n for n in notes if not n.get("isAbove", True)]
    
    return {
        "segmentEvents": {
            "MoveX": [],
            "MoveY": [],
            "Rotate": [],
            "Alpha": [],
            "Length": [],
            "Speed": []
        },
        "notesAbove": notes_above,
        "notesBelow": notes_below,
        "far": None
    }


def _split_into_segments(notes: List[Dict], segment_beats: float) -> List[Dict]:
    """按拍数将音符分成多个段落"""
    if not notes:
        return [_build_segment([])]
    
    # 找到第一个音符的时间
    first_time = notes[0]["time"]
    last_time = max(n["time"] + n.get("holdTime", 0.0) for n in notes)
    
    segments = []
    seg_start = first_time
    
    while seg_start <= last_time:
        seg_end = seg_start + segment_beats
        
        # 取该段内的音符
        seg_notes = [
            n for n in notes
            if seg_start <= n["time"] < seg_end
        ]
        
        if seg_notes:  # 只添加有音符的段落
            segments.append(_build_segment(seg_notes))
        
        seg_start = seg_end
    
    # 确保至少有一个段落
    if not segments:
        segments = [_build_segment(notes)]
    
    return segments


# ============================================================
# 输出 & 命令行
# ============================================================

def get_output_name(beatmap: OsuBeatmap) -> str:
    """生成输出文件名"""
    mode_names = {0: "std", 1: "taiko", 2: "catch", 3: "mania"}
    mode_str = mode_names.get(beatmap.mode, "unknown")
    
    title = beatmap.title_unicode or beatmap.title or "Unknown"
    version = beatmap.version_name or "Normal"
    
    # 清理文件名中的非法字符
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe_version = re.sub(r'[<>:"/\\|?*]', '_', version)
    
    return f"{safe_title} - {safe_version} ({mode_str}).json"


def print_summary(chart: Dict, beatmap: OsuBeatmap):
    """打印转换摘要"""
    total_notes = 0
    tap_count = 0
    hold_count = 0
    slide_count = 0
    
    for seg in chart["judgeSegments"]:
        for note in seg.get("notesAbove", []) + seg.get("notesBelow", []):
            total_notes += 1
            t = note.get("type", 0)
            if t == 0:
                tap_count += 1
            elif t == 1:
                hold_count += 1
            elif t == 2:
                slide_count += 1
    
    mode_names = {0: "osu! standard", 1: "osu!taiko", 2: "osu!catch", 3: "osu!mania"}
    
    print("=" * 50)
    print("osu! → ArellanoDreamWeaver 转换完成")
    print("=" * 50)
    print(f"  谱面: {beatmap.title_unicode or beatmap.title}")
    print(f"  艺术家: {beatmap.artist_unicode or beatmap.artist}")
    print(f"  谱师: {beatmap.creator}")
    print(f"  难度: {beatmap.version_name}")
    print(f"  模式: {mode_names.get(beatmap.mode, f'Unknown({beatmap.mode})')}")
    print(f"  CS: {beatmap.circle_size}  OD: {beatmap.overall_difficulty}")
    print(f"  AR: {beatmap.approach_rate}  HP: {beatmap.hp_drain_rate}")
    print(f"  滑块倍率: {beatmap.slider_multiplier}x")
    print("-" * 50)
    print(f"  段落数: {len(chart['judgeSegments'])}")
    print(f"  BPM 变化点: {len(chart['bpmList'])}")
    if chart['bpmList']:
        bpms = [b['num'] / b['den'] for b in chart['bpmList']]
        print(f"  BPM 范围: {min(bpms):.1f} ~ {max(bpms):.1f}")
    print(f"  总音符数: {total_notes}")
    print(f"    Tap:   {tap_count} ({tap_count / max(total_notes, 1) * 100:.1f}%)")
    print(f"    Hold:  {hold_count} ({hold_count / max(total_notes, 1) * 100:.1f}%)")
    print(f"    Slide: {slide_count} ({slide_count / max(total_notes, 1) * 100:.1f}%)")
    print(f"  音乐长度: {chart['musicLength']:.2f} beats")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="osu! 谱面 (.osu) → ArellanoDreamWeaver ChartData 转换工具"
    )
    parser.add_argument('input', nargs='?', help='输入 .osu 文件路径')
    parser.add_argument('output', nargs='?', help='输出 .json 文件路径 (可选)')
    parser.add_argument('--dir', help='批量转换目录 (所有 .osu 文件)')
    parser.add_argument('--outdir', help='批量转换输出目录')
    parser.add_argument('--no-split', action='store_true',
                        help='不分割段落 (所有音符放入单个 judgeSegment)')
    parser.add_argument('--segment-beats', type=float, default=16.0,
                        help='每段拍数 (默认: 16)')
    parser.add_argument('--pretty', action='store_true', default=True,
                        help='格式化 JSON 输出 (默认开启)')
    
    args = parser.parse_args()
    
    if args.dir:
        # 批量转换
        input_dir = Path(args.dir)
        if not input_dir.is_dir():
            print(f"[ERROR] 目录不存在: {args.dir}")
            sys.exit(1)
        
        out_dir = Path(args.outdir) if args.outdir else input_dir / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        osu_files = sorted(input_dir.glob("*.osu"))
        if not osu_files:
            print(f"[ERROR] 目录中没有 .osu 文件: {args.dir}")
            sys.exit(1)
        
        print(f"找到 {len(osu_files)} 个 .osu 文件")
        
        for osu_file in osu_files:
            try:
                beatmap = parse_osu_file(str(osu_file))
                chart = convert_osu_to_chart(
                    beatmap,
                    split_segments=not args.no_split,
                    segment_beats=args.segment_beats,
                )
                
                # 保持原始文件名，只改后缀
                out_name = osu_file.with_suffix('.json').name
                out_path = out_dir / out_name
                
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(chart, f, indent=2, ensure_ascii=False)
                
                print(f"\n[OK] {osu_file.name} → {out_name}")
                print_summary(chart, beatmap)
                
            except Exception as e:
                print(f"\n[ERROR] {osu_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    elif args.input:
        # 单文件转换
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] 文件不存在: {args.input}")
            sys.exit(1)
        
        beatmap = parse_osu_file(str(input_path))
        chart = convert_osu_to_chart(
            beatmap,
            split_segments=not args.no_split,
            segment_beats=args.segment_beats,
        )
        
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = input_path.with_suffix('.json')
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(chart, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] {input_path.name} → {out_path.name}")
        print_summary(chart, beatmap)
        print(f"\n输出文件: {out_path}")
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
