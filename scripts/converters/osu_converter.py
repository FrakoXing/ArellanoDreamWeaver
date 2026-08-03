"""
osu! 谱面格式转换器
====================
将 osu! (.osu) 谱面转换为 ArellanoDreamWeaver 的 ChartData 格式。

支持模式:
- osu!mania (mode=3): 完全映射，最推荐
- osu!standard (mode=0): 基础映射，需要手动调整

使用方法:
  python scripts/converters/osu_converter.py --input song.osu --output data/processed/
  python scripts/converters/osu_converter.py --dir ./osu_maps/ --output data/processed/
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import argparse

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.chart.structures import (
    ChartData, JudgeSegment, SegmentEvents, Note, BPM,
    NoteType, Event, AnimationCurve, CurveKey, Boundary
)


# ============================================================
# osu! 数据结构
# ============================================================

@dataclass
class OsuTimingPoint:
    """osu! TimingPoint (BPM/速度变化点)"""
    time: float = 0.0              # 时间 (ms)
    beat_length: float = 500.0     # 每拍时长 (ms), 负数表示继承
    meter: int = 4                 # 拍号
    uninherited: bool = True       # True=BPM点, False=速度继承点


@dataclass
class OsuHitObject:
    """osu! HitObject (音符)"""
    x: float = 0.0                 # X坐标 (mania: 列号, standard: 像素)
    y: float = 0.0                 # Y坐标 (standard: 像素)
    time: float = 0.0              # 开始时间 (ms)
    end_time: float = 0.0          # 结束时间 (ms, 仅长按/滑条)
    obj_type: int = 1              # 1=circle, 2=slider, 8=spinner, 128=hold
    hit_sound: int = 0             # 音效
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OsuBeatmap:
    """解析后的 osu! 谱面数据"""
    # 文件头
    format_version: int = 14
    
    # General
    audio_filename: str = ""
    mode: int = 0                  # 0=standard, 3=mania
    audio_lead_in: int = 0
    
    # Metadata
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version: str = ""
    tags: str = ""
    
    # Difficulty
    hp_drain: float = 5.0
    circle_size: float = 4.0      # mania: 键数
    overall_difficulty: float = 5.0
    approach_rate: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    
    # Timing
    timing_points: List[OsuTimingPoint] = field(default_factory=list)
    
    # Objects
    hit_objects: List[OsuHitObject] = field(default_factory=list)


# ============================================================
# osu! 解析器
# ============================================================

class OsuParser:
    """解析 .osu 文件"""
    
    def parse(self, filepath: str) -> OsuBeatmap:
        """解析 .osu 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        beatmap = OsuBeatmap()
        
        # 解析格式版本
        header_match = re.match(r'osu file format v(\d+)', content)
        if header_match:
            beatmap.format_version = int(header_match.group(1))
        
        # 分段解析
        sections = self._split_sections(content)
        
        for section_name, section_content in sections.items():
            if section_name == 'General':
                self._parse_general(section_content, beatmap)
            elif section_name == 'Metadata':
                self._parse_metadata(section_content, beatmap)
            elif section_name == 'Difficulty':
                self._parse_difficulty(section_content, beatmap)
            elif section_name == 'TimingPoints':
                self._parse_timing_points(section_content, beatmap)
            elif section_name == 'HitObjects':
                self._parse_hit_objects(section_content, beatmap)
        
        return beatmap
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """按 [SectionName] 分割文件"""
        sections = {}
        current_section = None
        current_lines = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('//'):
                continue
            
            # 检查是否是 section 标题
            section_match = re.match(r'^\[(\w+)\]$', line)
            if section_match:
                if current_section:
                    sections[current_section] = '\n'.join(current_lines)
                current_section = section_match.group(1)
                current_lines = []
            elif current_section:
                current_lines.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_lines)
        
        return sections
    
    def _parse_general(self, content: str, beatmap: OsuBeatmap):
        """解析 [General] 段"""
        for line in content.split('\n'):
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            
            if key == 'AudioFilename':
                beatmap.audio_filename = value
            elif key == 'Mode':
                beatmap.mode = int(value)
            elif key == 'AudioLeadIn':
                beatmap.audio_lead_in = int(value)
    
    def _parse_metadata(self, content: str, beatmap: OsuBeatmap):
        """解析 [Metadata] 段"""
        for line in content.split('\n'):
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            
            if key == 'Title':
                beatmap.title = value
            elif key == 'TitleUnicode':
                beatmap.title_unicode = value
            elif key == 'Artist':
                beatmap.artist = value
            elif key == 'ArtistUnicode':
                beatmap.artist_unicode = value
            elif key == 'Creator':
                beatmap.creator = value
            elif key == 'Version':
                beatmap.version = value
            elif key == 'Tags':
                beatmap.tags = value
    
    def _parse_difficulty(self, content: str, beatmap: OsuBeatmap):
        """解析 [Difficulty] 段"""
        for line in content.split('\n'):
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            
            if key == 'HPDrainRate':
                beatmap.hp_drain = float(value)
            elif key == 'CircleSize':
                beatmap.circle_size = float(value)
            elif key == 'OverallDifficulty':
                beatmap.overall_difficulty = float(value)
            elif key == 'ApproachRate':
                beatmap.approach_rate = float(value)
            elif key == 'SliderMultiplier':
                beatmap.slider_multiplier = float(value)
            elif key == 'SliderTickRate':
                beatmap.slider_tick_rate = float(value)
    
    def _parse_timing_points(self, content: str, beatmap: OsuBeatmap):
        """解析 [TimingPoints] 段"""
        for line in content.split('\n'):
            parts = line.split(',')
            if len(parts) < 2:
                continue
            
            time = float(parts[0])
            beat_length = float(parts[1])
            meter = int(parts[2]) if len(parts) > 2 else 4
            uninherited = int(parts[6]) if len(parts) > 6 else 1
            
            beatmap.timing_points.append(OsuTimingPoint(
                time=time,
                beat_length=beat_length,
                meter=meter,
                uninherited=(uninherited == 1)
            ))
    
    def _parse_hit_objects(self, content: str, beatmap: OsuBeatmap):
        """解析 [HitObjects] 段"""
        for line in content.split('\n'):
            parts = line.split(',')
            if len(parts) < 5:
                continue
            
            x = float(parts[0])
            y = float(parts[1])
            time = float(parts[2])
            obj_type = int(parts[3])
            hit_sound = int(parts[4])
            
            obj = OsuHitObject(
                x=x, y=y, time=time,
                obj_type=obj_type, hit_sound=hit_sound
            )
            
            # 解析附加参数
            if len(parts) > 5:
                extras_str = ','.join(parts[5:])
                self._parse_object_extras(obj, extras_str, beatmap.mode)
            
            beatmap.hit_objects.append(obj)
    
    def _parse_object_extras(self, obj: OsuHitObject, extras: str, mode: int):
        """解析 HitObject 附加参数"""
        parts = extras.split(',')
        
        if mode == 3:
            # osu!mania: 格式 x,y,time,type,hitSound,endTime:extras
            # 对于长按 (type & 128): endTime 在 extras 第一个参数，用冒号分隔
            if obj.obj_type & 128:  # 长按
                end_time_part = parts[0].split(':')[0]
                obj.end_time = float(end_time_part)
                if len(parts) > 1:
                    obj.extras['extras'] = ','.join(parts[1:])
            elif len(parts) > 0:
                obj.extras['extras'] = parts[0]
        else:
            # osu!standard / osu!taiko: 处理滑条长度
            if obj.obj_type & 2:  # 滑条
                # 滑条格式: curveType|curvePoints,slides,length,...
                if len(parts) >= 3:
                    obj.extras['slider_length'] = float(parts[2])  # 像素长度
                    obj.extras['slider_curve'] = parts[0]
                    obj.extras['slides'] = int(parts[1])
                    # end_time 延迟到转换时计算 (需要 BPM 信息)
            elif obj.obj_type & 8:  # Spinner
                if len(parts) > 0:
                    obj.end_time = float(parts[0])


# ============================================================
# 转换器: osu! → ChartData
# ============================================================

class OsuConverter:
    """
    将 osu! 谱面转换为 ChartData
    
    支持:
    - osu!mania (mode=3): 列 → positionX, 长按 → hold
    - osu!standard (mode=0): 简化映射
    """
    
    def __init__(self, max_notes_per_segment: int = 1000):
        self.max_notes_per_segment = max_notes_per_segment
    
    def convert(self, beatmap: OsuBeatmap) -> ChartData:
        """将 osu!Beatmap 转换为 ChartData"""
        chart = ChartData()
        
        # 设置元数据
        chart.musicLength = self._estimate_music_length(beatmap)
        chart.beatSubdivision = 4
        
        # 转换 BPM
        if beatmap.timing_points:
            bpm = self._get_base_bpm(beatmap)
            chart.add_bpm(bpm)
        else:
            chart.add_bpm(120.0)
        
        # 根据模式选择转换策略
        if beatmap.mode == 3:
            segments = self._convert_mania(beatmap)
        elif beatmap.mode == 2:
            segments = self._convert_taiko(beatmap)
        else:
            segments = self._convert_standard(beatmap)
        
        chart.judgeSegments = segments
        
        return chart
    
    def _get_base_bpm(self, beatmap: OsuBeatmap) -> float:
        """获取基础 BPM"""
        for tp in beatmap.timing_points:
            if tp.uninherited and tp.beat_length > 0:
                return 60000.0 / tp.beat_length
        return 120.0
    
    def _ms_to_beats(self, time_ms: float, timing_points: List[OsuTimingPoint]) -> float:
        """将毫秒转换为拍数（基于 TimingPoints）"""
        if not timing_points:
            return time_ms / 500.0  # 默认 120 BPM
        
        # 找到最近的 BPM 点
        current_bpm = 120.0
        current_beat_length = 500.0
        base_time = 0.0
        base_beat = 0.0
        
        for tp in sorted(timing_points, key=lambda x: x.time):
            if tp.time > time_ms:
                break
            
            if tp.uninherited and tp.beat_length > 0:
                # 计算到上一个 BPM 点之间的拍数
                if current_beat_length > 0:
                    beats_elapsed = (tp.time - base_time) / current_beat_length
                    base_beat += beats_elapsed
                base_time = tp.time
                current_beat_length = tp.beat_length
                current_bpm = 60000.0 / tp.beat_length
        
        # 计算最终拍数
        if current_beat_length > 0:
            beats_elapsed = (time_ms - base_time) / current_beat_length
            return base_beat + beats_elapsed
        
        return time_ms / 500.0
    
    def _get_slider_end_time(self, obj: OsuHitObject, timing_points: List[OsuTimingPoint], 
                              slider_multiplier: float) -> float:
        """
        计算滑条结束时间 (ms)
        
        公式: duration = slider_length / (100 * SliderMultiplier * velocity_mult) * beat_length
        """
        if 'slider_length' not in obj.extras:
            return obj.time
        
        slider_length = obj.extras['slider_length']
        slides = obj.extras.get('slides', 1)
        
        # 找到当前时刻的 BPM 和速度倍率
        current_beat_length = 500.0  # 默认 120 BPM
        velocity_mult = 1.0  # 默认正常速度
        
        for tp in sorted(timing_points, key=lambda x: x.time):
            if tp.time > obj.time:
                break
            if tp.uninherited and tp.beat_length > 0:
                current_beat_length = tp.beat_length
            elif not tp.uninherited and tp.beat_length < 0:
                # 继承时间点: beat_length = -100 * velocity_mult
                velocity_mult = -100.0 / tp.beat_length
        
        # 滑条速度 (像素/拍)
        sv = 100.0 * slider_multiplier * velocity_mult
        
        if sv <= 0:
            return obj.time
        
        # 滑条持续拍数
        beats_for_slider = slider_length / sv
        
        # 滑条持续时间 (ms)
        duration_ms = beats_for_slider * current_beat_length
        
        return obj.time + duration_ms * slides
    
    def _estimate_music_length(self, beatmap: OsuBeatmap) -> float:
        """估算音乐时长 (秒)"""
        if not beatmap.hit_objects:
            return 180.0
        
        max_time = max(obj.end_time for obj in beatmap.hit_objects) or \
                   max(obj.time for obj in beatmap.hit_objects)
        return (max_time + 5000) / 1000.0  # 加 5 秒余量，转秒
    
    def _convert_mania(self, beatmap: OsuBeatmap) -> List[JudgeSegment]:
        """
        转换 osu!mania 谱面
        
        映射:
        - 列号 (0~CS-1) → positionX (归一化到 -1~1)
        - Circle (type & 1) → Tap
        - Hold (type & 128) → Hold
        - 时间 (ms) → beats (拍数)
        """
        notes = []
        timing_points = beatmap.timing_points
        num_columns = max(int(beatmap.circle_size), 4)  # 最少 4 键
        
        for obj in beatmap.hit_objects:
            # 时间转换 (ms → beats)
            time_beats = self._ms_to_beats(obj.time, timing_points)
            
            # 列 → positionX
            column = int(obj.x) % num_columns
            position_x = (column / (num_columns - 1)) * 2 - 1 if num_columns > 1 else 0.0
            
            # 确定音符类型
            if obj.obj_type & 128:  # 长按
                note_type = NoteType.HOLD
                end_time_beats = self._ms_to_beats(obj.end_time, timing_points)
                hold_time = end_time_beats - time_beats
            elif obj.obj_type & 1:  # 单击
                note_type = NoteType.TAP
                hold_time = 0.0
            else:
                continue  # 跳过其他类型
            
            note = Note(
                type=int(note_type),
                time=float(time_beats),
                holdTime=float(hold_time),
                positionX=float(position_x),
                isAbove=True,
                isFakeNote=False,
                offset=0.0,
                hasOther=False
            )
            notes.append(note)
        
        # 排序
        notes.sort(key=lambda n: n.time)
        
        # 标记同拍其他音
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                if abs(notes[j].time - notes[i].time) < 0.05:
                    notes[i].hasOther = True
                    notes[j].hasOther = True
                else:
                    break
        
        # 分配到段落 (简单起见，全部放入一个段落)
        segment = JudgeSegment(notesAbove=notes)
        return [segment]
    
    def _convert_taiko(self, beatmap: OsuBeatmap) -> List[JudgeSegment]:
        """
        转换 osu!taiko 谱面 (mode=2)
        
        osu!taiko 映射:
        - 所有音符都是 Tap (taiko 没有长按)
        - X 坐标决定 Don(左半) / Kat(右半)
        - Circle (type & 1) → Tap
        - Slider (type & 2) → Tap (taiko 中滑条也是单点)
        - Big Don (type & 5) / Big Kat (type & 6) → 普通 Tap（位置信息保留）
        - 滑条长度/重复次数 → 忽略
        """
        notes = []
        timing_points = beatmap.timing_points
        
        for obj in beatmap.hit_objects:
            time_beats = self._ms_to_beats(obj.time, timing_points)
            
            # X坐标归一化 (0~512 → -1~1)
            position_x = max(-1.0, min(1.0, (obj.x - 256) / 256))
            
            # 判断 Don (左半) / Kat (右半) - 用 positionX 体现
            # 所有音符均为 Tap
            note = Note(
                type=int(NoteType.TAP),
                time=float(time_beats),
                holdTime=0.0,
                positionX=float(position_x),
                isAbove=True,
                isFakeNote=False,
                offset=0.0,
                hasOther=False
            )
            notes.append(note)
        
        notes.sort(key=lambda n: n.time)
        
        # 标记同拍其他音
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                if abs(notes[j].time - notes[i].time) < 0.05:
                    notes[i].hasOther = True
                    notes[j].hasOther = True
                else:
                    break
        
        segment = JudgeSegment(notesAbove=notes)
        return [segment]
    
    def _convert_standard(self, beatmap: OsuBeatmap) -> List[JudgeSegment]:
        """
        转换 osu!standard 谱面
        
        标准模式是 2D 瞄准，到 1D 轨道映射有信息损失。
        映射策略: 将 X 坐标作为 positionX (中心为0)
        """
        notes = []
        timing_points = beatmap.timing_points
        
        for obj in beatmap.hit_objects:
            time_beats = self._ms_to_beats(obj.time, timing_points)
            
            # X 坐标归一化到 -1~1 (假设 512 宽度)
            position_x = max(-1.0, min(1.0, (obj.x - 256) / 256))
            
            # 确定音符类型
            if obj.obj_type & 2:  # 滑条
                note_type = NoteType.HOLD
                end_time_beats = self._ms_to_beats(obj.end_time, timing_points)
                hold_time = end_time_beats - time_beats
            elif obj.obj_type & 1:  # 单击
                note_type = NoteType.TAP
                hold_time = 0.0
            elif obj.obj_type & 8:  # 转盘
                note_type = NoteType.HOLD
                end_time_beats = self._ms_to_beats(obj.end_time, timing_points)
                hold_time = end_time_beats - time_beats
                position_x = 0.0  # 居中
            else:
                continue
            
            # 根据 Y 坐标判断上下轨道
            is_above = obj.y < 192  # 屏幕上半部分为上方
            
            note = Note(
                type=int(note_type),
                time=float(time_beats),
                holdTime=float(hold_time),
                positionX=float(position_x),
                isAbove=is_above,
                isFakeNote=False,
                offset=0.0,
                hasOther=False
            )
            notes.append(note)
        
        notes.sort(key=lambda n: n.time)
        
        segment = JudgeSegment(notesAbove=[n for n in notes if n.isAbove],
                                notesBelow=[n for n in notes if not n.isAbove])
        return [segment]


# ============================================================
# 命令行工具
# ============================================================

def convert_file(input_path: str, output_dir: str = None, no_audio: bool = False) -> str:
    """
    转换单个 .osu 文件
    
    Args:
        input_path: .osu 文件路径
        output_dir: 输出目录 (默认: data/processed/)
        no_audio: 不复制音频文件
        
    Returns:
        输出文件路径
    """
    input_path = Path(input_path)
    if input_path.suffix.lower() != '.osu':
        raise ValueError(f"不是 .osu 文件: {input_path}")
    
    output_dir = Path(output_dir) if output_dir else \
                 PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析
    parser = OsuParser()
    beatmap = parser.parse(str(input_path))
    
    # 转换
    converter = OsuConverter()
    chart = converter.convert(beatmap)
    
    # 生成输出文件名
    mode_label = "mania" if beatmap.mode == 3 else "std"
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', beatmap.title)
    safe_version = re.sub(r'[<>:"/\\|?*]', '_', beatmap.version)
    output_name = f"{safe_title} - {safe_version} ({mode_label}).json"
    output_path = output_dir / output_name
    
    # 保存
    chart.save_to_file(str(output_path))
    
    # 复制音频文件
    audio_path = None
    if not no_audio and beatmap.audio_filename:
        audio_dir = output_dir.parent / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_src = input_path.parent / beatmap.audio_filename
        if audio_src.exists():
            audio_dst = audio_dir / f"{safe_title} - {safe_version} ({mode_label}){audio_src.suffix}"
            import shutil
            shutil.copy2(str(audio_src), str(audio_dst))
            audio_path = str(audio_dst)
            print(f"  🎵 音频已复制: {audio_dst}")
    
    # 打印统计
    notes = chart.get_all_notes()
    print(f"  📊 谱面: {beatmap.title} [{beatmap.version}]")
    print(f"    模式: {'osu!mania' if beatmap.mode == 3 else 'osu!standard'}")
    print(f"    音符数: {len(notes)}")
    print(f"    BPM: {chart.default_bpm:.1f}")
    print(f"    时长: {chart.musicLength:.1f}s")
    print(f"    输出: {output_path}")
    
    return str(output_path)


def convert_directory(input_dir: str, output_dir: str = None, 
                      no_audio: bool = False, max_files: int = 0):
    """
    转换目录下所有 .osu 文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        no_audio: 不复制音频文件
        max_files: 最大文件数 (0=不限)
    """
    input_dir = Path(input_dir)
    osu_files = list(input_dir.rglob("*.osu"))
    
    if max_files > 0:
        osu_files = osu_files[:max_files]
    
    if not osu_files:
        print(f"❌ 在 {input_dir} 中未找到 .osu 文件")
        return
    
    print(f"找到 {len(osu_files)} 个 .osu 文件，开始转换...")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for i, osu_file in enumerate(osu_files, 1):
        try:
            print(f"[{i}/{len(osu_files)}] {osu_file.name}")
            convert_file(str(osu_file), output_dir, no_audio)
            success += 1
        except Exception as e:
            print(f"  ❌ 转换失败: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"✅ 转换完成! 成功: {success}, 失败: {failed}")
    
    # 统计
    output_path = Path(output_dir) if output_dir else PROJECT_ROOT / 'data' / 'processed'
    json_files = list(output_path.glob("*.json"))
    total_notes = 0
    for jf in json_files:
        try:
            c = ChartData.load_from_file(str(jf))
            total_notes += c.total_notes
        except:
            pass
    
    print(f"  输出目录: {output_path}")
    print(f"  生成文件: {len(json_files)}")
    print(f"  总音符数: {total_notes}")


def main():
    parser = argparse.ArgumentParser(
        description="osu! 谱面 → ArellanoDreamWeaver 格式转换器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  python scripts/converters/osu_converter.py --input song.osu
  
  # 批量转换目录
  python scripts/converters/osu_converter.py --dir ./osu_maps/ --output data/processed/
  
  # 批量转换（不复制音频）
  python scripts/converters/osu_converter.py --dir ./osu_maps/ --no-audio
  
  # 限制转换数量
  python scripts/converters/osu_converter.py --dir ./osu_maps/ --max 10
        """
    )
    
    parser.add_argument('--input', '-i', type=str, help='输入的 .osu 文件路径')
    parser.add_argument('--dir', '-d', type=str, help='输入目录 (批量转换)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出目录 (默认: data/processed/)')
    parser.add_argument('--no-audio', action='store_true',
                        help='不复制音频文件')
    parser.add_argument('--max', type=int, default=0,
                        help='最大转换文件数 (0=不限)')
    
    args = parser.parse_args()
    
    if args.input:
        result = convert_file(args.input, args.output, args.no_audio)
        print(f"\n✅ 转换完成: {result}")
    elif args.dir:
        convert_directory(args.dir, args.output, args.no_audio, args.max)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()