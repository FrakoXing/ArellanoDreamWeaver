#!/usr/bin/env python3
"""
统一谱面 → Token 序列转换器
============================

将 PEC / osu! / Arellano 三种谱面格式直接转换为模型可用的 Token 序列，
跳过 ChartData JSON 中间步骤。

设计思路:
  每种格式的音符 → 统一的中间表示 (Note) → 离散 Token 序列

Token 词汇表 (与 NoteDecoder 对齐):
  0: PAD    1: SOS    2: EOS    3: MASK
  4: TAP    5: HOLD   6: SLIDE
  7-70:  时间偏移 (相对上一音符, 0~15.75拍, 0.25拍精度)
  71-79: X位置 (9档: -1.0, -0.75, ..., 0, ..., 0.75, 1.0)
  80-111: Hold时长 (0~7.75拍, 0.25拍精度)
  112: ABOVE (音符在判定线正面)
  113: BELOW (音符在判定线背面)
  116: GAME_PHIRA   (Phira/Phigros 来源)
  117: GAME_OSU     (osu! 来源)
  118: GAME_ARELLANO (Arellano 来源)

用法:
  # 转换单个 PEC 文件
  python unified_tokenizer.py chart.pec

  # 转换 osu! 文件
  python unified_tokenizer.py song.osu

  # 转换 Arellano JSON
  python unified_tokenizer.py chart.json

  # 批量转换目录
  python unified_tokenizer.py --dir ./charts/ --outdir data/tokens/

  # 输出为 PyTorch .pt 文件 (可直接用于训练)
  python unified_tokenizer.py chart.pec --format pt

依赖: numpy, torch (可选, 仅 --format pt 需要)
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union


# ============================================================
# Token 常量 (与 NoteDecoder 完全对齐)
# ============================================================

PAD = 0
SOS = 1
EOS = 2
MASK = 3

TYPE_TAP = 4
TYPE_HOLD = 5
TYPE_SLIDE = 6

TIME_OFFSET_START = 7
TIME_OFFSET_END = 70     # 64 个值, 0~15.75 拍, 步长 0.25

POS_X_START = 71
POS_X_END = 79           # 9 个值: -1.0, -0.75, ..., 1.0

HOLD_TIME_START = 80
HOLD_TIME_END = 111      # 32 个值, 0~7.75 拍, 步长 0.25

NOTE_ABOVE = 112
NOTE_BELOW = 113

# 游戏来源 Token (扩展)
GAME_PHIRA = 116
GAME_OSU = 117
GAME_ARELLANO = 118

VOCAB_SIZE = 128  # 与 NoteDecoder.DEFAULT_VOCAB_SIZE 一致

GAME_TOKEN_MAP = {
    "phira": GAME_PHIRA,
    "pec": GAME_PHIRA,      # PEC 是 Phira/Phigros 格式
    "osu": GAME_OSU,
    "arellano": GAME_ARELLANO,
}


# ============================================================
# 统一音符结构
# ============================================================

@dataclass
class UniNote:
    """统一音符表示 - 所有格式解析后都转成这个"""
    type: int             # 0=Tap, 1=Hold, 2=Slide
    time: float           # 拍数 (绝对)
    hold_time: float      # Hold 时长 (拍)
    position_x: float     # X 位置 (-1.0 ~ 1.0)
    is_above: bool        # 是否在判定线正面
    is_fake: bool = False # 是否假音符


# ============================================================
# Tokenizer: UniNote → Token 序列
# ============================================================

class ChartTokenizer:
    """将音符列表编码为 Token 序列"""

    TIME_STEP = 0.25       # 时间量化步长 (拍)
    TIME_MAX = 15.75       # 最大时间偏移
    POS_BINS = 9           # 位置档数
    HOLD_STEP = 0.25       # Hold 时长量化步长
    HOLD_MAX = 7.75        # 最大 Hold 时长

    def __init__(self, game: str = "phira", max_time_delta: float = 15.75):
        self.game = game
        self.game_token = GAME_TOKEN_MAP.get(game, GAME_PHIRA)
        self.max_time_delta = max_time_delta

    def quantize_time(self, delta: float) -> int:
        """量化时间偏移 → Token ID"""
        steps = round(delta / self.TIME_STEP)
        steps = max(0, min(steps, int(self.TIME_MAX / self.TIME_STEP)))
        return TIME_OFFSET_START + steps

    def quantize_pos(self, x: float) -> int:
        """量化 X 位置 → Token ID"""
        # -1.0 → 71, -0.75 → 72, ..., 0 → 75, ..., 1.0 → 79
        normalized = (x + 1.0) / 2.0  # 0 ~ 1
        idx = round(normalized * (self.POS_BINS - 1))
        idx = max(0, min(idx, self.POS_BINS - 1))
        return POS_X_START + idx

    def quantize_hold(self, hold: float) -> int:
        """量化 Hold 时长 → Token ID"""
        steps = round(hold / self.HOLD_STEP)
        steps = max(0, min(steps, int(self.HOLD_MAX / self.HOLD_STEP)))
        return HOLD_TIME_START + steps

    def type_token(self, note_type: int) -> int:
        """音符类型 → Token ID"""
        return TYPE_TAP + note_type  # 0→4, 1→5, 2→6

    def encode(self, notes: List[UniNote], include_game_token: bool = True) -> List[int]:
        """
        将音符列表编码为 Token 序列

        每个音符编码为:
          [TYPE] [TIME_DELTA] [POS_X] [HOLD_TIME?] [ABOVE/BELOW]

        Returns:
            List[int]: Token ID 列表
        """
        tokens = []

        # 游戏来源 Token
        if include_game_token:
            tokens.append(self.game_token)

        # SOS
        tokens.append(SOS)

        prev_time = 0.0

        for note in notes:
            if note.is_fake:
                continue  # 跳过假音符

            # 1. 音符类型
            tokens.append(self.type_token(note.type))

            # 2. 时间偏移 (相对于上一音符)
            delta = max(0.0, note.time - prev_time)
            # 如果偏移超过最大值, 插入多个最大偏移 Token
            while delta > self.max_time_delta:
                tokens.append(self.quantize_time(self.max_time_delta))
                delta -= self.max_time_delta
                prev_time += self.max_time_delta
            tokens.append(self.quantize_time(delta))

            # 3. X 位置
            tokens.append(self.quantize_pos(note.position_x))

            # 4. Hold 时长 (仅 Hold/Slide)
            if note.type in (1, 2) and note.hold_time > 0:
                tokens.append(self.quantize_hold(note.hold_time))

            # 5. 方向
            tokens.append(NOTE_ABOVE if note.is_above else NOTE_BELOW)

            # 更新 prev_time
            prev_time = note.time

        # EOS
        tokens.append(EOS)

        return tokens

    def decode(self, tokens: List[int]) -> List[UniNote]:
        """从 Token 序列解码回音符列表 (用于验证)"""
        notes = []
        i = 0

        # 跳过游戏 Token
        if i < len(tokens) and tokens[i] >= GAME_PHIRA:
            i += 1

        # 跳过 SOS
        if i < len(tokens) and tokens[i] == SOS:
            i += 1

        prev_time = 0.0

        while i < len(tokens):
            if tokens[i] == EOS:
                break
            if tokens[i] == PAD:
                i += 1
                continue

            # 类型
            if tokens[i] in (TYPE_TAP, TYPE_HOLD, TYPE_SLIDE):
                note_type = tokens[i] - TYPE_TAP
                i += 1
            else:
                i += 1
                continue

            # 时间偏移
            if i < len(tokens) and TIME_OFFSET_START <= tokens[i] <= TIME_OFFSET_END:
                delta = (tokens[i] - TIME_OFFSET_START) * self.TIME_STEP
                i += 1
            else:
                delta = 0.0

            time = prev_time + delta

            # X 位置
            if i < len(tokens) and POS_X_START <= tokens[i] <= POS_X_END:
                pos_idx = tokens[i] - POS_X_START
                pos_x = (pos_idx / (self.POS_BINS - 1)) * 2.0 - 1.0
                i += 1
            else:
                pos_x = 0.0

            # Hold 时长
            hold_time = 0.0
            if note_type in (1, 2) and i < len(tokens) and HOLD_TIME_START <= tokens[i] <= HOLD_TIME_END:
                hold_time = (tokens[i] - HOLD_TIME_START) * self.HOLD_STEP
                i += 1

            # 方向
            is_above = True
            if i < len(tokens) and tokens[i] in (NOTE_ABOVE, NOTE_BELOW):
                is_above = (tokens[i] == NOTE_ABOVE)
                i += 1

            notes.append(UniNote(
                type=note_type,
                time=time,
                hold_time=hold_time,
                position_x=pos_x,
                is_above=is_above,
            ))

            prev_time = time

        return notes


# ============================================================
# PEC 解析器 (Phigros / Phira)
# ============================================================

class PECParser:
    """
    解析 PEC (PhiEditer Chart) 格式

    格式参考: https://pgrfm.miraheze.org/wiki/PEC

    结构:
      第一行: offset (毫秒, 需减 175)
      bp beat bpm           - BPM 变化
      cp line beat x y      - 判定线位置事件
      cm line start end x y easing - 位置运动
      cd line beat degree   - 旋转事件
      cr line start end degree easing - 旋转运动
      ca line beat alpha    - 透明度事件
      cv line beat speed    - 速度事件
      n1 line beat x dir fake - Tap
      n2 line start end x dir fake - Hold
      n3 line beat x dir fake - Flick
      n4 line beat x dir fake - Drag
      # speed               - 音符速度
      & width               - 音符宽度
    """

    def parse(self, text: str) -> Tuple[List[UniNote], List[Dict]]:
        """
        解析 PEC 文本

        Returns:
            (notes, bpm_changes)
        """
        lines = text.strip().split('\n')
        if not lines:
            return [], []

        # 第一行: offset
        try:
            raw_offset = int(lines[0].strip())
            offset_ms = raw_offset - 175  # PEC 约定
        except (ValueError, IndexError):
            offset_ms = 0

        notes = []
        bpm_changes = [{"beat": 0.0, "bpm": 120.0}]

        i = 1
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith('//'):
                continue

            parts = line.split()
            if not parts:
                continue

            cmd = parts[0]

            try:
                # BPM 变化
                if cmd == 'bp' and len(parts) >= 3:
                    beat = float(parts[1])
                    bpm = float(parts[2])
                    bpm_changes.append({"beat": beat, "bpm": bpm})

                # Tap 音符
                elif cmd == 'n1' and len(parts) >= 5:
                    # n1 line beat x direction fake
                    beat = float(parts[2])
                    x_raw = float(parts[3])
                    direction = int(parts[4])
                    is_fake = int(parts[5]) if len(parts) > 5 else 0

                    # x 范围 ±1024, 归一化到 -1~1
                    pos_x = max(-1.0, min(1.0, x_raw / 1024.0))
                    is_above = (direction == 1)

                    notes.append(UniNote(
                        type=0,  # Tap
                        time=beat,
                        hold_time=0.0,
                        position_x=pos_x,
                        is_above=is_above,
                        is_fake=(is_fake == 1),
                    ))

                # Hold 音符
                elif cmd == 'n2' and len(parts) >= 6:
                    # n2 line startBeat endBeat x direction fake
                    start_beat = float(parts[2])
                    end_beat = float(parts[3])
                    x_raw = float(parts[4])
                    direction = int(parts[5])
                    is_fake = int(parts[6]) if len(parts) > 6 else 0

                    pos_x = max(-1.0, min(1.0, x_raw / 1024.0))
                    is_above = (direction == 1)

                    notes.append(UniNote(
                        type=1,  # Hold
                        time=start_beat,
                        hold_time=max(0.0, end_beat - start_beat),
                        position_x=pos_x,
                        is_above=is_above,
                        is_fake=(is_fake == 1),
                    ))

                # Flick 音符
                elif cmd == 'n3' and len(parts) >= 5:
                    beat = float(parts[2])
                    x_raw = float(parts[3])
                    direction = int(parts[4])
                    is_fake = int(parts[5]) if len(parts) > 5 else 0

                    pos_x = max(-1.0, min(1.0, x_raw / 1024.0))
                    is_above = (direction == 1)

                    notes.append(UniNote(
                        type=2,  # Flick → Slide
                        time=beat,
                        hold_time=0.0,
                        position_x=pos_x,
                        is_above=is_above,
                        is_fake=(is_fake == 1),
                    ))

                # Drag 音符
                elif cmd == 'n4' and len(parts) >= 5:
                    beat = float(parts[2])
                    x_raw = float(parts[3])
                    direction = int(parts[4])
                    is_fake = int(parts[5]) if len(parts) > 5 else 0

                    pos_x = max(-1.0, min(1.0, x_raw / 1024.0))
                    is_above = (direction == 1)

                    notes.append(UniNote(
                        type=0,  # Drag → Tap
                        time=beat,
                        hold_time=0.0,
                        position_x=pos_x,
                        is_above=is_above,
                        is_fake=(is_fake == 1),
                    ))

                # 跳过 # 和 & 行 (速度/宽度修饰符, 属于上一个音符)
                elif cmd in ('#', '&'):
                    pass

                # 其他事件 (cp/cm/cd/cr/ca/cf/cv) 忽略
                # 这些是判定线事件, 不影响音符数据

            except (ValueError, IndexError):
                continue

        # 按时间排序
        notes.sort(key=lambda n: n.time)

        return notes, bpm_changes

    def parse_file(self, filepath: str) -> Tuple[List[UniNote], List[Dict]]:
        """解析 PEC 文件"""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return self.parse(f.read())


# ============================================================
# osu! 解析器
# ============================================================

class OsuParser:
    """
    解析 osu! (.osu) 格式

    支持:
    - osu!mania (mode=3): 列→positionX, 长按→Hold
    - osu!standard (mode=0): X→positionX, 滑条→Slide, 圆圈→Tap
    """

    def parse(self, filepath: str) -> Tuple[List[UniNote], Dict[str, Any]]:
        """解析 .osu 文件"""
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

        mode = 0
        circle_size = 4.0
        slider_multiplier = 1.4
        timing_points = []  # [(time_ms, beat_length, uninherited), ...]
        hit_objects = []    # [(x, y, time_ms, type_flags, end_time_ms), ...]
        current_section = ""

        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            # 章节标题
            m = re.match(r'^\[(\w+)\]$', line)
            if m:
                current_section = m.group(1)
                continue

            if current_section == 'General':
                if ':' in line:
                    k, _, v = line.partition(':')
                    if k.strip() == 'Mode':
                        mode = int(v.strip()) if v.strip() else 0

            elif current_section == 'Difficulty':
                if ':' in line:
                    k, _, v = line.partition(':')
                    k = k.strip()
                    v = v.strip()
                    if k == 'CircleSize':
                        circle_size = float(v) if v else 4.0
                    elif k == 'SliderMultiplier':
                        slider_multiplier = float(v) if v else 1.4

            elif current_section == 'TimingPoints':
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        bl = float(parts[1])
                        uninherited = (int(parts[6]) == 1) if len(parts) > 6 else True
                        timing_points.append((t, bl, uninherited))
                    except (ValueError, IndexError):
                        pass

            elif current_section == 'HitObjects':
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        x = int(parts[0])
                        y = int(parts[1])
                        t = float(parts[2])
                        flags = int(parts[3])
                        end_time = 0.0

                        # mania 长按
                        if flags & 128 and len(parts) > 5:
                            end_time = float(parts[5].split(':')[0])
                        # slider
                        elif flags & 2 and len(parts) > 7:
                            # 计算 slider 结束时间
                            slider_length = float(parts[7])
                            slides = int(parts[6])
                            # 找到当前 BPM
                            bpm = 120.0
                            for tp_t, tp_bl, tp_un in timing_points:
                                if tp_t > t:
                                    break
                                if tp_un and tp_bl > 0:
                                    bpm = 60000.0 / tp_bl
                            beat_len_ms = 60000.0 / bpm if bpm > 0 else 500.0
                            if slider_multiplier > 0:
                                duration_ms = (slider_length / (slider_multiplier * 100.0)) * beat_len_ms * slides
                                end_time = t + duration_ms
                        # spinner
                        elif flags & 8 and len(parts) > 5:
                            end_time = float(parts[5])

                        hit_objects.append((x, y, t, flags, end_time))
                    except (ValueError, IndexError):
                        pass

        # 构建 beat map
        beat_map = self._build_beat_map(timing_points)

        # 转换音符
        notes = []
        num_keys = max(1, int(round(circle_size)))

        for x, y, t_ms, flags, end_ms in hit_objects:
            time_beats = self._ms_to_beats(t_ms, beat_map)

            if mode == 3:
                # osu!mania
                column = int(x) % num_keys
                pos_x = (column / (num_keys - 1)) * 2 - 1 if num_keys > 1 else 0.0

                if flags & 128:
                    end_beats = self._ms_to_beats(end_ms, beat_map)
                    notes.append(UniNote(
                        type=1, time=time_beats,
                        hold_time=max(0.0, end_beats - time_beats),
                        position_x=pos_x, is_above=True,
                    ))
                elif flags & 1:
                    notes.append(UniNote(
                        type=0, time=time_beats,
                        hold_time=0.0,
                        position_x=pos_x, is_above=True,
                    ))
            else:
                # osu!standard
                pos_x = max(-1.0, min(1.0, (x - 256) / 256.0))
                is_above = y < 192

                if flags & 2:
                    # slider → Slide
                    end_beats = self._ms_to_beats(end_ms, beat_map)
                    notes.append(UniNote(
                        type=2, time=time_beats,
                        hold_time=max(0.0, end_beats - time_beats),
                        position_x=pos_x, is_above=is_above,
                    ))
                elif flags & 8:
                    # spinner → Slide (长hold)
                    end_beats = self._ms_to_beats(end_ms, beat_map)
                    notes.append(UniNote(
                        type=2, time=time_beats,
                        hold_time=max(0.0, end_beats - time_beats),
                        position_x=0.0, is_above=True,
                    ))
                elif flags & 1:
                    notes.append(UniNote(
                        type=0, time=time_beats,
                        hold_time=0.0,
                        position_x=pos_x, is_above=is_above,
                    ))

        notes.sort(key=lambda n: n.time)

        meta = {
            "mode": mode,
            "circle_size": circle_size,
            "slider_multiplier": slider_multiplier,
        }

        return notes, meta

    def _build_beat_map(self, timing_points):
        """构建 ms→beats 映射"""
        uninherited = [(t, bl) for t, bl, u in timing_points if u and bl > 0]
        if not uninherited:
            return [(0.0, 0.0, 120.0)]

        beat_map = []
        acc_beats = 0.0
        prev_time = 0.0

        for i, (t, bl) in enumerate(uninherited):
            bpm = 60000.0 / bl
            if i == 0:
                if t > 0:
                    beat_map.append((0.0, 0.0, bpm))
                beat_map.append((t, acc_beats, bpm))
                prev_time = t
            else:
                dt = t - prev_time
                prev_bpm = beat_map[-1][2]
                acc_beats += dt * prev_bpm / 60000.0
                beat_map.append((t, acc_beats, bpm))
                prev_time = t

        return beat_map

    def _ms_to_beats(self, time_ms, beat_map):
        """ms → beats"""
        for i in range(len(beat_map) - 1, -1, -1):
            if time_ms >= beat_map[i][0]:
                base_t, base_b, bpm = beat_map[i]
                return base_b + (time_ms - base_t) * bpm / 60000.0
        _, base_b, bpm = beat_map[0]
        return base_b + (time_ms - beat_map[0][0]) * bpm / 60000.0


# ============================================================
# Phira JSON 解析器
# ============================================================

class PhiraJsonParser:
    """
    解析 Phira 的 chart.json 格式
    
    格式参考: https://pgrfm.miraheze.org/wiki/PEZ
    
    Phira JSON 结构:
    {
      "BPMList": [{"beat": 0.0, "bpm": 120.0}, ...],
      "META": {"offset": 0.0, ...},
      "judgeLineList": [
        {
          "notesAbove": [{"type": 1, "time": 0.0, "positionX": 0.0, ...}, ...],
          "notesBelow": [...]
        },
        ...
      ]
    }
    """
    
    def parse(self, filepath: str) -> Tuple[List[UniNote], Dict]:
        """解析 Phira JSON 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        notes = []
        
        # 获取 offset (毫秒)
        offset_ms = 0.0
        if 'META' in data:
            offset_ms = float(data['META'].get('offset', 0.0))
        
        # 解析判定线列表
        judge_lines = data.get('judgeLineList', [])
        
        for line_data in judge_lines:
            # 解析上方音符
            for nd in line_data.get('notesAbove', []):
                note = self._parse_note(nd, is_above=True)
                if note:
                    notes.append(note)
            
            # 解析下方音符
            for nd in line_data.get('notesBelow', []):
                note = self._parse_note(nd, is_above=False)
                if note:
                    notes.append(note)
        
        notes.sort(key=lambda n: n.time)
        
        meta = {
            "offset_ms": offset_ms,
            "bpm_list": data.get('BPMList', []),
        }
        
        return notes, meta
    
    def _parse_note(self, nd: Dict, is_above: bool) -> Optional[UniNote]:
        """解析单个音符"""
        try:
            note_type_raw = int(nd.get('type', 1))
            
            # Phira 音符类型:
            # 1 = Tap
            # 2 = Hold
            # 3 = Flick
            # 4 = Drag
            type_map = {1: 0, 2: 1, 3: 2, 4: 0}  # Tap, Hold, Slide, Drag→Tap
            note_type = type_map.get(note_type_raw, 0)
            
            time = float(nd.get('time', 0.0))
            position_x = float(nd.get('positionX', 0.0)) / 1024.0  # 归一化
            position_x = max(-1.0, min(1.0, position_x))
            
            # Hold 时长
            hold_time = 0.0
            if note_type_raw == 2:  # Hold
                end_time = float(nd.get('endTime', time))
                hold_time = max(0.0, end_time - time)
            
            # 是否假音符
            is_fake = bool(nd.get('isFake', 0))
            
            return UniNote(
                type=note_type,
                time=time,
                hold_time=hold_time,
                position_x=position_x,
                is_above=is_above,
                is_fake=is_fake,
            )
        except (ValueError, TypeError):
            return None


# ============================================================
# Arellano JSON 解析器
# ============================================================

class ArellanoParser:
    """
    解析 Arellano 原生 ChartData JSON 格式

    支持两种格式:
    1. ChartData JSON (带 judgeSegments)
    2. 简化 JSON (带 notes 列表)
    """

    def parse(self, filepath: str) -> Tuple[List[UniNote], Dict]:
        """解析 Arellano JSON 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        notes = []

        if 'judgeSegments' in data:
            # ChartData 格式
            for seg in data['judgeSegments']:
                for nd in seg.get('notesAbove', []):
                    notes.append(self._note_from_dict(nd, is_above=True))
                for nd in seg.get('notesBelow', []):
                    notes.append(self._note_from_dict(nd, is_above=False))
        elif 'notes' in data:
            # 简化格式
            for nd in data['notes']:
                is_above = nd.get('isAbove', True)
                notes.append(self._note_from_dict(nd, is_above=is_above))
        else:
            raise ValueError("无法识别的 JSON 格式 (需要 judgeSegments 或 notes)")

        notes.sort(key=lambda n: n.time)
        return notes, {}

    def _note_from_dict(self, nd: Dict, is_above: bool = True) -> UniNote:
        """从字典创建 UniNote"""
        return UniNote(
            type=int(nd.get('type', 0)),
            time=float(nd.get('time', 0.0)),
            hold_time=float(nd.get('holdTime', 0.0)),
            position_x=float(nd.get('positionX', 0.0)),
            is_above=is_above,
            is_fake=bool(nd.get('isFakeNote', False)),
        )


# ============================================================
# 自动检测 & 统一入口
# ============================================================

def detect_format(filepath: str) -> str:
    """根据文件扩展名和内容检测格式"""
    ext = Path(filepath).suffix.lower()

    if ext == '.osu':
        return 'osu'
    elif ext == '.pec':
        return 'pec'
    elif ext == '.pez':
        return 'pez'  # ZIP 包, 需要解压
    elif ext == '.json':
        # 尝试区分 Phira JSON 和 Arellano JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(2000)  # 读取前 2000 字符
                if 'judgeLineList' in content or 'BPMList' in content:
                    return 'phira_json'
                elif 'judgeSegments' in content or 'notes' in content:
                    return 'arellano'
                else:
                    return 'phira_json'  # 默认尝试 Phira 格式
        except Exception:
            return 'phira_json'
    else:
        # 尝试内容检测
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                first_line = f.readline().strip()
                if first_line.startswith('osu file format'):
                    return 'osu'
                # PEC 第一行是数字 (offset)
                try:
                    int(first_line)
                    return 'pec'
                except ValueError:
                    pass
        except Exception:
            pass

    raise ValueError(f"无法识别文件格式: {filepath}")


def parse_chart(filepath: str, fmt: str = None) -> Tuple[List[UniNote], str, Dict]:
    """
    解析谱面文件

    Args:
        filepath: 文件路径
        fmt: 强制指定格式 ('pec', 'osu', 'arellano', 'phira_json')

    Returns:
        (notes, game_name, meta)
    """
    if fmt is None:
        fmt = detect_format(filepath)

    if fmt == 'pec':
        parser = PECParser()
        notes, bpm_changes = parser.parse_file(filepath)
        return notes, 'phira', {"bpm_changes": bpm_changes}

    elif fmt == 'phira_json':
        parser = PhiraJsonParser()
        notes, meta = parser.parse(filepath)
        return notes, 'phira', meta

    elif fmt == 'osu':
        parser = OsuParser()
        notes, meta = parser.parse(filepath)
        return notes, 'osu', meta

    elif fmt == 'arellano':
        parser = ArellanoParser()
        notes, meta = parser.parse(filepath)
        return notes, 'arellano', meta

    else:
        raise ValueError(f"不支持的格式: {fmt}")


# ============================================================
# 输出
# ============================================================

def save_tokens(tokens: List[int], output_path: str, fmt: str = "json"):
    """保存 Token 序列"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"tokens": tokens, "length": len(tokens)}, f)

    elif fmt == "pt":
        try:
            import torch
            tensor = torch.tensor(tokens, dtype=torch.long)
            torch.save(tensor, output_path)
        except ImportError:
            print("[WARN] torch 未安装, 回退到 JSON 格式")
            save_tokens(tokens, str(output_path.with_suffix('.json')), "json")
            return

    elif fmt == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(str(t) for t in tokens))

    elif fmt == "bin":
        # 紧凑二进制: uint16 序列
        with open(output_path, 'wb') as f:
            # 写入长度头
            f.write(struct.pack('<I', len(tokens)))
            for t in tokens:
                f.write(struct.pack('<H', t))


# ============================================================
# 命令行
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="统一谱面 → Token 序列转换器 (PEC / osu! / Arellano)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件 (自动检测格式)
  python unified_tokenizer.py chart.pec
  python unified_tokenizer.py song.osu
  python unified_tokenizer.py chart.json

  # 批量转换
  python unified_tokenizer.py --dir ./charts/ --outdir data/tokens/

  # 输出为 PyTorch tensor
  python unified_tokenizer.py chart.pec --format pt

  # 输出为紧凑二进制
  python unified_tokenizer.py chart.pec --format bin

  # 验证: 编码后解码, 对比原始音符
  python unified_tokenizer.py chart.pec --verify

支持的文件格式:
  .pec  - Phigros/Phira PEC 格式
  .osu  - osu! 谱面 (mania/standard)
  .json - Phira JSON / Arellano ChartData JSON
        """
    )

    parser.add_argument('input', nargs='?', help='输入文件路径')
    parser.add_argument('--dir', '-d', help='批量转换目录')
    parser.add_argument('--outdir', '-o', default='data/tokens',
                        help='输出目录 (默认: data/tokens)')
    parser.add_argument('--format', '-f', choices=['json', 'pt', 'txt', 'bin'],
                        default='json', help='输出格式 (默认: json)')
    parser.add_argument('--game', '-g', choices=['phira', 'osu', 'arellano'],
                        help='强制指定游戏来源 (默认: 自动检测)')
    parser.add_argument('--verify', '-v', action='store_true',
                        help='编码后解码验证')
    parser.add_argument('--stats', '-s', action='store_true',
                        help='打印统计信息')

    args = parser.parse_args()

    if args.dir:
        # 批量模式
        input_dir = Path(args.dir)
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有支持的文件
        files = []
        for ext in ('*.pec', '*.osu', '*.json'):
            files.extend(input_dir.rglob(ext))
        files.sort()

        if not files:
            print(f"❌ 目录中没有找到支持的谱面文件: {args.dir}")
            sys.exit(1)

        print(f"📥 统一谱面 → Token 批量转换")
        print(f"{'=' * 60}")
        print(f"  输入目录: {input_dir.absolute()}")
        print(f"  输出目录: {out_dir.absolute()}")
        print(f"  输出格式: {args.format}")
        print(f"  文件数量: {len(files)}")
        print(f"{'=' * 60}\n")

        ok = 0
        fail = 0

        for i, fpath in enumerate(files):
            print(f"[{i+1}/{len(files)}] {fpath.name}")
            try:
                fmt = args.game if args.game else None
                notes, game, meta = parse_chart(str(fpath), fmt)

                if not notes:
                    print(f"  ⚠️  没有音符, 跳过")
                    fail += 1
                    continue

                tokenizer = ChartTokenizer(game=game)
                tokens = tokenizer.encode(notes)

                # 输出文件
                out_name = fpath.stem + f".{args.format}"
                out_path = out_dir / out_name
                save_tokens(tokens, str(out_path), args.format)

                print(f"  ✅ {game} | {len(notes)} notes → {len(tokens)} tokens")

                if args.stats:
                    tap = sum(1 for n in notes if n.type == 0)
                    hold = sum(1 for n in notes if n.type == 1)
                    slide = sum(1 for n in notes if n.type == 2)
                    print(f"     Tap:{tap} Hold:{hold} Slide:{slide}")

                ok += 1

            except Exception as e:
                print(f"  ❌ {e}")
                fail += 1

        print(f"\n{'=' * 60}")
        print(f"📊 完成! 成功: {ok}, 失败: {fail}")

    elif args.input:
        # 单文件模式
        fpath = args.input
        try:
            fmt = args.game if args.game else None
            notes, game, meta = parse_chart(fpath, fmt)

            if not notes:
                print(f"❌ 谱面中没有音符")
                sys.exit(1)

            tokenizer = ChartTokenizer(game=game)
            tokens = tokenizer.encode(notes)

            print(f"📄 文件: {fpath}")
            print(f"🎮 来源: {game}")
            print(f"🎵 音符: {len(notes)}")
            print(f"📝 Token: {len(tokens)}")

            tap = sum(1 for n in notes if n.type == 0)
            hold = sum(1 for n in notes if n.type == 1)
            slide = sum(1 for n in notes if n.type == 2)
            print(f"   Tap:{tap}  Hold:{hold}  Slide:{slide}")

            # 保存
            out_path = Path(args.outdir) / (Path(fpath).stem + f".{args.format}")
            save_tokens(tokens, str(out_path), args.format)
            print(f"💾 输出: {out_path}")

            # 验证
            if args.verify:
                decoded = tokenizer.decode(tokens)
                print(f"\n🔍 验证:")
                print(f"   原始音符: {len(notes)}")
                print(f"   解码音符: {len(decoded)}")
                if len(notes) == len(decoded):
                    max_err = 0.0
                    for orig, dec in zip(notes, decoded):
                        max_err = max(max_err, abs(orig.time - dec.time))
                        max_err = max(max_err, abs(orig.position_x - dec.position_x))
                    print(f"   最大误差: {max_err:.4f}")
                else:
                    print(f"   ⚠️  数量不匹配!")

        except Exception as e:
            print(f"❌ {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
