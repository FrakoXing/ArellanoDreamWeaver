"""
谱面编解码模块
==============
提供谱面数据的编码/解码工具
"""

import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.chart.structures import ChartData, Note, NoteType


class NoteCodec:
    """
    音符编解码器
    
    将音符数据编码为模型可用的数值表示，或从模型输出解码为音符
    """
    
    # 音符类型映射
    NOTE_TYPE_MAP = {
        0: NoteType.TAP,
        1: NoteType.HOLD,
        2: NoteType.SLIDE,
        3: None  # none/empty
    }
    
    REVERSE_TYPE_MAP = {v: k for k, v in NOTE_TYPE_MAP.items() if v is not None}
    REVERSE_TYPE_MAP[None] = 3
    
    def __init__(
        self,
        num_types: int = 4,
        position_range: tuple = (-1.0, 1.0),
        max_hold_time: float = 4.0
    ):
        """
        初始化编解码器
        
        Args:
            num_types: 音符类型数量 (包括 none)
            position_range: X位置范围
            max_hold_time: 最大 Hold 时长
        """
        self.num_types = num_types
        self.position_range = position_range
        self.max_hold_time = max_hold_time
    
    def encode_note(self, note: Note, time_offset: float = 0.0) -> np.ndarray:
        """
        编码单个音符
        
        Args:
            note: 音符对象
            time_offset: 时间偏移
            
        Returns:
            编码向量 [type, position_x, hold_time, is_above, is_fake]
        """
        # 类型编码 (one-hot)
        type_idx = self.REVERSE_TYPE_MAP.get(note.type, 3)
        type_onehot = np.zeros(self.num_types, dtype=np.float32)
        type_onehot[type_idx] = 1.0
        
        # 位置归一化
        pos_normalized = (note.positionX - self.position_range[0]) / \
                         (self.position_range[1] - self.position_range[0])
        
        # Hold 时长归一化
        hold_normalized = min(note.holdTime / self.max_hold_time, 1.0)
        
        # 组合特征
        features = np.concatenate([
            type_onehot,                    # [num_types]
            [pos_normalized],               # [1]
            [hold_normalized],              # [1]
            [float(note.isAbove)],          # [1]
            [float(note.isFakeNote)]        # [1]
        ])
        
        return features
    
    def decode_note(
        self,
        features: np.ndarray,
        time: float = 0.0
    ) -> Optional[Note]:
        """
        从特征向量解码音符
        
        Args:
            features: 特征向量
            time: 时间位置
            
        Returns:
            Note 对象，如果是 none 类型则返回 None
        """
        # 解析类型
        type_probs = features[:self.num_types]
        type_idx = np.argmax(type_probs)
        
        if type_idx == self.num_types - 1:  # none 类型
            return None
        
        note_type = self.NOTE_TYPE_MAP.get(type_idx, NoteType.TAP)
        
        # 解析位置
        pos_normalized = features[self.num_types]
        position_x = pos_normalized * (self.position_range[1] - self.position_range[0]) + \
                     self.position_range[0]
        
        # 解析 Hold 时长
        hold_normalized = features[self.num_types + 1]
        hold_time = hold_normalized * self.max_hold_time
        
        # 解析其他属性
        is_above = features[self.num_types + 2] > 0.5
        is_fake = features[self.num_types + 3] > 0.5
        
        return Note(
            type=note_type,
            time=time,
            holdTime=hold_time,
            positionX=position_x,
            isAbove=is_above,
            isFakeNote=is_fake
        )
    
    def encode_sequence(
        self,
        notes: List[Note],
        max_len: int = 4096
    ) -> np.ndarray:
        """
        编码音符序列
        
        Args:
            notes: 音符列表
            max_len: 最大序列长度
            
        Returns:
            编码矩阵 [max_len, feature_dim]
        """
        feature_dim = self.num_types + 4  # type + pos + hold + above + fake
        sequence = np.zeros((max_len, feature_dim), dtype=np.float32)
        
        for i, note in enumerate(notes[:max_len]):
            sequence[i] = self.encode_note(note)
        
        return sequence


class EventCodec:
    """
    事件编解码器
    
    处理谱面事件（MoveX, MoveY, Rotate 等）的编码/解码
    """
    
    EVENT_TYPES = ["MoveX", "MoveY", "Rotate", "Alpha", "Length"]
    
    def __init__(self, max_events_per_type: int = 128):
        """
        初始化事件编解码器
        
        Args:
            max_events_per_type: 每种事件类型的最大数量
        """
        self.max_events_per_type = max_events_per_type
    
    def encode_events(
        self,
        segment_events: Any,
        max_len: int = 128
    ) -> np.ndarray:
        """
        编码段落事件
        
        Args:
            segment_events: SegmentEvents 对象
            max_len: 最大事件数
            
        Returns:
            编码矩阵 [max_len, event_dim]
        """
        event_dim = 1 + 1 + len(self.EVENT_TYPES)  # beat + value + type_onehot
        encoded = np.zeros((max_len, event_dim), dtype=np.float32)
        
        events_list = []
        
        # 收集所有事件
        for event_type in self.EVENT_TYPES:
            events = getattr(segment_events, event_type, [])
            for event in events:
                events_list.append((event.beat, event.value, event_type))
        
        # 按时间排序
        events_list.sort(key=lambda x: x[0])
        
        # 编码
        type_to_idx = {t: i for i, t in enumerate(self.EVENT_TYPES)}
        
        for i, (beat, value, event_type) in enumerate(events_list[:max_len]):
            # 类型 one-hot
            type_idx = type_to_idx.get(event_type, 0)
            type_onehot = np.zeros(len(self.EVENT_TYPES), dtype=np.float32)
            type_onehot[type_idx] = 1.0
            
            encoded[i] = np.concatenate([
                [beat, value],  # beat + value
                type_onehot     # type one-hot
            ])
        
        return encoded
    
    def decode_events(
        self,
        encoded: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        从编码矩阵解码事件
        
        Args:
            encoded: 编码矩阵
            threshold: 事件存在阈值
            
        Returns:
            事件字典 {event_type: [{"beat": float, "value": float}, ...]}
        """
        events = {t: [] for t in self.EVENT_TYPES}
        
        for row in encoded:
            beat = row[0]
            value = row[1]
            type_probs = row[2:]
            
            # 检查是否有事件
            if np.max(type_probs) < threshold:
                continue
            
            type_idx = np.argmax(type_probs)
            event_type = self.EVENT_TYPES[type_idx]
            
            events[event_type].append({
                "beat": float(beat),
                "value": float(value)
            })
        
        return events


# ============================================================
# 便捷函数
# ============================================================

def encode_chart(
    chart: ChartData,
    max_notes: int = 4096,
    max_events: int = 128
) -> Dict[str, np.ndarray]:
    """
    编码完整谱面
    
    Args:
        chart: ChartData 对象
        max_notes: 最大音符数
        max_events: 最大事件数
        
    Returns:
        编码字典 {
            "notes": np.ndarray,
            "events": List[np.ndarray]
        }
    """
    note_codec = NoteCodec()
    event_codec = EventCodec()
    
    # 编码所有音符
    all_notes = chart.get_all_notes()
    notes_encoded = note_codec.encode_sequence(all_notes, max_notes)
    
    # 编码每段的事件
    events_encoded = []
    for segment in chart.judgeSegments:
        events = event_codec.encode_events(segment.segmentEvents, max_events)
        events_encoded.append(events)
    
    return {
        "notes": notes_encoded,
        "events": events_encoded
    }


if __name__ == "__main__":
    # 测试
    print("NoteCodec 测试:")
    codec = NoteCodec()
    
    note = Note(type=0, time=1.0, positionX=0.5, isAbove=True)
    encoded = codec.encode_note(note)
    print(f"  编码形状: {encoded.shape}")
    print(f"  编码值: {encoded}")
    
    decoded = codec.decode_note(encoded, time=1.0)
    print(f"  解码结果: {decoded}")
