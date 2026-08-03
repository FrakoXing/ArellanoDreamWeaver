"""
ArellanoDreamWeaver - 谱面数据结构定义
======================================
完全兼容 ArellanoDreamEdit 的 ChartData JSON 格式
支持 Unity AnimationCurve 和多层事件系统
"""
import json
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from enum import IntEnum
import numpy as np


class NoteType(IntEnum):
    """音符类型 (与 DreamEdit 完全一致)"""
    TAP = 0       # 单击
    HOLD = 1      # 长按
    SLIDE = 2     # 滑动


@dataclass
class CurveKey:
    """
    Unity AnimationCurve 关键帧
    """
    time: float = 0.0
    value: float = 0.0
    inTangent: float = 0.0
    outTangent: float = 0.0
    inWeight: float = 0.0
    outWeight: float = 0.0
    weightedMode: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurveKey':
        if not isinstance(data, dict):
            return cls()
        return cls(
            time=float(data.get("time", 0.0)),
            value=float(data.get("value", 0.0)),
            inTangent=float(data.get("inTangent", 0.0)),
            outTangent=float(data.get("outTangent", 0.0)),
            inWeight=float(data.get("inWeight", 0.0)),
            outWeight=float(data.get("outWeight", 0.0)),
            weightedMode=int(data.get("weightedMode", 0))
        )


@dataclass
class AnimationCurve:
    """
    Unity AnimationCurve (用于事件和 far 曲线)
    """
    keys: List[CurveKey] = field(default_factory=list)
    preWrapMode: int = 8
    postWrapMode: int = 8
    
    @classmethod
    def from_dict(cls, data: Any) -> Optional['AnimationCurve']:
        if not isinstance(data, dict):
            return None
        keys_data = data.get("keys", [])
        keys = [CurveKey.from_dict(k) if isinstance(k, dict) else CurveKey() for k in keys_data]
        return cls(
            keys=keys,
            preWrapMode=int(data.get("preWrapMode", 8)),
            postWrapMode=int(data.get("postWrapMode", 8))
        )


@dataclass
class Event:
    """
    段落事件 (兼容 Unity 的 Event 格式)
    
    使用 StartTime/EndTime 和 StartValue/EndValue 表示
    线性插值，可选的 AnimationCurve 用于非线性的过渡
    """
    StartTime: float = 0.0
    EndTime: float = 0.0
    StartValue: float = 0.0
    EndValue: float = 0.0
    Curve: Optional[AnimationCurve] = None
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        result = {
            "StartTime": self.StartTime,
            "EndTime": self.EndTime,
            "StartValue": self.StartValue,
            "EndValue": self.EndValue
        }
        if self.Curve:
            result["Curve"] = {
                "keys": [{"time": k.time, "value": k.value, "inTangent": k.inTangent,
                          "outTangent": k.outTangent, "inWeight": k.inWeight,
                          "outWeight": k.outWeight, "weightedMode": k.weightedMode}
                         for k in self.Curve.keys],
                "preWrapMode": self.Curve.preWrapMode,
                "postWrapMode": self.Curve.postWrapMode
            }
        return result
    
    @classmethod
    def from_dict(cls, data: Any) -> 'Event':
        if isinstance(data, dict):
            return cls(
                StartTime=float(data.get("StartTime", 0.0)),
                EndTime=float(data.get("EndTime", 0.0)),
                StartValue=float(data.get("StartValue", 0.0)),
                EndValue=float(data.get("EndValue", 0.0)),
                Curve=AnimationCurve.from_dict(data.get("Curve"))
            )
        # 如果数据格式不对，返回空事件
        return cls()


@dataclass
class SegmentEvents:
    """
    段落事件集合 (兼容 Unity 多层事件系统)
    
    每个事件类型是 List[List[Event]]:
    - 外层列表 = 多个轨道 (layers)
    - 内层列表 = 该轨道上的事件序列
    同一时刻所有轨道的当前值相加得到最终值。
    """
    MoveX: List[List[Event]] = field(default_factory=list)
    MoveY: List[List[Event]] = field(default_factory=list)
    Rotate: List[List[Event]] = field(default_factory=list)
    Alpha: List[List[Event]] = field(default_factory=list)
    Length: List[List[Event]] = field(default_factory=list)
    Speed: List[List[Event]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, list]:
        """转换为字典 (用于 JSON 序列化)"""
        return {
            "MoveX": [[e.to_dict() for e in layer] for layer in self.MoveX],
            "MoveY": [[e.to_dict() for e in layer] for layer in self.MoveY],
            "Rotate": [[e.to_dict() for e in layer] for layer in self.Rotate],
            "Alpha": [[e.to_dict() for e in layer] for layer in self.Alpha],
            "Length": [[e.to_dict() for e in layer] for layer in self.Length],
            "Speed": [[e.to_dict() for e in layer] for layer in self.Speed]
        }
    
    @classmethod
    def from_dict(cls, data: Any) -> 'SegmentEvents':
        if not isinstance(data, dict):
            return cls()
        
        seg = cls()
        for event_type in ['MoveX', 'MoveY', 'Rotate', 'Alpha', 'Length', 'Speed']:
            raw = data.get(event_type, [])
            if isinstance(raw, list):
                layers = []
                for layer in raw:
                    if isinstance(layer, list):
                        # 标准格式: List[List[Event]]
                        events = [Event.from_dict(e) for e in layer if isinstance(e, dict)]
                        layers.append(events)
                    elif isinstance(layer, dict):
                        # 单层格式: List[Event] (非标准但兼容)
                        layers.append([Event.from_dict(layer)])
                setattr(seg, event_type, layers)
            # 其他类型保持默认空列表
        return seg


@dataclass
class BoundaryEvents:
    """
    边界事件集合 (与 SegmentEvents 结构相同)
    """
    MoveX: List[List[Event]] = field(default_factory=list)
    MoveY: List[List[Event]] = field(default_factory=list)
    Rotate: List[List[Event]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, list]:
        """转换为字典"""
        return {
            "MoveX": [[e.to_dict() for e in layer] for layer in self.MoveX],
            "MoveY": [[e.to_dict() for e in layer] for layer in self.MoveY],
            "Rotate": [[e.to_dict() for e in layer] for layer in self.Rotate]
        }
    
    @classmethod
    def from_dict(cls, data: Any) -> 'BoundaryEvents':
        if not isinstance(data, dict):
            return cls()
        be = cls()
        for event_type in ['MoveX', 'MoveY', 'Rotate']:
            raw = data.get(event_type, [])
            if isinstance(raw, list):
                layers = []
                for layer in raw:
                    if isinstance(layer, list):
                        events = [Event.from_dict(e) for e in layer if isinstance(e, dict)]
                        layers.append(events)
                    elif isinstance(layer, dict):
                        layers.append([Event.from_dict(layer)])
                setattr(be, event_type, layers)
        return be


@dataclass
class BPM:
    """
    BPM 数据结构 (兼容 DreamEdit 的分数存储方式)
    """
    num: int = 120000       # 默认 BPM = 120.0 (存储为 120000/1000)
    den: int = 1000         # 默认精度分母
    
    @property
    def value(self) -> float:
        return self.num / self.den if self.den != 0 else 0.0
    
    @classmethod
    def from_float(cls, value: float, denominator: int = 1000) -> 'BPM':
        num = int(value * denominator)
        return cls(num=num, den=denominator)
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典 (用于 JSON 序列化)"""
        return {"num": self.num, "den": self.den}
    
    @classmethod
    def from_dict(cls, data: Any) -> 'BPM':
        if isinstance(data, dict):
            return cls(num=int(data.get("num", 120000)), den=int(data.get("den", 1000)))
        return cls()


@dataclass
class Note:
    """
    音符数据结构 (完全兼容 ArellanoDreamEdit)
    """
    type: int = NoteType.TAP          # 0=Tap, 1=Hold, 2=Slide
    time: float = 0.0                 # 时间位置
    holdTime: float = 0.0             # 长按时长
    positionX: float = 0.0           # X位置 [-1, 1]
    isFakeNote: bool = False          # 是否假音
    offset: float = 0.0               # 偏移量
    isAbove: bool = True              # true=上方, false=下方
    hasOther: bool = False            # 同时间是否有其他音
    
    # 运行时属性 (不参与序列化)
    floorPosition: float = -0.1
    holdEndFloorPosition: float = -0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于 JSON 序列化)"""
        return {
            "type": self.type,
            "time": self.time,
            "holdTime": self.holdTime,
            "positionX": self.positionX,
            "isFakeNote": self.isFakeNote,
            "offset": self.offset,
            "isAbove": self.isAbove,
            "hasOther": self.hasOther
        }
    
    @classmethod
    def from_dict(cls, data: Any) -> 'Note':
        if isinstance(data, dict):
            return cls(
                type=int(data.get("type", 0)),
                time=float(data.get("time", 0.0)),
                holdTime=float(data.get("holdTime", 0.0)),
                positionX=float(data.get("positionX", 0.0)),
                isFakeNote=bool(data.get("isFakeNote", False)),
                offset=float(data.get("offset", 0.0)),
                isAbove=bool(data.get("isAbove", True)),
                hasOther=bool(data.get("hasOther", False))
            )
        return cls()
    
    @property
    def note_type_name(self) -> str:
        names = {0: "Tap", 1: "Hold", 2: "Slide"}
        return names.get(self.type, "Unknown")
    
    def __repr__(self):
        return (f"Note(type={self.note_type_name}, "
                f"time={self.time:.3f}, pos={self.positionX:.2f}, "
                f"above={self.isAbove})")


@dataclass
class JudgeSegment:
    """
    判定段落
    """
    segmentEvents: SegmentEvents = field(default_factory=SegmentEvents)
    notesAbove: List[Note] = field(default_factory=list)
    notesBelow: List[Note] = field(default_factory=list)
    far: Optional[AnimationCurve] = None
    
    @property
    def all_notes(self) -> List[Note]:
        return self.notesAbove + self.notesBelow
    
    @property
    def note_count(self) -> int:
        return len(self.notesAbove) + len(self.notesBelow)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于 JSON 序列化)"""
        return {
            "segmentEvents": self.segmentEvents.to_dict() if self.segmentEvents else {},
            "notesAbove": [n.to_dict() for n in self.notesAbove],
            "notesBelow": [n.to_dict() for n in self.notesBelow],
            "far": None  # 简化: 不序列化 far 曲线
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JudgeSegment':
        if not isinstance(data, dict):
            return cls()
        
        # 解析事件
        seg_events = SegmentEvents.from_dict(data.get("segmentEvents"))
        
        # 解析音符
        notes_above = []
        for n in data.get("notesAbove", []):
            if isinstance(n, dict):
                notes_above.append(Note.from_dict(n))
        
        notes_below = []
        for n in data.get("notesBelow", []):
            if isinstance(n, dict):
                notes_below.append(Note.from_dict(n))
        
        # 解析 far (AnimationCurve)
        far = AnimationCurve.from_dict(data.get("far"))
        
        return cls(
            segmentEvents=seg_events,
            notesAbove=notes_above,
            notesBelow=notes_below,
            far=far
        )


@dataclass
class Boundary:
    """
    边界点 (兼容 ArellanoDreamEdit)
    """
    BoundaryEvents: BoundaryEvents = field(default_factory=BoundaryEvents)
    StartTime: float = 0.0
    EndTime: float = 0.0
    DefaultX: float = 0.0
    DefaultY: float = 0.0
    Direction: bool = False
    IncludedSegments: Optional[Any] = None
    includedSegmentIndices: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "BoundaryEvents": self.BoundaryEvents.to_dict() if self.BoundaryEvents else {},
            "StartTime": self.StartTime,
            "EndTime": self.EndTime,
            "DefaultX": self.DefaultX,
            "DefaultY": self.DefaultY,
            "Direction": self.Direction,
            "IncludedSegments": self.IncludedSegments,
            "includedSegmentIndices": self.includedSegmentIndices
        }
    
    @classmethod
    def from_dict(cls, data: Any) -> 'Boundary':
        if not isinstance(data, dict):
            return cls()
        return cls(
            BoundaryEvents=BoundaryEvents.from_dict(data.get("BoundaryEvents")),
            StartTime=float(data.get("StartTime", 0.0)),
            EndTime=float(data.get("EndTime", 0.0)),
            DefaultX=float(data.get("DefaultX", 0.0)),
            DefaultY=float(data.get("DefaultY", 0.0)),
            Direction=bool(data.get("Direction", False)),
            IncludedSegments=data.get("IncludedSegments"),
            includedSegmentIndices=[int(i) for i in data.get("includedSegmentIndices", [])]
        )


# ============================================================
# 主数据结构: ChartData
# ============================================================

@dataclass
class ChartData:
    """
    完整的谱面数据 (编辑格式 - beats域)
    
    与 ArellanoDreamEdit 完全兼容的主数据结构。
    """
    
    judgeSegments: List[JudgeSegment] = field(default_factory=list)
    BoundaryList: List[Boundary] = field(default_factory=list)
    bpmList: List[BPM] = field(default_factory=list)
    
    offset: float = 0.0
    musicLength: float = 0.0
    beatSubdivision: int = 4
    
    @property
    def total_notes(self) -> int:
        return sum(seg.note_count for seg in self.judgeSegments)
    
    @property
    def total_segments(self) -> int:
        return len(self.judgeSegments)
    
    def get_all_notes(self) -> List[Note]:
        all_notes = []
        for seg in self.judgeSegments:
            all_notes.extend(seg.all_notes)
        all_notes.sort(key=lambda n: n.time)
        return all_notes
    
    def get_notes_by_type(self, note_type: NoteType) -> List[Note]:
        return [n for n in self.get_all_notes() if n.type == note_type]
    
    def add_note(self, note: Note, segment_index: int = -1):
        if not self.judgeSegments:
            self.judgeSegments.append(JudgeSegment())
        idx = segment_index if segment_index >= 0 else len(self.judgeSegments) - 1
        if note.isAbove:
            self.judgeSegments[idx].notesAbove.append(note)
        else:
            self.judgeSegments[idx].notesBelow.append(note)
    
    def add_bpm(self, bpm_value: float, beat_position: float = 0.0):
        new_bpm = BPM.from_float(bpm_value)
        self.bpmList.append(new_bpm)
    
    @property
    def default_bpm(self) -> float:
        if self.bpmList:
            return self.bpmList[0].value
        return 120.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartData':
        if not isinstance(data, dict):
            return cls()
        
        segments = []
        for seg_data in data.get("judgeSegments", []):
            segments.append(JudgeSegment.from_dict(seg_data))
        
        boundaries = []
        for b_data in data.get("BoundaryList", []):
            boundaries.append(Boundary.from_dict(b_data))
        
        bpms = []
        for b_data in data.get("bpmList", []):
            bpms.append(BPM.from_dict(b_data))
        
        return cls(
            judgeSegments=segments,
            BoundaryList=boundaries,
            bpmList=bpms,
            offset=float(data.get("offset", 0.0)),
            musicLength=float(data.get("musicLength", 0.0)),
            beatSubdivision=int(data.get("beatSubdivision", 4))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于 JSON 序列化)"""
        return {
            "judgeSegments": [s.to_dict() for s in self.judgeSegments],
            "BoundaryList": [b.to_dict() for b in self.BoundaryList],
            "bpmList": [b.to_dict() for b in self.bpmList],
            "offset": self.offset,
            "musicLength": self.musicLength,
            "beatSubdivision": self.beatSubdivision
        }
    
    def to_json_string(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json_string(cls, json_str: str) -> 'ChartData':
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json_string())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ChartData':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json_string(f.read())
    
    def get_summary(self) -> str:
        notes = self.get_all_notes()
        tap_count = sum(1 for n in notes if n.type == NoteType.TAP)
        hold_count = sum(1 for n in notes if n.type == NoteType.HOLD)
        slide_count = sum(1 for n in notes if n.type == NoteType.SLIDE)
        
        summary = [
            f"📊 谱面统计",
            f"  总音符数: {len(notes)}",
            f"  - Tap: {tap_count} ({tap_count/max(len(notes),1)*100:.1f}%)",
            f"  - Hold: {hold_count} ({hold_count/max(len(notes),1)*100:.1f}%)",
            f"  - Slide: {slide_count} ({slide_count/max(len(notes),1)*100:.1f}%)",
            f"  段落数: {len(self.judgeSegments)}",
            f"  BPM变化点: {len(self.bpmList)} (默认: {self.default_bpm})",
            f"  音乐时长: {self.musicLength:.1f}s"
        ]
        return "\n".join(summary)


# ============================================================
# 张量表示 (用于 Diffusion 模型)
# ============================================================

def chart_to_tensor(chart: ChartData, max_len: int = 4096, feature_dim: int = 8) -> np.ndarray:
    """
    将 ChartData 转换为张量表示 (供 Diffusion 模型使用)
    
    输出形状: [max_len, feature_dim]
    
    特征维度:
    - 0: note_type (0=none, 0.33=tap, 0.66=hold, 1.0=slide)
    - 1: position_x (-1 到 1 归一化到 0~1)
    - 2: is_above (0 或 1)
    - 3: hold_time (归一化)
    - 4: intensity (基于周围音符密度)
    - 5: is_fake (0 或 1)
    - 6: has_other (0 或 1)
    - 7: 时间间隔 (距离上一个音符的相对距离)
    """
    tensor = np.zeros((max_len, feature_dim), dtype=np.float32)
    
    notes = chart.get_all_notes()
    if not notes:
        return tensor
    
    max_hold_time = max((n.holdTime for n in notes if n.holdTime > 0), default=1.0)
    max_time = max(n.time for n in notes) if notes else 1.0
    
    prev_time = 0.0
    for i, note in enumerate(notes[:max_len]):
        idx = min(i, max_len - 1)
        
        tensor[idx, 0] = note.type / 3.0
        tensor[idx, 1] = (note.positionX + 1) / 2
        tensor[idx, 2] = 1.0 if note.isAbove else 0.0
        tensor[idx, 3] = note.holdTime / max(max_hold_time, 1e-6)
        
        time_gap = note.time - prev_time
        tensor[idx, 4] = np.clip(1.0 / (time_gap + 0.1), 0, 5) / 5.0
        tensor[idx, 5] = 1.0 if note.isFakeNote else 0.0
        tensor[idx, 6] = 1.0 if note.hasOther else 0.0
        tensor[idx, 7] = time_gap / max(max_time, 1e-6)
        
        prev_time = note.time
    
    return tensor


def tensor_to_chart(tensor: np.ndarray, base_bpm: float = 120.0) -> ChartData:
    """
    将张量表示转回 ChartData (逆向操作)
    """
    chart = ChartData()
    chart.add_bpm(base_bpm)
    
    for i in range(tensor.shape[0]):
        row = tensor[i]
        
        note_type_val = row[0]
        if note_type_val < 0.15:
            continue
        
        if note_type_val < 0.5:
            note_type = NoteType.TAP
        elif note_type_val < 0.83:
            note_type = NoteType.HOLD
        else:
            note_type = NoteType.SLIDE
        
        position_x = row[1] * 2 - 1
        is_above = row[2] > 0.5
        hold_time = row[3]
        
        note = Note(
            type=int(note_type),
            time=float(i),
            holdTime=float(hold_time * 4),
            positionX=float(position_x),
            isAbove=bool(is_above)
        )
        
        chart.add_note(note)
    
    return chart


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("=" * 60)
    print("ArellanoDreamWeaver - Chart Data Structures Test")
    print("=" * 60)
    
    # 测试加载实际谱面文件
    test_file = Path(__file__).parent.parent.parent / "data" / "processed" / "Chart.json"
    if test_file.exists():
        print(f"\n加载实际谱面: {test_file}")
        try:
            chart = ChartData.load_from_file(str(test_file))
            print(f"✅ 加载成功!")
            print(chart.get_summary())
            
            # 测试 tensor 转换
            tensor = chart_to_tensor(chart, max_len=4096)
            print(f"\n张量形状: {tensor.shape}")
            non_zero = np.count_nonzero(tensor)
            print(f"非零元素: {non_zero} ({non_zero/tensor.size*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"测试文件不存在: {test_file}")
        print("使用默认测试...")
        
        chart = ChartData(
            offset=0.0,
            musicLength=180.0,
            beatSubdivision=4
        )
        chart.add_bpm(128.0)
        
        test_notes = [
            Note(type=NoteType.TAP, time=0.0, positionX=-0.5, isAbove=True),
            Note(type=NoteType.TAP, time=1.0, positionX=0.0, isAbove=False),
            Note(type=NoteType.HOLD, time=2.0, holdTime=2.0, positionX=0.5, isAbove=True),
            Note(type=NoteType.SLIDE, time=4.0, positionX=-0.8, isAbove=False),
            Note(type=NoteType.TAP, time=4.5, positionX=0.8, isAbove=True),
        ]
        
        for note in test_notes:
            chart.add_note(note)
        
        print("\n谱面摘要:")
        print(chart.get_summary())
        
        tensor = chart_to_tensor(chart, max_len=256)
        print(f"\n张量形状: {tensor.shape}")
        print(f"非零元素: {np.count_nonzero(tensor)}")
    
    print("\n✅ 测试完成!")