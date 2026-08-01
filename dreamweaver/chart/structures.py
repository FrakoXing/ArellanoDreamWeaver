"""
ArellanoDreamWeaver - 谱面数据结构定义
======================================
完全兼容 ArellanoDreamEdit 的 ChartData JSON 格式
提供 Python 端的数据类、验证、转换工具
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
class BPM:
    """
    BPM 数据结构 (兼容 DreamEdit 的分数存储方式)
    
    使用分数避免浮点精度丢失:
    - _numerator: 总拍号的分子 (整数部分*分母 + 余数分子)
    - _denominator: 分母
    """
    num: int = 120000       # 默认 BPM = 120.0 (存储为 120000/1000)
    den: int = 1000         # 默认精度分母
    
    @property
    def value(self) -> float:
        """获取实际 BPM 浮点值"""
        return self.num / self.den if self.den != 0 else 0.0
    
    @value.setter
    def value(self, v: float):
        """设置 BPM 值，自动转为分数"""
        # 使用合理的默认精度
        self.denominator = 1000
        self._numerator = int(v * 1000)
    
    @property
    def integer(self) -> int:
        """整数部分"""
        return abs(self.num) // self.den
    
    @property
    def molecule(self) -> int:
        """余数部分的分子"""
        return abs(self.num) % self.den
    
    @property
    def denominator(self) -> int:
        return self.den
    
    @denominator.setter
    def denominator(self, d: int):
        if d > 0:
            self.den = d
    
    @classmethod
    def from_float(cls, value: float, denominator: int = 1000) -> 'BPM':
        """从浮点数创建 BPM 对象"""
        num = int(value * denominator)
        return cls(num=num, den=denominator)
    
    def to_dict(self) -> Dict[str, int]:
        """序列化为字典 (JSON 兼容)"""
        return {"num": self.num, "den": self.den}
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'BPM':
        """从字典反序列化"""
        return cls(num=data.get("num", 120000), den=data.get("den", 1000))


@dataclass
class Note:
    """
    音符数据结构 (完全兼容 ArellanoDreamEdit)
    
    属性说明:
    - type: 0=Tap(单击), 1=Hold(长按), 2=Slide(滑动)
    - time: 时间位置 (编辑格式=拍号, 播放格式=秒数)
    - holdTime: 长按时长 (同上单位)
    - positionX: X位置 [-1, 1]
    - isFakeNote: 是否假音(装饰用,不判定)
    - offset: 偏移量
    - isAbove: true=上方轨道, false=下方轨道
    - hasOther: 同一时间是否有其他音
    """
    type: int = NoteType.TAP          # 音符类型
    time: float = 0.0                 # 时间位置
    holdTime: float = 0.0             # 长按时长
    positionX: float = 0.0           # X位置 [-1, 1]
    isFakeNote: bool = False          # 是否假音
    offset: float = 0.0               # 偏移量
    isAbove: bool = True              # true=上方, false=下方
    hasOther: bool = False            # 同时间是否有其他音
    
    # 运行时属性 (不参与序列化)
    floorPosition: float = -0.1       # 运行时地板位置
    holdEndFloorPosition: float = -0.1  # Hold尾地板位置
    
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
    def from_dict(cls, data: Dict[str, Any]) -> 'Note':
        """从字典创建 Note 对象"""
        return cls(
            type=data.get("type", 0),
            time=data.get("time", 0.0),
            holdTime=data.get("holdTime", 0.0),
            positionX=data.get("positionX", 0.0),
            isFakeNote=data.get("isFakeNote", False),
            offset=data.get("offset", 0.0),
            isAbove=data.get("isAbove", True),
            hasOther=data.get("hasOther", False)
        )
    
    @property
    def note_type_name(self) -> str:
        """获取音符类型名称"""
        names = {0: "Tap", 1: "Hold", 2: "Slide"}
        return names.get(self.type, "Unknown")
    
    def __repr__(self):
        return (f"Note(type={self.note_type_name}, "
                f"time={self.time:.3f}, pos={self.positionX:.2f}, "
                f"above={self.isAbove})")


@dataclass
class Event:
    """段落事件 (MoveX/Y, Rotate, Alpha, Length 等)"""
    beat: float = 0.0      # 事件触发节拍位置
    value: float = 0.0     # 事件值
    
    def to_dict(self) -> Dict[str, float]:
        return {"beat": self.beat, "value": self.value}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Event':
        return cls(beat=data["beat"], value=data["value"])


@dataclass
class SegmentEvents:
    """段落事件集合"""
    MoveX: List[Event] = field(default_factory=list)      # X轴位移事件
    MoveY: List[Event] = field(default_factory=list)      # Y轴位移事件
    Rotate: List[Event] = field(default_factory=list)     # 旋转事件
    Alpha: List[Event] = field(default_factory=list)      # 透明度事件
    Length: List[Event] = field(default_factory=list)     # 长度缩放事件
    
    def to_dict(self) -> Dict[str, List]:
        return {
            "MoveX": [e.to_dict() for e in self.MoveX],
            "MoveY": [e.to_dict() for e in self.MoveY],
            "Rotate": [e.to_dict() for e in self.Rotate],
            "Alpha": [e.to_dict() for e in self.Alpha],
            "Length": [e.to_dict() for e in self.Length]
        }


@dataclass
class JudgeSegment:
    """
    判定段落
    
    一个谱面由多个 JudgeSegment 组成，
    每个 segment 包含上方/下方的音符列表和该段的事件
    """
    segmentEvents: SegmentEvents = field(default_factory=SegmentEvents)
    notesAbove: List[Note] = field(default_factory=list)   # 上方轨道音符
    notesBelow: List[Note] = field(default_factory=list)   # 下方轨道音符
    far: Optional[List[float]] = None                       # far曲线点 (可选)
    
    @property
    def all_notes(self) -> List[Note]:
        """获取所有音符"""
        return self.notesAbove + self.notesBelow
    
    @property
    def note_count(self) -> int:
        """音符总数"""
        return len(self.notesAbove) + len(self.notesBelow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segmentEvents": self.segmentEvents.to_dict(),
            "notesAbove": [n.to_dict() for n in self.notesAbove],
            "notesBelow": [n.to_dict() for n in self.notesBelow],
            "far": self.far
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JudgeSegment':
        events_data = data.get("segmentEvents", {})
        
        # 解析 SegmentEvents
        seg_events = SegmentEvents()
        if isinstance(events_data, dict):
            for event_type in ['MoveX', 'MoveY', 'Rotate', 'Alpha', 'Length']:
                events_list = events_data.get(event_type, [])
                setattr(seg_events, event_type, 
                        [Event.from_dict(e) if isinstance(e, dict) else Event(**e) 
                         for e in events_list])
        
        # 解析 Notes
        notes_above = [Note.from_dict(n) if isinstance(n, dict) else Note(**n) 
                       for n in data.get("notesAbove", [])]
        notes_below = [Note.from_dict(n) if isinstance(n, dict) else Note(**n) 
                       for n in data.get("notesBelow", [])]
        
        return cls(
            segmentEvents=seg_events,
            notesAbove=notes_above,
            notesBelow=notes_below,
            far=data.get("far")
        )


@dataclass
class Boundary:
    """边界点"""
    beat: float = 0.0
    judgeLine: int = 0
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {"beat": self.beat, "judgeLine": self.judgeLine}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Boundary':
        return cls(beat=data.get("beat", 0.0), judgeLine=data.get("judgeLine", 0))


# ============================================================
# 主数据结构: ChartData
# ============================================================

@dataclass
class ChartData:
    """
    完整的谱面数据 (编辑格式 - beats域)
    
    这是与 ArellanoDreamEdit 完全兼容的主数据结构。
    所有时间单位在编辑模式下使用 BEATS (拍号)。
    """
    
    # 核心数据
    judgeSegments: List[JudgeSegment] = field(default_factory=list)
    BoundaryList: List[Boundary] = field(default_factory=list)
    bpmList: List[BPM] = field(default_factory=list)
    
    # 全局参数
    offset: float = 0.0             # 全局偏移 (秒)
    musicLength: float = 0.0        # 音乐总长度 (秒)
    beatSubdivision: int = 4         # 节拍细分 (默认4分音符)
    
    # === 便捷方法 ===
    
    @property
    def total_notes(self) -> int:
        """总音符数"""
        return sum(seg.note_count for seg in self.judgeSegments)
    
    @property
    def total_segments(self) -> int:
        """总段落数"""
        return len(self.judgeSegments)
    
    def get_all_notes(self) -> List[Note]:
        """获取所有音符 (按时间排序)"""
        all_notes = []
        for seg in self.judgeSegments:
            all_notes.extend(seg.all_notes)
        
        # 按时间排序
        all_notes.sort(key=lambda n: n.time)
        return all_notes
    
    def get_notes_by_type(self, note_type: NoteType) -> List[Note]:
        """按类型筛选音符"""
        return [n for n in self.get_all_notes() if n.type == note_type]
    
    def add_note(self, note: Note, segment_index: int = -1):
        """添加音符到指定段落"""
        if not self.judgeSegments:
            self.judgeSegments.append(JudgeSegment())
        
        idx = segment_index if segment_index >= 0 else len(self.judgeSegments) - 1
        
        if note.isAbove:
            self.judgeSegments[idx].notesAbove.append(note)
        else:
            self.judgeSegments[idx].notesBelow.append(note)
    
    def add_bpm(self, bpm_value: float, beat_position: float = 0.0):
        """添加 BPM 变化点"""
        new_bpm = BPM.from_float(bpm_value)
        self.bpmList.append(new_bpm)
    
    @property
    def default_bpm(self) -> float:
        """获取默认 BPM (第一个 BPM 点)"""
        if self.bpmList:
            return self.bpmList[0].value
        return 120.0  # 默认 120 BPM
    
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartData':
        """从字典创建 ChartData (从 JSON 反序列化)"""
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
            offset=data.get("offset", 0.0),
            musicLength=data.get("musicLength", 0.0),
            beatSubdivision=data.get("beatSubdivision", 4)
        )
    
    def to_json_string(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json_string(cls, json_str: str) -> 'ChartData':
        """从 JSON 字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str):
        """保存到 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json_string())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ChartData':
        """从 JSON 文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json_string(f.read())
    
    def get_summary(self) -> str:
        """生成谱面摘要信息"""
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
    
    # 计算归一化参数
    max_hold_time = max((n.holdTime for n in notes if n.holdTime > 0), default=1.0)
    max_time = max(n.time for n in notes) if notes else 1.0
    
    prev_time = 0.0
    for i, note in enumerate(notes[:max_len]):
        # 截断到 max_len
        idx = min(i, max_len - 1)
        
        tensor[idx, 0] = note.type / 3.0  # 音符类型
        tensor[idx, 1] = (note.positionX + 1) / 2  # X位置归一化
        tensor[idx, 2] = 1.0 if note.isAbove else 0.0  # 上方/下方
        tensor[idx, 3] = note.holdTime / max(max_hold_time, 1e-6)  # Hold时长
        
        # 强度 (简化版: 基于时间间隔)
        time_gap = note.time - prev_time
        tensor[idx, 4] = np.clip(1.0 / (time_gap + 0.1), 0, 5) / 5.0  # 密度指标
        tensor[idx, 5] = 1.0 if note.isFakeNote else 0.0
        tensor[idx, 6] = 1.0 if note.hasOther else 0.0
        tensor[idx, 7] = time_gap / max(max_time, 1e-6)  # 相对时间间隔
        
        prev_time = note.time
    
    return tensor


def tensor_to_chart(tensor: np.ndarray, base_bpm: float = 120.0) -> ChartData:
    """
    将张量表示转回 ChartData (逆向操作)
    
    这是一个有损转换，主要用于可视化生成的结果
    """
    chart = ChartData()
    chart.add_bpm(base_bpm)
    
    max_time = tensor.shape[0]
    
    for i in range(tensor.shape[0]):
        row = tensor[i]
        
        note_type_val = row[0]
        if note_type_val < 0.15:
            continue  # 无音符
        
        # 反归一化
        if note_type_val < 0.5:
            note_type = NoteType.TAP
        elif note_type_val < 0.83:
            note_type = NoteType.HOLD
        else:
            note_type = NoteType.SLIDE
        
        position_x = row[1] * 2 - 1  # 反归一化 [-1, 1]
        is_above = row[2] > 0.5
        hold_time = row[3]  # 已经是归一化的
        
        note = Note(
            type=int(note_type),
            time=float(i) / max_time * 100.0,  # 简单映射到 100 拍
            holdTime=float(hold_time * 4),       # 假设最大 4 拍
            positionX=float(position_x),
            isAbove=bool(is_above)
        )
        
        chart.add_note(note)
    
    return chart


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ArellanoDreamWeaver - Chart Data Structures Test")
    print("=" * 60)
    
    # 创建示例谱面
    chart = ChartData(
        offset=0.0,
        musicLength=180.0,
        beatSubdivision=4
    )
    
    # 添加 BPM
    chart.add_bpm(128.0)
    
    # 添加一些测试音符
    test_notes = [
        Note(type=NoteType.TAP, time=0.0, positionX=-0.5, isAbove=True),
        Note(type=NoteType.TAP, time=1.0, positionX=0.0, isAbove=False),
        Note(type=NoteType.HOLD, time=2.0, holdTime=2.0, positionX=0.5, isAbove=True),
        Note(type=NoteType.SLIDE, time=4.0, positionX=-0.8, isAbove=False),
        Note(type=NoteType.TAP, time=4.5, positionX=0.8, isAbove=True),
    ]
    
    for note in test_notes:
        chart.add_note(note)
    
    # 打印信息
    print("\n原始谱面:")
    print(chart.get_summary())
    print(f"\n音符列表:")
    for note in chart.get_all_notes():
        print(f"  {note}")
    
    # JSON 序列化 & 反序列化
    print("\n--- JSON 序列化测试 ---")
    json_str = chart.to_json_string()
    print(f"JSON 长度: {len(json_str)} 字符")
    print(f"JSON 前200字符:\n{json_str[:200]}...")
    
    chart_loaded = ChartData.from_json_string(json_str)
    assert chart_loaded.total_notes == chart.total_notes
    print(f"\n✅ JSON 往返转换成功! 音符数: {chart_loaded.total_notes}")
    
    # 张量转换
    print("\n--- 张量表示测试 ---")
    tensor = chart_to_tensor(chart, max_len=256)
    print(f"张量形状: {tensor.shape}")
    print(f"非零元素数: {np.count_nonzero(tensor)}")
    
    # 逆转换
    chart_recovered = tensor_to_chart(tensor)
    print(f"恢复的谱面音符数: {chart_recovered.total_notes}")
    
    print("\n✅ 所有测试通过!")
