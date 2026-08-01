"""
ArellanoDreamWeaver - LLM 提示词解析器
======================================
将用户的自然语言提示词转化为结构化的制谱参数
支持: OpenAI GPT-4 / 本地模型
"""
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum


class NoteStyle(Enum):
    """音符风格/类型"""
    TAP = "tap"
    HOLD = "hold"
    SLIDE = "slide"
    
class DifficultyLevel(Enum):
    """难度等级 (与 ArellanoDreamEdit 对应)"""
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"
    INSANE = "insane"
    ANOTHER = "another"
    LEGACY = "legacy"


class MusicGenre(Enum):
    """音乐流派"""
    POP = "pop"
    ROCK = "rock"
    ELECTRONIC = "electronic"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    HIPHOP = "hiphop"
    METAL = "metal"
    RNB = "rnb"
    FOLK = "folk"
    AMBIENT = "ambient"


@dataclass
class ChartGenerationParams:
    """
    制谱参数 - 从提示词解析出的结构化参数
    
    这些参数将被传递给 Transformer + Diffusion 模型作为条件输入
    """
    # === 基础参数 ===
    bpm: Optional[float] = None           # BPM 值
    time_signature: str = "4/4"          # 拍号
    
    # === 音乐属性 ===
    genre: Optional[MusicGenre] = None   # 流派
    mood: Optional[str] = None            # 情绪 (sad, happy, energetic, calm...)
    energy_level: Optional[int] = None     # 能量等级 1-10
    tempo_feeling: Optional[str] = None   # 节奏感觉 (fast, slow, moderate...)
    
    # === 谱面难度 & 复杂度 ===
    difficulty: Optional[DifficultyLevel] = None
    note_density: Optional[float] = None   # 音符密度 0.0-1.0
    complexity_score: Optional[float] = None  # 综合复杂度 0.0-1.0
    
    # === 特殊技巧要求 ===
    note_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        "tap": 0.5,
        "hold": 0.25,
        "slide": 0.25
    })  # 各类音符比例
    
    special_patterns: List[str] = field(default_factory=list)  # 特殊模式
    # e.g., ["stream", "jump", "trill", "jack", "staircase"]
    
    # === 空间布局 ===
    lane_usage: Optional[str] = None       # 轨道使用模式 (full, center, alternating...)
    vertical_range: Optional[str] = None    # 垂直范围 (full_upper, full_lower, mixed)
    
    # === 风格描述 (原始文本) ===
    description: Optional[str] = None      # 用户原始描述的摘要
    keywords: List[str] = field(default_factory=list)  # 提取的关键词
    
    # === 高级选项 ===
    creativity: float = 0.5                # 创造性 vs 规范性 (0=严格规范, 1=高度创意)
    randomness: float = 0.3                # 随机性程度
    
    # 元数据
    raw_prompt: Optional[str] = None       # 原始提示词
    confidence_scores: Dict[str, float] = field(default_factory=dict)  # 各字段解析置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于 JSON 序列化)"""
        d = asdict(self)
        
        # 处理枚举类型
        for key, value in d.items():
            if isinstance(value, MusicGenre):
                d[key] = value.value
            elif isinstance(value, DifficultyLevel):
                d[key] = value.value
        
        return d
    
    def to_condition_tensor(self):
        """
        转换为模型可用的条件张量表示
        这个方法将在 encoder 中被调用
        """
        import torch
        import numpy as np
        
        features = []
        
        # BPM 归一化到 [0, 1]
        if self.bpm is not None:
            features.append(self.bpm / 300.0)  # 假设最大 BPM 300
        else:
            features.append(0.5)  # 默认值 (120 BPM)
        
        # 能量等级
        features.append((self.energy_level or 5) / 10.0)
        
        # 音符密度
        features.append(self.note_density or 0.5)
        
        # 复杂度分数
        features.append(self.complexity_score or 0.5)
        
        # 创造性和随机性
        features.append(self.creativity)
        features.append(self.randomness)
        
        # 难度编码 (one-hot)
        difficulty_map = {
            None: 0,
            DifficultyLevel.EASY: 0,
            DifficultyLevel.NORMAL: 1,
            DifficultyLevel.HARD: 2,
            DifficultyLevel.INSANE: 3,
            DifficultyLevel.ANOTHER: 4,
            DifficultyLevel.LEGACY: 5
        }
        features.append(difficulty_map.get(self.difficulty, 0) / 5.0)
        
        return torch.tensor([features], dtype=torch.float32)


# ============================================================
# 提示词解析模板
# ============================================================

SYSTEM_PROMPT = """你是一个专业的音乐游戏谱面设计 AI 助手。你的任务是将用户关于制谱的自然语言描述，转化为结构化的参数。

## 你的输出格式必须是严格的 JSON:

```json
{
  "bpm": <数字或null>,
  "time_signature": "<拍号>",
  
  "genre": "<pop|rock|electronic|classical|jazz|hiphop|metal|rnb|folk|ambient|null>",
  "mood": "<情绪描述词或null>",
  "energy_level": <1-10整数或null>,
  "tempo_feeling": "<fast|slow|moderate|null>",
  
  "difficulty": "<easy|normal|hard|insane|another|legacy|null>",
  "note_density": <0.0-1.0浮点数>,
  "complexity_score": <0.0-1.0浮点数>,
  
  "note_type_distribution": {
    "tap": <比例>,
    "hold": <比例>,
    "slide": <比例>
  },
  
  "special_patterns": ["<模式名>", ...],
  
  "lane_usage": "<full|center|alternating|...|null>",
  "vertical_range": "<full_upper|full_lower|mixed|null>",
  
  "description": "<对用户意图的简短总结>",
  "keywords": ["<关键词>", ...],
  
  "creativity": <0.0-1.0>,
  "randomness": <0.0-1.0>,
  
  "confidence_scores": {
    "bpm": <0.0-1.0>,
    "genre": <0.0-1.0>,
    "difficulty": <0.0-1.0>
  }
}
```

## 解析规则:
1. 如果用户没有明确提到某个参数，设为 null (除了 note_density 和 complexity_score 可以根据上下文推断)
2. confidence_scores 表示你对每个参数提取的确信程度 (0-1)
3. 对于模糊的描述，适当降低 confidence 并给出合理的推断值
4. 注意音乐术语和谱面术语的区别

## 示例:
用户: "帮我生成一段128 BPM的电子音乐谱面，要有很多滑动音符，难度中等偏上"
输出: 
```json
{
  "bpm": 128,
  "genre": "electronic",
  "difficulty": "hard",
  "note_type_distribution": {"tap": 0.3, "hold": 0.3, "slide": 0.4},
  "special_patterns": ["stream"],
  "description": "电子音乐风格，128 BPM，中等偏高难度，强调滑动音符",
  "keywords": ["electronic", "128BPM", "slide", "hard"],
  "confidence_scores": {"bpm": 0.95, "genre": 0.9, "difficulty": 0.8}
}
```
"""


# ============================================================
# LLM 接口实现
# ============================================================

class PromptParser:
    """
    提示词解析器 - 使用 LLM 将自然语言转为结构化参数
    
    支持:
    - OpenAI GPT-4/GPT-3.5 API
    - 本部部署的兼容 OpenAI 格式的模型
    """
    
    def __init__(
        self,
        provider: str = "openai",      # openai, local
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,       # 低温度以确保结构化输出稳定
        max_tokens: int = 2048
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # API 配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        # 初始化客户端 (延迟加载)
        self._client = None
    
    @property
    def client(self):
        """懒初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError(
                    "请安装 openai 包: pip install openai\n"
                    "或者设置 provider='local' 使用本地模型"
                )
        return self._client
    
    def _build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """构建对话消息列表"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    
    def parse(
        self,
        prompt: str,
        use_fallback: bool = True
    ) -> ChartGenerationParams:
        """
        解析用户提示词为结构化参数
        
        Args:
            prompt: 用户自然语言提示词
            use_fallback: 如果 LLM 解析失败是否回退到规则匹配
            
        Returns:
            ChartGenerationParams 结构化参数
        """
        print(f"🔮 正在解析提示词: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_messages(prompt),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # 构建参数对象
            params = self._dict_to_params(data, prompt)
            
            print(f"✅ 解析成功!")
            print(f"   风格: {params.genre}")
            print(f"   难度: {params.difficulty}")
            print(f"   BPM: {params.bpm}")
            
            return params
            
        except Exception as e:
            print(f"⚠️  LLM 解析失败: {e}")
            if use_fallback:
                print("   回退到规则匹配...")
                return self._rule_based_parse(prompt)
            raise
    
    def _dict_to_params(self, data: Dict, raw_prompt: str) -> ChartGenerationParams:
        """将 LLM 输出字典转为 ChartGenerationParams 对象"""
        # 处理枚举
        genre = None
        if data.get("genre"):
            try:
                genre = MusicGenre(data["genre"])
            except ValueError:
                genre = None
        
        difficulty = None
        if data.get("difficulty"):
            try:
                difficulty = DifficultyLevel(data["difficulty"])
            except ValueError:
                difficulty = None
        
        return ChartGenerationParams(
            bpm=data.get("bpm"),
            time_signature=data.get("time_signature", "4/4"),
            genre=genre,
            mood=data.get("mood"),
            energy_level=data.get("energy_level"),
            tempo_feeling=data.get("tempo_feeling"),
            difficulty=difficulty,
            note_density=data.get("note_density"),
            complexity_score=data.get("complexity_score"),
            note_type_distribution=data.get("note_type_distribution", {}),
            special_patterns=data.get("special_patterns", []),
            lane_usage=data.get("lane_usage"),
            vertical_range=data.get("vertical_range"),
            description=data.get("description"),
            keywords=data.get("keywords", []),
            creativity=data.get("creativity", 0.5),
            randomness=data.get("randomness", 0.3),
            raw_prompt=raw_prompt,
            confidence_scores=data.get("confidence_scores", {})
        )
    
    def _rule_based_parse(self, prompt: str) -> ChartGenerationParams:
        """
        基于规则的简单解析 (当 LLM 不可用时的 fallback)
        使用正则表达式和关键词匹配提取基本信息
        """
        import re
        
        params = ChartGenerationParams(raw_prompt=prompt)
        prompt_lower = prompt.lower()
        
        # === BPM 提取 ===
        bpm_pattern = r'(\d+)\s*bpm'
        bpm_match = re.search(bpm_pattern, prompt_lower)
        if bpm_match:
            params.bpm = int(bpm_match.group(1))
        
        # === 难度提取 ===
        difficulty_keywords = {
            'easy': ['简单', '容易', 'ez', 'easy', '新手', '入门'],
            'normal': ['普通', '正常', 'nm', 'normal', '中等'],
            'hard': ['困难', '难', 'hd', 'hard', '挑战'],
            'insane': ['疯狂', '很难', 'in', 'insane'],
            'another': ['另一个', 'at', 'another', '极难'],
            'legacy': ['遗产', 'legacy', '传说']
        }
        
        for level, keywords in difficulty_keywords.items():
            for kw in keywords:
                if kw in prompt_lower:
                    try:
                        params.difficulty = DifficultyLevel(level)
                        break
                    except ValueError:
                        pass
        
        # === 流派提取 ===
        genre_keywords = {
            'pop': ['流行', 'pop', 'pop music'],
            'rock': ['摇滚', 'rock'],
            'electronic': ['电子', '电音', 'edm', 'electronic', 'techno', 'house'],
            'classical': ['古典', '钢琴', 'classical', 'piano'],
            'jazz': ['爵士', 'jazz'],
            'hiphop': ['嘻哈', '说唱', 'hip-hop', 'hiphop', 'rap'],
            'metal': ['金属', '重金属', 'metal', 'heavy metal'],
            'rnb': ['节奏布鲁斯', 'r&b', 'rnb'],
            'folk': ['民谣', 'folk'],
            'ambient': ['氛围', '环境', 'ambient']
        }
        
        for genre, keywords in genre_keywords.items():
            for kw in keywords:
                if kw in prompt_lower:
                    try:
                        params.genre = MusicGenre(genre)
                        break
                    except ValueError:
                        pass
        
        # === 特殊模式提取 ===
        pattern_keywords = {
            'stream': ['连打', 'stream', '连续'],
            'jump': ['跳', '跳跃', 'jump', '大跨度'],
            'trill': ['颤音', '交替', 'trill'],
            'jack': ['同键连打', 'jack'],
            'staircase': ['楼梯', '阶梯', 'staircase'],
            'long_hold': ['长按', '长条', 'long hold']
        }
        
        for pattern, kws in pattern_keywords.items():
            for kw in kws:
                if kw in prompt_lower:
                    params.special_patterns.append(pattern)
        
        # === 情绪/能量 ===
        mood_keywords = {
            'sad': ['悲伤', '难过', '悲伤', '忧郁', 'sad'],
            'happy': ['快乐', '开心', '愉快', 'happy'],
            'energetic': ['激情', '激昂', '充满活力', 'energetic', '激烈'],
            'calm': ['平静', '安静', '舒缓', 'calm', 'peaceful'],
            'intense': ['紧张', '激烈', 'intense']
        }
        
        for mood, kws in mood_keywords.items():
            for kw in kws:
                if kw in prompt_lower:
                    params.mood = mood
                    break
        
        # 设置默认推断值
        if params.note_density is None:
            if params.difficulty in [DifficultyLevel.ANOTHER, DifficultyLevel.LEGACY]:
                params.note_density = 0.85
            elif params.difficulty == DifficultyLevel.INSANE:
                params.note_density = 0.7
            elif params.difficulty == DifficultyLevel.HARD:
                params.note_density = 0.55
            else:
                params.note_density = 0.35
        
        params.description = f"(Rule-based parsing) {prompt}"
        params.keywords = list(set(params.special_patterns))
        
        return params
    
    def batch_parse(self, prompts: List[str]) -> List[ChartGenerationParams]:
        """批量解析多个提示词"""
        return [self.parse(p) for p in prompts]


def create_default_parser() -> PromptParser:
    """创建默认配置的解析器"""
    return PromptParser(
        provider="openai",
        model_name="gpt-4-turbo",
        temperature=0.3
    )


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ArellanoDreamWeaver - Prompt Parser Test")
    print("=" * 60)
    
    parser = create_default_parser()
    
    test_prompts = [
        "生成一段 120 BPM 的流行歌曲谱面，难度中等，要有一些长按音符",
        "帮我做一首重金属风格的谱面，180 BPM，非常困难，包含大量滑动音符和连打",
        "制作一首悲伤的钢琴曲谱面，简单难度，音符稀疏一些",
        "128 BPM 电子音乐，需要很多 stream 模式"
    ]
    
    print("\n📝 测试提示词解析:\n")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}] {prompt}")
        print("-" * 40)
        
        try:
            params = parser.parse(prompt, use_fallback=True)
            
            print(f"\n解析结果:")
            print(f"  BPM: {params.bpm}")
            print(f"  Genre: {params.genre}")
            print(f"  Difficulty: {params.difficulty}")
            print(f"  Mood: {params.mood}")
            print(f"  Note Density: {params.note_density}")
            print(f"  Special Patterns: {params.special_patterns}")
            print(f"  Description: {params.description}")
            
            # 测试转换为条件张量
            cond_tensor = params.to_condition_tensor()
            print(f"  Condition Tensor Shape: {cond_tensor.shape}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\n" + "=" * 60 + "\n")
