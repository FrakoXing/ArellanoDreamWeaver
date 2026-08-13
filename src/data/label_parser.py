# -*- coding: utf-8 -*-
"""
定数标签解析器
==============

从 token 文件名解析定数（难度等级）标签，并完成跨游戏归一化。

文件命名规范（普查得出的实际格式）:
  Phigros: {id}_{档位}Lv.{定数}_{曲名}_{作者}.pt
           如 10018_IN Lv.12_别让我担心_小鱼仔娱乐.pt
               10048_IN.Lv14_Broken_贾小贾.pt          (点号分隔)
               10061_AT  Lv.16_Poppy_Fyato.pt           (双空格)
               10062_IN lv.13_東の空..._.pt             (小写 lv)
               10093_IN Lv. 14_安逸之野_.pt             (Lv 后空格)
  osu!mania: {id}_{keys} LV.{star}_{曲名}_{作者}.pt
           如 10100_4K LV.25_S.A.T.E.L.L.I.T.E_小子.pt

两套定数尺度不兼容:
  Phigros 定数: 0.0 ~ 16.0  (IN 12-14, AT 15-16, LEGACY 17+)
  osu!mania 星级: 0 ~ 50+   (4K/5K/6K/7K)

统一归一化方案: 每个游戏独立 min-max 归一化到 [0, 1]，
训练用归一化值，推理时按 Game Token 反归一化回原始尺度。
"""

import re
import json
from pathlib import Path
from typing import Optional, Dict, Any


# ============================================================
# 归一化范围 (持久化到 data/rating/normalization.json)
# ============================================================

NORM_RANGES: Dict[str, Dict[str, float]] = {
    "phira":    {"min": 0.0, "max": 16.0},   # 16 是真正上限, 17 截断
    "osu":      {"min": 0.0, "max": 50.0},   # 4K Lv.50+ 罕见
    "arellano": {"min": 0.0, "max": 16.0},   # Arellano 沿用 Phigros 尺度
}

# 合法范围上限 (超过视为脏数据/恶搞)
PHIRA_MAX = 17.0   # AT Lv.17 是合法极限, >17 多为恶搞
OSU_MAX = 50.0


# ============================================================
# 档位识别
# ============================================================

# Phigros 标准档位前缀 (大小写不敏感)
PHIRA_PREFIXES = ("IN", "AT", "HD", "EZ", "NM", "SP", "LEGACY")

# osu!mania 键数前缀
OSU_KEYS = ("4K", "5K", "6K", "7K")


# ============================================================
# 脏数据黑名单
# ============================================================

# 已知恶搞/误匹配的定数字符串 (从 10281 文件普查中收集)
BLACKLIST_RAW_LEVELS = {
    "114514", "115414", "514", "495",
    "999", "333", "167", "168",
    "2024", "2025", "2085",
    "10086", "51121", "51522", "5071",
    "0419", "520", "1314",
}


# ============================================================
# 正则: 抽取 Lv 后的数字
# ============================================================

# 匹配 Lv / LV / lv / L.V 等变体 + 可选点号/空格 + 数字(可带小数)
# 例: "Lv.12" "LV.25" "lv.13" "Lv 14" "Lv. 14" "Lv14"
LV_PATTERN = re.compile(r'[Ll][Vv]\.?\s*(\d+(?:\.\d+)?)')


# ============================================================
# 核心解析函数
# ============================================================

def detect_game(diff_part: str) -> Optional[str]:
    """
    从文件名的"档位段"识别游戏来源。

    Args:
        diff_part: 文件名按 '_' 切分后的第二段 (如 "IN Lv.12", "4K LV.25")

    Returns:
        "phira" / "osu" / None
    """
    upper = diff_part.upper().strip()

    # 先检测 osu!mania 键数 (必须是段开头)
    for k in OSU_KEYS:
        if upper.startswith(k):
            return "osu"

    # 检测 Phigros 标准档位
    for prefix in PHIRA_PREFIXES:
        if upper.startswith(prefix):
            return "phira"

    # 非标准档位名 (ORIGINAL/Cecilia/IF/XT 等社区自定义) 但带 Lv 的,
    # 默认归为 Phigros 生态
    if LV_PATTERN.search(diff_part):
        return "phira"

    return None


def detect_osu_keys(diff_part: str) -> Optional[str]:
    """检测 osu!mania 键数 (4K/5K/6K/7K)"""
    upper = diff_part.upper().strip()
    for k in OSU_KEYS:
        if upper.startswith(k):
            return k
    return None


def parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    解析 token 文件名 → 定数标签字典。

    Args:
        filename: 文件名 (如 "10018_IN Lv.12_别让我担心_小鱼仔娱乐.pt")

    Returns:
        成功: {
            "game": "phira" | "osu",
            "osu_keys": "4K" | None,
            "level": float,          # 原始定数 (如 12.0)
            "normalized": float,     # 归一化值 [0, 1]
            "raw_filename": str,
            "source": "filename",
        }
        失败 (无标签/脏数据): None
    """
    stem = Path(filename).stem
    parts = stem.split('_', 2)
    if len(parts) < 2:
        return None

    id_part = parts[0]
    # 第一段必须是纯数字 ID (如 10018)
    if not id_part.isdigit():
        return None

    diff_part = parts[1]
    # 后续内容 (用于 Lv 搜索, 包含档位段+曲名段)
    rest = '_'.join(parts[1:])

    # 识别游戏
    game = detect_game(diff_part)
    if game is None:
        return None

    osu_keys = detect_osu_keys(diff_part) if game == "osu" else None

    # 抽取 Lv 数字
    m = LV_PATTERN.search(rest)
    if not m:
        return None

    level_str = m.group(1)

    # 黑名单过滤
    if level_str in BLACKLIST_RAW_LEVELS:
        return None

    try:
        level = float(level_str)
    except ValueError:
        return None

    # 范围校验 (脏数据过滤的第二道防线)
    if game == "phira":
        if not (0.0 <= level <= PHIRA_MAX):
            return None
    else:  # osu
        if not (0.0 <= level <= OSU_MAX):
            return None

    normalized = normalize_level(level, game)

    return {
        "game": game,
        "osu_keys": osu_keys,
        "level": level,
        "normalized": normalized,
        "raw_filename": filename,
        "source": "filename",
    }


# ============================================================
# 归一化 / 反归一化
# ============================================================

def normalize_level(level: float, game: str) -> float:
    """原始定数 → 归一化值 [0, 1]"""
    r = NORM_RANGES.get(game, NORM_RANGES["phira"])
    clipped = max(r["min"], min(r["max"], level))
    return (clipped - r["min"]) / (r["max"] - r["min"])


def denormalize_level(normalized: float, game: str) -> float:
    """归一化值 [0, 1] → 原始定数"""
    r = NORM_RANGES.get(game, NORM_RANGES["phira"])
    clipped = max(0.0, min(1.0, normalized))
    return clipped * (r["max"] - r["min"]) + r["min"]


# ============================================================
# Game Token ID ↔ 游戏名映射
# ============================================================

# 与 unified_tokenizer.py / note_decoder.py 对齐
GAME_TOKEN_PHIRA = 116
GAME_TOKEN_OSU = 117
GAME_TOKEN_ARELLANO = 118

GAME_TOKEN_TO_NAME = {
    GAME_TOKEN_PHIRA: "phira",
    GAME_TOKEN_OSU: "osu",
    GAME_TOKEN_ARELLANO: "arellano",
}

GAME_NAME_TO_TOKEN = {v: k for k, v in GAME_TOKEN_TO_NAME.items()}


def game_token_to_name(token_id: int) -> str:
    """Game Token ID → 游戏名 (未知则归 phira)"""
    return GAME_TOKEN_TO_NAME.get(int(token_id), "phira")


def game_name_to_token(name: str) -> int:
    """游戏名 → Game Token ID"""
    return GAME_NAME_TO_TOKEN.get(name, GAME_TOKEN_PHIRA)


# ============================================================
# 持久化辅助
# ============================================================

def save_normalization_file(path: str) -> None:
    """保存归一化范围到 JSON (供 Unity 端加载)"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(NORM_RANGES, f, indent=2, ensure_ascii=False)


def load_normalization_file(path: str) -> Dict[str, Dict[str, float]]:
    """加载归一化范围 JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    test_cases = [
        ("10018_IN Lv.12_别让我担心_小鱼仔娱乐.pt", "phira", None, 12.0, 0.75),
        ("10048_IN.Lv14_Broken_贾小贾.pt", "phira", None, 14.0, 0.875),
        ("10061_AT  Lv.16_Poppy_Fyato.pt", "phira", None, 16.0, 1.0),
        ("10062_IN lv.13_東の空から始まる世界_屑影酱.pt", "phira", None, 13.0, 0.8125),
        ("10093_IN Lv. 14_安逸之野_hanjuelai123.pt", "phira", None, 14.0, 0.875),
        ("10100_4K LV.25_S.A.T.E.L.L.I.T.E_小子想干啥.pt", "osu", "4K", 25.0, 0.5),
        ("10047_SP Lv.___ロストワンの号哭_沙尘暴_未来老子.pt", None, None, None, None),
        ("12391_SP Lv.114514_xxx.pt", None, None, None, None),
        ("10046_超人_音弾超人ゴリライザー_sJDeviko.pt", None, None, None, None),
        ("10104_IN.不会填_Water Quench_水我只认家乡的.pt", None, None, None, None),
    ]

    print("=" * 70)
    print("label_parser 自测")
    print("=" * 70)
    ok = 0
    for fname, exp_game, exp_keys, exp_level, exp_norm in test_cases:
        r = parse_filename(fname)
        if exp_level is None:
            passed = r is None
        else:
            passed = (r is not None
                      and r["game"] == exp_game
                      and r["osu_keys"] == exp_keys
                      and abs(r["level"] - exp_level) < 1e-6
                      and abs(r["normalized"] - exp_norm) < 1e-6)
        status = "✓" if passed else "✗"
        ok += passed
        print(f"  {status} {fname}")
        if not passed and r is not None:
            print(f"      got: game={r['game']}, keys={r['osu_keys']}, "
                  f"level={r['level']}, norm={r['normalized']}")
    print(f"\n  {ok}/{len(test_cases)} passed")
