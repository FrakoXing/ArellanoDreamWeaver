# -*- coding: utf-8 -*-
"""
评级模块共享加载器
==================

绕过 src/data/__init__.py (后者导入了依赖 librosa 的 dataset.py,
评级模型不需要 librosa)。

所有 scripts/ 下的评级脚本统一通过本模块访问 label_parser /
handcrafted_features, 避免重复 importlib 样板代码。

用法:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from rating_common import parse_filename, compute_features, ...
"""

import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = PROJECT_ROOT / "src" / "data"


def _load_module(name: str, path: Path):
    """从文件路径直接加载模块 (不触发包 __init__)"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 加载同级数据模块
_label_parser = _load_module("label_parser", _DATA_DIR / "label_parser.py")
_handcrafted_features = _load_module("handcrafted_features", _DATA_DIR / "handcrafted_features.py")

# === 导出 label_parser 符号 ===
parse_filename = _label_parser.parse_filename
detect_game = _label_parser.detect_game
normalize_level = _label_parser.normalize_level
denormalize_level = _label_parser.denormalize_level
NORM_RANGES = _label_parser.NORM_RANGES
save_normalization_file = _label_parser.save_normalization_file
load_normalization_file = _label_parser.load_normalization_file
game_token_to_name = _label_parser.game_token_to_name
game_name_to_token = _label_parser.game_name_to_token
GAME_TOKEN_PHIRA = _label_parser.GAME_TOKEN_PHIRA
GAME_TOKEN_OSU = _label_parser.GAME_TOKEN_OSU
GAME_TOKEN_ARELLANO = _label_parser.GAME_TOKEN_ARELLANO

# === 导出 handcrafted_features 符号 ===
compute_features = _handcrafted_features.compute_features
compute_features_torch = _handcrafted_features.compute_features_torch
compute_feature_stats = _handcrafted_features.compute_feature_stats
FEATURE_DIM = _handcrafted_features.FEATURE_DIM
