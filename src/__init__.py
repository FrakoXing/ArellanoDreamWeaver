"""
ArellanoDreamWeaver Package Init
"""

__version__ = "0.1.0"
__author__ = "Arellano Team"

import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def get_project_root() -> Path:
    return PROJECT_ROOT
