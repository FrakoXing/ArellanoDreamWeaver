"""
ArellanoDreamWeaver - 包安装配置
================================
支持 pip install -e . 开发模式安装
"""
from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# 读取依赖
req_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if req_path.exists():
    install_requires = [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="arellano-dreamweaver",
    version="0.1.0-alpha",
    author="Arellano Team",
    description="基于多任务 Transformer + Diffusion 的 AI 辅助制谱模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    url="https://github.com/Arellano/ArellanoDreamWeaver",
    
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "checkpoints", "data"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
        ],
        "gpu": [
            "flash-attn>=2.0.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "dreamweaver-init=dreamweaver:init_dreamweaver",
            "dreamweaver-generate=scripts:generate.main",
            "dreamweaver-analyze=scripts:analyze_prompt.main",
        ]
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    keywords=[
        "music game",
        "chart generation",
        "diffusion model",
        "transformer",
        "AI music",
        "rhythm game"
    ]
)
