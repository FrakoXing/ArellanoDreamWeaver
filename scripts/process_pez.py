"""
PEZ 谱面包处理脚本
==================

处理 Phira 的 .pez 谱面包文件，提取谱面和音频。

PEZ 格式:
  - ZIP 压缩包
  - 包含谱面文件 (chart.json 或 chart.pec)
  - 包含音频文件 (music.mp3, music.ogg 等)
  - 可能包含封面图片 (cover.jpg/png)

用法:
  # 处理所有 PEZ 文件
  python scripts/process_pez.py

  # 处理单个 PEZ 文件
  python scripts/process_pez.py path/to/chart.pez

  # 指定输入输出目录
  python scripts/process_pez.py --input data/raw_pez --output data/raw
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PEZ 文件处理
# ============================================================

class PEZProcessor:
    """
    PEZ 谱面包处理器
    
    处理流程:
      1. 解压 PEZ 文件 (ZIP 格式)
      2. 识别谱面文件 (chart.json / chart.pec)
      3. 识别音频文件 (music.mp3 / music.ogg)
      4. 重命名并输出到指定目录
    """
    
    # 支持的谱面文件扩展名
    CHART_EXTENSIONS = {'.json', '.pec'}
    
    # 支持的音频文件扩展名
    AUDIO_EXTENSIONS = {'.mp3', '.ogg', '.wav', '.flac'}
    
    # 支持的封面文件扩展名
    COVER_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    
    def __init__(self, input_dir: str = "data/raw_pez", output_dir: str = "data/raw", audio_dir: str = "data/audio"):
        self.input_dir = PROJECT_ROOT / input_dir
        self.output_dir = PROJECT_ROOT / output_dir
        self.audio_dir = PROJECT_ROOT / audio_dir
        
        # 创建目录
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all(self) -> dict:
        """处理所有 PEZ 文件"""
        pez_files = list(self.input_dir.glob("*.pez"))
        
        if not pez_files:
            print(f"❌ 没有找到 PEZ 文件: {self.input_dir}")
            return {"total": 0, "success": 0, "failed": 0}
        
        print(f"📦 找到 {len(pez_files)} 个 PEZ 文件")
        
        stats = {"total": len(pez_files), "success": 0, "failed": 0, "skipped": 0}
        
        for pez_path in pez_files:
            try:
                result = self.process_single(pez_path)
                if result:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                stats["failed"] += 1
        
        print(f"\n{'='*50}")
        print(f"✅ 处理完成")
        print(f"   总计: {stats['total']}")
        print(f"   成功: {stats['success']}")
        print(f"   失败: {stats['failed']}")
        print(f"   跳过: {stats['skipped']}")
        print(f"{'='*50}")
        
        return stats
    
    def process_single(self, pez_path: Path) -> bool:
        """
        处理单个 PEZ 文件
        
        Args:
            pez_path: PEZ 文件路径
            
        Returns:
            bool: 是否成功
        """
        print(f"\n📦 处理: {pez_path.name}")
        
        # 获取基础文件名 (不含扩展名)
        base_name = pez_path.stem
        
        # 临时解压目录
        extract_dir = self.input_dir / f"_temp_{base_name}"
        
        try:
            # Step 1: 解压 PEZ 文件
            if not self._extract_pez(pez_path, extract_dir):
                return False
            
            # Step 2: 查找谱面和音频文件
            chart_file, audio_file = self._find_files(extract_dir)
            
            if not chart_file:
                print(f"  ⚠️  未找到谱面文件")
                return False
            
            if not audio_file:
                print(f"  ⚠️  未找到音频文件")
                return False
            
            print(f"  📄 谱面: {chart_file.name} ({self._get_file_size(chart_file):.1f}KB)")
            print(f"  🎵 音频: {audio_file.name} ({self._get_file_size(audio_file):.1f}KB)")
            
            # Step 3: 确定输出文件名
            # 使用 PEZ 文件名作为基础，避免冲突
            output_stem = base_name
            
            # Step 4: 复制谱面文件
            chart_ext = chart_file.suffix
            output_chart = self.output_dir / f"{output_stem}{chart_ext}"
            
            if output_chart.exists():
                print(f"  ⏭️  谱面已存在: {output_chart.name}")
            else:
                shutil.copy2(chart_file, output_chart)
                print(f"  ✅ 谱面 → {output_chart.name}")
            
            # Step 5: 复制并重命名音频文件
            audio_ext = audio_file.suffix
            output_audio = self.audio_dir / f"{output_stem}{audio_ext}"
            
            if output_audio.exists():
                print(f"  ⏭️  音频已存在: {output_audio.name}")
            else:
                shutil.copy2(audio_file, output_audio)
                print(f"  ✅ 音频 → {output_audio.name}")
            
            return True
            
        finally:
            # 清理临时目录
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
    
    def _extract_pez(self, pez_path: Path, extract_dir: Path) -> bool:
        """解压 PEZ 文件"""
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(pez_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            return True
        except zipfile.BadZipFile:
            print(f"  ❌ 无效的 ZIP 文件")
            return False
        except Exception as e:
            print(f"  ❌ 解压失败: {e}")
            return False
    
    def _find_files(self, extract_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        在解压目录中查找谱面和音频文件
        
        Returns:
            (chart_file, audio_file) 或 (None, None)
        """
        chart_file = None
        audio_file = None
        
        # 递归查找所有文件
        for file_path in extract_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            ext = file_path.suffix.lower()
            name_lower = file_path.name.lower()
            
            # 查找谱面文件
            if ext in self.CHART_EXTENSIONS:
                # 优先选择 chart.json 或 chart.pec
                if name_lower in ('chart.json', 'chart.pec'):
                    chart_file = file_path
                elif chart_file is None:
                    chart_file = file_path
            
            # 查找音频文件
            elif ext in self.AUDIO_EXTENSIONS:
                # 优先选择 music.mp3 或 music.ogg
                if name_lower.startswith('music'):
                    audio_file = file_path
                elif audio_file is None:
                    audio_file = file_path
        
        return chart_file, audio_file
    
    def _get_file_size(self, path: Path) -> float:
        """获取文件大小 (KB)"""
        return path.stat().st_size / 1024


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PEZ 谱面包处理脚本")
    
    parser.add_argument("pez_file", nargs="?", default=None,
                        help="单个 PEZ 文件路径 (可选)")
    parser.add_argument("--input", "-i", default="data/raw_pez",
                        help="PEZ 输入目录 (默认: data/raw_pez)")
    parser.add_argument("--output", "-o", default="data/raw",
                        help="谱面输出目录 (默认: data/raw)")
    parser.add_argument("--audio", "-a", default="data/audio",
                        help="音频输出目录 (默认: data/audio)")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = PEZProcessor(
        input_dir=args.input,
        output_dir=args.output,
        audio_dir=args.audio,
    )
    
    # 处理
    if args.pez_file:
        # 处理单个文件
        pez_path = Path(args.pez_file)
        if not pez_path.exists():
            print(f"❌ 文件不存在: {pez_path}")
            sys.exit(1)
        
        if not pez_path.suffix.lower() == '.pez':
            print(f"⚠️  文件扩展名不是 .pez: {pez_path}")
        
        success = processor.process_single(pez_path)
        if success:
            print("\n✅ 处理成功")
        else:
            print("\n❌ 处理失败")
            sys.exit(1)
    else:
        # 处理所有文件
        processor.process_all()


if __name__ == "__main__":
    main()
