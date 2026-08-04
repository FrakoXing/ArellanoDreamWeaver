#!/usr/bin/env python3
"""
OSZ 自动处理脚本
================

从 .osz 文件中提取 .osu 谱面和音频，自动整理到训练数据目录。

.osz 是 osu! 的谱面包格式 (ZIP)，包含:
  - .osu 谱面文件
  - 音频文件 (mp3/ogg)
  - 背景图片等

用法:
  # 处理 data/raw_osz/ 下所有 osz 文件
  python scripts/process_osz.py

  # 指定输入输出目录
  python scripts/process_osz.py --input ./osz_files --output-osu ./osu --output-audio ./audio

  # 处理单个文件
  python scripts/process_osz.py --file song.osz
"""

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List


# ============================================================
# 配置
# ============================================================

DEFAULT_OSZ_DIR = "./data/raw_osz"
DEFAULT_OSU_DIR = "./data/raw"
DEFAULT_AUDIO_DIR = "./data/audio"


# ============================================================
# 工具函数
# ============================================================

def safe_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    # 替换 Windows 文件名非法字符
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # 去掉首尾空格和点
    safe = safe.strip(' .')
    # 限制长度
    if len(safe) > 200:
        safe = safe[:200]
    return safe or "unknown"


def extract_audio_filename(osu_content: str) -> Optional[str]:
    """
    从 .osu 文件内容中提取 AudioFilename
    
    .osu 文件格式:
      [General]
      AudioFilename: audio.mp3
      ...
    """
    for line in osu_content.split('\n'):
        line = line.strip()
        if line.startswith('AudioFilename:'):
            audio = line[len('AudioFilename:'):].strip()
            # 去掉可能的注释
            if '//' in audio:
                audio = audio.split('//')[0].strip()
            return audio
    return None


def extract_osu_metadata(osu_content: str) -> dict:
    """提取 .osu 文件的元数据"""
    metadata = {
        'title': '',
        'artist': '',
        'creator': '',
        'version': '',
    }
    
    for line in osu_content.split('\n'):
        line = line.strip()
        if line.startswith('Title:'):
            metadata['title'] = line[len('Title:'):].strip()
        elif line.startswith('Artist:'):
            metadata['artist'] = line[len('Artist:'):].strip()
        elif line.startswith('Creator:'):
            metadata['creator'] = line[len('Creator:'):].strip()
        elif line.startswith('Version:'):
            metadata['version'] = line[len('Version:'):].strip()
    
    return metadata


# ============================================================
# OSZ 处理器
# ============================================================

class OSZProcessor:
    def __init__(self, osu_dir: str, audio_dir: str):
        self.osu_dir = Path(osu_dir)
        self.audio_dir = Path(audio_dir)
        self.osu_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total': 0,
            'processed': 0,
            'no_audio': 0,
            'no_osu': 0,
            'error': 0,
        }
    
    def process_osz(self, osz_path: Path) -> List[dict]:
        """
        处理单个 .osz 文件
        
        Returns:
            List of {"osu": str, "audio": str, "status": str, "message": str}
        """
        results = []
        
        try:
            with zipfile.ZipFile(osz_path, 'r') as zf:
                # 列出所有文件
                all_files = zf.namelist()
                
                # 找 .osu 文件
                osu_files = [f for f in all_files if f.lower().endswith('.osu')]
                
                if not osu_files:
                    self.stats['no_osu'] += 1
                    return [{"osu": "", "audio": "", "status": "no_osu", 
                             "message": "osz 中没有 .osu 文件"}]
                
                # 找音频文件 (mp3, ogg, wav)
                audio_exts = {'.mp3', '.ogg', '.wav', '.flac'}
                audio_files = [f for f in all_files 
                              if Path(f).suffix.lower() in audio_exts]
                
                # 处理每个 .osu 文件
                for osu_file in osu_files:
                    result = {"osu": "", "audio": "", "status": "unknown", "message": ""}
                    
                    try:
                        # 读取 .osu 内容
                        osu_content = zf.read(osu_file).decode('utf-8', errors='replace')
                        
                        # 提取音频文件名
                        audio_name = extract_audio_filename(osu_content)
                        
                        # 生成新的文件名 (基于 .osu 文件名)
                        osu_stem = Path(osu_file).stem
                        new_osu_name = safe_filename(osu_stem) + '.osu'
                        new_audio_name = safe_filename(osu_stem)
                        
                        # 保存 .osu 文件
                        osu_dest = self.osu_dir / new_osu_name
                        with open(osu_dest, 'w', encoding='utf-8') as f:
                            f.write(osu_content)
                        result['osu'] = str(osu_dest)
                        
                        # 查找并保存音频文件
                        audio_found = False
                        
                        if audio_name:
                            # 在 osz 中查找匹配的音频
                            for af in audio_files:
                                if Path(af).name.lower() == audio_name.lower():
                                    # 找到匹配的音频
                                    audio_ext = Path(af).suffix
                                    audio_dest = self.audio_dir / (new_audio_name + audio_ext)
                                    
                                    # 提取音频
                                    audio_data = zf.read(af)
                                    with open(audio_dest, 'wb') as f:
                                        f.write(audio_data)
                                    
                                    result['audio'] = str(audio_dest)
                                    audio_found = True
                                    break
                        
                        # 如果没有找到匹配的音频，尝试用第一个音频文件
                        if not audio_found and audio_files:
                            af = audio_files[0]
                            audio_ext = Path(af).suffix
                            audio_dest = self.audio_dir / (new_audio_name + audio_ext)
                            
                            audio_data = zf.read(af)
                            with open(audio_dest, 'wb') as f:
                                f.write(audio_data)
                            
                            result['audio'] = str(audio_dest)
                            audio_found = True
                        
                        if not audio_found:
                            self.stats['no_audio'] += 1
                            result['status'] = 'no_audio'
                            result['message'] = f"未找到音频文件 (osu: {new_osu_name})"
                        else:
                            self.stats['processed'] += 1
                            result['status'] = 'ok'
                            
                            # 提取元数据用于显示
                            meta = extract_osu_metadata(osu_content)
                            title = meta.get('title', osu_stem)
                            artist = meta.get('artist', '')
                            result['message'] = f"{artist} - {title}"
                    
                    except Exception as e:
                        result['status'] = 'error'
                        result['message'] = f"处理 {osu_file} 失败: {e}"
                        self.stats['error'] += 1
                    
                    results.append(result)
        
        except zipfile.BadZipFile:
            return [{"osu": "", "audio": "", "status": "error", 
                     "message": f"无效的 ZIP 文件: {osz_path.name}"}]
        except Exception as e:
            return [{"osu": "", "audio": "", "status": "error", 
                     "message": f"处理失败: {e}"}]
        
        return results
    
    def process_directory(self, input_dir: str, verbose: bool = True):
        """处理目录下所有 .osz 文件"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"❌ 目录不存在: {input_dir}")
            return
        
        # 收集所有 .osz 文件
        osz_files = list(input_path.glob('*.osz'))
        osz_files.sort()
        
        if not osz_files:
            print(f"⚠️  目录中没有 .osz 文件: {input_dir}")
            return
        
        self.stats['total'] = len(osz_files)
        
        print(f"📦 OSZ 自动处理")
        print(f"{'=' * 60}")
        print(f"  输入目录:   {input_path.absolute()}")
        print(f"  OSU 输出:   {self.osu_dir.absolute()}")
        print(f"  音频输出:   {self.audio_dir.absolute()}")
        print(f"  文件数量:   {len(osz_files)}")
        print(f"{'=' * 60}\n")
        
        for i, osz_path in enumerate(osz_files):
            if verbose:
                print(f"[{i+1}/{len(osz_files)}] {osz_path.name}")
            
            results = self.process_osz(osz_path)
            
            for r in results:
                if verbose:
                    status_icon = {
                        'ok': '✅',
                        'no_osu': '❌',
                        'no_audio': '⚠️',
                        'error': '❌',
                    }.get(r['status'], '?')
                    print(f"  {status_icon} {r['message']}")
                    if r['osu']:
                        print(f"     📄 → {Path(r['osu']).name}")
                    if r['audio']:
                        print(f"     🎵 → {Path(r['audio']).name}")
        
        # 打印统计
        print(f"\n{'=' * 60}")
        print(f"📊 处理完成!")
        print(f"{'=' * 60}")
        print(f"  总计 OSZ:     {self.stats['total']}")
        print(f"  成功处理:     {self.stats['processed']}")
        print(f"  无音频:       {self.stats['no_audio']}")
        print(f"  无 OSU:       {self.stats['no_osu']}")
        print(f"  错误:         {self.stats['error']}")
        print(f"{'=' * 60}")


# ============================================================
# 命令行
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="OSZ 自动处理脚本 - 提取 .osu 和音频到训练数据目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理 data/raw_osz/ 下所有 osz 文件
  python scripts/process_osz.py

  # 指定输入输出目录
  python scripts/process_osz.py \\
      --input ./my_osz_files \\
      --output-osu ./data/raw \\
      --output-audio ./data/audio

  # 处理单个文件
  python scripts/process_osz.py --file song.osz

  # 静默模式 (不显示每个文件的处理详情)
  python scripts/process_osz.py --quiet

目录结构:
  data/
  ├── raw_osz/      ← 放 .osz 文件在这里
  ├── raw/          ← .osu 文件输出到这里
  └── audio/        ← 音频文件输出到这里
        """
    )
    
    parser.add_argument('--input', '-i', default=DEFAULT_OSZ_DIR,
                        help=f'输入目录 (默认: {DEFAULT_OSZ_DIR})')
    parser.add_argument('--output-osu', default=DEFAULT_OSU_DIR,
                        help=f'OSU 输出目录 (默认: {DEFAULT_OSU_DIR})')
    parser.add_argument('--output-audio', default=DEFAULT_AUDIO_DIR,
                        help=f'音频输出目录 (默认: {DEFAULT_AUDIO_DIR})')
    parser.add_argument('--file', '-f',
                        help='处理单个 .osz 文件')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式')
    
    args = parser.parse_args()
    
    processor = OSZProcessor(
        osu_dir=args.output_osu,
        audio_dir=args.output_audio,
    )
    
    if args.file:
        # 处理单个文件
        osz_path = Path(args.file)
        if not osz_path.exists():
            print(f"❌ 文件不存在: {args.file}")
            sys.exit(1)
        
        print(f"📦 处理单个文件: {osz_path.name}")
        results = processor.process_osz(osz_path)
        
        for r in results:
            status_icon = {
                'ok': '✅',
                'no_osu': '❌',
                'no_audio': '⚠️',
                'error': '❌',
            }.get(r['status'], '?')
            print(f"  {status_icon} {r['message']}")
            if r['osu']:
                print(f"     📄 → {r['osu']}")
            if r['audio']:
                print(f"     🎵 → {r['audio']}")
    else:
        # 处理目录
        processor.process_directory(args.input, verbose=not args.quiet)


if __name__ == "__main__":
    main()
