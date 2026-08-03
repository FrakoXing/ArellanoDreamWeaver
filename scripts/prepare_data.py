"""
数据预处理脚本
==============
功能:
1. 检查谱面-音频文件配对情况
2. 批量提取音频特征并缓存
3. 生成数据集统计报告
4. 验证数据完整性

用法:
    python scripts/prepare_data.py --check          # 检查配对情况
    python scripts/prepare_data.py --process        # 处理音频特征
    python scripts/prepare_data.py --stats          # 生成统计报告
    python scripts/prepare_data.py --all            # 执行所有步骤
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dreamweaver.chart.structures import ChartData
from src.data.audio_processor import AudioProcessor


class DataPreparator:
    """数据预处理器"""
    
    def __init__(
        self,
        chart_dir: str = "data/processed",
        audio_dir: str = "data/audio",
        cache_dir: str = "data/cache",
        sample_rate: int = 22050,
        n_mels: int = 128
    ):
        self.chart_dir = Path(chart_dir)
        self.audio_dir = Path(audio_dir)
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # 创建目录
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 音频处理器
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        
        # 支持的文件格式
        self.chart_extensions = ['.json']
        self.audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
    
    def find_chart_files(self) -> List[Path]:
        """查找所有谱面文件"""
        charts = []
        for ext in self.chart_extensions:
            charts.extend(self.chart_dir.glob(f"*{ext}"))
        return sorted(charts)
    
    def find_audio_files(self) -> List[Path]:
        """查找所有音频文件"""
        audios = []
        for ext in self.audio_extensions:
            audios.extend(self.audio_dir.glob(f"*{ext}"))
        return sorted(audios)
    
    def check_pairing(self) -> Dict[str, any]:
        """
        检查谱面-音频配对情况
        
        Returns:
            {
                "total_charts": int,
                "total_audio": int,
                "matched": List[Tuple[Path, Path]],
                "unmatched_charts": List[Path],
                "unmatched_audio": List[Path]
            }
        """
        print("\n" + "="*60)
        print("📋 检查谱面-音频配对情况")
        print("="*60)
        
        chart_files = self.find_chart_files()
        audio_files = self.find_audio_files()
        
        print(f"\n找到谱面文件: {len(chart_files)} 个")
        print(f"找到音频文件: {len(audio_files)} 个")
        
        # 建立音频文件索引 (按文件名)
        audio_index = {}
        for audio_path in audio_files:
            base_name = audio_path.stem.lower()
            audio_index[base_name] = audio_path
        
        # 匹配
        matched = []
        unmatched_charts = []
        
        for chart_path in chart_files:
            chart_name = chart_path.stem.lower()
            if chart_name in audio_index:
                matched.append((chart_path, audio_index[chart_name]))
            else:
                unmatched_charts.append(chart_path)
        
        # 未匹配的音频
        matched_audio_names = {m[1].stem.lower() for m in matched}
        unmatched_audio = [
            a for a in audio_files 
            if a.stem.lower() not in {m[0].stem.lower() for m in matched}
        ]
        
        # 打印结果
        print(f"\n✅ 已匹配: {len(matched)} 对")
        if matched:
            for chart, audio in matched[:5]:  # 只显示前5个
                print(f"   {chart.name} ↔ {audio.name}")
            if len(matched) > 5:
                print(f"   ... 还有 {len(matched) - 5} 对")
        
        if unmatched_charts:
            print(f"\n⚠️  缺少音频的谱面: {len(unmatched_charts)} 个")
            for chart in unmatched_charts[:5]:
                print(f"   ❌ {chart.name}")
            if len(unmatched_charts) > 5:
                print(f"   ... 还有 {len(unmatched_charts) - 5} 个")
        
        if unmatched_audio:
            print(f"\n📁 未使用的音频: {len(unmatched_audio)} 个")
            for audio in unmatched_audio[:5]:
                print(f"   {audio.name}")
            if len(unmatched_audio) > 5:
                print(f"   ... 还有 {len(unmatched_audio) - 5} 个")
        
        return {
            "total_charts": len(chart_files),
            "total_audio": len(audio_files),
            "matched": matched,
            "unmatched_charts": unmatched_charts,
            "unmatched_audio": unmatched_audio
        }
    
    def process_audio_features(
        self,
        target_steps: int = 4096,
        force_reprocess: bool = False
    ) -> Dict[str, any]:
        """
        批量处理音频特征
        
        Args:
            target_steps: 目标时间步数
            force_reprocess: 是否强制重新处理
            
        Returns:
            处理统计信息
        """
        print("\n" + "="*60)
        print("🎵 处理音频特征")
        print("="*60)
        
        pairing = self.check_pairing()
        matched = pairing["matched"]
        
        if not matched:
            print("\n❌ 没有匹配的谱面-音频对，无法处理")
            return {"processed": 0, "failed": 0, "skipped": 0}
        
        processed = 0
        failed = 0
        skipped = 0
        
        for i, (chart_path, audio_path) in enumerate(matched):
            cache_file = self.cache_dir / f"{chart_path.stem}.npy"
            
            # 检查缓存
            if cache_file.exists() and not force_reprocess:
                print(f"[{i+1}/{len(matched)}] ⏭️  跳过 (已缓存): {audio_path.name}")
                skipped += 1
                continue
            
            # 处理音频
            try:
                print(f"[{i+1}/{len(matched)}] 🎵 处理: {audio_path.name}")
                
                # 加载谱面获取时长
                chart = ChartData.load_from_file(str(chart_path))
                duration = chart.musicLength if chart.musicLength > 0 else 180.0
                
                # 提取特征
                features = self.audio_processor.align_to_chart(
                    audio_path,
                    chart_duration=duration,
                    target_steps=target_steps
                )
                
                # 保存缓存
                np.save(cache_file, features.numpy())
                print(f"   ✅ 保存: {cache_file.name} (shape: {features.shape})")
                processed += 1
                
            except Exception as e:
                print(f"   ❌ 失败: {e}")
                failed += 1
        
        print(f"\n📊 处理完成:")
        print(f"   成功: {processed}")
        print(f"   失败: {failed}")
        print(f"   跳过: {skipped}")
        
        return {
            "processed": processed,
            "failed": failed,
            "skipped": skipped
        }
    
    def generate_stats(self) -> Dict[str, any]:
        """
        生成数据集统计报告
        
        Returns:
            统计信息字典
        """
        print("\n" + "="*60)
        print("📊 生成数据集统计报告")
        print("="*60)
        
        chart_files = self.find_chart_files()
        
        if not chart_files:
            print("\n❌ 没有找到谱面文件")
            return {}
        
        stats = {
            "total_charts": len(chart_files),
            "total_notes": 0,
            "total_duration": 0.0,
            "note_types": defaultdict(int),
            "bpm_distribution": [],
            "difficulty_distribution": defaultdict(int),
            "note_density": []  # 音符/秒
        }
        
        for chart_path in chart_files:
            try:
                chart = ChartData.load_from_file(str(chart_path))
                
                # 基本统计
                notes = chart.get_all_notes()
                stats["total_notes"] += len(notes)
                stats["total_duration"] += chart.musicLength
                
                # 音符类型
                for note in notes:
                    type_name = {0: "tap", 1: "hold", 2: "slide"}.get(note.type, "unknown")
                    stats["note_types"][type_name] += 1
                
                # BPM
                stats["bpm_distribution"].append(chart.default_bpm)
                
                # 音符密度
                if chart.musicLength > 0:
                    density = len(notes) / chart.musicLength
                    stats["note_density"].append(density)
                    
                    # 难度估计
                    if density < 2.0:
                        stats["difficulty_distribution"]["easy"] += 1
                    elif density < 4.0:
                        stats["difficulty_distribution"]["medium"] += 1
                    elif density < 6.0:
                        stats["difficulty_distribution"]["hard"] += 1
                    elif density < 8.0:
                        stats["difficulty_distribution"]["expert"] += 1
                    else:
                        stats["difficulty_distribution"]["master"] += 1
                
            except Exception as e:
                print(f"⚠️  处理 {chart_path.name} 失败: {e}")
        
        # 打印统计
        print(f"\n📈 数据集概览:")
        print(f"   谱面总数: {stats['total_charts']}")
        print(f"   音符总数: {stats['total_notes']}")
        print(f"   总时长: {stats['total_duration']:.1f} 秒 ({stats['total_duration']/60:.1f} 分钟)")
        
        print(f"\n🎵 音符类型分布:")
        for type_name, count in stats["note_types"].items():
            pct = count / max(stats["total_notes"], 1) * 100
            print(f"   {type_name}: {count} ({pct:.1f}%)")
        
        if stats["bpm_distribution"]:
            bpms = stats["bpm_distribution"]
            print(f"\n🎼 BPM 分布:")
            print(f"   最小: {min(bpms):.1f}")
            print(f"   最大: {max(bpms):.1f}")
            print(f"   平均: {np.mean(bpms):.1f}")
        
        if stats["note_density"]:
            densities = stats["note_density"]
            print(f"\n📊 音符密度 (音符/秒):")
            print(f"   最小: {min(densities):.2f}")
            print(f"   最大: {max(densities):.2f}")
            print(f"   平均: {np.mean(densities):.2f}")
        
        print(f"\n🎯 难度分布:")
        for diff, count in stats["difficulty_distribution"].items():
            pct = count / max(stats["total_charts"], 1) * 100
            print(f"   {diff}: {count} ({pct:.1f}%)")
        
        # 保存统计报告
        stats_file = self.chart_dir.parent / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # 转换 defaultdict 为普通 dict
            stats_copy = dict(stats)
            stats_copy["note_types"] = dict(stats["note_types"])
            stats_copy["difficulty_distribution"] = dict(stats["difficulty_distribution"])
            json.dump(stats_copy, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 统计报告已保存: {stats_file}")
        
        return stats
    
    def validate_data(self) -> bool:
        """
        验证数据完整性
        
        Returns:
            是否通过验证
        """
        print("\n" + "="*60)
        print("🔍 验证数据完整性")
        print("="*60)
        
        chart_files = self.find_chart_files()
        
        if not chart_files:
            print("\n❌ 没有找到谱面文件")
            return False
        
        valid_count = 0
        invalid_count = 0
        
        for chart_path in chart_files:
            try:
                chart = ChartData.load_from_file(str(chart_path))
                
                # 基本检查
                if not chart.judgeSegments:
                    print(f"⚠️  {chart_path.name}: 没有 judgeSegments")
                    invalid_count += 1
                    continue
                
                if chart.total_notes == 0:
                    print(f"⚠️  {chart_path.name}: 没有音符")
                    invalid_count += 1
                    continue
                
                valid_count += 1
                
            except Exception as e:
                print(f"❌ {chart_path.name}: 加载失败 - {e}")
                invalid_count += 1
        
        print(f"\n📊 验证结果:")
        print(f"   有效: {valid_count}")
        print(f"   无效: {invalid_count}")
        
        return invalid_count == 0


def main():
    parser = argparse.ArgumentParser(description="数据预处理工具")
    parser.add_argument("--chart-dir", type=str, default="data/processed",
                        help="谱面文件目录")
    parser.add_argument("--audio-dir", type=str, default="data/audio",
                        help="音频文件目录")
    parser.add_argument("--cache-dir", type=str, default="data/cache",
                        help="缓存目录")
    parser.add_argument("--check", action="store_true",
                        help="检查配对情况")
    parser.add_argument("--process", action="store_true",
                        help="处理音频特征")
    parser.add_argument("--stats", action="store_true",
                        help="生成统计报告")
    parser.add_argument("--validate", action="store_true",
                        help="验证数据完整性")
    parser.add_argument("--all", action="store_true",
                        help="执行所有步骤")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理")
    
    args = parser.parse_args()
    
    preparator = DataPreparator(
        chart_dir=args.chart_dir,
        audio_dir=args.audio_dir,
        cache_dir=args.cache_dir
    )
    
    if args.all or args.check:
        preparator.check_pairing()
    
    if args.all or args.validate:
        preparator.validate_data()
    
    if args.all or args.process:
        preparator.process_audio_features(force_reprocess=args.force)
    
    if args.all or args.stats:
        preparator.generate_stats()
    
    if not (args.check or args.process or args.stats or args.validate or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()
