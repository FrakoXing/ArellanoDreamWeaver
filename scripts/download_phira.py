#!/usr/bin/env python3
"""
Phira 谱面批量下载工具
======================
从 api.phira.cn 批量下载 .pez 谱面文件

用法:
  python download_phira.py --start 1 --end 50000 --output ./phira_charts
  python download_phira.py --ids 46274 46275 46276 --output ./phira_charts
  python download_phira.py --start 40000 --end 50000 --output ./phira_charts --concurrent 5
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ============================================================
# 配置
# ============================================================
API_BASE = "https://api.phira.cn"
CHART_API = f"{API_BASE}/chart/{{chart_id}}"
DEFAULT_OUTPUT = "./phira_charts"
DEFAULT_CONCURRENT = 3
REQUEST_TIMEOUT = 60  # 秒 (慢网络需要更长时间)
RETRY_COUNT = 3       # 重试次数
RETRY_DELAY = 3       # 重试间隔 (秒)

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


def fetch_json(url: str, timeout: int = REQUEST_TIMEOUT, retries: int = RETRY_COUNT) -> dict:
    """请求 JSON API，返回解析后的 dict (带重试)"""
    req = Request(url, headers={
        "User-Agent": "PhiraBatchDownloader/1.0",
        "Accept": "application/json"
    })
    
    last_error = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except HTTPError:
            raise  # HTTP 错误不重试
        except (URLError, TimeoutError, Exception) as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
    
    raise last_error


def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """下载文件到指定路径"""
    req = Request(url, headers={
        "User-Agent": "PhiraBatchDownloader/1.0"
    })
    try:
        with urlopen(req, timeout=timeout) as resp:
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception:
        # 下载失败，删除不完整的文件
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# ============================================================
# 核心下载逻辑
# ============================================================

class PhiraDownloader:
    def __init__(self, output_dir: str, concurrent: int = DEFAULT_CONCURRENT,
                 timeout: int = REQUEST_TIMEOUT, delay: float = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.concurrent = concurrent
        self.timeout = timeout
        self.delay = delay
        
        # 统计
        self.lock = Lock()
        self.stats = {
            "total": 0,
            "downloaded": 0,
            "not_found": 0,
            "error": 0,
            "skipped_exists": 0,
        }
        
        # 进度记录文件
        self.progress_file = self.output_dir / ".download_progress.json"
        self.downloaded_ids = self._load_progress()
    
    def _load_progress(self) -> set:
        """加载已下载记录（断点续传）"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("downloaded_ids", []))
            except Exception:
                pass
        return set()
    
    def _save_progress(self):
        """保存下载进度"""
        with self.lock:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump({
                    "downloaded_ids": sorted(self.downloaded_ids),
                    "stats": self.stats
                }, f, indent=2)
    
    def _update_stats(self, key: str):
        with self.lock:
            self.stats[key] += 1
    
    def download_one(self, chart_id: int) -> dict:
        """
        下载单个谱面
        
        返回: {"id": int, "status": str, "message": str, "file": str}
        """
        result = {"id": chart_id, "status": "unknown", "message": "", "file": ""}
        
        # 延迟请求 (避免太快被 ban)
        if self.delay > 0:
            time.sleep(self.delay)
        
        # 1. 请求谱面信息
        api_url = CHART_API.format(chart_id=chart_id)
        try:
            info = fetch_json(api_url, timeout=self.timeout)
        except HTTPError as e:
            if e.code == 404:
                self._update_stats("not_found")
                result["status"] = "not_found"
                result["message"] = "谱面不存在 (HTTP 404)"
                return result
            else:
                self._update_stats("error")
                result["status"] = "error"
                result["message"] = f"HTTP 错误 {e.code}"
                return result
        except (URLError, TimeoutError, Exception) as e:
            self._update_stats("error")
            result["status"] = "error"
            result["message"] = f"请求失败: {e}"
            return result
        
        # 2. 检查是否 NOT_FOUND
        if info.get("code") == "NOT_FOUND":
            self._update_stats("not_found")
            result["status"] = "not_found"
            result["message"] = "谱面不存在 (NOT_FOUND)"
            return result
        
        # 3. 提取信息
        chart_name = info.get("name", f"chart_{chart_id}")
        chart_level = info.get("level", "unknown")
        charter = info.get("charter", "unknown")
        file_url = info.get("file")
        
        if not file_url:
            self._update_stats("error")
            result["status"] = "error"
            result["message"] = "没有文件下载链接"
            return result
        
        # 4. 构建文件名: ID_难度_谱名_谱师.pez
        filename = f"{chart_id}_{safe_filename(chart_level)}_{safe_filename(chart_name)}_{safe_filename(charter)}.pez"
        dest_path = self.output_dir / filename
        
        # 5. 检查是否已下载
        if chart_id in self.downloaded_ids or dest_path.exists():
            self._update_stats("skipped_exists")
            result["status"] = "skipped"
            result["message"] = f"已存在: {filename}"
            result["file"] = str(dest_path)
            return result
        
        # 6. 下载文件
        success = download_file(file_url, str(dest_path))
        if not success:
            # 重试一次
            time.sleep(1)
            success = download_file(file_url, str(dest_path))
        
        if success:
            self._update_stats("downloaded")
            with self.lock:
                self.downloaded_ids.add(chart_id)
            result["status"] = "ok"
            result["message"] = f"{chart_name} [{chart_level}] by {charter}"
            result["file"] = str(dest_path)
            
            # 同时保存元信息 JSON
            meta_path = dest_path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        else:
            self._update_stats("error")
            result["status"] = "error"
            result["message"] = "文件下载失败"
        
        return result
    
    def run(self, chart_ids: list):
        """批量下载"""
        self.stats["total"] = len(chart_ids)
        
        print(f"📥 Phira 谱面批量下载")
        print(f"{'=' * 60}")
        print(f"  目标数量: {len(chart_ids)}")
        print(f"  并发数:   {self.concurrent}")
        print(f"  输出目录: {self.output_dir.absolute()}")
        print(f"  已下载过: {len(self.downloaded_ids)}")
        print(f"{'=' * 60}\n")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.concurrent) as executor:
            futures = {
                executor.submit(self.download_one, cid): cid 
                for cid in chart_ids
            }
            
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                
                # 进度显示
                progress = done_count / len(chart_ids) * 100
                status_icon = {
                    "ok": "✅",
                    "not_found": "❌",
                    "error": "⚠️",
                    "skipped": "⏭️",
                }.get(result["status"], "?")
                
                print(f"[{done_count}/{len(chart_ids)}] ({progress:.1f}%) "
                      f"{status_icon} ID={result['id']} — {result['message']}")
                
                # 每 50 个保存一次进度
                if done_count % 50 == 0:
                    self._save_progress()
        
        # 最终保存进度
        self._save_progress()
        
        elapsed = time.time() - start_time
        
        # 打印统计
        print(f"\n{'=' * 60}")
        print(f"📊 下载完成!")
        print(f"{'=' * 60}")
        print(f"  总计:     {self.stats['total']}")
        print(f"  成功下载: {self.stats['downloaded']}")
        print(f"  不存在:   {self.stats['not_found']}")
        print(f"  已跳过:   {self.stats['skipped_exists']}")
        print(f"  失败:     {self.stats['error']}")
        print(f"  耗时:     {elapsed:.1f}s")
        if self.stats['downloaded'] > 0:
            print(f"  平均速度: {elapsed / self.stats['downloaded']:.2f}s/个")
        print(f"{'=' * 60}")


# ============================================================
# 命令行
# ============================================================

def parse_id_range(spec: str) -> list:
    """
    解析 ID 范围 specification
    
    支持格式:
      "1-100"       → 1 到 100
      "1,2,3"       → [1, 2, 3]
      "1-100,200"   → 1到100 + 200
      "46274"       → [46274]
    """
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser(
        description="Phira 谱面批量下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 ID 1 到 50000 的所有谱面
  python download_phira.py --start 1 --end 50000

  # 下载指定 ID
  python download_phira.py --ids 46274 46275 46276

  # 混合范围
  python download_phira.py --range "1-1000,5000-6000,46274"

  # 自定义并发数和输出目录
  python download_phira.py --start 1 --end 10000 --concurrent 5 --output ./my_charts
        """
    )
    
    # ID 选择方式 (三选一)
    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--ids", nargs="+", type=int,
                          help="指定要下载的谱面 ID 列表")
    id_group.add_argument("--range", type=str,
                          help="ID 范围，如 '1-1000,5000-6000,46274'")
    id_group.add_argument("--start", type=int,
                          help="起始 ID (配合 --end 使用)")
    
    parser.add_argument("--end", type=int,
                        help="结束 ID (配合 --start 使用)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"输出目录 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--concurrent", "-c", type=int, default=DEFAULT_CONCURRENT,
                        help=f"并发下载数 (默认: {DEFAULT_CONCURRENT})")
    
    args = parser.parse_args()
    
    # 解析 ID 列表
    if args.ids:
        chart_ids = sorted(set(args.ids))
    elif args.range:
        chart_ids = parse_id_range(args.range)
    elif args.start and args.end:
        chart_ids = list(range(args.start, args.end + 1))
    else:
        parser.error("请指定 --ids, --range, 或 --start + --end")
        return
    
    if not chart_ids:
        print("[ERROR] 没有要下载的 ID")
        return
    
    # 开始下载
    downloader = PhiraDownloader(
        output_dir=args.output,
        concurrent=args.concurrent
    )
    downloader.run(chart_ids)


if __name__ == "__main__":
    main()
