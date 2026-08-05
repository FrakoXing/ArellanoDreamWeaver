#!/usr/bin/env python3
"""
Phira 谱面批量下载工具 - 防丢数据终极修复版
解决：Ctrl+C、断电、强制杀进程导致下载记录丢失问题
三层持久化：内存集合 + 增量日志兜底 + 主进度原子文件
"""
import argparse
import asyncio
import json
import os
import sys
import time
import re
import signal
from pathlib import Path
from threading import Lock as SyncLock

# ============================================================
# 依赖检查
# ============================================================
try:
    import aiohttp
except ImportError:
    print("需要安装 aiohttp: pip install aiohttp")
    sys.exit(1)

# ============================================================
# 全局配置
# ============================================================
API_BASE = "https://api.phira.cn"
CHART_API = f"{API_BASE}/chart/{{chart_id}}"
DEFAULT_OUTPUT = "./phira_charts"
DEFAULT_CONCURRENT = 16
REQUEST_TIMEOUT = 30
RETRY_COUNT = 3
RETRY_DELAY = 1
CHUNK_SIZE = 65536
PROGRESS_SAVE_INTERVAL = 100
TCP_LIMIT = 100
TCP_LIMIT_PER_HOST = 30
DNS_TTL = 600
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://phira.cn/"
}

# ============================================================
# 工具函数
# ============================================================
def safe_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    safe = safe.strip(' .')
    return safe[:200] if len(safe) > 200 else safe or "unknown"

async def async_path_exists(path: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, os.path.exists, path)

def clean_tmp_files(dir_path: Path):
    for f in dir_path.glob("*.tmp"):
        try:
            os.remove(f)
        except Exception:
            pass

# 同步追加写入增量日志（兜底防丢）
sync_log_lock = SyncLock()
def append_incr_log(log_path: Path, cid: int, tag: str):
    """tag: ok / missing / error"""
    with sync_log_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())},{cid},{tag}\n")

# ============================================================
# 下载核心类（三层持久化防丢）
# ============================================================
class PhiraDownloader:
    def __init__(self, output_dir: str, concurrent: int = DEFAULT_CONCURRENT, timeout: int = REQUEST_TIMEOUT):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.concurrent = concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # 进度文件
        self.progress_file = self.output_dir / ".download_progress.json"
        self.incr_log = self.output_dir / ".incr_progress.log"  # 增量兜底日志
        self.save_lock = asyncio.Lock()
        self.stat_lock = asyncio.Lock()

        # 内存缓存
        self.downloaded_ids = set()
        self.missing_ids = set()
        self.stats = {"total": 0, "downloaded": 0, "not_found": 0, "error": 0, "skipped_exists": 0}
        self.downloaded_bytes = 0

        # 运行状态
        self.semaphore = asyncio.Semaphore(concurrent)
        self.done_count = 0
        self.total_count = 0
        self.start_time = 0
        self.force_exit = False
        self.all_tasks = []  # 保存全部任务，用于优雅等待退出

        # 加载进度 + 增量日志修复丢失数据
        self._load_all_progress()

    def _load_all_progress(self):
        """加载主进度 + 增量日志兜底修复丢失ID"""
        # 1. 加载主进度文件
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.downloaded_ids = set(data.get("downloaded_ids", []))
                    self.missing_ids = set(data.get("missing_ids", []))
            except Exception:
                print("[警告] 主进度文件损坏，将使用增量日志恢复数据")

        # 2. 解析增量日志，修复丢失记录（核心防丢）
        if self.incr_log.exists():
            try:
                with open(self.incr_log, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        if len(parts) < 3:
                            continue
                        _, cid_str, tag = parts
                        try:
                            cid = int(cid_str)
                            if tag == "ok":
                                self.downloaded_ids.add(cid)
                            elif tag == "missing":
                                self.missing_ids.add(cid)
                        except ValueError:
                            continue
                print(f"[恢复] 从增量日志加载：成功{len(self.downloaded_ids)} 黑名单{len(self.missing_ids)}")
            except Exception as e:
                print(f"[警告] 增量日志读取失败: {e}")

    def _save_progress_sync(self):
        """原子写入主进度文件"""
        tmp_path = str(self.progress_file) + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "downloaded_ids": sorted(self.downloaded_ids),
                    "missing_ids": sorted(self.missing_ids),
                    "stats": self.stats
                }, f, indent=2)
            os.replace(tmp_path, self.progress_file)
        except Exception as e:
            print(f"\n[警告] 主进度保存失败: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def _save_progress(self):
        async with self.save_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_progress_sync)

    async def _fetch_json(self, session: aiohttp.ClientSession, url: str) -> dict:
        for attempt in range(RETRY_COUNT):
            try:
                async with session.get(url, timeout=self.timeout) as resp:
                    if resp.status == 404:
                        raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=404)
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ClientResponseError) as e:
                if attempt == RETRY_COUNT - 1:
                    raise
                await asyncio.sleep(RETRY_DELAY)

    async def _download_file(self, session: aiohttp.ClientSession, url: str, dest_path: str) -> int:
        tmp_path = dest_path + ".tmp"
        try:
            for attempt in range(RETRY_COUNT):
                try:
                    async with session.get(url, timeout=self.timeout) as resp:
                        resp.raise_for_status()
                        total_bytes = 0
                        loop = asyncio.get_running_loop()
                        f = await loop.run_in_executor(None, open, tmp_path, "wb")
                        try:
                            async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                                await loop.run_in_executor(None, f.write, chunk)
                                total_bytes += len(chunk)
                        finally:
                            await loop.run_in_executor(None, f.close)
                        await loop.run_in_executor(None, os.replace, tmp_path, dest_path)
                        return total_bytes
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    if attempt == RETRY_COUNT - 1:
                        raise
                    await asyncio.sleep(RETRY_DELAY)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    async def download_one(self, session: aiohttp.ClientSession, chart_id: int) -> dict:
        res = {"id": chart_id, "status": "unknown", "message": "", "file": ""}
        async with self.semaphore:
            # 黑名单直接跳过
            if chart_id in self.missing_ids:
                async with self.stat_lock:
                    self.stats["not_found"] += 1
                res["status"] = "not_found"
                res["message"] = "黑名单不存在"
                return res

            api_url = CHART_API.format(chart_id=chart_id)
            # 请求谱面信息
            try:
                info = await self._fetch_json(session, api_url)
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    # 记录黑名单 + 写入增量日志兜底
                    async with self.stat_lock:
                        self.stats["not_found"] += 1
                    self.missing_ids.add(chart_id)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, append_incr_log, self.incr_log, chart_id, "missing")
                    res["status"] = "not_found"
                    res["message"] = "404不存在"
                    return res
                else:
                    async with self.stat_lock:
                        self.stats["error"] += 1
                    res["status"] = "error"
                    res["message"] = f"HTTP {e.status}"
                    return res
            except Exception as e:
                async with self.stat_lock:
                    self.stats["error"] += 1
                res["status"] = "error"
                res["message"] = f"网络失败: {str(e)}"
                return res

            if info.get("code") == "NOT_FOUND":
                async with self.stat_lock:
                    self.stats["not_found"] += 1
                self.missing_ids.add(chart_id)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, append_incr_log, self.incr_log, chart_id, "missing")
                res["status"] = "not_found"
                return res

            file_url = info.get("file")
            if not file_url:
                async with self.stat_lock:
                    self.stats["error"] += 1
                res["status"] = "error"
                res["message"] = "无下载链接"
                return res

            chart_name = info.get("name", f"chart_{chart_id}")
            chart_level = info.get("level", "unknown")
            charter = info.get("charter", "unknown")
            filename = f"{chart_id}_{safe_filename(chart_level)}_{safe_filename(chart_name)}_{safe_filename(charter)}.pez"
            dest_path = str(self.output_dir / filename)

            dest_exists = await async_path_exists(dest_path)
            if chart_id in self.downloaded_ids or dest_exists:
                async with self.stat_lock:
                    self.stats["skipped_exists"] += 1
                res["status"] = "skipped"
                res["file"] = dest_path
                return res

            # 执行下载
            try:
                bytes_down = await self._download_file(session, file_url, dest_path)
            except Exception as e:
                async with self.stat_lock:
                    self.stats["error"] += 1
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, append_incr_log, self.incr_log, chart_id, "error")
                res["status"] = "error"
                res["message"] = f"下载失败: {str(e)}"
                return res

            # === 下载成功：立刻写入增量日志兜底（防止丢数据核心）===
            async with self.stat_lock:
                self.stats["downloaded"] += 1
                self.downloaded_ids.add(chart_id)
                self.downloaded_bytes += bytes_down
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, append_incr_log, self.incr_log, chart_id, "ok")

            res["status"] = "ok"
            res["message"] = f"{chart_name} [{chart_level}]"
            res["file"] = dest_path
            return res

    async def _progress_reporter(self):
        while self.done_count < self.total_count and not self.force_exit:
            await asyncio.sleep(2)
            elapsed = time.time() - self.start_time
            speed = self.downloaded_bytes / elapsed / 1024 / 1024 if elapsed else 0
            rate = self.done_count / elapsed if elapsed else 0
            eta = (self.total_count - self.done_count) / rate if rate else 0
            sys.stdout.write(
                f"\r  [{self.done_count}/{self.total_count}] {self.done_count/self.total_count*100:.1f}% | {speed:.2f}MB/s | ETA:{eta:.0f}s   "
            )
            sys.stdout.flush()

    def handle_exit_signal(self, *_):
        print("\n\n[!] 收到终止信号，等待当前任务完成并保存全部进度，请勿强制关闭！")
        self.force_exit = True

    async def run_async(self, chart_ids: list):
        self.stats["total"] = len(chart_ids)
        self.total_count = len(chart_ids)

        # 注册退出信号
        signal.signal(signal.SIGINT, self.handle_exit_signal)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, self.handle_exit_signal)

        print(f"📥 Phira 谱面下载（防丢数据终极版）")
        print("="*60)
        print(f"总数：{len(chart_ids)} | 并发：{self.concurrent}")
        print(f"已记录成功：{len(self.downloaded_ids)} | 黑名单无效ID：{len(self.missing_ids)}")
        print(f"输出目录：{self.output_dir.absolute()}")
        print("="*60 + "\n")

        self.start_time = time.time()
        connector = aiohttp.TCPConnector(limit=TCP_LIMIT, limit_per_host=TCP_LIMIT_PER_HOST, ttl_dns_cache=DNS_TTL, use_dns_cache=True)

        try:
            async with aiohttp.ClientSession(connector=connector, headers=HEADERS) as session:
                reporter = asyncio.create_task(self._progress_reporter())
                # 保存全部任务列表，用于中断后等待完成
                self.all_tasks = [asyncio.create_task(self.download_one(session, cid)) for cid in chart_ids]

                # 遍历所有任务，不中途break，即使触发exit也要等正在运行的协程结束
                for coro in asyncio.as_completed(self.all_tasks):
                    result = await coro
                    self.done_count += 1
                    # 每间隔批量保存主进度
                    if self.done_count % PROGRESS_SAVE_INTERVAL == 0:
                        await self._save_progress()
                    # 触发退出标记不中断当前任务，仅不再新建
                    if self.force_exit:
                        print(f"\n[暂停] 已接收终止信号，等待剩余{self.total_count - self.done_count}个任务完成...")

                reporter.cancel()
                try:
                    await reporter
                except asyncio.CancelledError:
                    pass
        finally:
            # 程序结束强制全量写入主进度文件
            print("\n[持久化] 正在写入完整进度文件...")
            await self._save_progress()
            clean_tmp_files(self.output_dir)
            print("[完成] 临时文件清理完毕，所有记录已落地磁盘")

        # 统计输出
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print("📊 下载汇总")
        print("="*60)
        print(f"总任务：{self.stats['total']}")
        print(f"成功下载：{self.stats['downloaded']}")
        print(f"不存在黑名单：{self.stats['not_found']}")
        print(f"跳过已存在：{self.stats['skipped_exists']}")
        print(f"下载失败：{self.stats['error']}")
        print(f"总耗时：{elapsed:.1f}s")
        if self.stats["downloaded"] > 0:
            total_mb = self.downloaded_bytes / 1024 / 1024
            print(f"总流量：{total_mb:.1f} MB | 平均带宽：{total_mb/elapsed:.2f} MB/s")
        print("="*60)

# ============================================================
# 命令行解析
# ============================================================
def parse_id_range(spec: str) -> list:
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-", 1)
            ids.extend(range(int(s.strip()), int(e.strip()) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))

def main():
    parser = argparse.ArgumentParser(
        description="Phira批量谱面下载 - 防丢失数据修复版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
python download_phira.py --start 1 --end 1000 -c 16
python download_phira.py --range "1-500,1000-1200"
python download_phira.py --ids 123 456 789
"""
    )
    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--ids", nargs="+", type=int)
    id_group.add_argument("--range", type=str)
    id_group.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("-c", "--concurrent", type=int, default=DEFAULT_CONCURRENT)
    args = parser.parse_args()

    # 参数校验
    if args.start is not None and args.end is None:
        parser.error("--start 必须搭配 --end 使用")
    if args.end is not None and args.start is None:
        parser.error("--end 必须搭配 --start 使用")

    # 生成ID列表
    if args.ids:
        chart_ids = sorted(set(args.ids))
    elif args.range:
        chart_ids = parse_id_range(args.range)
    elif args.start and args.end:
        chart_ids = list(range(args.start, args.end + 1))
    else:
        parser.error("必须指定 --ids / --range / --start+--end")

    if not chart_ids:
        print("[错误] 无有效下载ID")
        return

    # Windows事件循环兼容
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    dl = PhiraDownloader(output_dir=args.output, concurrent=args.concurrent)
    asyncio.run(dl.run_async(chart_ids))

if __name__ == "__main__":
    main()