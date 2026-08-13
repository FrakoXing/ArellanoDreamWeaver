# -*- coding: utf-8 -*-
import json
from pathlib import Path

progress_file = Path('data/raw_pez/.download_progress.json')
incr_log = Path('data/raw_pez/.incr_progress.log')

if progress_file.exists():
    with open(progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    downloaded = data.get('downloaded_ids', [])
    missing = data.get('missing_ids', [])
    print(f'进度文件:')
    print(f'  已下载 ID 数量: {len(downloaded)}')
    print(f'  黑名单 ID 数量: {len(missing)}')
    if downloaded:
        print(f'  已下载 ID 示例: {downloaded[:10]}')
    if missing:
        print(f'  黑名单 ID 示例: {missing[:10]}')
else:
    print('进度文件不存在')

if incr_log.exists():
    with open(incr_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'\n增量日志:')
    print(f'  总行数: {len(lines)}')
    if lines:
        print(f'  前 5 行: {lines[:5]}')
else:
    print('\n增量日志不存在')

# 检查输出目录中的文件
pez_files = list(Path('data/raw_pez').glob('*.pez'))
print(f'\n输出目录中的 .pez 文件数量: {len(pez_files)}')
if pez_files:
    print(f'  文件示例: {[f.name for f in pez_files[:5]]}')
