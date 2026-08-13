# -*- coding: utf-8 -*-
"""
检查训练数据中的 time_offset 分布，找出为什么 time_offset=15.75bt 占 31.53%
"""
import json
from pathlib import Path
from collections import Counter

token_dir = Path('data/tokens')
time_offset_counts = Counter()
total_time_offsets = 0
files_with_high_max_offset = []

for f in token_dir.glob('*.json'):
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        tokens = data.get('tokens', [])
        
        # 统计这个文件中的 time_offset 分布
        file_offsets = Counter()
        for t in tokens:
            if 7 <= t <= 70:  # time_offset tokens
                time_offset_counts[t] += 1
                file_offsets[t] += 1
                total_time_offsets += 1
        
        # 检查是否有异常高的 time_offset
        if file_offsets:
            max_offset = max(file_offsets.keys())
            if max_offset == 70:  # time_offset=15.75bt
                files_with_high_max_offset.append((f.name, file_offsets[70], sum(file_offsets.values())))
                
    except Exception as e:
        pass

print(f'Total time_offset tokens: {total_time_offsets}')
print(f'\n=== Time Offset Distribution ===')
for tid in range(7, 71):
    count = time_offset_counts.get(tid, 0)
    pct = count / total_time_offsets * 100 if total_time_offsets > 0 else 0
    offset_beats = (tid - 7) * 0.25
    bar = '█' * int(pct / 2)
    print(f'  {offset_beats:5.2f}bt ({tid:3d}): {count:8d} ({pct:5.2f}%) {bar}')

print(f'\n=== Files with time_offset=15.75bt ===')
print(f'Found {len(files_with_high_max_offset)} files with max offset=15.75bt')
for fname, count_70, total_offsets in files_with_high_max_offset[:20]:
    pct = count_70 / total_offsets * 100 if total_offsets > 0 else 0
    print(f'  {fname}: {count_70}/{total_offsets} ({pct:.1f}%) are 15.75bt')
