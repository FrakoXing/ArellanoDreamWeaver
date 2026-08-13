# -*- coding: utf-8 -*-
import json
from pathlib import Path

# 查看几个训练数据文件的详细内容
token_dir = Path('data/tokens')
files = list(token_dir.glob('*.json'))[:5]

for f in files:
    print(f'\n=== {f.name} ===')
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    tokens = data.get('tokens', [])
    print(f'Total tokens: {len(tokens)}')
    print(f'First 50 tokens: {tokens[:50]}')
    
    # 统计这个文件中的note类型
    from collections import Counter
    counter = Counter(tokens)
    
    tap = counter.get(4, 0)
    hold = counter.get(5, 0)
    slide = counter.get(6, 0)
    total_notes = tap + hold + slide
    
    print(f'Notes: TAP={tap}, HOLD={hold}, SLIDE={slide}, total={total_notes}')
    
    # 看看time_offset分布
    time_offsets = {k: v for k, v in counter.items() if 7 <= k <= 70}
    print(f'Time offset tokens: {len(time_offsets)} unique values')
    if time_offsets:
        max_offset = max(time_offsets.keys())
        print(f'  Max time_offset token: {max_offset} (={((max_offset-7)*0.25):.2f}bt, count={time_offsets[max_offset]})')
