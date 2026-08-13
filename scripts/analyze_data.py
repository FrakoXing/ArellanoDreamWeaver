# -*- coding: utf-8 -*-
import json
from pathlib import Path
from collections import Counter

# 统计训练数据中的token分布
token_dir = Path('data/tokens')
all_tokens = []
files_count = 0

for f in token_dir.glob('*.json'):
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        files_count += 1
        
        if isinstance(data, dict) and 'tokens' in data:
            all_tokens.extend(data['tokens'])
        elif isinstance(data, list):
            all_tokens.extend(data)
    except Exception as e:
        pass

print(f'Files processed: {files_count}')
print(f'Total tokens: {len(all_tokens)}')

# Token ID 含义 (来自 note_decoder.py)
# 0=PAD, 1=SOS, 2=EOS, 3=MASK
# 4=TAP, 5=HOLD, 6=SLIDE
# 7-70=time_offset, 71-79=pos_x, 80-111=hold_time
# 112=ABOVE, 113=BELOW

counter = Counter(all_tokens)

print(f'\n=== Token Distribution ===')
print(f'Special tokens:')
for tid, name in [(0,'PAD'), (1,'SOS'), (2,'EOS'), (3,'MASK')]:
    c = counter.get(tid, 0)
    print(f'  {name}({tid}): {c} ({c/len(all_tokens)*100:.2f}%)')

print(f'\nNote type tokens:')
for tid, name in [(4,'TAP'), (5,'HOLD'), (6,'SLIDE')]:
    c = counter.get(tid, 0)
    print(f'  {name}({tid}): {c} ({c/len(all_tokens)*100:.2f}%)')

print(f'\nDirection tokens:')
for tid, name in [(112,'ABOVE'), (113,'BELOW')]:
    c = counter.get(tid, 0)
    print(f'  {name}({tid}): {c} ({c/len(all_tokens)*100:.2f}%)')

print(f'\nTime offset tokens (7-70): {sum(counter.get(i,0) for i in range(7,71))}')
print(f'Pos X tokens (71-79): {sum(counter.get(i,0) for i in range(71,80))}')
print(f'Hold time tokens (80-111): {sum(counter.get(i,0) for i in range(80,112))}')

# Top 20 most common tokens
print(f'\n=== Top 20 Most Common Tokens ===')
for tid, count in counter.most_common(20):
    if tid == 0: name = 'PAD'
    elif tid == 1: name = 'SOS'
    elif tid == 2: name = 'EOS'
    elif tid == 3: name = 'MASK'
    elif tid == 4: name = 'TAP'
    elif tid == 5: name = 'HOLD'
    elif tid == 6: name = 'SLIDE'
    elif tid == 112: name = 'ABOVE'
    elif tid == 113: name = 'BELOW'
    elif 7 <= tid <= 70: name = f'time_offset={((tid-7)*0.25):.2f}bt'
    elif 71 <= tid <= 79: name = f'x={[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.0][tid-71]}'
    elif 80 <= tid <= 111: name = f'hold_time={((tid-80)*0.25):.2f}bt'
    else: name = f'token_{tid}'
    print(f'  {name}({tid}): {count} ({count/len(all_tokens)*100:.2f}%)')
