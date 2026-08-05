import json, os, glob

token_dir = 'data/tokens'
token_files = glob.glob(os.path.join(token_dir, '*.json'))

lengths = []
total_tokens = 0
file_count = 0

for f in token_files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        tokens = data.get('tokens', [])
        length = len(tokens)
        lengths.append(length)
        total_tokens += length
        file_count += 1
    except:
        pass

lengths.sort()

print(f"=== Token Data Stats ===")
print(f"Total files: {file_count}")
print(f"Total tokens: {total_tokens:,}")
print(f"Avg tokens/file: {total_tokens // max(1, file_count):,}")
print()
print(f"=== Sequence Length Distribution ===")
print(f"Min: {min(lengths) if lengths else 0}")
print(f"Max: {max(lengths) if lengths else 0}")
print(f"Median: {lengths[len(lengths)//2] if lengths else 0}")
print(f"Mean: {sum(lengths)/len(lengths):.0f}" if lengths else "")

# Percentiles
for p in [10, 25, 50, 75, 90, 95, 99]:
    idx = int(len(lengths) * p / 100)
    idx = min(idx, len(lengths) - 1)
    print(f"  P{p}: {lengths[idx]}")

# Bucket distribution
buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, 20000), (20000, 50000), (50000, 999999)]
print(f"\n=== Length Buckets ===")
for lo, hi in buckets:
    count = sum(1 for l in lengths if lo <= l < hi)
    if count > 0:
        label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"
        print(f"  {label:>12}: {count} files ({count*100/len(lengths):.1f}%)")

# Estimate training data in bytes
# Each token is an int, typically stored as int64 (8 bytes) in training
data_bytes = total_tokens * 8
print(f"\n=== Memory Estimate ===")
print(f"Raw token data: {data_bytes / 1024 / 1024:.1f} MB")
print(f"With attention (seq_len=2048, batch=32): ~{data_bytes * 4 / 1024 / 1024:.1f} MB (4x for gradients)")
