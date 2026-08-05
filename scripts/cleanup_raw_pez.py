import os, glob

d = r'F:\Arellano\ArellanoDreamWeaver\data\raw_pez'
removed = 0
for f in glob.glob(os.path.join(d, '*')):
    if os.path.basename(f) not in ['.download_progress.json', '.incr_progress.log']:
        try:
            os.remove(f)
            removed += 1
        except Exception as e:
            print(f'Error removing {f}: {e}')
print(f'Removed {removed} files, kept progress files')
