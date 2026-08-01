import os
config_dir = r'F:\Arellano\ArellanoDreamWeaver\configs'
for fname in os.listdir(config_dir):
    if fname.endswith('.yaml') or fname.endswith('.yml'):
        fpath = os.path.join(config_dir, fname)
        with open(fpath, 'rb') as f:
            raw = f.read()
        # 修复 \r\r\n -> \r\n
        fixed = raw.replace(b'\r\r\n', b'\r\n')
        # 也修复可能残留的 \r\r
        fixed = fixed.replace(b'\r\r', b'\r')
        if fixed != raw:
            with open(fpath, 'wb') as f:
                f.write(fixed)
            print(f'{fname}: fixed line endings')
        else:
            print(f'{fname}: OK')
print('Done')