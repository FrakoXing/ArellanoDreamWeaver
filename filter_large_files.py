#!/usr/bin/env python3
"""从 Git 历史中删除大文件"""
import subprocess
import sys

# 需要删除的大文件
large_files = [
    "data/tokens/15345_ATLv.15_Nebulae Colliderz_烈華斯林.json",
    "data/tokens/14684_FP Lv.2024_SkyFire_BiliUID_544310035.json"
]

print("开始从 Git 历史中删除大文件...")
print(f"文件列表: {large_files}")

# 构建 git rm 命令
git_rm_cmd = "git rm --cached --ignore-unmatch " + " ".join(f'"{f}"' for f in large_files)
print(f"\n执行命令: {git_rm_cmd}")

# 执行 git filter-branch
cmd = [
    "git", "filter-branch", "--force",
    "--index-filter", git_rm_cmd,
    "--prune-empty",
    "--tag-name-filter", "cat",
    "--", "--all"
]

print(f"\n运行: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("\n✅ 成功!")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
except subprocess.CalledProcessError as e:
    print(f"\n❌ 失败!")
    print(f"返回码: {e.returncode}")
    print(f"STDOUT: {e.stdout}")
    print(f"STDERR: {e.stderr}")
    sys.exit(1)

print("\n清理引用...")
subprocess.run(["git", "for-each-ref", "--format=delete %(refname)", "refs/original/"], check=False)
subprocess.run(["git", "reflog", "expire", "--expire=now", "--all"], check=False)
subprocess.run(["git", "gc", "--prune=now", "--aggressive"], check=False)

print("\n✅ 完成! 现在可以推送了。")
