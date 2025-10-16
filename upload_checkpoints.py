#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

from huggingface_hub import (
    HfApi,
    create_repo,
    CommitOperationAdd,
)

# ---------- 工具函数 ----------

def parse_size(s: str) -> int:
    """
    支持如 512M / 2GB / 10G / 1048576 这样的输入，返回字节数。
    """
    s = s.strip().upper().replace("B", "")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s[-1:] in units:
        return int(float(s[:-1]) * units[s[-1:]])
    return int(float(s))  # 纯字节

def iter_files(root: Path) -> Iterator[Tuple[Path, str, int]]:
    """
    遍历 root 下所有文件，返回 (绝对路径, 相对路径, 文件字节数)
    """
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(root)).replace(os.sep, "/")
            try:
                size = p.stat().st_size
            except OSError:
                continue
            yield p, rel, size

def human(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024 or unit == "PB":
            return f"{n:.2f}{unit}"
        n /= 1024

# ---------- 主逻辑 ----------

def main():
    parser = argparse.ArgumentParser(
        description="Upload a local folder to a Hugging Face model repo in robust batches."
    )
    parser.add_argument("--local", required=True, help="本地目录，例如 /home/ubuntu/Diversified-Reasoning/store")
    parser.add_argument("--repo", required=True, help="目标仓库，例如 guanning-ai/DSGRPO-7B")
    parser.add_argument("--path-in-repo", default="", help="上传到仓库内的子路径（默认仓库根）")
    parser.add_argument("--private", action="store_true", help="若仓库不存在则创建为私有")
    parser.add_argument("--chunk-bytes", default="10GB", help="每次提交的体量阈值，默认 10GB（支持 512M / 10GB 等）")
    parser.add_argument("--skip-existing", action="store_true", help="若远端已存在同名路径则跳过（简单断点续传）")
    parser.add_argument("--branch", default="main", help="提交到哪个分支，默认 main")
    args = parser.parse_args()

    local_root = Path(args.local)
    if not local_root.exists() or not local_root.is_dir():
        print(f"[错误] 本地目录不存在或不是目录：{local_root}", file=sys.stderr)
        sys.exit(1)

    chunk_limit = parse_size(args.chunk_bytes)
    repo_id = args.repo
    path_in_repo = args.path_in_repo.strip().strip("/")

    api = HfApi()

    # 1) 确保仓库存在
    print(f"[信息] 确保仓库存在：{repo_id} (model)")
    create_repo(
        repo_id=repo_id,
        exist_ok=True,
        repo_type="model",
        private=args.private,
    )

    # 2) 远端已有文件列表（用于 --skip-existing）
    existing_paths = set()
    if args.skip_existing:
        try:
            print("[信息] 拉取远端文件列表（用于跳过已存在路径）...")
            existing_paths = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=args.branch))
            # 如果设了 path_in_repo，仅比较该前缀下
            if path_in_repo:
                existing_paths = {p for p in existing_paths if p.startswith(path_in_repo + "/")}
        except Exception as e:
            print(f"[警告] 获取远端文件列表失败：{e}，继续上传但无法跳过已存在路径。")
            existing_paths = set()

    # 3) 遍历本地文件并分批提交
    batch_ops: List[CommitOperationAdd] = []
    batch_bytes = 0
    total_files = 0
    total_bytes = 0
    batch_idx = 0

    def flush_batch():
        nonlocal batch_ops, batch_bytes, batch_idx
        if not batch_ops:
            return
        batch_idx += 1
        msg = f"Upload batch #{batch_idx} ({len(batch_ops)} files, {human(batch_bytes)})"
        print(f"[提交] {msg} ...")
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=batch_ops,
            commit_message=msg,
            revision=args.branch,
        )
        print(f"[完成] batch #{batch_idx}")
        batch_ops = []
        batch_bytes = 0

    print(f"[信息] 开始扫描并上传：{local_root}")
    for abs_p, rel_p, sz in iter_files(local_root):
        repo_path = f"{path_in_repo}/{rel_p}" if path_in_repo else rel_p

        if args.skip_existing and repo_path in existing_paths:
            # 简单跳过同名路径（不校验内容/大小）
            print(f"[跳过] 已存在：{repo_path}")
            continue

        op = CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(abs_p))
        batch_ops.append(op)
        batch_bytes += sz
        total_files += 1
        total_bytes += sz

        # 达到阈值就提交一次
        if batch_bytes >= chunk_limit:
            flush_batch()

    # 提交尾批
    flush_batch()

    print(f"[完成] 全部上传结束。文件数：{total_files}，总大小：{human(total_bytes)}")
    if total_files == 0:
        print("[提示] 没有需要上传的文件（可能全部已存在或目录为空）。")

if __name__ == "__main__":
    main()
