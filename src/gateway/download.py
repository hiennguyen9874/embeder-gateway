"""Downloader: fetch exactly the Checkpoint files named in YAML into models/<org>--<repo>/."""

import os
import subprocess
from pathlib import Path

from gateway.config import Registry

LFS_POINTER = b"version https://git-lfs.github.com/spec/"


def _is_downloaded(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as f:
        return not f.read(len(LFS_POINTER)).startswith(LFS_POINTER)


def download(registry: Registry) -> None:
    for c in registry.checkpoints.values():
        if all(_is_downloaded(c.dir / f) for f in c.files):
            print(f"{c.name}: up to date in {c.dir}")
            continue
        if not (c.dir / ".git").exists():
            subprocess.run(
                ["git", "clone", f"https://huggingface.co/{c.repo}", str(c.dir)],
                env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
                check=True,
            )
        subprocess.run(["git", "lfs", "pull", "--include", ",".join(c.files)], cwd=c.dir, check=True)
        missing = [f for f in c.files if not _is_downloaded(c.dir / f)]
        if missing:
            raise RuntimeError(f"{c.name}: {missing} not found in {c.repo}")
        print(f"{c.name}: downloaded {c.files} into {c.dir}")
