"""Run manifest: everything needed to say exactly what produced a result.

Every run record will embed this dict (from C03 on). `code_hash` covers the
Python sources, so two runs with the same hash ran the same code even if git
was dirty at the time.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from config import CFG, ROOT
from llm import is_reasoning_model, sampling_params

PACKAGES = ["openai", "agentdojo", "numpy", "torch", "sentence-transformers",
            "transformers", "scikit-learn", "pydantic", "pyyaml"]


def code_hash(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _git(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def build_manifest() -> dict:
    commit = _git(ROOT, "rev-parse", "HEAD")
    dirty = _git(ROOT, "status", "--porcelain")
    return {
        "code_hash": code_hash(),
        "git_commit": commit,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "env_prefix": sys.prefix,
        "packages": {p: _version(p) for p in PACKAGES},
        "llm": {
            "model": CFG.model,
            "reasoning_model": is_reasoning_model(CFG.model),
            "sampling": sampling_params(CFG, CFG.model),
            "embed_model": CFG.embed_model,
            "xgguard_model": CFG.xgguard_model,
        },
        "benchmark": {
            "agentdojo_version": _version("agentdojo"),
            "agentdojo_benchmark": CFG.agentdojo_benchmark,
            "agentdojo_suite": CFG.agentdojo_suite,
        },
        "xgguard": {
            "dir": str(CFG.xgguard_dir),
            "commit": _git(CFG.xgguard_dir, "rev-parse", "HEAD"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_manifest(), indent=2))
