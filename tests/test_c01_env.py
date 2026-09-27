"""C01 checks (EXPERIMENT_PLAN.md §9.3). Run with:  conda run -n mastrust pytest"""
import importlib
import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest

from config import CFG
from llm import is_reasoning_model, sampling_params

ROOT = Path(__file__).resolve().parents[1]


def test_env_is_isolated():
    assert not site.ENABLE_USER_SITE, "~/.local site-packages leak in; use `conda run -n mastrust`"
    assert not os.environ.get("PYTHONPATH"), "PYTHONPATH leaks in; use `conda run -n mastrust`"
    # C06: fixed string hashing, so set iteration order is the same in every process
    assert sys.flags.hash_randomization == 0, "PYTHONHASHSEED=0 missing; use `conda run -n mastrust`"


@pytest.mark.parametrize("mod", ["openai", "agentdojo", "numpy", "torch", "sentence_transformers",
                                 "transformers", "sklearn", "yaml", "pydantic", "mpmath"])
def test_packages_come_from_env(mod):
    assert importlib.import_module(mod).__file__.startswith(sys.prefix)


def test_cuda_available():
    import torch
    assert torch.cuda.is_available()


def test_agentdojo_workspace_suite():
    from agentdojo.task_suite.load_suites import get_suite
    suite = get_suite(CFG.agentdojo_benchmark, CFG.agentdojo_suite)
    assert (len(suite.user_tasks), len(suite.injection_tasks), len(suite.tools)) == (40, 14, 24)


def test_models_are_pinned_snapshots():
    assert CFG.model == "gpt-5-mini-2025-08-07"
    assert CFG.xgguard_model == "gpt-4o-mini-2024-07-18"


def test_sampling_params_match_what_the_api_accepts():
    # gpt-5-mini returns HTTP 400 for temperature != 1 and for max_tokens.
    p = sampling_params(CFG, "gpt-5-mini-2025-08-07")
    assert "temperature" not in p and "max_tokens" not in p
    assert set(p) == {"reasoning_effort", "max_completion_tokens"}
    assert set(sampling_params(CFG, "gpt-4o-mini-2024-07-18")) == {"temperature", "max_tokens"}
    assert not is_reasoning_model("gpt-5-chat-latest")


def _run(*args, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, "run.py", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


def test_manifest():
    r = _run("--manifest")
    assert r.returncode == 0, r.stderr
    m = json.loads(r.stdout)
    assert len(m["code_hash"]) == 64
    assert m["llm"]["model"] == CFG.model and m["llm"]["reasoning_model"] is True
    assert m["packages"]["agentdojo"] == "0.1.35"
    assert m["benchmark"]["agentdojo_benchmark"] == "v1.2.2"
    # local patches live on a branch; what must not move is the upstream base,
    # and the patched code must be committed
    assert m["xgguard"]["upstream_base"] == "86e1121512f76800f80d4687e492c7f99f049929"
    assert m["xgguard"]["dirty"] is False


def test_fake_llm_smoke(tmp_path):
    r = _run("--fake-llm", "--quiet", env_extra={"MAS_RUNS_DIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr[-2000:]
    assert list(tmp_path.glob("run_*/transcript.json")), "no transcript written"
