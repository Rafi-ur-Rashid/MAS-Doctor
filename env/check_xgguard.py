"""Smoke test for the xgguard env. Run: conda run -n xgguard python env/check_xgguard.py"""
import os
import site
import sys

XG = os.environ.get("XGGUARD_DIR", "/scratch/mur5028/XG-Guard")
sys.path.insert(0, XG)
os.chdir(XG)

import numpy, openai, sentence_transformers, torch, torch_geometric, transformers  # noqa: E401
from torch_scatter import scatter_mean

assert not site.ENABLE_USER_SITE, "user site-packages leak into the env"
assert not os.environ.get("PYTHONPATH"), "PYTHONPATH leaks into the env"
expected = {"torch": "2.5.1+cu124", "torch_geometric": "2.6.1", "sentence_transformers": "3.3.1",
            "transformers": "4.44.2", "numpy": "1.26.4", "openai": "1.58.1"}
got = {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
       "sentence_transformers": sentence_transformers.__version__,
       "transformers": transformers.__version__, "numpy": numpy.__version__,
       "openai": openai.__version__}
assert got == expected, f"version drift: {got}"
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(4, 3).cuda()
assert scatter_mean(x, torch.tensor([0, 0, 1, 1]).cuda(), dim=0).shape == (2, 3)

import modules.Dominant, modules.TAM, modules.train_un2, utils.utils  # noqa: E401,F401
print("xgguard env OK:", got)
