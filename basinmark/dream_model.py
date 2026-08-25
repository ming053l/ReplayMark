"""Dream-7B-Instruct wrapper for ReplayMark's shifted denoiser.

Dream reads position i's prediction from the RAW logits at i-1 -- its own
generation_utils.py does `logits = cat([logits[:, :1], logits[:, :-1]], dim=1)` right
after every forward. `_Shifted` reproduces that once, here, so every caller keeps the
LLaDA convention that `.logits[:, i]` predicts position i and the rest of the pipeline
runs unchanged.
"""
import os

import torch

from .model import BasinModel

DREAM_MASK_ID = 151666      # config.json mask_token_id; vocab_size 152064 (Qwen tokenizer)
DEFAULT_DREAM_MODEL = os.environ.get(
    "REPLAYMARK_DREAM_MODEL", "Dream-org/Dream-v0-Instruct-7B"
)


def _snapshot():
    """Backward-compatible model resolver for older experiment scripts."""
    return DEFAULT_DREAM_MODEL


class _Shifted:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, x):
        out = self.inner(x)
        out.logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        return out


class DreamModel(BasinModel):
    mask_id = DREAM_MASK_ID

    def __init__(self, path=None, dtype=torch.float16, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        path = path or _snapshot()
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        inner = AutoModel.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        # same sm75 SDPA fallback note as BasinModel
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        self.model = _Shifted(inner)
        self.device = device
