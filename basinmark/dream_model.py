"""Dream-7B-Instruct wrapper: BasinModel's interface over Dream's shifted denoiser.

Dream reads position i's prediction from the RAW logits at i-1 -- its own
generation_utils.py does `logits = cat([logits[:, :1], logits[:, :-1]], dim=1)` right
after every forward. `_Shifted` reproduces that once, here, so every caller keeps the
LLaDA convention that `.logits[:, i]` predicts position i and the rest of the pipeline
(ResampleMark, kgw_generate, BasinModel.generate) runs unchanged.
"""
import glob
import os

import torch

from .model import BasinModel

DREAM_MASK_ID = 151666      # config.json mask_token_id; vocab_size 152064 (Qwen tokenizer)


def _snapshot():
    hits = glob.glob("/ssd2/ming/hf_cache/hub/models--Dream-org--Dream-v0-Instruct-7B/"
                     "snapshots/*/")
    if not hits:
        raise FileNotFoundError("Dream-v0-Instruct-7B snapshot not found in hf_cache")
    return hits[0]


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
        os.environ.setdefault("HF_HOME", "/ssd2/ming/hf_cache")
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
