"""Shared prompt construction, so every method in this repo sees the same inputs.

dgMARK's `data_utils.prepare_prompts` truncates each C4 document to 300 characters at a
word boundary. Early experiments took the first 40 tokens instead, which
makes any cross-method comparison unsound. This is the single definition everything now
uses; the baseline's published protocol wins, since it is the one that was published.
"""
import gzip
import json
import os
from pathlib import Path

import torch

C4 = os.environ.get(
    "REPLAYMARK_C4",
    str(Path(__file__).resolve().parents[1] / "data" / "c4-validation.json.gz"),
)


def c4_prompts(tok, n, max_chars=300, min_tokens=20, skip=0, src_min=0, path=None):
    """Load the shared C4 prompt cohort.

    Set ``REPLAYMARK_C4`` or pass ``path`` when the dataset is outside the repository.
    """
    source = Path(path or C4).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"C4 file not found at {source}. Set REPLAYMARK_C4 or pass path=."
        )
    out = []
    with gzip.open(source, "rt") as f:
        for line in f:
            text = json.loads(line)["text"]
            cur = ""
            for w in text.split():
                if len(cur) + len(w) + 1 > max_chars:
                    break
                cur = w if not cur else cur + " " + w
            ids = tok(cur)["input_ids"]
            if len(ids) < min_tokens:
                continue
            # For long-form generation: a source document too short to support the target
            # length yields a degenerate continuation in EVERY arm (measured: a prompt
            # window where 12/16 documents collapsed). Key-independent, applied to all
            # methods identically, in the spirit of dgMARK's min-length retention.
            if src_min and len(tok(text)["input_ids"]) < src_min:
                continue
            if skip > 0:
                skip -= 1
                continue
            out.append(torch.tensor(ids, dtype=torch.long)[None])
            if len(out) == n:
                return out
    return out
