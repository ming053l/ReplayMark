"""Shared prompt construction, so every method in this repo sees the same inputs.

dgMARK's `data_utils.prepare_prompts` truncates each C4 document to 300 characters at a
word boundary. BasinMark's earlier experiments took the first 40 tokens instead, which
makes any cross-method comparison unsound. This is the single definition everything now
uses; the baseline's published protocol wins, since it is the one that was published.
"""
import gzip, json
import torch

C4 = "/ssd2/ming/basinmark/data/c4-validation.json.gz"


def c4_prompts(tok, n, max_chars=300, min_tokens=20, skip=0, src_min=0):
    out = []
    with gzip.open(C4, "rt") as f:
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
