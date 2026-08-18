"""LLaDA-8B wrapper: masked-denoiser log-probs + plain generation."""
import os
import numpy as np
import torch
import torch.nn.functional as F

SNAP = ("/ssd1/ming/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/"
        "snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
MASK_ID = 126336


class BasinModel:
    def __init__(self, path=SNAP, dtype=torch.float16, device="cuda"):
        os.environ.setdefault("HF_HOME", "/ssd1/ming/hf_cache")
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        # sm75 has no flash kernel; upstream disables mem-efficient SDPA (a note written
        # for A100s), so torch silently falls back to the math backend and materialises
        # the full L x L attention matrix per layer. Same bug as MMaDA-Parallel.
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        self.device = device

    # ---------- core probe ----------
    @torch.no_grad()
    def logprobs_rows(self, ids_batch, rows, chunk=2):
        """ids_batch: LongTensor [B, L] already corrupted. rows: LongTensor [R] positions.
        Returns log-softmax [B, R, V] (fp32)."""
        outs = []
        for s in range(0, ids_batch.shape[0], chunk):
            logits = self.model(ids_batch[s:s + chunk].to(self.device)).logits
            outs.append(F.log_softmax(logits[:, rows, :].float(), dim=-1).cpu())
            del logits
        return torch.cat(outs, 0)

    def corrupt(self, ids, positions):
        out = ids.clone()
        out[..., positions] = MASK_ID
        return out

    # ---------- generation ----------
    def build_prompt(self, question):
        m = [{"role": "user", "content": question}]
        s = self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        return torch.tensor(self.tok(s)["input_ids"], dtype=torch.long)[None]

    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=128, steps=128, block_len=32, temperature=0.0,
                 seed=0):
        """LLaDA low-confidence-remasking sampler (semi-autoregressive over blocks)."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        B, P = prompt_ids.shape
        x = torch.full((B, P + gen_len), MASK_ID, dtype=torch.long, device=self.device)
        x[:, :P] = prompt_ids.to(self.device)
        n_blocks = gen_len // block_len
        steps_pb = max(1, steps // n_blocks)
        for b in range(n_blocks):
            lo, hi = P + b * block_len, P + (b + 1) * block_len
            blk_mask = x[:, lo:hi] == MASK_ID
            per_step = _split_evenly(blk_mask.sum(1, keepdim=True), steps_pb)
            for t in range(steps_pb):
                m_idx = x == MASK_ID
                logits = self.model(x).logits
                if temperature > 0:
                    noise = torch.rand(logits.shape, device=logits.device, generator=g)
                    logits = logits / temperature + (-torch.log(-torch.log(noise + 1e-10) + 1e-10))
                x0 = logits.argmax(-1)
                conf = F.softmax(logits.float(), -1).max(-1).values
                del logits
                conf = torch.where(m_idx, conf, torch.full_like(conf, -np.inf))
                conf[:, hi:] = -np.inf
                x0 = torch.where(m_idx, x0, x)
                for i in range(B):
                    k = int(per_step[i, t])
                    if k <= 0:
                        continue
                    sel = torch.topk(conf[i], k=k).indices
                    x[i, sel] = x0[i, sel]
        return x


def _split_evenly(counts, steps):
    base = counts // steps
    out = base.repeat(1, steps)
    rem = (counts % steps).squeeze(1)
    for i in range(out.shape[0]):
        out[i, :int(rem[i])] += 1
    return out
