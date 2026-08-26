"""LLaDA wrapper used by ReplayMark generation and replay."""
import os
import numpy as np
import torch
import torch.nn.functional as F

SNAP = os.environ.get("REPLAYMARK_LLADA_MODEL", "GSAI-ML/LLaDA-8B-Instruct")
MASK_ID = 126336


class LLaDAModel:
    mask_id = MASK_ID

    def __init__(self, path=None, dtype=torch.float16, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        path = path or SNAP
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        # Enable memory-efficient SDPA where the installed PyTorch build supports it.
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        self.device = device

    # ---------- core probe ----------
    @torch.no_grad()
    def logprobs_rows(self, ids_batch, rows, chunk=2, dtype=torch.float32):
        """ids_batch: LongTensor [B, L] already corrupted. rows: LongTensor [R] positions.
        Returns log-softmax [B, R, V].

        `dtype=torch.float64` matters wherever two arms' log-probs are subtracted. This
        model is confident enough that a chosen token's log-prob is often within float32's
        resolution of zero under both arms, so the difference underflows to exactly zero
        and the position becomes unusable. Measured: 28-34 % of positions tie in float32.
        Converting after this function has already rounded does not help -- the precision
        has to be kept here.
        """
        outs = []
        for s in range(0, ids_batch.shape[0], chunk):
            logits = self.model(ids_batch[s:s + chunk].to(self.device)).logits
            outs.append(F.log_softmax(logits[:, rows, :].to(dtype), dim=-1).cpu())
            del logits
        return torch.cat(outs, 0)

    def corrupt(self, ids, positions):
        out = ids.clone()
        out[..., positions] = self.mask_id
        return out

    # ---------- generation ----------
    def build_prompt(self, question):
        m = [{"role": "user", "content": question}]
        s = self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        return torch.tensor(self.tok(s)["input_ids"], dtype=torch.long)[None]

    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=128, steps=128, block_len=32, temperature=0.0,
                 seed=0, remasking="low_confidence"):
        """LLaDA's reference sampler, matching GSAI-ML/LLaDA `generate.py`.

        Low-confidence remasking ranks sampled tokens by their probability under the clean
        model distribution.
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        B, P = prompt_ids.shape
        x = torch.full((B, P + gen_len), self.mask_id, dtype=torch.long,
                       device=self.device)
        x[:, :P] = prompt_ids.to(self.device)
        n_blocks = max(1, gen_len // block_len)
        steps_pb = max(1, steps // n_blocks)
        for b in range(n_blocks):
            lo, hi = P + b * block_len, P + (b + 1) * block_len
            per_step = _split_evenly((x[:, lo:hi] == self.mask_id).sum(1, keepdim=True),
                                     steps_pb)
            for t in range(steps_pb):
                m_idx = x == self.mask_id
                logits = self.model(x).logits
                if temperature > 0:
                    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float64,
                                   generator=g)
                    x0 = (logits.double() / temperature
                          + (-torch.log(-torch.log(u)))).argmax(-1)
                else:
                    x0 = logits.argmax(-1)
                if remasking == "low_confidence":
                    conf = F.softmax(logits.double(), dim=-1).gather(
                        -1, x0.unsqueeze(-1)).squeeze(-1)
                else:
                    conf = torch.rand(x0.shape, device=x0.device, generator=g).double()
                del logits
                conf = torch.where(m_idx, conf, torch.full_like(conf, -np.inf))
                conf[:, hi:] = -np.inf
                x0 = torch.where(m_idx, x0, x)
                for i in range(B):
                    k = int(per_step[i, t])
                    if k > 0:
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
