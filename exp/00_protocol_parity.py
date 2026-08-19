"""One log that shows the two no-watermark controls side by side.

'Matched protocol' has already been wrong once here: dgMARK sets steps = gen_length
(generate.py:136) while our harness ran 128 steps for 256 tokens, which moved the reference
perplexity from 8.65 to 23.41 and would have flattered every ratio measured against it.
This prints the literal configuration of both controls and their reference perplexity in
one place, so the claim can be checked rather than asserted.

The remaining known difference is the sampling rule: dgMARK's multinomial path draws from
the top-k renormalised distribution with its own temperature default, ours draws from the
full vocabulary at temperature 0.8. Both are printed; neither is silently called equal.
"""
import sys, csv, json, glob
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch

from basinmark.model import BasinModel, SNAP
from basinmark.data import c4_prompts

GEN, BLK, NS = 256, 32, 12
OUT = "/ssd1/ming/basinmark/results/baselines"

DGMARK = dict(
    checkpoint="GSAI-ML/LLaDA-8B-Instruct (local snapshot)",
    prompts="C4 realnewslike, truncated to 300 characters at a word boundary",
    gen_length=256, steps="= gen_length = 256 (generate.py:136)", block_length=32,
    temperature="0.0 (argparse default)", sampling="multinomial over top-k, k=3",
    transfer="low-confidence remasking, p(x0) under the clean softmax",
    ppl="GPT-2-large over the generated span",
)
RETRACE = dict(
    checkpoint="GSAI-ML/LLaDA-8B-Instruct (local snapshot)",
    prompts="C4 realnewslike, truncated to 300 characters at a word boundary",
    gen_length=256, steps="256 (matched)", block_length=32,
    temperature="0.8", sampling="multinomial over the full vocabulary",
    transfer="low-confidence remasking, p(x0) under the clean softmax",
    ppl="GPT-2-large over the generated span",
)


def main():
    import os
    os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M = BasinModel()
    tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
    gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                              torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def ppl(texts):
        o = []
        for t in texts:
            ids = tk(t, return_tensors="pt", truncation=True,
                     max_length=512).input_ids.cuda()
            o.append(float(gm(ids, labels=ids).loss) if ids.shape[1] >= 8 else np.nan)
        return float(np.exp(np.nanmean(o)))

    print("field                dgMARK control                      ReTrace control")
    print("-" * 96)
    for k in DGMARK:
        print(f"{k:<20} {str(DGMARK[k]):<35} {str(RETRACE[k])}")

    prompts = c4_prompts(M.tok, NS)
    ours = [M.generate(p, gen_len=GEN, steps=GEN, block_len=BLK, temperature=0.8,
                       seed=3000 + i).cpu() for i, p in enumerate(prompts)]
    txt = [M.tok.decode(x[0, p.shape[1]:p.shape[1] + GEN], skip_special_tokens=True)
           for x, p in zip(ours, prompts)]
    p_ours = ppl(txt)

    hits = sorted(glob.glob(f"{OUT}/dgmark_original*.csv"))
    p_dg = float("nan")
    if hits:
        rows = list(csv.DictReader(open(hits[0], newline="", encoding="utf-8")))
        p_dg = ppl([r["generated"] for r in rows])

    print("-" * 96)
    print(f"reference perplexity  {p_dg:<35.2f} {p_ours:.2f}")
    print(f"\nratio between the two controls: {p_ours / p_dg:.3f}   "
          f"(1.00 would mean the two decoding regimes are interchangeable; they are not, "
          f"and each watermark is therefore scored against its own)")
    json.dump(dict(dgmark=DGMARK, retrace=RETRACE, ppl_dgmark=p_dg, ppl_retrace=p_ours),
              open("/ssd1/ming/basinmark/results/protocol_parity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
