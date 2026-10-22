# ReplayMark

**Probe-and-Replay Model-Response Watermarking for Diffusion Language Models**

Chia-Ming Lee · Wen-Hsin Tsai · Tung Yuk Ying · Chih-Chung Hsu

**NeurIPS 2026 FLMSec Workshop · ★ Distinguished Paper Award**

[Project Page](https://ming053l.github.io/ReplayMark/) ·
[Paper](https://openreview.net/pdf?id=fmXX4kVZ30) ·
[OpenReview](https://openreview.net/forum?id=fmXX4kVZ30) ·
[BibTeX](citation.bib)

ReplayMark embeds a text watermark through a diffusion language model's response to
partially hidden context. During generation, it favors plausible tokens whose responses
match a secret key. To verify the finished text, it reconstructs the probes and measures
that agreement using the same checkpoint.

## How it works

A diffusion language model predicts masked tokens from the visible context. Hiding
different parts of that context changes the probabilities of candidate tokens. ReplayMark
uses the difference between two such predictions as a **model response**.

1. **Probe:** mask the current block and future suffix, then hide different subsets of
   the completed prefix.
2. **Select:** find probe pairs and positions with probability mass on both response
   directions, without consulting the keyed target direction.
3. **Embed:** prefer plausible live samples whose response matches the keyed direction.
   If all guided proposals fail, take an unconditional fallback sample.
4. **Replay:** rebuild the probes from the finished sequence and count keyed matches
   with an exact binomial test.

The checkpoint weights remain unchanged. Verification does not require saved generation
logits, rejected samples, or the generation random seed. It **does** require the original
checkpoint/tokenizer, key, document nonce when used, and matching replay settings.

The released detector takes the **full token sequence, including the prompt**, plus its
prompt and continuation lengths. Retain the original token IDs for exact replay; decoding
and re-tokenizing text can change token boundaries.

## Installation

Use Python 3.10+ and a PyTorch installation suitable for your CUDA setup. Model inference
requires enough GPU memory for the checkpoint and probe activations. CPU tests use a tiny
stand-in model and do not download checkpoints.

```bash
git clone https://github.com/ming053l/ReplayMark.git
cd ReplayMark
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package pins `transformers==4.46.2`. The wrappers load model-provided code through
`trust_remote_code=True`. Use a fixed local snapshot when reproducible replay is required.

Optional plotting/evaluation dependencies:

```bash
python -m pip install -e '.[evaluation]'
```

## Quick start

Generate a short LLaDA completion and replay it immediately:

```bash
python -m examples.quickstart \
  --prompt "Explain why the sky appears blue." \
  --generation-length 64 \
  --probe-batch-size 1
```

For Dream:

```bash
python -m examples.quickstart \
  --backend dream \
  --prompt "Explain why the sky appears blue." \
  --generation-length 64 \
  --probe-batch-size 1
```

The installed `replaymark-demo` command exposes the same arguments. Each run prints the
continuation, number of presence carriers, match rate, and p-value. A short completion
may contain too few carriers for significant detection; it is a functional demo.

`--model` accepts a Hugging Face ID or local checkpoint directory. Defaults can also be
set before starting Python:

```bash
export REPLAYMARK_LLADA_MODEL=GSAI-ML/LLaDA-8B-Instruct
export REPLAYMARK_DREAM_MODEL=Dream-org/Dream-v0-Instruct-7B
export REPLAYMARK_C4=/path/to/c4-validation.json.gz
```

### Python API

```python
from replaymark import LLaDAModel, ReplayMark

model = LLaDAModel()  # or DreamModel()
prompt_ids = model.build_prompt("Explain why the sky appears blue.")
key = b"replace-with-your-secret-key"
settings = dict(
    block_len=32,
    n_patterns=8,
    n_payload_bits=1,
    sync_frac=1.0,       # use all selected carriers for presence detection
    s_min=0.5,
    retries=16,
    p_floor=0.05,
    nonce="unique-document-id",
    probe_batch_size=1,
)
writer = ReplayMark(model, key, **settings)
ids = writer.generate(prompt_ids, gen_len=64, steps=64, seed=0)

# This can run later, with the same checkpoint and replay settings.
verifier = ReplayMark(model, key, **settings)
report = verifier.detect(ids, prompt_len=prompt_ids.shape[1], gen_len=64)
print(report["p_value"], report["n_sync"], report["rate_sync"])
```

The constructor retains legacy payload-oriented defaults for experiment compatibility.
The settings above explicitly select the presence-only setup used in the quick start.
Use a private key and a fresh document nonce outside the demo, and retain the nonce for
verification. No presence carriers yields `p_value=1.0`.

### Configuration and memory

| Setting | Meaning |
|:--|:--|
| `block_len` | Tokens per generation/replay block; `gen_len` must be a positive multiple |
| `steps` | Generation steps; must be positive and divisible by the number of blocks |
| `n_patterns` | Number of hidden-context patterns; at least two |
| `probe_batch_size` | Probe sequences per forward (default 2); use 1 to reduce peak GPU memory |
| `s_min` | Strict lower threshold for two-sided carrier mass |
| `retries` | Maximum guided proposals per carrier, followed by one fallback if exhausted |
| `p_floor` | Minimum proposal probability relative to the live distribution's maximum |
| `temperature` | Positive sampling temperature; also used for carrier mass calculation |

ReplayMark currently processes one sequence at a time. Smaller probe batches reduce
activation memory but do not reduce the number of probes. Probe log-probabilities are
stored on CPU; the response bank can still consume substantial host RAM. Preserve the
checkpoint, precision, and settings between embedding and replay. Memory requirements
and replay agreement should be measured on the target hardware.

## Main results

The primary setting uses 512 generated tokens, 32-token blocks, eight probe patterns,
`R=16`, and `kappa=0.05`.

| Checkpoint | TPR at 1% FPR | TPR at 0.1% FPR | PPL ratio | Verification calls |
|:--|:--:|:--:|:--:|:--:|
| LLaDA-8B-Instruct | 0.80 | 0.60 | 0.74x | 144 |
| Dream-7B-Instruct | 0.90 | 0.90 | 1.09x | 144 |

The same configuration is transferred from LLaDA to Dream without retuning. Verification is
checkpoint-specific and computationally heavier than token-local detection. Dispersed editing and
full-document paraphrasing reduce detection substantially; the paper reports these limitations
directly.

## Scope and limitations

- The finite-sample presence test relies on text being independent of the document key,
  domain-separated keyed coins, and carrier selection independent of the scored direction.
  It is not a blanket guarantee against adaptive, key-aware attacks.
- Checkpoint-specific verification is more expensive than token-local counting. Quantization
  or a different checkpoint can change response directions.
- Dispersed edits and full-document paraphrasing can substantially reduce detection.
- Detection of the keyed signal does not establish factual correctness of the text.

## Results and reproducibility

The measured outputs are committed under `results/`. The main paper entries are:

| Result | Experiment or source |
|:--|:--|
| LLaDA detectability and quality | `results/29_clean.json` |
| Dream transfer | `results/36_dream.json` |
| GSM8K, MMLU, HumanEval | `results/32_*`, `39_*`, `40_*`, `41_*` |
| Editing evaluation | `results/33_robust.json`, `49_paraphrase.json` |
| Probe-count sweep | `results/47_*.json` |
| Human-text calibration | `results/48_fpr*.json` |
| Per-document carrier statistics | `results/48_carrier_stats.json` |

The repository includes saved measurements and plotting scripts; it does not yet provide
a single end-to-end runner for all paper experiments. The quick start above is a smoke
test, not a reproduction of the full evaluation.

The measured outputs are retained for auditability. Figure scripts under `exp/` rebuild the
result-driven plots used by the manuscript.

Run the lightweight primitive tests with:

```bash
python -m unittest discover -s tests
```

## Paper builds

```bash
cd paper

# Anonymous workshop version
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Non-anonymous public version
pdflatex main_public.tex
bibtex main_public
pdflatex main_public.tex
pdflatex main_public.tex
```

The two entry points share the same technical body. `main_public.tex` only switches the workshop
style, author block, notice, and PDF metadata.

## Repository layout

```text
replaymark/resample.py       ReplayMark generation and exact replay detector
replaymark/challenges.py     keyed probes, directions, carrier selection primitives
replaymark/model.py          LLaDA wrapper and reference decoder
replaymark/dream_model.py    Dream wrapper with shifted prediction positions
replaymark/data.py           shared C4 prompt construction
examples/quickstart.py      minimal generation and replay example
tests/                      CPU-only primitive tests
exp/                        result-driven paper figure scripts
results/                    measured outputs used by the paper
paper/                      anonymous and public manuscript builds
```

## Citation

If you use ReplayMark, please cite the paper. A standalone [citation.bib](citation.bib)
and GitHub-compatible [CITATION.cff](CITATION.cff) are included.

```bibtex
@inproceedings{lee2026replaymark,
  title     = {{ReplayMark}: Probe-and-Replay Model-Response Watermarking for Diffusion Language Models},
  author    = {Lee, Chia-Ming and Tsai, Wen-Hsin and Tung, Yuk Ying and Hsu, Chih-Chung},
  booktitle = {NeurIPS 2026 Workshop on Foundations of Language Model Security},
  year      = {2026},
  url       = {https://openreview.net/forum?id=fmXX4kVZ30}
}
```

## License

ReplayMark is released under the [MIT License](LICENSE). Model checkpoints retain their
respective licenses.
