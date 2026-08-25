# ReplayMark

**Probe-and-replay model-response watermarking for diffusion language models.**

[Paper](paper/main_public.pdf) | [Anonymous workshop version](paper/main.pdf) |
[Paper source](paper/main.tex) | [Results](results)

ReplayMark embeds a keyed signal in reproducible masked-DLM responses and verifies it by
replaying the same checkpoint. It does not modify the checkpoint, decoder, block schedule, or
within-block commit order.

## Method in one minute

ReplayMark uses arbitrary-mask prediction, context sensitivity, and checkpoint replayability:

1. Re-mask the active block, its suffix, and keyed subsets of the completed prefix.
2. Compare the resulting conditional token probabilities.
3. Select positions where plausible tokens occur in both response directions.
4. Reveal the keyed direction only after selection, then accept the first suitable live draw.
5. Reconstruct the same queries from the completed text and apply an exact binomial test.

Selection is symmetric with respect to the later keyed direction. Text produced independently of
the document key therefore has a finite-sample false-positive probability no greater than the
chosen level, even though each document can contain a different number of carriers.

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

## Installation

Python 3.10 and a CUDA GPU with enough memory for an 8B checkpoint are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The wrappers download checkpoints from Hugging Face by default. Local paths can be supplied on the
command line or through environment variables:

```bash
export REPLAYMARK_LLADA_MODEL=GSAI-ML/LLaDA-8B-Instruct
export REPLAYMARK_DREAM_MODEL=Dream-org/Dream-v0-Instruct-7B
export REPLAYMARK_C4=/path/to/c4-validation.json.gz
```

## Quick start

The example generates one short LLaDA completion and immediately replays it:

```bash
python examples/quickstart.py \
  --prompt "Explain why the sky appears blue." \
  --generation-length 64
```

The public API is `replaymark.ReplayMark`.

## Reproducing the paper

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

The repository is available at <https://github.com/ming053l/ReplayMark>.

## License

ReplayMark is released under the [MIT License](LICENSE).
