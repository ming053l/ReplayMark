"""Generate and verify one ReplayMark completion."""

import argparse
import sys
from pathlib import Path

# Also support direct execution from a source checkout before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replaymark import LLaDAModel, ReplayMark


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=None, help="Hugging Face ID or local LLaDA path")
    parser.add_argument("--generation-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--key", default="replaymark-demo-key")
    parser.add_argument("--nonce", default="demo-document")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.generation_length % args.block_length:
        raise ValueError("generation length must be divisible by block length")

    model = LLaDAModel(path=args.model)
    prompt_ids = model.build_prompt(args.prompt)
    config = dict(
        block_len=args.block_length,
        n_patterns=8,
        sync_frac=1.0,
        n_payload_bits=1,
        s_min=0.5,
        retries=16,
        p_floor=0.05,
        nonce=args.nonce,
    )
    key = args.key.encode("utf-8")
    writer = ReplayMark(model, key, **config)
    output = writer.generate(
        prompt_ids,
        gen_len=args.generation_length,
        steps=args.generation_length,
        seed=args.seed,
    )

    verifier = ReplayMark(model, key, **config)
    report = verifier.detect(
        output.cpu(), prompt_ids.shape[1], args.generation_length
    )
    continuation = output[0, prompt_ids.shape[1]:]
    print(model.tok.decode(continuation, skip_special_tokens=True))
    print(
        f"carriers={report['n_sync']} "
        f"rate={report['rate_sync']:.3f} "
        f"p_value={report['p_value']:.6g}"
    )


if __name__ == "__main__":
    main()
