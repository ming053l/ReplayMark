"""CPU regression tests using a deterministic masked-language-model stand-in."""
import unittest
from types import SimpleNamespace

import torch

from replaymark import ReplayMark
from replaymark.model import LLaDAModel


class ToyModel(LLaDAModel):
    mask_id = 7

    def __init__(self):
        self.device = "cpu"
        self.batch_sizes = []
        self.model = self.forward

    def forward(self, ids):
        self.batch_sizes.append(ids.shape[0])
        vocab = torch.arange(8, dtype=torch.float64)
        context = ids.double().sum(1)[:, None, None]
        positions = torch.arange(ids.shape[1])[None, :, None]
        logits = torch.sin(context * (vocab + 1) * .2 + positions)
        logits[..., self.mask_id] = -100
        return SimpleNamespace(logits=logits)


class ReplayTests(unittest.TestCase):
    def make(self, **kwargs):
        config = dict(block_len=4, n_patterns=4, sync_frac=1,
                      n_payload_bits=1, s_min=0, retries=2)
        config.update(kwargs)
        return ReplayMark(ToyModel(), b"test-key", **config)

    def test_probe_chunking_preserves_bank_and_bounds_batch(self):
        ids = torch.tensor([[1, 2, 3, 4, 7, 7, 7, 7]])
        one, two = self.make(probe_batch_size=1), self.make(probe_batch_size=2)
        a = one._build_response_bank(ids, 4, 8)
        b = two._build_response_bank(ids, 4, 8)
        for left, right in zip(a, b):
            torch.testing.assert_close(torch.as_tensor(left), torch.as_tensor(right), rtol=0, atol=0)
        self.assertEqual(max(one.M.batch_sizes), 1)
        self.assertEqual(max(two.M.batch_sizes), 2)

    def test_generation_and_replay_are_reproducible(self):
        prompt = torch.tensor([[1, 2, 3, 4]])
        writer = self.make()
        output = writer.generate(prompt, gen_len=8, steps=8, seed=5)
        self.assertFalse((output[:, 4:] == writer.mask_id).any())
        torch.testing.assert_close(output, self.make().generate(prompt, gen_len=8, steps=8, seed=5))
        report = writer.detect(output, 4, 8)
        self.assertEqual(report, self.make().detect(output, 4, 8))
        self.assertEqual(report["n_seqs"], 10)
        self.assertTrue(0 <= report["p_value"] <= 1)
        # A block's bank must ignore its own final tokens and the future suffix.
        masked = torch.cat([prompt, torch.full((1, 8), writer.mask_id)], dim=1)
        a = writer._build_response_bank(masked, 4, 12)
        b = writer._build_response_bank(output, 4, 12)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)

    def test_invalid_configuration_fails_early(self):
        for config in ({"block_len": 0}, {"n_patterns": 1}, {"retries": 0},
                       {"temperature": 0}, {"temperature": float("nan")},
                       {"probe_batch_size": 0}, {"sync_frac": 2},
                       {"ctx_window": -1}, {"challenge": "typo"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.make(**config)

    def test_invalid_lengths_fail_before_forward(self):
        wm = self.make()
        prompt = torch.tensor([[1, 2]])
        for length in (0, 3, 6):
            with self.subTest(length=length), self.assertRaises(ValueError):
                wm.generate(prompt, gen_len=length)
        with self.assertRaises(ValueError):
            wm.generate(prompt, gen_len=8, steps=3)
        with self.assertRaises(ValueError):
            wm.detect(torch.ones((1, 10), dtype=torch.long), 2, 4)
        with self.assertRaises(ValueError):
            wm.generate(prompt.repeat(2, 1), gen_len=8)
        self.assertEqual(wm.M.batch_sizes, [])


if __name__ == "__main__":
    unittest.main()
