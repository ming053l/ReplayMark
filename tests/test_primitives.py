import unittest

import numpy as np

from basinmark.challenges import orientation_bits, score, tie_bits
from basinmark.prng import stream


class PrimitiveTests(unittest.TestCase):
    def test_keyed_stream_is_reproducible_and_domain_separated(self):
        key = b"unit-test-key"
        first = stream(key, "probe", 3).integers(0, 2**31, size=16)
        repeat = stream(key, "probe", 3).integers(0, 2**31, size=16)
        other = stream(key, "carrier", 3).integers(0, 2**31, size=16)
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, other))

    def test_orientation_and_tie_domains_are_deterministic(self):
        key = b"unit-test-key"
        positions = np.arange(32)
        self.assertEqual(
            orientation_bits(key, positions), orientation_bits(key, positions)
        )
        self.assertEqual(tie_bits(key, positions), tie_bits(key, positions))

    def test_score_uses_the_independent_tie_coin_only_at_zero(self):
        contrast = np.array([2.0, -3.0, 0.0, 0.0])
        direction = np.array([1.0, 1.0, -1.0, 1.0])
        tie = np.array([0, 0, 1, 0])
        observed, ties = score(contrast, direction, tie)
        np.testing.assert_array_equal(observed, np.array([1, 0, 1, 0]))
        self.assertEqual(ties, 2)


if __name__ == "__main__":
    unittest.main()
