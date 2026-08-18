"""Keyed, deterministic pattern derivation."""
import hashlib, hmac
import numpy as np


def stream(key: bytes, *labels) -> np.random.Generator:
    """A numpy Generator seeded by HMAC(key, labels). Deterministic across runs."""
    msg = b"|".join(str(x).encode() for x in labels)
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    seed = np.frombuffer(digest, dtype=np.uint32)
    return np.random.default_rng(seed)


def probe_pattern(key: bytes, j: int, n: int, probe_rate: float, ctx_rate: float):
    """Positions (S, D0, D1) for probe j over a length-n token span.

    S   -- probe positions, masked in *both* arms, energy read here.
    D0  -- context ablation for arm 0, masked additionally in arm 0 only.
    D1  -- context ablation for arm 1, same size, disjoint from S and D0.

    D0/D1 are drawn exchangeably, which is what gives the detector its exact
    Binomial(M, 1/2) null (see DESIGN.md section 1).
    """
    rng = stream(key, "probe", j, n)
    perm = rng.permutation(n)
    n_s = max(1, int(round(probe_rate * n)))
    n_d = max(1, int(round(ctx_rate * n)))
    if n_s + 2 * n_d > n:
        raise ValueError(f"probe_rate+2*ctx_rate too large for n={n}")
    S = np.sort(perm[:n_s])
    D0 = np.sort(perm[n_s:n_s + n_d])
    D1 = np.sort(perm[n_s + n_d:n_s + 2 * n_d])
    return S, D0, D1


def payload_bits(key: bytes, n_bits: int, message: int = 0) -> np.ndarray:
    """Target signs s_j in {-1,+1} for each probe: keyed one-time pad over the message."""
    rng = stream(key, "pad", n_bits)
    pad = rng.integers(0, 2, size=n_bits)
    msg = np.array([(message >> i) & 1 for i in range(n_bits)], dtype=np.int64)
    bits = pad ^ msg
    return np.where(bits == 1, 1, -1)


def partition_patterns(key: bytes, n: int, n_probes: int, ctx_rate: float):
    """Disjoint probe sets: a keyed partition of the span into n_probes blocks.

    Overlapping probe sets leave a position subject to several probes pulling it in
    opposite directions, and the joint objective then averages to nothing. A partition
    removes the contention entirely, at the cost of a smaller |S_j| per bit.

    D0/D1 are still drawn exchangeably from the complement of S_j, so the exact
    Binomial(M, 1/2) null is unchanged.
    """
    rng = stream(key, "part", n, n_probes)
    perm = rng.permutation(n)
    n_d = max(1, int(round(ctx_rate * n)))
    out = []
    for j, S in enumerate(np.array_split(perm, n_probes)):
        S = np.sort(S)
        comp = np.setdiff1d(np.arange(n), S)
        q = stream(key, "ctx", j, n).permutation(len(comp))
        out.append((S, np.sort(comp[q[:n_d]]), np.sort(comp[q[n_d:2 * n_d]])))
    return out
