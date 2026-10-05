import hashlib

import numpy as np
import pytest


class HashEmbedder:
    """Deterministic stand-in encoder for unit tests (no weights needed)."""
    name = "hash"

    def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
            out.append(rng.normal(size=16))
        return np.stack(out) if out else np.zeros((0, 16))


@pytest.fixture
def emb():
    return HashEmbedder()
