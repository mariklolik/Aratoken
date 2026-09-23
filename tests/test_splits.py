from collections import Counter
from pathlib import Path

import pytest

from aratoken.splits import split_of

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def test_split_proportions():
    counts = Counter(split_of(f"<urn:uuid:{i}>") for i in range(20000))
    assert abs(counts["lm-train"] / 20000 - 0.60) < 0.02
    assert abs(counts["tok-train"] / 20000 - 0.30) < 0.02
    assert abs(counts["dev"] / 20000 - 0.05) < 0.01
    assert abs(counts["test"] / 20000 - 0.05) < 0.01


@pytest.mark.parametrize("split", ["test", "dev"])
def test_released_ids_hash_to_their_split(split):
    ids = (SPLITS / f"{split}_ids.txt").read_text().split()
    assert ids and all(split_of(doc_id) == split for doc_id in ids)
