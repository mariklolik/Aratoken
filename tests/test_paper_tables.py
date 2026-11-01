import pytest

from scripts.paper.make_tables import table_algorithms


@pytest.mark.parametrize("scored,rate,display", [
    (0, 0.0, "--"),
    (None, 0.0, "--"),
    (500, 0.0, "0.00"),
    (500, 1.25, "1.25"),
])
def test_oov_display_distinguishes_unscored_and_measured(scored, rate, display):
    metrics = {"macro": {"fertility": {"mean": 1.5}, "compression": {"mean": 4.25}},
               "oov_rate_percent": rate}
    if scored is not None:
        metrics["oov_words_scored"] = scored
    rendered = table_algorithms({"results": {"bpe_drop": {"own_norm": metrics}}})
    assert f"BPE & 1.500 & 4.25 & {display} " + r"\\" in rendered.splitlines()
