"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Kaggle T4 of blob 99829b51 (2026-09-19); Colab T4 of blobs 4cf243c1 (2026-10-08),
adef085a and 324e664e (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "pixart_sigma_generation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_expectations_and_run_status_sentences_are_gone():
    md = _markdown()
    for stale in ("this version takes about 19 minutes", "No run of this activity is recorded yet",
                  "about three to four minutes on a T4", "took about two minutes", "in both: up, down, up",
                  "trained for a few minutes", "should take roughly half the time"):
        assert stale not in md, stale


def test_confirming_run_of_324e664e_is_recorded():
    md = _markdown()
    assert "blobs `adef085a` and `324e664e`, 25 October 2026" in md
    assert "0.108927 → 0.108929 in the third" in md and "rose at epoch 4 in the third, which kept epoch 3" in md
    assert "0.105811, 0.105809 and 0.105832" in md
    assert "Dark-eyed Junco 62.12 → 61.67, 60.99 and 60.74" in md
    assert "took between 17.1 and 19.0 minutes" in md
