"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Runs: Kaggle T4 of blob 99829b51 (2026-09-19);
Colab T4 of blobs 4cf243c1 (2026-10-08) and adef085a (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "pixart_sigma_generation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_single_run_values_are_gone():
    md = _markdown()
    for stale in ("label accuracy moved from 0.50 to 0.58, one of twelve images (in",
                  "the mean reference similarity rose from 65.57 to 69.27 and prompt",
                  "Training took 119.9 s.</details>", "no run of this version is recorded yet",
                  "Releasing the reloaded pipeline brought GPU memory back from 3.8 GB to 2.22 GB in that run."):
        assert stale not in md, stale


def test_values_that_moved_are_given_per_run():
    md = _markdown()
    assert "rose from 65.57 to 69.27, 68.22 and 68.88" in md
    assert "but stayed at 0.50 in the Colab T4 run of blob `adef085a`" in md
    assert "Dark-eyed Junco 62.12 → 61.67, 60.99 and 60.74" in md
    assert "1,140.3 s for blob `4cf243c1` on 2026-10-08, 1,118.9 s for blob `adef085a`" in md
