"""Final prose pass (Kurt's ruling, 2026-10-26): the prose-rule items flagged on the confirming run are fixed.

Run-status sentences, timing promises and unlabelled expectations are removed or recast
as records of named runs.
Fails on PR #9 head a3e57bf.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "pixart_sigma_generation_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_final_prose_pass():
    md = _markdown(NB)
    for gone in (
        'has not yet been run on Colab',
        'MSE around 0.1..0.3',
        '0.36 at t = 100',
        '0.0002 at t = 900',
        'the test loss fell by 0.14 %',
    ):
        assert gone not in md, gone
    for kept in (
        'Look for: the denoising MSE, label accuracy',
        'for t = 100 and t = 900), so',
        'Check the size and the per-species rows printed above.',
    ):
        assert kept in md, kept
