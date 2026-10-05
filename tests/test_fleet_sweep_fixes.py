"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['pixart_sigma_generation_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'pixart_sigma_generation_colab': 5}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


# --- repository-specific: SWP-G checkpoint numbers, SWP-F frozen LoRA snapshot, SWP-B BYOD path ----------------------

import os  # noqa: E402

NB = NOTEBOOKS[0]
RECORD = ROOT / "docs" / "release-verification.md"


def test_swp_g_checkpoint_answers_quote_the_recorded_run():
    """Every number in a Check-your-reasoning answer is in the recorded 2026-09-19 Kaggle T4 run."""
    markdown = "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")
    checks = re.findall(r"<details><summary>Check your reasoning</summary>(.*?)</details>", markdown, re.S)
    assert len(checks) == 6
    record = RECORD.read_text(encoding="utf-8")
    for number in ("0.105983", "0.105838", "0.109093", "0.108941", "26.79", "28.01", "65.57", "68.95", "30.25", "0.917", "88.66", "0.0"):
        assert number in record, number
    assert "0.105983" in checks[2] and "0.109093" in checks[3] and "0.105838" in checks[4] and "mean_abs_pixel_diff: 0.0" in checks[5]
    assert "epoch 3" in checks[3].lower() or "best epoch was 3" in checks[3]


class _Tensor:
    def __init__(self, value, device="cuda", dtype="float16"):
        self.value, self.device, self.dtype = value, device, dtype

    def detach(self):
        return self

    def cpu(self):
        return _Tensor(self.value, "cpu", self.dtype)

    def clone(self):
        return _Tensor(self.value, self.device, self.dtype)

    def to(self, device, dtype):
        return _Tensor(self.value, device, dtype)


class _Transformer:
    def __init__(self):
        self.state = {"blocks.0.attn.to_q.lora_A.weight": _Tensor(1.0), "blocks.0.attn.to_q.lora_B.weight": _Tensor(0.0), "blocks.0.attn.to_q.weight": _Tensor(5.0)}
        self.loaded = []

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, merged, strict=True):
        assert strict and set(merged) == set(self.state)
        self.state = dict(merged)
        self.loaded.append({k: v.value for k, v in merged.items()})

    def eval(self):
        return self


def _section_6_head(nb: dict) -> str:
    source = _cell(nb, "def restore_frozen_lora():")
    return source[: source.index("scorer = ClipScorer(")]


def test_swp_f_section_6_takes_the_frozen_snapshot_once_and_restores_it_on_a_rerun(capsys):
    """SWP-F: the first run of Section 6 snapshots the untrained LoRA; a re-run after training restores it and forgets the adapter."""
    head = _section_6_head(_nb(NB))
    pipe = types.SimpleNamespace(transformer=_Transformer(), adapter=None)
    ns = {"pipe": pipe, "lora_parameter_names": lambda t: [k for k in t.state_dict() if "lora_" in k]}
    exec(compile(head, "<section 6>", "exec"), ns)
    assert set(ns["frozen_lora_state"]) == {"blocks.0.attn.to_q.lora_A.weight", "blocks.0.attn.to_q.lora_B.weight"}
    assert ns["frozen_lora_state"]["blocks.0.attn.to_q.lora_B.weight"].device == "cpu"
    assert "'frozen_lora_snapshot': 'taken'" in capsys.readouterr().out
    # Section 7 trains in place.
    pipe.transformer.state["blocks.0.attn.to_q.lora_B.weight"] = _Tensor(3.0)
    pipe.adapter = {"best_epoch": 3}
    exec(compile(head, "<section 6 again>", "exec"), ns)
    assert pipe.transformer.state["blocks.0.attn.to_q.lora_B.weight"].value == 0.0
    assert pipe.transformer.state["blocks.0.attn.to_q.weight"].value == 5.0  # the base weights are untouched
    assert pipe.transformer.loaded[-1]["blocks.0.attn.to_q.lora_B.weight"] == 0.0
    assert pipe.adapter is None
    assert "'frozen_lora_snapshot': 'restored'" in capsys.readouterr().out


def test_swp_f_section_7_restores_the_frozen_lora_before_training_and_checks_epoch_0():
    s7 = _cell(_nb(NB), "adapt_result = pipe.adapt(")
    assert s7.index("restore_frozen_lora()") < s7.index("adapt_result = pipe.adapt(")
    check = s7[s7.index("epoch0 = adapt_result") : s7.index("print({'trainable_parameters'")]
    ns = {"adapt_result": {"history": [{"val_loss": 0.109093}]}, "frozen_val": {"denoising_mse": 0.109093}}
    exec(check, ns)  # identical: no error
    ns["adapt_result"] = {"history": [{"val_loss": 0.1}]}
    with pytest.raises(RuntimeError, match="Epoch 0 is not the Section 6 frozen model"):
        exec(check, ns)
    ns["adapt_result"] = {"history": [{"val_loss": None}]}
    exec(check, ns)  # no validation split: nothing to compare


def _byod_file(nb: dict):
    source = _cell(nb, "def byod_file(")
    helper = source[source.index("def byod_file(") : source.index("os.makedirs('outputs'")]
    ns = {"os": os, "Path": Path}
    exec(helper, ns)
    return ns["byod_file"]


def test_swp_b_byod_path_reads_a_zip_and_refusals_name_the_file(tmp_path, monkeypatch):
    helper = _byod_file(_nb(NB))
    good = tmp_path / "mine.zip"
    good.write_bytes(b"PK")
    assert helper(str(good), "zip", (".zip",)) == good
    with pytest.raises(FileNotFoundError, match="missing.zip"):
        helper(str(tmp_path / "missing.zip"), "zip")
    bad = tmp_path / "wrong.tar"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"wrong.tar: expected a"):
        helper(str(bad), "zip", (".zip",))
    _no_colab(monkeypatch)
    with pytest.raises(RuntimeError, match="only in Google Colab"):
        helper("", "zip")


def test_swp_b_cancelled_upload_is_named_and_one_upload_is_saved(tmp_path, monkeypatch):
    helper = _byod_file(_nb(NB))
    _colab(monkeypatch, lambda: {})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Upload exactly one"):
        helper("", "zip")
    _colab(monkeypatch, lambda: {"up.zip": b"1"})
    saved = helper("", "zip", (".zip",))
    assert saved.read_bytes() == b"1" and saved.parent.name == "work"


def test_swp_b_byod_gate_is_off_by_default_and_has_a_path_field():
    code = "\n".join(c["source"] for c in _code_cells(_nb(NB)))
    assert "USE_BYOD = False  # @param" in code and "BYOD_PATH = ''  # @param" in code
    assert "next(iter(uploaded.items()))" not in code.split("def byod_file(")[0]
