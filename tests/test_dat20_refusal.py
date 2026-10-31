"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import contextlib
import importlib.machinery
import io
import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from test_pixart_sigma_generation_colab_review_fixes import _one_caption_zip, _package_namespace, _section_4

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/pixart_sigma_generation_colab.ipynb"
BYOD_CELL = "pixart_sigma_generation-17"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "pixart_sigma_generation"
NB_DATA = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
EARLIER = {"train_records": "old", "test_records": "old", "frozen_scores": "old", "adapt_result": "old", "parity": "old", "result_payload": "old"}


def _seed(tmp_path: Path) -> dict:
    (tmp_path / "outputs" / f"{STEM}_lora").mkdir(parents=True)
    for name in ("sample_captions.csv", "result.json", "frozen_grid.png"):
        (tmp_path / "outputs" / f"{STEM}_{name}").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / "unrelated.txt").write_text("kept", encoding="utf-8")
    return {**_package_namespace(), **EARLIER, "pipe": "kept"}


def _run(monkeypatch, tmp_path, byod_path: str, ns: dict | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    ns = ns if ns is not None else _seed(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_section_4(NB_DATA, use_byod=True, byod_path=byod_path), "section-4", "exec"), ns)
    return ns


def _colab(monkeypatch, uploaded):
    colab = types.ModuleType("google.colab")
    colab.__spec__ = importlib.machinery.ModuleSpec("google.colab", None, is_package=True)
    colab.__path__ = []
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    _one_caption_zip(tmp_path, 5, name="five.zip")
    return {
        "folder without captions.csv (the scripted /etc run)": ("etc", "the folder must contain captions.csv"),
        "not a zip": ("fake.zip", "is not a zip archive"),
        "too few images to split": ("five.zip", "at least 6 distinct images"),
        "missing path": ("absent.zip", "does not exist"),
    }


@pytest.mark.parametrize("case", ["folder without captions.csv (the scripted /etc run)", "not a zip", "too few images to split", "missing path"])
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _cases(tmp_path)[case]
    ns = _seed(tmp_path)
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, str(tmp_path / file_name), ns)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "run this cell again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == ["unrelated.txt"]


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): The upload dialog exists only on Google Colab")


@pytest.mark.parametrize(("uploaded", "file_name"), [({}, "(no file)"), ({"a.zip": b"1", "b.zip": b"2"}, "a.zip, b.zip")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name):
    _colab(monkeypatch, uploaded)
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith(f"Refused: {file_name}: Upload exactly one .zip")


def test_byod_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    archive = _one_caption_zip(tmp_path, 7)
    ns = _seed(tmp_path)

    def broken(_path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(monkeypatch, tmp_path, str(archive), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert set(EARLIER) <= set(ns) and (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing removed


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    """The pinned sample is not user input: a digest mismatch keeps its plain ValueError and traceback."""
    monkeypatch.chdir(tmp_path)
    ns = _seed(tmp_path)

    def tampered(cache_dir):
        raise ValueError("sha256 mismatch for the pinned photo")

    ns["fetch_sample_dataset"] = tampered
    with pytest.raises(ValueError, match="sha256 mismatch") as caught:
        exec(compile(_section_4(NB_DATA, use_byod=False), "section-4", "exec"), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert set(EARLIER) <= set(ns)


def test_good_byod_still_loads(monkeypatch, tmp_path):
    ns = _run(monkeypatch, tmp_path, str(_one_caption_zip(tmp_path, 7)))
    assert ns["data_source"] == "BYOD (byod.zip)" and len(ns["train_records"]) == 5


@pytest.mark.skipif(not Path("/etc").is_dir() or Path("/etc/captions.csv").exists(), reason="needs a POSIX /etc, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc': a folder present in every fresh runtime, without captions.csv."""
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "/etc")
    assert _refusal_line(caught.value).startswith("Refused: etc: /etc: the folder must contain captions.csv")
