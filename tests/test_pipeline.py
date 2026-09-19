"""Offline tests for the three snapshot manifests, staging, the dataset contract and the artifact-manifest
rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import synthetic_image, synthetic_records
from pixart_sigma_generation_pipeline import (
    BASE_ID,
    BASE_REVISION,
    INPUT_SCHEMA,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGETS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    RESOLUTION,
    SCORER_REVISION,
    PixArtSigmaPipeline,
    dataset_digest,
    preprocess_image,
    stage_missing_base_files,
    stage_missing_files,
    stage_missing_scorer_files,
    validate_dataset,
    validate_inputs,
    validate_prompts,
    verify_base_snapshot,
    verify_scorer_snapshot,
    verify_snapshot,
)
from pixart_sigma_generation_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_FILE = "transformer/diffusion_pytorch_model.safetensors"


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": "k",
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and committed manifests -------------------------------------------------------------------


def test_identity_is_immutable_and_committed_manifests_agree():
    assert len(MODEL_REVISION) == len(BASE_REVISION) == len(SCORER_REVISION) == 40
    for key, model_id, revision in (
        (pl.MODEL_KEY, MODEL_ID, MODEL_REVISION),
        (pl.BASE_KEY, BASE_ID, BASE_REVISION),
        (pl.SCORER_KEY, pl.SCORER_ID, SCORER_REVISION),
    ):
        manifest = json.loads((ROOT / "weights" / key / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (model_id, revision, key)
        assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
        assert all(len(e["sha256"]) == 64 for e in manifest["files"])
        assert not any(e["path"].endswith((".bin", ".pt", ".pth", ".ckpt", ".pickle", ".py")) for e in manifest["files"])
    transformer = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = {"README.md", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"}
    assert {e["path"] for e in transformer["files"]} == expected
    base = json.loads((ROOT / "weights" / pl.BASE_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    components = {"text_encoder/model-00001-of-00002.safetensors", "vae/diffusion_pytorch_model.safetensors"}
    assert components | {"tokenizer/spiece.model", "scheduler/scheduler_config.json"} <= {e["path"] for e in base["files"]}


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    manifest = _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"transformer/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    assert verify_snapshot(root)["files"] == manifest["files"]
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"tensorz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"tensors-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_snapshot_refuses_executable_file_types(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"transformer/diffusion_pytorch_model.bin": b"pickle"})
    with pytest.raises(ValueError, match="unexpected file type"):
        verify_snapshot(root)


def test_each_snapshot_has_its_own_identity(tmp_path, forbid_model_imports):
    base = tmp_path / "base"
    _write_snapshot(base, BASE_ID, BASE_REVISION, {"vae/config.json": b"{}"})
    assert verify_base_snapshot(base)["modelId"] == BASE_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(base)
    scorer = tmp_path / "scorer"
    _write_snapshot(scorer, pl.SCORER_ID, SCORER_REVISION, {"config.json": b"{}"})
    assert verify_scorer_snapshot(scorer)["modelId"] == pl.SCORER_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_base_snapshot(scorer)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"transformer/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    (root / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"tensors")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [WEIGHTS_FILE]
    assert calls == ["transformer/diffusion_pytorch_model.safetensors"]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_stage_refuses_a_manifest_naming_another_model(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=lambda rel, dst: None)
    base = tmp_path / "base"
    _write_snapshot(base, MODEL_ID, MODEL_REVISION, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_base_files(base, allow_download=True, downloader=lambda rel, dst: None)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_scorer_files(base, allow_download=True, downloader=lambda rel, dst: None)


# --- dataset contract -------------------------------------------------------------------------------------


def test_validate_dataset_reports_shape_and_digest(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["n_captions"] == 2
    assert report["shorter_side"] == {"min": 480, "max": 480} and report["centre_cropped"] == 6
    assert report["resolution"] == RESOLUTION and len(report["digest"]) == 64
    assert report["digest"] == dataset_digest(records)
    assert INPUT_SCHEMA["image_side"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]


def test_validate_dataset_refusals_name_the_rule(forbid_model_imports):
    records = synthetic_records(6)
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"})
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="missing 'caption'"):
        validate_dataset([{"id": "a", "image": records[0]["image"]}, *records[1:]])
    with pytest.raises(ValueError, match="caption must be a non-empty string"):
        validate_dataset([{**records[0], "caption": "   "}, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records])
    with pytest.raises(ValueError, match="image sides must be within"):
        validate_dataset([{**records[0], "image": synthetic_image(width=200, height=200)}, *records[1:]])
    with pytest.raises(ValueError, match="image must be a PIL"):
        validate_dataset([{**records[0], "image": b"bytes"}, *records[1:]])
    with pytest.raises(ValueError, match="image file not found"):
        validate_dataset([{**records[0], "image": "nope.jpg"}, *records[1:]])


def test_validate_inputs_reports_the_crop(tmp_path, forbid_model_imports):
    image = synthetic_image(width=800, height=600)
    path = tmp_path / "photo.jpg"
    image.save(path)
    report = validate_inputs({"id": "p", "image": str(path), "caption": "a photo"})
    assert report["size"] == (800, 600) and report["resized_to"] == (683, 512) and report["centre_crop"] == (512, 512)
    out = preprocess_image(image)
    assert out.size == (RESOLUTION, RESOLUTION) and out.mode == "RGB"
    assert preprocess_image(Image.new("RGB", (512, 512), (10, 20, 30))).getpixel((0, 0)) == (10, 20, 30)


def test_validate_prompts(forbid_model_imports):
    assert validate_prompts(["  a bird ", "a bird"]) == ["a bird", "a bird"]
    with pytest.raises(ValueError, match="non-empty list"):
        validate_prompts([])
    with pytest.raises(ValueError, match="prompts\\[1\\]"):
        validate_prompts(["ok", ""])


# --- artifact manifest (static checks, no weights) -------------------------------------------------------


def _good_manifest(root: Path) -> dict:
    (root / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    return {
        "format": pl.ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "key": pl.MODEL_KEY,
            "components": {"id": BASE_ID, "revision": BASE_REVISION, "key": pl.BASE_KEY},
        },
        "adapter": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "targets": list(LORA_TARGETS)},
        "tensors": ["transformer_blocks.0.attn1.to_q.lora_A.default.weight"],
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
    }


def test_artifact_manifest_static_checks(tmp_path, forbid_model_imports):
    good = _good_manifest(tmp_path)
    assert PixArtSigmaPipeline.check_artifact_manifest(tmp_path, good) == (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).resolve()
    cases = {
        "format": ({**good, "format": "other"}, "artifact format"),
        "version": ({**good, "format_version": "2.0"}, "format_version"),
        "base": ({**good, "base_model": {**good["base_model"], "revision": "0" * 40}}, "different base model"),
        "components": (
            {**good, "base_model": {**good["base_model"], "components": {"id": BASE_ID, "revision": "0" * 40}}},
            "components",
        ),
        "two files": ({**good, "files": good["files"] * 2}, "exactly one weights file"),
        "other name": ({**good, "files": [{**good["files"][0], "path": "weights.safetensors"}]}, "must be named"),
        "traversal": ({**good, "files": [{**good["files"][0], "path": "../adapter.safetensors"}]}, "must be named"),
        "rank": ({**good, "adapter": {**good["adapter"], "rank": 16}}, "must declare rank"),
        "targets": ({**good, "adapter": {**good["adapter"], "targets": ["to_q"]}}, "must declare rank"),
        "tensors": ({**good, "tensors": "all"}, "list its tensors"),
    }
    for name, (manifest, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            PixArtSigmaPipeline.check_artifact_manifest(tmp_path, manifest)
        del name
