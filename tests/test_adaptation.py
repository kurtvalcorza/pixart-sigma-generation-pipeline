"""Adaptation, evaluation and artifact tests on a stub transformer/VAE (torch + diffusers required, no weights):
the training loop, epoch selection, the transactional guarantee and the artifact round trip with its refusals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from conftest import synthetic_records  # noqa: E402
from pixart_sigma_generation_pipeline import LORA_TENSORS, PixArtSigmaPipeline, lora_parameter_names  # noqa: E402
from pixart_sigma_generation_pipeline import pipeline as pl  # noqa: E402

RANK, DIM, EMB = 2, 6, 8


class _StubTransformer(torch.nn.Module):
    """Names follow peft's layout; the output depends on the LoRA tensors so training moves them."""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        for block in range(2):
            for attn in ("attn1", "attn2"):
                for target in ("to_q", "to_k"):
                    stem = f"transformer_blocks.{block}.{attn}.{target}".replace(".", "__")
                    self.register_parameter(f"{stem}__lora_A__default__weight", torch.nn.Parameter(torch.randn(RANK, DIM) * 0.1))
                    self.register_parameter(f"{stem}__lora_B__default__weight", torch.nn.Parameter(torch.zeros(DIM, RANK)))

    def named_parameters(self, *args, **kwargs):  # type: ignore[override]
        for name, param in super().named_parameters(*args, **kwargs):
            yield name.replace("__", "."), param

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {k.replace("__", "."): v for k, v in super().state_dict(*args, **kwargs).items()}

    def load_state_dict(self, state, strict=True):  # type: ignore[override]
        return super().load_state_dict({k.replace(".", "__"): v for k, v in state.items()}, strict=strict)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask, timestep, added_cond_kwargs, return_dict):
        gain = 1.0
        for name, param in self.named_parameters():
            if ".lora_B." in name:
                a = dict(self.named_parameters())[name.replace("lora_B", "lora_A")]
                gain = gain + (param @ a).mean()
        pred = hidden_states * gain + 0.01 * encoder_hidden_states.mean()
        return (torch.cat([pred, pred], dim=1),)


class _StubVAE:
    config = SimpleNamespace(scaling_factor=0.13025)

    def encode(self, pixels):
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, 4)[:, :3]
        latent = torch.cat([pooled, pooled[:, :1]], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None: latent))


def _pipeline(monkeypatch) -> PixArtSigmaPipeline:
    transformer = _StubTransformer()
    names = lora_parameter_names(transformer)
    monkeypatch.setattr(pl, "LORA_PARAMETERS", sum(dict(transformer.named_parameters())[n].numel() for n in names))
    pipe = PixArtSigmaPipeline(
        transformer=transformer,
        vae=_StubVAE(),
        scheduler_config={
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
        },
        tokenizer=None,
        text_encoder_dir=Path("unused"),
        device="cpu",
        dtype=torch.float32,
        weights_dir=Path("unused"),
        base_dir=Path("unused"),
        source="stub",
        use_lora=True,
    )
    generator = torch.Generator().manual_seed(0)
    for prompt in (pl.NEGATIVE_PROMPT, "a photo of a red bird", "a photo of a blue bird"):
        embeds = torch.randn(4, EMB, generator=generator)
        pipe._prompt_cache[prompt] = {"embeds": embeds, "mask": torch.ones(4, dtype=torch.long), "tokens": 4}
    return pipe


def test_stub_matches_the_contract_shape():
    names = lora_parameter_names(_StubTransformer())
    assert len(names) == 16 and all(".lora_" in n for n in names) and LORA_TENSORS == 448


def test_evaluate_is_paired_and_seeded(monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    first = pipe.evaluate(records, seed=3)
    second = pipe.evaluate(records, seed=3)
    assert first["denoising_mse"] == second["denoising_mse"] and first["per_record"] == second["per_record"]
    assert set(first["by_timestep"]) == {str(t) for t in pl.EVAL_TIMESTEPS} and first["adapted"] is False
    assert pipe.evaluate(records, seed=4)["denoising_mse"] != first["denoising_mse"]


def test_adapt_trains_only_lora_and_keeps_the_best_epoch(monkeypatch):
    pipe = _pipeline(monkeypatch)
    train, val = synthetic_records(6), synthetic_records(2, seed=50)
    before = pipe.evaluate(val, seed=0)["denoising_mse"]
    base_before = pipe.transformer.base.clone()
    seen = []
    result = pipe.adapt(train, val, epochs=3, lr=1e-2, seed=0, progress=seen.append)
    assert [e["epoch"] for e in seen] == [0, 1, 2, 3] and seen[0]["note"].startswith("frozen model")
    assert result["history"][0]["val_loss"] == before
    assert result["best_epoch"] == min(range(4), key=lambda i: result["history"][i]["val_loss"])
    assert result["n_trainable"] == pl.LORA_PARAMETERS and result["n_steps"] == 18 and result["precision"] == "float32"
    assert torch.equal(pipe.transformer.base, base_before)
    assert pipe.adapter is not None and all(not p.requires_grad for p in pipe.transformer.parameters())
    assert pipe.evaluate(val, seed=0)["denoising_mse"] == result["history"][result["best_epoch"]]["val_loss"]


def test_adapt_is_transactional_when_the_progress_callback_raises(monkeypatch):
    pipe = _pipeline(monkeypatch)
    initial = {k: v.clone() for k, v in pipe.transformer.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipe.adapt(synthetic_records(4), None, epochs=2, progress=boom)
    assert pipe.adapter is None
    assert all(torch.equal(initial[k], v) for k, v in pipe.transformer.state_dict().items())
    assert all(not p.requires_grad for p in pipe.transformer.parameters())


def test_adapt_refusals(monkeypatch):
    pipe = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(synthetic_records(4), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(synthetic_records(4), lr=1.0)
    with pytest.raises(ValueError, match="4..2000"):
        pipe.adapt(synthetic_records(3))
    pipe.use_lora = False
    with pytest.raises(ValueError, match="use_lora=True"):
        pipe.adapt(synthetic_records(4))
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact("unused")


def test_artifact_round_trip_and_refusals(tmp_path, monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    pipe.adapt(records, epochs=1, lr=1e-2)
    adapted = pipe.evaluate(records, seed=0)["denoising_mse"]
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"tutorial": "test"})
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == pl.ARTIFACT_FORMAT and len(manifest["tensors"]) == 16
    assert manifest["metadata"] == {"tutorial": "test"}
    assert manifest["base_model"]["components"]["revision"] == pl.BASE_REVISION
    fresh = _pipeline(monkeypatch)
    assert fresh.evaluate(records, seed=0)["denoising_mse"] != adapted
    fresh.load_artifact(out)
    assert fresh.evaluate(records, seed=0)["denoising_mse"] == adapted and fresh.adapter["best_epoch"] == 1
    # digest mismatch
    (out / pl.ARTIFACT_WEIGHTS_NAME).write_bytes((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest or size"):
        _pipeline(monkeypatch).load_artifact(out)
    # tensor list outside the scope
    pipe.save_artifact(out)
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["tensors"] = manifest["tensors"][:-1] + ["base"]
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _pipeline(monkeypatch).load_artifact(out)
    # a pipeline without the adapter attached refuses
    pipe.save_artifact(out)
    plain = _pipeline(monkeypatch)
    plain.use_lora = False
    with pytest.raises(ValueError, match="use_lora=True"):
        plain.load_artifact(out)
    digest = hashlib.sha256((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes()).hexdigest()
    assert digest == json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text())["files"][0]["sha256"]
