"""PixArt-Σ XL-2 512-MS (`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`) DIMER pipeline: verified snapshots, text-to-image
generation, held-out denoising-loss evaluation, and bounded LoRA fine-tuning of the diffusion transformer to a
user's captioned images with a portable adapter.

PixArt-Σ (Chen et al., 2024) is a 0.6 B-parameter Diffusion Transformer (DiT): a T5-XXL encoder turns the prompt
into 300 token embeddings, a 28-block transformer denoises a 4-channel latent (the SDXL VAE's 8× downsampled
image) under classifier-free guidance, and the VAE decodes the latent to pixels. Three pinned snapshots are
used, each with its own manifest and staged/verified separately:

* the **transformer** (`MODEL_ID`, 2.44 GB safetensors, float32 as shipped, run in float16 on CUDA);
* the **text encoder, tokenizer, VAE and scheduler** the upstream authors publish beside every Σ checkpoint
  (`BASE_ID`, 19.4 GB, the T5-XXL encoder stored in float32 and loaded in float16 on CUDA);
* the **scorer** used only by evaluation (`SCORER_ID`, an MIT-licensed CLIP ViT-B/32, 605 MB).

Every file is safetensors or plain JSON/text: nothing is unpickled and no Hub-hosted code is executed (the model
classes come from `diffusers` and `transformers` on PyPI).

The adaptation contract is LoRA on the query/key/value/output projections of the self- and cross-attention of
all 28 blocks (rank 8, 448 tensors, 4,128,768 parameters); the transformer, VAE and text encoder stay frozen.
Training minimises the noise-prediction MSE on the user's images; the held-out metric is the same MSE at fixed
timesteps and fixed noise, so the frozen and adapted models are compared on identical inputs. Everything
model-related is imported lazily so that snapshot verification and input validation run (and can refuse) before
`torch`, `diffusers` or `transformers` are imported (fleet RTM-001).
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
MODEL_REVISION = "76fb7eb5a9314bc1e4e479d2f13447517fca9be4"
MODEL_LICENSE = "openrail++"
MODEL_KEY = "pixart-sigma-xl-2-512-ms"
ARTIFACT_FORMAT = "org.valcorza.pixart-sigma-generation.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_DIR = _WEIGHTS_ROOT / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The Σ checkpoints ship only the transformer; the authors publish the T5-XXL encoder, tokenizer, SDXL VAE and
# scheduler once, in a separate repository that every Σ pipeline composes with. Pinned as a second snapshot.
BASE_ID = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
BASE_REVISION = "2c17b4e85261cd549b4068d086b7c2ba9d468e9f"
BASE_LICENSE = "openrail"
BASE_KEY = "pixart-sigma-t5-vae"
BASE_WEIGHTS_DIR = _WEIGHTS_ROOT / BASE_KEY
# Evaluation-only scorer (never trained, never part of generation): a CLIP ViT-B/32 served as safetensors.
SCORER_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SCORER_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
SCORER_LICENSE = "mit"
SCORER_KEY = "clip-vit-b-32-laion2b"
SCORER_WEIGHTS_DIR = _WEIGHTS_ROOT / SCORER_KEY

# Architecture and contract facts (transformer/config.json of the pinned snapshot).
TRANSFORMER_PARAMETERS = 610_856_096
TRANSFORMER_TENSORS = 603
NUM_BLOCKS = 28
HIDDEN_SIZE = 1152
RESOLUTION = 512  # the 512-MS checkpoint's training resolution; images are resized + centre-cropped to it
LATENT_CHANNELS = 4
VAE_SCALE = 8
MAX_PROMPT_TOKENS = 300  # T5 sequence length the Σ models were trained with
MAX_CAPTION_CHARS = 1_000
MIN_IMAGE_SIDE = 256
MAX_IMAGE_SIDE = 4_096
MIN_TRAIN_RECORDS = 4
MAX_RECORDS = 2_000
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 4.5
MAX_STEPS = 100
MAX_GUIDANCE = 20.0
NUM_TRAIN_TIMESTEPS = 1000
EVAL_TIMESTEPS: tuple[int, ...] = (100, 300, 500, 700, 900)  # fixed timesteps of the held-out denoising loss
LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGETS: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
LORA_TENSORS = 448  # 28 blocks × (attn1 + attn2) × 4 projections × (A, B)
LORA_PARAMETERS = 4_128_768
NEGATIVE_PROMPT = ""


# --------------------------------------------------------------------------------------------------
# manifests and staging (three pinned snapshots)
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if not entry["path"].endswith((".safetensors", ".json", ".model", ".txt", ".md")):
            raise ValueError(f"{entry['path']}: unexpected file type in a code-free snapshot")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the transformer snapshot against its DIMER manifest (size + SHA-256 of every listed file)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def verify_base_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the text-encoder / tokenizer / VAE / scheduler snapshot against its own manifest."""
    root = Path(path) if path is not None else BASE_WEIGHTS_DIR
    return _verify_manifest(root, BASE_ID, BASE_REVISION)


def verify_scorer_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the CLIP scorer snapshot against its own manifest."""
    root = Path(path) if path is not None else SCORER_WEIGHTS_DIR
    return _verify_manifest(root, SCORER_ID, SCORER_REVISION)


def _hub_download(relative_path: str, root: Path, model_id: str, revision: str) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(model_id, relative_path, revision=revision, local_dir=str(root))


def _stage_missing(
    root: Path,
    model_id: str,
    revision: str,
    allow_download: bool,
    downloader: Callable[[str, Path], None] | None,
) -> list[str]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != model_id or manifest.get("revision") != revision:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {model_id}@{revision}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {revision}"
        )
    fetch = downloader or (lambda rel, dst: _hub_download(rel, dst, model_id, revision))
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch transformer-manifest entries that are absent locally (a fresh clone commits the manifest and
    git-ignores the 2.44 GB safetensors)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _stage_missing(root, MODEL_ID, MODEL_REVISION, allow_download, downloader)


def stage_missing_base_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Same for the 19.4 GB text-encoder / tokenizer / VAE / scheduler snapshot at BASE_REVISION."""
    root = Path(path) if path is not None else BASE_WEIGHTS_DIR
    return _stage_missing(root, BASE_ID, BASE_REVISION, allow_download, downloader)


def stage_missing_scorer_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Same for the CLIP scorer snapshot at SCORER_REVISION."""
    root = Path(path) if path is not None else SCORER_WEIGHTS_DIR
    return _stage_missing(root, SCORER_ID, SCORER_REVISION, allow_download, downloader)


# --------------------------------------------------------------------------------------------------
# captioned-image records and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": "{id, image, caption}: a PIL image (or a path to one) and the caption used to generate it",
    "image_side": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "resolution": RESOLUTION,
    "preprocessing": (
        f"each image is resized so its shorter side is {RESOLUTION} px and centre-cropped to {RESOLUTION}×{RESOLUTION}; "
        "the crop is reported per record (VAL7). Nothing else is changed"
    ),
    "caption_chars": [1, MAX_CAPTION_CHARS],
    "prompt_tokens": MAX_PROMPT_TOKENS,
    "records": [MIN_TRAIN_RECORDS, MAX_RECORDS],
    "generation": {"steps": [1, MAX_STEPS], "guidance_scale": [1.0, MAX_GUIDANCE], "size": RESOLUTION},
    "validation": (
        "record shape, image decodability and side limits, caption length and duplicate ids only. Nothing checks "
        "that a caption describes its image, that the images are photographs, or that the prompt is one the model "
        "can render -- any RGB image with any string is accepted"
    ),
}


def _check_record(record: Any, index: int) -> dict[str, Any]:
    from PIL import Image

    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/caption")
    for key in ("id", "image", "caption"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid, image, caption = record["id"], record["image"], record["caption"]
    if not isinstance(rid, str) or not rid or len(rid) > 64:
        raise ValueError(f"{label}: id must be a non-empty string of at most 64 characters")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"{label}: image sides must be within {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px, got {image.size}")
    if not isinstance(caption, str) or not caption.strip() or len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"{label}: caption must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
    item = {"id": rid, "image": image.convert("RGB"), "caption": caption.strip()}
    for key in ("label", "common_name", "scientific_name", "observer", "inat_photo_id", "inat_observation_url", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def image_digest(image: Any) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same photo matches."""
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes()).hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], image_digest(r["image"]), r["caption"]] for r in records]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_TRAIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a captioned-image dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, caption} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    crops = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        width, height = item["image"].size
        if width != height:
            crops += 1
        checked.append(item)
    sides = [min(r["image"].size) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "n_captions": len({r["caption"] for r in checked}),
        "shorter_side": {"min": min(sides), "max": max(sides)},
        "centre_cropped": crops,
        "resolution": RESOLUTION,
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record; returns its id, size, the crop it will get and the caption length."""
    item = _check_record(record, 0)
    width, height = item["image"].size
    short = min(width, height)
    scale = RESOLUTION / short
    return {
        "id": item["id"],
        "size": (width, height),
        "resized_to": (round(width * scale), round(height * scale)),
        "centre_crop": (RESOLUTION, RESOLUTION),
        "caption_chars": len(item["caption"]),
    }


def validate_prompts(prompts: Sequence[str]) -> list[str]:
    """Generation prompts: non-empty strings within the caption limit; duplicates are allowed."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence) or not prompts:
        raise ValueError("prompts must be a non-empty list of strings")
    out = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_CAPTION_CHARS:
            raise ValueError(f"prompts[{index}] must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
        out.append(prompt.strip())
    return out


def preprocess_image(image: Any) -> Any:
    """Resize the shorter side to RESOLUTION and centre-crop; returns a PIL RGB image of RESOLUTION²."""
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = RESOLUTION / min(width, height)
    new = (max(RESOLUTION, round(width * scale)), max(RESOLUTION, round(height * scale)))
    resized = rgb.resize(new, Image.Resampling.BICUBIC)
    left = (new[0] - RESOLUTION) // 2
    top = (new[1] - RESOLUTION) // 2
    return resized.crop((left, top, left + RESOLUTION, top + RESOLUTION))


# --------------------------------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------------------------------


def _lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, init_lora_weights="gaussian", target_modules=list(LORA_TARGETS))


def lora_parameter_names(transformer: Any) -> list[str]:
    """The exact tensor set the adaptation contract may change on a transformer built with the adapter."""
    return sorted(name for name, _param in transformer.named_parameters() if ".lora_A." in name or ".lora_B." in name)


def build_transformer(weights_dir: Path, *, dtype: Any, use_lora: bool) -> Any:
    """Load the pinned transformer from the verified snapshot; optionally attach the (untrained) LoRA adapter."""
    import torch
    from diffusers import PixArtTransformer2DModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PixArtTransformer2DModel.from_pretrained(str(weights_dir), subfolder="transformer", torch_dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != TRANSFORMER_PARAMETERS or len(model.state_dict()) != TRANSFORMER_TENSORS:
        raise ValueError(
            f"transformer has {n_params} parameters in {len(model.state_dict())} tensors; "
            f"expected {TRANSFORMER_PARAMETERS} / {TRANSFORMER_TENSORS}"
        )
    for param in model.parameters():
        param.requires_grad_(False)
    if use_lora:
        from peft import inject_adapter_in_model

        inject_adapter_in_model(_lora_config(), model, adapter_name="default")
        names = lora_parameter_names(model)
        if len(names) != LORA_TENSORS:
            raise ValueError(f"adapter attached {len(names)} LoRA tensors, expected {LORA_TENSORS}")
        for name, param in model.named_parameters():
            if name in set(names):
                param.data = param.data.to(torch.float32)  # trained in float32 under autocast
                param.requires_grad_(False)
    model.eval()
    return model


def _select_device(device: str | None) -> str:
    import torch

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    return device


@dataclass
class PixArtSigmaPipeline:
    """Text-to-image generation and bounded LoRA fine-tuning on top of the verified PixArt-Σ 512-MS snapshot."""

    transformer: Any
    vae: Any
    scheduler_config: dict[str, Any]
    tokenizer: Any
    text_encoder_dir: Path
    device: str
    dtype: Any
    weights_dir: Path
    base_dir: Path
    source: str
    use_lora: bool
    adapter: dict[str, Any] | None = None
    _text_encoder: Any = field(default=None, repr=False)
    _prompt_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
        allow_download: bool = False,
        use_lora: bool = False,
    ) -> PixArtSigmaPipeline:
        """Stage (when allowed) and verify both model snapshots, then load the transformer, VAE, tokenizer and
        scheduler config. The text encoder is loaded lazily by `encode_prompts` and released by
        `release_text_encoder`, because its 9.5 GB (float16) does not fit beside a training graph on a 16 GB GPU."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        base = Path(base_dir) if base_dir is not None else BASE_WEIGHTS_DIR
        stage_missing_files(root, allow_download=allow_download)
        stage_missing_base_files(base, allow_download=allow_download)
        verify_snapshot(root)
        verify_base_snapshot(base)
        import torch
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
        from transformers import AutoTokenizer

        chosen = _select_device(device)
        dtype = torch.float16 if chosen.startswith("cuda") else torch.float32
        transformer = build_transformer(root, dtype=dtype, use_lora=use_lora).to(chosen)
        # The SDXL VAE overflows in float16 (`force_upcast` in its config); it is kept in float32 throughout.
        vae = AutoencoderKL.from_pretrained(str(base), subfolder="vae", torch_dtype=torch.float32).to(chosen).eval()
        for param in vae.parameters():
            param.requires_grad_(False)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(str(base), subfolder="scheduler")
        tokenizer = AutoTokenizer.from_pretrained(str(base / "tokenizer"))
        return cls(
            transformer=transformer,
            vae=vae,
            scheduler_config=dict(scheduler.config),
            tokenizer=tokenizer,
            text_encoder_dir=base / "text_encoder",
            device=chosen,
            dtype=dtype,
            weights_dir=root,
            base_dir=base,
            source="local-snapshot (three manifests verified; safetensors only)",
            use_lora=use_lora,
        )

    # ---- prompts ---------------------------------------------------------------------------------------

    def encode_prompts(self, prompts: Sequence[str], *, text_encoder_device: str | None = None) -> dict[str, Any]:
        """Encode every distinct prompt (and the empty negative prompt) with the T5-XXL encoder into the
        pipeline's prompt cache. The encoder is loaded on first use — float16 on CUDA, float32 on CPU — on
        `text_encoder_device` (default: the pipeline device) and stays loaded until `release_text_encoder`."""
        import torch
        from transformers import T5EncoderModel

        wanted = [p for p in dict.fromkeys([NEGATIVE_PROMPT, *validate_prompts(list(prompts))]) if p not in self._prompt_cache]
        if not wanted:
            return {"encoded": 0, "cached": len(self._prompt_cache)}
        enc_device = text_encoder_device or self.device
        enc_dtype = torch.float16 if enc_device.startswith("cuda") else torch.float32
        if self._text_encoder is None:
            started = time.perf_counter()
            encoder = T5EncoderModel.from_pretrained(str(self.text_encoder_dir), torch_dtype=enc_dtype)
            self._text_encoder = encoder.to(enc_device).eval()
            load_seconds = round(time.perf_counter() - started, 1)
        else:
            load_seconds = 0.0
        started = time.perf_counter()
        with torch.inference_mode():
            for prompt in wanted:
                tokens = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=MAX_PROMPT_TOKENS,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                n_tokens = int(tokens.attention_mask.sum())
                mask = tokens.attention_mask.to(enc_device)
                embeds = self._text_encoder(tokens.input_ids.to(enc_device), attention_mask=mask)[0]
                self._prompt_cache[prompt] = {
                    "embeds": embeds[0].to("cpu", torch.float16 if self.dtype == torch.float16 else torch.float32),
                    "mask": mask[0].to("cpu"),
                    "tokens": n_tokens,
                }
        return {
            "encoded": len(wanted),
            "cached": len(self._prompt_cache),
            "encoder_device": enc_device,
            "encoder_dtype": str(enc_dtype).replace("torch.", ""),
            "load_seconds": load_seconds,
            "encode_seconds": round(time.perf_counter() - started, 1),
            "truncated": [p[:40] for p in wanted if self._prompt_cache[p]["tokens"] >= MAX_PROMPT_TOKENS],
        }

    def export_prompt_cache(self) -> dict[str, Any]:
        """The encoded prompts (CPU tensors keyed by prompt), so a second pipeline can reuse them without
        loading the 9.5 GB text encoder again (the notebook's fresh-reload step)."""
        return {
            k: {"embeds": v["embeds"].clone(), "mask": v["mask"].clone(), "tokens": v["tokens"]}
            for k, v in self._prompt_cache.items()
        }

    def import_prompt_cache(self, cache: Mapping[str, Mapping[str, Any]]) -> int:
        """Adopt prompt embeddings exported by `export_prompt_cache` from a pipeline of the same identity."""
        for prompt, entry in cache.items():
            if tuple(entry["embeds"].shape) != (MAX_PROMPT_TOKENS, 4096) or tuple(entry["mask"].shape) != (MAX_PROMPT_TOKENS,):
                raise ValueError(f"prompt cache entry for {prompt[:40]!r} has an unexpected shape")
            self._prompt_cache[prompt] = {"embeds": entry["embeds"], "mask": entry["mask"], "tokens": int(entry["tokens"])}
        return len(self._prompt_cache)

    def release_text_encoder(self) -> bool:
        """Drop the T5 encoder (the cached embeddings remain). Returns whether anything was released."""
        import torch

        had = self._text_encoder is not None
        self._text_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return had

    def _embeds(self, prompt: str) -> tuple[Any, Any]:
        if prompt not in self._prompt_cache:
            raise ValueError(f"prompt not encoded; call encode_prompts([...]) first: {prompt[:60]!r}")
        entry = self._prompt_cache[prompt]
        return entry["embeds"].to(self.device, self.dtype), entry["mask"].to(self.device)

    def _batch_embeds(self, prompts: Sequence[str]) -> tuple[Any, Any]:
        import torch

        pairs = [self._embeds(p) for p in prompts]
        return torch.stack([e for e, _ in pairs]), torch.stack([m for _, m in pairs])

    # ---- generation ------------------------------------------------------------------------------------

    def _diffusers_pipeline(self) -> Any:
        from diffusers import DPMSolverMultistepScheduler
        from diffusers import PixArtSigmaPipeline as _Upstream

        return _Upstream(
            tokenizer=self.tokenizer,
            text_encoder=None,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=DPMSolverMultistepScheduler.from_config(self.scheduler_config),
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        seed: int = 0,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
    ) -> dict[str, Any]:
        """Generate one RESOLUTION² image per prompt with classifier-free guidance; image i uses seed + i."""
        import numpy as np
        import torch

        prompts = validate_prompts(prompts)
        if not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be an int in 1..{MAX_STEPS}")
        if not 1.0 <= float(guidance_scale) <= MAX_GUIDANCE:
            raise ValueError(f"guidance_scale must be in 1..{MAX_GUIDANCE}")
        self.encode_prompts(prompts) if any(p not in self._prompt_cache for p in [NEGATIVE_PROMPT, *prompts]) else None
        pipe = self._diffusers_pipeline()
        pipe.set_progress_bar_config(disable=True)
        neg_embeds, neg_mask = self._embeds(NEGATIVE_PROMPT)
        started = time.perf_counter()
        images = []
        for index, prompt in enumerate(prompts):
            embeds, mask = self._embeds(prompt)
            generator = torch.Generator(device="cpu").manual_seed(seed + index)
            with torch.inference_mode():
                out = pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=embeds[None],
                    prompt_attention_mask=mask[None],
                    negative_prompt_embeds=neg_embeds[None],
                    negative_prompt_attention_mask=neg_mask[None],
                    num_inference_steps=steps,
                    guidance_scale=float(guidance_scale),
                    height=RESOLUTION,
                    width=RESOLUTION,
                    generator=generator,
                    output_type="pil",
                    use_resolution_binning=False,
                )
            image = out.images[0]
            array = np.asarray(image)
            images.append({"prompt": prompt, "seed": seed + index, "image": image, "pixel_mean": round(float(array.mean()), 3)})
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "steps": steps,
            "guidance_scale": float(guidance_scale),
            "size": (RESOLUTION, RESOLUTION),
            "scheduler": "DPMSolverMultistepScheduler (upstream config)",
            "precision": str(self.dtype).replace("torch.", ""),
            "images": images,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- latents and the denoising loss --------------------------------------------------------------

    def _latents(self, records: Sequence[Mapping[str, Any]], *, seed: int) -> Any:
        """VAE-encode preprocessed images to scaled latents (B, 4, 64, 64); the posterior sample is seeded."""
        import numpy as np
        import torch

        arrays = [np.asarray(preprocess_image(r["image"]), dtype=np.float32) / 127.5 - 1.0 for r in records]
        pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device, torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            posterior = self.vae.encode(pixels).latent_dist
            latents = posterior.sample(generator=generator) * self.vae.config.scaling_factor
        return latents.to(self.dtype)

    def _noise_scheduler(self) -> Any:
        from diffusers import DDPMScheduler

        keys = ("num_train_timesteps", "beta_start", "beta_end", "beta_schedule", "prediction_type", "trained_betas")
        return DDPMScheduler.from_config({k: v for k, v in self.scheduler_config.items() if k in keys})

    def _predict_noise(self, noisy: Any, timesteps: Any, embeds: Any, mask: Any) -> Any:
        out = self.transformer(
            hidden_states=noisy,
            encoder_hidden_states=embeds,
            encoder_attention_mask=mask,
            timestep=timesteps,
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
            return_dict=False,
        )[0]
        return out[:, :LATENT_CHANNELS]  # the remaining channels are the learned variance (unused here)

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, seed: int = 0, batch_size: int = 4) -> dict[str, Any]:
        """Held-out denoising MSE: every record is VAE-encoded, noised at each of EVAL_TIMESTEPS with a seeded
        noise tensor, and the transformer's noise prediction is scored against that noise. The same seed gives the
        same latents, noise and timesteps for the frozen and the adapted model, so the numbers are paired."""
        import torch

        checked = validate_dataset(records, min_records=1)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        self.encode_prompts([r["caption"] for r in checked])
        scheduler = self._noise_scheduler()
        started = time.perf_counter()
        per_timestep: dict[int, list[float]] = {t: [] for t in EVAL_TIMESTEPS}
        per_record: dict[str, float] = {}
        self.transformer.eval()
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            latents = self._latents(batch, seed=seed + start)
            embeds, mask = self._batch_embeds([r["caption"] for r in batch])
            record_losses = [0.0] * len(batch)
            for t in EVAL_TIMESTEPS:
                generator = torch.Generator(device="cpu").manual_seed(seed * 1_000 + t + start)
                noise = torch.randn(latents.shape, generator=generator).to(self.device, self.dtype)
                timesteps = torch.full((len(batch),), t, device=self.device, dtype=torch.long)
                noisy = scheduler.add_noise(latents.float(), noise.float(), timesteps).to(self.dtype)
                use_amp = self.dtype == torch.float16
                autocast = torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp)
                with torch.inference_mode(), autocast:
                    pred = self._predict_noise(noisy, timesteps, embeds, mask)
                loss = ((pred.float() - noise.float()) ** 2).mean(dim=(1, 2, 3))
                for i, value in enumerate(loss.tolist()):
                    per_timestep[t].append(value)
                    record_losses[i] += value / len(EVAL_TIMESTEPS)
            for record, value in zip(batch, record_losses, strict=True):
                per_record[record["id"]] = round(value, 6)
        by_t = {str(t): round(sum(v) / len(v), 6) for t, v in per_timestep.items()}
        mean = sum(per_record.values()) / len(per_record)
        return {
            "metric": "denoising_mse (noise-prediction MSE over the latent, mean over records and EVAL_TIMESTEPS)",
            "n_records": len(checked),
            "timesteps": list(EVAL_TIMESTEPS),
            "seed": seed,
            "denoising_mse": round(mean, 6),
            "by_timestep": by_t,
            "per_record": per_record,
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- adaptation ------------------------------------------------------------------------------------

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 4,
        lr: float = 1e-4,
        batch_size: int = 1,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded LoRA fine-tuning on the noise-prediction objective: every step draws one timestep per image
        uniformly from the training schedule and one noise tensor (both seeded), and minimises the MSE between the
        transformer's noise prediction and that noise. AdamW at a fixed learning rate on the 448 LoRA tensors only,
        float16 autocast with loss scaling on CUDA. Epoch 0 records the frozen model; the epoch with the lowest
        validation denoising MSE is kept."""
        if not self.use_lora:
            raise ValueError("adapt() needs a pipeline built with use_lora=True")
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be an int in 1..8")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        import torch

        self.encode_prompts([r["caption"] for r in train_checked] + ([r["caption"] for r in val_checked] if val_checked else []))
        torch.manual_seed(seed)
        started = time.perf_counter()
        model = self.transformer
        names = lora_parameter_names(model)
        name_set = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in name_set)
        params = [p for n, p in model.named_parameters() if n in name_set]
        n_trainable = sum(p.numel() for p in params)
        if n_trainable != LORA_PARAMETERS:
            raise ValueError(f"{n_trainable} trainable parameters, expected {LORA_PARAMETERS}")
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        use_amp = self.dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scheduler = self._noise_scheduler()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Latents are encoded once (the VAE posterior sample is seeded), so epochs differ only in noise/timesteps.
        latents_by_id = {}
        for start in range(0, len(train_checked), 4):
            batch = train_checked[start : start + 4]
            encoded = self._latents(batch, seed=seed + 10_000 + start)
            for record, latent in zip(batch, encoded, strict=True):
                latents_by_id[record["id"]] = latent
        try:
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "note": "frozen model (LoRA at initialisation: B = 0)"}
            entry["val_loss"] = self.evaluate(val_checked, seed=seed)["denoising_mse"] if val_checked else None
            history.append(entry)
            if progress:
                progress(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            best_epoch = 0
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    latents = torch.stack([latents_by_id[r["id"]] for r in batch])
                    embeds, mask = self._batch_embeds([r["caption"] for r in batch])
                    noise = torch.randn(latents.shape, generator=generator).to(self.device, self.dtype)
                    timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (len(batch),), generator=generator).to(self.device)
                    noisy = scheduler.add_noise(latents.float(), noise.float(), timesteps).to(self.dtype)
                    with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp):
                        pred = self._predict_noise(noisy, timesteps, embeds, mask)
                    loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                    optimiser.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses)}
                entry["val_loss"] = self.evaluate(val_checked, seed=seed)["denoising_mse"] if val_checked else None
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
                    best_epoch = epoch
        except BaseException:
            # Transactional: any failure leaves the transformer as it was before adapt() — LoRA restored to its
            # initial values, everything frozen, no adapter attached.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "method": "LoRA (peft)",
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "targets": list(LORA_TARGETS),
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": "AdamW (weight_decay 0, grad-norm clip 1.0)",
            "precision": "float16 autocast + GradScaler" if use_amp else "float32",
            "objective": "noise-prediction MSE, uniform timesteps",
            "n_train_records": len(train_checked),
            "n_steps": n_steps,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts -------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained LoRA tensors as safetensors with a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        state = self.transformer.state_dict()
        tensors = {k: v.detach().to("cpu", dtype=v.dtype).contiguous() for k, v in state.items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "components": {"id": BASE_ID, "revision": BASE_REVISION, "key": BASE_KEY},
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
        """Static checks before any weights work: format and version, the pinned base and component snapshots,
        exactly one weights entry named `adapter.safetensors` inside the artifact directory, and the pinned LoRA
        configuration. Returns the weights path."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        components = base.get("components", {})
        if (components.get("id"), components.get("revision")) != (BASE_ID, BASE_REVISION):
            raise ValueError("artifact records different text-encoder/VAE components")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        declared = None
        if isinstance(adapter, Mapping):
            declared = (adapter.get("rank"), adapter.get("alpha"), list(adapter.get("targets", [])))
        if declared != (LORA_RANK, LORA_ALPHA, list(LORA_TARGETS)):
            raise ValueError(
                f"artifact adapter must declare rank {LORA_RANK}, alpha {LORA_ALPHA} and targets {list(LORA_TARGETS)}"
            )
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, scope and digest, then overwrite exactly the LoRA tensors."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self.check_artifact_manifest(root, manifest)
        if not self.use_lora:
            raise ValueError("this adapter carries LoRA tensors; build the pipeline with use_lora=True")
        expected = lora_parameter_names(self.transformer)
        if sorted(manifest["tensors"]) != expected:
            raise ValueError(f"artifact tensor list does not match the {len(expected)} LoRA tensors of this model")
        entry = manifest["files"][0]
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the validated manifest")
        state = self.transformer.state_dict()
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, model has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()})
        self.transformer.load_state_dict(merged, strict=True)
        self.transformer.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> PixArtSigmaPipeline:
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.check_artifact_manifest(root, manifest)
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, base_dir=base_dir, allow_download=allow_download, use_lora=True
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
