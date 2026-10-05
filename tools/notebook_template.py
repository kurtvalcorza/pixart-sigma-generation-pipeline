"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E text-to-image fine-tuning workflow: the pinned PixArt-Σ 512-MS transformer,
its T5-XXL / SDXL-VAE components and the CLIP scorer are staged and digest-verified, 60 pinned CC0
iNaturalist bird photographs are fetched, validated and split, every prompt is encoded once and the text
encoder released, the frozen model is scored (held-out denoising loss, CLIP-scored generations) against the
real-photo ceiling, a bounded LoRA fine-tuning runs in the kernel, the held-out scores are read again in a
paired comparison, a new prompt is rendered, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "pixart-sigma-generation-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/pixart_sigma_generation_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PixArt--alpha%2FPixArt--Sigma--XL--2--512--MS-ffcc4d?style=flat",
        "https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-PixArt--alpha%2FPixArt--sigma-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/PixArt-alpha/PixArt-sigma",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2403.04692-b31b1b.svg", "https://arxiv.org/abs/2403.04692"),
]

TEMPLATE = {
    "package": "pixart_sigma_generation_pipeline",
    "repo_name": REPO,
    "stem": "pixart_sigma_generation",
    "notebook_name": "pixart_sigma_generation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) builds an isolated, "
        "hash-locked Python 3.12.12 environment with the pinned dependencies (torch, diffusers, transformers, peft, accelerate, "
        "sentencepiece, safetensors, huggingface-hub, numpy, pillow) — nothing is installed into the notebook kernel, so no "
        "restart is needed — stages and digest-verifies three pinned snapshots from the Hub — the 2.4 GB PixArt-Σ 512-MS transformer, "
        "the 19.4 GB T5-XXL encoder / tokenizer / SDXL VAE / scheduler the authors publish beside it, and a 0.6 GB CLIP scorer — "
        "loads the transformer in float16 with an untrained LoRA adapter attached, fetches 60 CC0 iNaturalist bird photographs "
        "as digest-verified JPEGs (6 MB, no credential), validates them and splits them 36 / 12 / 12 by seed, encodes every "
        "prompt with the T5 encoder and releases it, scores the frozen model — the held-out denoising loss on the validation and "
        "test photographs, and twelve generated images scored by CLIP against their prompts, the held-out photographs and the "
        "real-photo ceiling — runs a bounded LoRA fine-tuning (4 epochs over 36 images), scores the adapted model on identical "
        "inputs, renders a new prompt, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify generation parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On a T4 the whole path takes about "
        "fifteen minutes of model time after the 22 GB of downloads."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 (and `BYOD_PATH` to the zip's path on Kaggle or "
        "Jupyter; on Colab an empty path opens the upload dialog) and re-run from that cell to supply your own "
        "captioned photographs as a zip holding `captions.csv` (columns `id`, `file`, `caption`) beside the image files (JPEG or "
        "PNG, shorter side 256..4096 px; at least four images, and at least one caption with three or more images so a held-out "
        "record exists). Your records are split by caption into training, validation and test sets and flow through the same "
        "contract — validation, prompt encoding, frozen baseline, adaptation, held-out evaluation, generation, artifact export "
        "and reload parity. The expected schema, the ceilings and the privacy guidance are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "PixArtSigmaPipeline",
    "model_load": "PixArtSigmaPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, base_dir=BASE_WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), use_lora=True)",
    "weights_key": "pixart-sigma-xl-2-512-ms",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    # generator /2: the three weights directories derive from one shared root; the second and third pinned
    # snapshots (T5/VAE/scheduler components, CLIP scorer) are carried, staged and verified by the model cell.
    "rewrites": [
        [
            r"^_WEIGHTS_ROOT = Path\(__file__\)[^\n]*$",
            '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ]
    ],
    "extra_weights": [
        {
            "key": "pixart-sigma-t5-vae",
            "var": "BASE_MANIFEST",
            "dir": "BASE_WEIGHTS_DIR",
            "identity": ["BASE_ID", "BASE_REVISION"],
            "stage": "stage_missing_base_files",
            "verify": "verify_base_snapshot",
        },
        {
            "key": "clip-vit-b-32-laion2b",
            "var": "SCORER_MANIFEST",
            "dir": "SCORER_WEIGHTS_DIR",
            "identity": ["SCORER_ID", "SCORER_REVISION"],
            "stage": "stage_missing_scorer_files",
            "verify": "verify_scorer_snapshot",
        },
    ],
    "runtime_imports": ["torch", "diffusers", "transformers", "peft"],
    "title": "PixArt-Σ 512-MS — DIMER E2E text-to-image fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "text-to-image generation with a 0.6 B-parameter diffusion transformer, held-out denoising-loss and CLIP-scored evaluation, and bounded LoRA fine-tuning to a set of captioned photographs",
    "intro": (
        "PixArt-Σ (Chen et al., 2024) is a Diffusion Transformer: a frozen T5-XXL encoder turns the prompt into 300 token "
        "embeddings, 28 transformer blocks denoise a 4-channel latent — the SDXL VAE's 8× compressed image — with "
        "classifier-free guidance, and the VAE decodes the latent to pixels. The checkpoint here is **XL-2 512-MS**, the "
        "512 × 512 member of the family (the 1024 and 2K members are the same architecture at more tokens; they are not "
        "packaged). Three pinned snapshots make one generator: the transformer (2.4 GB), the components the authors publish "
        "once for every Σ checkpoint — the 19 GB float32 T5-XXL encoder, the tokenizer, the SDXL VAE and the scheduler — and, "
        "for evaluation only, a CLIP ViT-B/32 scorer.\n\n"
        "Two things about this row are handled in the open. **The text encoder does not fit beside a training graph on a "
        "16 GB GPU**, so Section 5 encodes every prompt the notebook will ever use — the 60 captions, the generation prompts, "
        "the empty negative prompt — once, keeps the embeddings, and releases the 9.5 GB encoder before anything is trained; "
        "the reloaded pipeline in Section 9 adopts the same embeddings rather than loading it again. **Generation has no ground "
        "truth**, so the notebook reads three kinds of number and says what each is: the held-out *denoising loss* (the "
        "training objective, measured on photographs the model never trained on, with identical noise for the frozen and the "
        "adapted model), CLIP scores of generated images (prompt alignment, which species CLIP thinks it sees, and closeness to "
        "the held-out real photographs), and the same CLIP scores on the real photographs themselves — the ceiling. None of "
        "these is a human judgement of image quality."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and scorer modules; stage and digest-verify three "
        "pinned snapshots; fetch, validate and split a small real captioned-photograph dataset; encode prompts with a "
        "text encoder and release it; read a held-out denoising loss and CLIP-scored generations against a real-photo "
        "ceiling; run a bounded LoRA fine-tuning of a diffusion transformer with explicit hyperparameters; compare the "
        "adapted and frozen models on identical held-out inputs; render a new prompt; and export a safetensors adapter that "
        "reloads against the pinned base with verified generation parity."
    ),
    "exclusions": (
        "the 1024-MS and 2K-MS checkpoints, PixArt-α, ControlNet or image-to-image conditioning, full fine-tuning, DreamBooth "
        "identifiers, safety filtering of prompts or images, human preference or FID/KID benchmarks, prompt engineering, and any "
        "claim that a CLIP score or a denoising loss measures image quality. The repository exposes none of these."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met the idea of a diffusion model (noise → latent → image), and wants to see what a small LoRA fine-tuning of a 0.6 B-parameter diffusion transformer does and does not change — read through a held-out training loss and CLIP scores, not through a human judgement of pictures. The audience is students and practitioners deciding whether a few dozen captioned photographs can steer a text-to-image model towards their own visual domain; no prior experience with PixArt-Σ, diffusers or peft is assumed — each term is explained where it first matters and again in the **Glossary**. A GPU with 15 GB (a Colab T4) is required; CPU-only runtimes are not supported.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | captioned photographs as `{{id, image, caption}}` records; the default is 60 pinned CC0 iNaturalist photographs of six bird species, split 36 / 12 / 12 by seed, with one template caption per species |\n| Model | PixArt-Σ XL-2 512-MS: a frozen T5-XXL text encoder, a 28-block diffusion transformer denoising the SDXL VAE's 4-channel latent, and a rank-8 LoRA (4.1 M parameters, 0.7 %) on the attention projections — the only trained part |\n| Output | a held-out denoising MSE and CLIP scores (prompt similarity, label accuracy, reference similarity) for the frozen and the adapted model against the real-photo ceiling, two image grids, a rendering of a new prompt, a 16 MB safetensors adapter that reloads with verified parity, and `result.json` |\n\n**How to use this notebook.** Choose a **GPU** runtime (T4 or better), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the three verified snapshots — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded Kaggle T4 run of 19 September 2026. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Budget about 30 minutes: the 22 GB of downloads dominate, then about fifteen minutes of model time.\n\n**Roadmap:** 1–3 infrastructure → 4 the photographs, the data contract and three refusals *(core concept: a record is validated before any model work)* → 5 encode the prompts and release the 9.5 GB encoder *(engineering: fitting a training graph on 16 GB)* → 6 the frozen model: held-out denoising loss and CLIP-scored generations against the real-photo ceiling *(core concept: three kinds of number and what each measures)* → 7 bounded LoRA fine-tuning *(core concept: what 0.7 % of the parameters can move)* → 8 the paired held-out comparison *(evaluation practice: identical inputs before and after)* → 9 a new prompt, export and reload parity *(engineering)* → conclude."
    )]},
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with PixArt-Σ, diffusers or peft. Latents, guidance, the denoising loss, LoRA, the CLIP scores and reload parity are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **GPU** runtime on **Linux x86_64** (Google Colab T4 or better, Kaggle, or a Linux Jupyter server with a CUDA GPU of at least 15 GB). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The T5-XXL encoder runs in float16 (9.5 GB) while it encodes prompts and is then released; the transformer runs in float16 (1.2 GB) with the LoRA parameters in float32; the SDXL VAE stays in float32 (0.33 GB). CPU-only runtimes are not supported for this notebook: the encoder would need 19 GB of RAM in float32 and the diffusion steps take minutes per image. About 25 GB of disk is needed for the three snapshots.",
        "- **Knowledge:** what a latent diffusion model does at inference (noise → latent → image), what classifier-free guidance is, what a LoRA adapter changes and what it does not, and why a training loss is not a quality score.",
        "- **Weights:** the transformer, the T5-XXL encoder, the SDXL VAE and the CLIP scorer are all safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers` and `peft` on PyPI. The Σ weights are released under the CreativeML Open RAIL++-M licence, which carries use-based restrictions; the scorer is MIT.",
        "- **Data contract:** a record is `{id, image, caption}` — an RGB image with shorter side 256..4096 px (resized so the shorter side is 512 px and centre-cropped to 512 × 512; the crop is reported) and a caption of 1..1000 characters (truncated to 300 T5 tokens; truncations are reported). Validation is structural: nothing checks that a caption describes its image or that the model can render it.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, validation and splits\n\n"
                "The default dataset is 60 research-grade iNaturalist photographs of six common North American birds — 10 per "
                "species, one per observer per species, every one CC0 — fetched by photo id from the open-data bucket and "
                "refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). Each photo's caption is generated from its "
                "species by one template, so the adaptation teaches the generator what six names look like in this kind of "
                "photograph. `build_sample_dataset` draws a seeded stratified split — 6 / 2 / 2 per species for training, "
                "validation and test — and `dataset_manifest` validates every split, checks that no image appears twice and "
                "records a digest.\n\n"
                "Look for: 36 / 12 / 12 records, six distinct captions, a shorter side around 300..500 px (every photo is "
                "centre-cropped to 512²), a written `outputs/{stem}_sample_captions.csv` in the shape BYOD expects, and "
                "three refusal probes — a missing caption, a 200 px image, a duplicate id — each rejected before the model runs.\n\n"
                "**Predict:** three probes are deliberately broken. Will each be refused, and by what — a model, or a rule that runs before any model work? Which of the three would a model have silently accepted?"
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = \'\'  # @param {{type:"string"}}\n'
                '\n'
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ValueError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                'if USE_BYOD:\n'
                "    byod_path = byod_file(BYOD_PATH, 'captioned-photograph zip (captions.csv beside the image files)', ('.zip',))\n"
                '    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n'
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/inat-birds')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'captions': dataset_report['splits']['train']['n_captions'], 'disjoint': dataset_report['disjoint']}})\n"
                "print({{'shorter_side': dataset_report['splits']['train']['shorter_side'], 'centre_cropped': dataset_report['splits']['train']['centre_cropped'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "print({{'first_test_record': validate_inputs(test_records[0]), 'caption': test_records[0]['caption']}})\n"
                "prompts = sample_prompts(train_records)\n"
                "print({{'prompts': prompts}})\n"
                "sample_csv = write_dataset_csv(test_records, 'outputs/{stem}_sample_captions.csv')\n"
                "print({{'sample_csv': str(sample_csv)}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'missing caption': [{{'id': r['id'], 'image': r['image']}} for r in train_records[:4]],\n"
                "    'image too small': [{{**train_records[0], 'image': Image.new('RGB', (200, 200))}}, *train_records[1:4]],\n"
                "    'duplicate id': [train_records[0], *train_records[:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>All three are refused, and none by the model: `validate_dataset` runs before any snapshot is loaded and names the rule — a record without a caption, a shorter side of 200 px (below the 256 px floor), an id that appears twice. A model would have accepted the 200 px image after resizing and the duplicate without comment; only the missing caption would have failed, and later, inside the training loop. The default split is 36 / 12 / 12 with six captions by construction, so those numbers do not depend on the run.</details>'
            ),
        },
        {
            "md": (
                "## 5. Encode every prompt, then release the text encoder\n\n"
                "`pipe.encode_prompts` loads the T5-XXL encoder from the verified component snapshot (float16 on CUDA), "
                "tokenises each distinct prompt to 300 tokens with the pinned SentencePiece model, runs the encoder once per "
                "prompt and keeps the (300, 4096) embeddings and attention mask on the CPU. Encoded here: the six training "
                "captions (which are also the validation and test captions and the generation prompts), one new prompt for "
                "Section 9, and the empty negative prompt that classifier-free guidance needs. `release_text_encoder` then "
                "drops the 9.5 GB encoder so the transformer, the VAE, the scorer and a training graph fit on a 16 GB GPU.\n\n"
                "Look for: the encoder loading in about a minute from the 19 GB float32 shards, seven or eight prompts encoded "
                "in seconds, no truncation, and the GPU memory falling back by about 9.5 GB after the release.\n\n"
                "**Predict:** seven or eight prompts, 300 tokens each. Will any caption be truncated, and after `release_text_encoder` will the GPU hold more or less than half of what it held with the encoder loaded?"
            ),
            "code": (
                "import time\n\n"
                "NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'\n\n"
                "def gpu_memory_gb():\n"
                "    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None\n\n"
                "all_prompts = sample_prompts(train_records + val_records + test_records) + [NEW_PROMPT]\n"
                "encode_report = pipe.encode_prompts(all_prompts)\n"
                "print({{**encode_report, 'gpu_memory_gb_with_encoder': gpu_memory_gb()}})\n"
                "released = pipe.release_text_encoder()\n"
                "print({{'text_encoder_released': released, 'gpu_memory_gb_after_release': gpu_memory_gb(), 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', '')}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>No truncation: the template captions are far shorter than 300 tokens, and the report shows `truncated: 0`. After the release the GPU holds the float16 transformer (about 1.2 GB) and the float32 VAE (0.33 GB), well under half of what it held with the 9.5 GB encoder; the exact figures depend on the runtime, so read them from the printed `gpu_memory_gb_*` values. The recorded run encoded the prompts in seconds and loaded the encoder in about a minute from the 19 GB float32 shards.</details>'
            ),
        },
        {
            "md": (
                "## 6. The frozen model: held-out denoising loss and CLIP-scored generations\n\n"
                "The pipeline was built with `use_lora=True`: the adapter's B matrices start at zero, so until Section 7 this is "
                "the pretrained model. Two kinds of number are read here and kept for the comparison.\n\n"
                "**Held-out denoising loss** (`pipe.evaluate`): each held-out photograph is VAE-encoded, noised at five fixed "
                "timesteps (100, 300, 500, 700, 900) with a seeded noise tensor, and the transformer's noise prediction is "
                "scored against that noise (MSE over the latent). It is the training objective measured on photographs the "
                "model never trains on; the same seed gives the same latents, noise and timesteps later, so the adapted number "
                "is a paired comparison, not a re-draw.\n\n"
                "**CLIP-scored generations** (`pipe.generate` + `score_generations`): two images per training caption at fixed "
                "seeds (20 DPM-Solver++ steps, guidance 4.5), scored by the frozen CLIP ViT-B/32 on prompt alignment (cosine × "
                "100), zero-shot label accuracy (which of the six captions is nearest — an argmax with no threshold) and "
                "similarity to the mean embedding of the held-out real photographs of that species. `real_photo_baseline` "
                "scores the real test photographs the same way: the ceiling these numbers could reach. Look for: a "
                "denoising MSE around 0.1..0.3, label accuracy well below the real photographs' 1.0 for at least some species, "
                "and a first grid of twelve generated birds.\n\n"
                "**Predict:** the twelve generated birds are scored by the same CLIP as the real test photographs. Will the frozen model's label accuracy reach the real photographs' value, and which of the three CLIP numbers will be furthest from its ceiling?"
            ),
            "code": (
                "STEPS = 20  # @param {{type:\"integer\"}}\n"
                "GUIDANCE_SCALE = 4.5  # @param {{type:\"number\"}}\n"
                "IMAGES_PER_PROMPT = 2  # @param {{type:\"integer\"}}\n"
                "EVAL_SEED = 0\n\n"
                "def grid(images, path, columns=6):\n"
                "    tiles = [im.resize((256, 256)) for im in images]\n"
                "    rows = (len(tiles) + columns - 1) // columns\n"
                "    sheet = Image.new('RGB', (256 * columns, 256 * rows), 'white')\n"
                "    for i, tile in enumerate(tiles):\n"
                "        sheet.paste(tile, (256 * (i % columns), 256 * (i // columns)))\n"
                "    sheet.save(path)\n"
                "    return path\n\n"
                '# Sections 6 and 7 describe the frozen model and what one adaptation adds to it, so both start from the untrained\n'
                '# LoRA (B = 0). The snapshot is taken once, before any training; a re-run restores it instead of scoring or training\n'
                '# whatever an earlier Section 7 left in the transformer.\n'
                'def restore_frozen_lora():\n'
                '    """Put the LoRA tensors back to the frozen snapshot and forget any adapter, so the pipeline is the pretrained model again."""\n'
                '    state = pipe.transformer.state_dict()\n'
                '    merged = {{**state, **{{name: value.to(state[name].device, state[name].dtype) for name, value in frozen_lora_state.items()}}}}\n'
                '    pipe.transformer.load_state_dict(merged, strict=True)\n'
                '    pipe.transformer.eval()\n'
                '    pipe.adapter = None\n'
                '\n'
                "if 'frozen_lora_state' not in globals():\n"
                '    lora_names = set(lora_parameter_names(pipe.transformer))\n'
                '    frozen_lora_state = {{name: value.detach().cpu().clone() for name, value in pipe.transformer.state_dict().items() if name in lora_names}}\n'
                "    print({{'frozen_lora_snapshot': 'taken', 'tensors': len(frozen_lora_state), 'adapted': pipe.adapter is not None}})\n"
                'else:\n'
                '    restore_frozen_lora()\n'
                "    print({{'frozen_lora_snapshot': 'restored', 'tensors': len(frozen_lora_state), 'adapted': pipe.adapter is not None}})\n"
                '\n'
                'scorer = ClipScorer(weights_dir=SCORER_WEIGHTS_DIR, device=pipe.device)\n'
                't0 = time.perf_counter()\n'
                'frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n'
                "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "print({{'frozen_denoising_mse': {{'validation': frozen_val['denoising_mse'], 'test': frozen_test['denoising_mse']}}, 'by_timestep_test': frozen_test['by_timestep'], 'seconds': round(time.perf_counter() - t0, 1)}})\n\n"
                "generation_prompts = [p for p in prompts for _ in range(IMAGES_PER_PROMPT)]\n"
                "t0 = time.perf_counter()\n"
                "frozen_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "print({{'generated': len(frozen_generation['images']), 'steps': frozen_generation['steps'], 'guidance_scale': frozen_generation['guidance_scale'], 'seconds': frozen_generation['seconds'], 'adapted': frozen_generation['model']['adapted']}})\n"
                "frozen_scores = score_generations(scorer, frozen_generation['images'], references=test_records)\n"
                "real_ceiling = real_photo_baseline(scorer, test_records)\n"
                "print({{'frozen_generations': {{k: frozen_scores[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "print({{'real_photo_ceiling': {{k: real_ceiling[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "for entry in frozen_scores['per_image'][::IMAGES_PER_PROMPT]:\n"
                "    print({{'prompt': entry['prompt'][:42], 'clip': entry['clip_prompt_similarity'], 'nearest': entry['nearest_prompt'][13:40], 'correct': entry['correct'], 'reference_similarity': entry['reference_similarity']}})\n"
                "print({{'grid': str(grid([g['image'] for g in frozen_generation['images']], 'outputs/{stem}_frozen_grid.jpg'))}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>No. In the recorded run the frozen model's twelve generations scored label accuracy 0.50 against the real photographs' 0.917, prompt similarity 26.79 against 30.25 and reference similarity 65.57 against 88.66. Reference similarity is the furthest from its ceiling: CLIP agrees that the pictures show birds matching their prompts about as well as it agrees with real photographs, but the generated birds are not close to the held-out photographs of each species. The frozen held-out test denoising MSE was 0.105983.</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded LoRA fine-tuning\n\n"
                "`pipe.adapt` trains the 448 LoRA tensors (rank 8, 4,128,768 parameters — 0.7 % of the transformer) that "
                "`peft` attached to the query, key, value and output projections of the self- and cross-attention of every "
                "block, and nothing else; the transformer, the VAE and the text encoder are frozen. Each step takes one "
                "training photograph's latent (VAE-encoded once, seeded), draws a timestep uniformly from the 1,000-step "
                "schedule and a noise tensor (both seeded), adds the noise, and minimises the MSE between the predicted and "
                "the true noise; AdamW at a fixed learning rate, gradient-norm clipping at 1.0, float16 autocast with loss "
                "scaling on CUDA. Epoch 0 records the frozen model's validation loss, and the epoch with the lowest validation "
                "denoising loss is kept.\n\n"
                "Watch the validation denoising MSE fall from epoch 0; four epochs over 36 images (144 steps) take about three "
                "minutes on a T4. The training loss is a noisy per-step average over random timesteps and is not the quality "
                "signal — the paired held-out numbers in Section 8 are. The cell first restores the frozen LoRA snapshot from Section 6, so "
                "a second run (after changing `EPOCHS` or `LEARNING_RATE`) trains the pretrained model again, not the previous adaptation, and "
                "it checks that epoch 0 reproduces Section 6's frozen validation loss.\n\n"
                "**Predict:** four epochs over 36 photographs. Will the validation denoising MSE fall at every epoch, and by how much — percent, or a fraction of a percent?"
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 1  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_denoising_mse': entry['val_loss']}}\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                'restore_frozen_lora()  # every run of this cell trains the frozen model, never an earlier adaptation\n'
                't0 = time.perf_counter()\n'
                'adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=EVAL_SEED, progress=report)\n'
                'adapt_seconds = round(time.perf_counter() - t0, 1)\n'
                "epoch0 = adapt_result['history'][0]['val_loss']\n"
                "if epoch0 is not None and abs(epoch0 - frozen_val['denoising_mse']) > 1e-4:\n"
                '    raise RuntimeError(f"Epoch 0 is not the Section 6 frozen model (epoch 0 {{epoch0!r}} vs frozen validation {{frozen_val[\'denoising_mse\']!r}}): run Sections 6 and 7 again in order.")\n'
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['precision'], 'seconds': adapt_seconds, 'gpu_memory_gb': gpu_memory_gb()}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Not at every epoch, and by a fraction of a percent. In the recorded run epoch 0 (the frozen model) scored a validation denoising MSE of 0.109093 and the best epoch was 3 at 0.108941 — a drop of 0.00015, about 0.14 % — so epoch 4 was not lower than epoch 3 and was not kept. The training loss printed per epoch is a noisy average over random timesteps; it is the validation column that selects the epoch. On a T4 the four epochs took about three minutes.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation: the paired comparison\n\n"
                "The test photographs were never used for training or epoch selection. The adapted model is scored exactly as "
                "the frozen model was in Section 6 — the same seed, so the same latents, noise and timesteps, and the same "
                "twelve prompt/seed pairs for generation — and the table puts the frozen, the adapted and the real-photo "
                "numbers side by side. The cell asserts only what the procedure guarantees — the kept epoch's validation loss is no "
                "higher than the frozen model's (epoch 0) and the re-scored validation loss matches the history — and prints the test "
                "comparison without a directional assertion: a lower test denoising MSE is what to look for, not what is promised. Look "
                "too for "
                "the generated birds moving towards the held-out photographs: higher reference similarity and label accuracy, "
                "and a second grid to compare with the first by eye. Twelve images per model from one seeded run give no "
                "dispersion estimate; these are sample-sanity numbers that show the adaptation contract works, not a benchmark, "
                "and CLIP agreement is not a human judgement of quality.\n\n"
                "**Predict:** write down a direction for the test denoising MSE and for each of the three CLIP numbers. Which of the CLIP numbers do you expect not to move at all on twelve images?"
            ),
            "code": (
                "adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "adapted_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "adapted_scores = score_generations(scorer, adapted_generation['images'], references=test_records)\n"
                "comparison = {{\n"
                "    'denoising_mse_validation': {{'frozen': frozen_val['denoising_mse'], 'adapted': adapted_val['denoising_mse']}},\n"
                "    'denoising_mse_test': {{'frozen': frozen_test['denoising_mse'], 'adapted': adapted_test['denoising_mse']}},\n"
                "    'denoising_mse_test_by_timestep': {{t: {{'frozen': frozen_test['by_timestep'][t], 'adapted': adapted_test['by_timestep'][t]}} for t in adapted_test['by_timestep']}},\n"
                "    'clip_prompt_similarity': {{'frozen': frozen_scores['clip_prompt_similarity'], 'adapted': adapted_scores['clip_prompt_similarity'], 'real_photos': real_ceiling['clip_prompt_similarity']}},\n"
                "    'label_accuracy': {{'frozen': frozen_scores['label_accuracy'], 'adapted': adapted_scores['label_accuracy'], 'real_photos': real_ceiling['label_accuracy']}},\n"
                "    'reference_similarity': {{'frozen': frozen_scores['reference_similarity'], 'adapted': adapted_scores['reference_similarity'], 'real_photos': real_ceiling['reference_similarity']}},\n"
                "}}\n"
                "for name, row in comparison.items():\n"
                "    print({{name: row}})\n"
                "for before, after in zip(frozen_scores['per_image'][::IMAGES_PER_PROMPT], adapted_scores['per_image'][::IMAGES_PER_PROMPT]):\n"
                "    print({{'prompt': before['prompt'][:42], 'reference_similarity': {{'frozen': before['reference_similarity'], 'adapted': after['reference_similarity']}}, 'correct': {{'frozen': before['correct'], 'adapted': after['correct']}}}})\n"
                "print({{'grid': str(grid([g['image'] for g in adapted_generation['images']], 'outputs/{stem}_adapted_grid.jpg'))}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'components': {{'id': BASE_ID, 'revision': BASE_REVISION}},\n"
                "    'scorer': frozen_scores['scorer'],\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'generation': {{'steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'images_per_prompt': IMAGES_PER_PROMPT, 'seed': 1000}},\n"
                "    'frozen': {{'validation': frozen_val, 'test': frozen_test, 'generations': frozen_scores}},\n"
                "    'adapted': {{'validation': adapted_val, 'test': adapted_test, 'generations': adapted_scores}},\n"
                "    'real_photo_ceiling': real_ceiling,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "best = adapt_result['history'][adapt_result['best_epoch']]\n"
                "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['denoising_mse'] - best['val_loss']) < 1e-4\n"
                "print({{'test_denoising_mse_change': round(adapted_test['denoising_mse'] - frozen_test['denoising_mse'], 6), 'note': 'held-out observation, not asserted'}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded run the test denoising MSE fell from 0.105983 to 0.105838 (−0.000145), prompt similarity rose 26.79 → 28.01, reference similarity rose 65.57 → 68.95, and label accuracy stayed at 0.50: with twelve images, one changed nearest-caption vote moves that number by 0.083, so an unchanged value is the most likely outcome of a small adaptation. Every CLIP number is still far from the real-photo ceiling (30.25 / 0.917 / 88.66); these are sample-sanity numbers that show the paired contract works, not a quality claim.</details>'
            ),
        },
        {
            "md": (
                "## 9. A new prompt, artifact export and fresh reload\n\n"
                "The adapted model renders `NEW_PROMPT` — a composition that appears in no training caption — at two seeds; "
                "the CLIP prompt similarity is printed as a sanity check, not an evaluation.\n\n"
                "`pipe.save_artifact` writes the 448 trained tensors (about 16 MB) as `adapter.safetensors` with a "
                "`manifest.json` recording the artifact format, the transformer's id and revision, the component snapshot's id "
                "and revision, the LoRA configuration, the tensor names, the file size and SHA-256, the training configuration "
                "and the epoch history (OUT8). `PixArtSigmaPipeline.from_artifact` re-verifies both snapshots, checks the "
                "manifest, the LoRA scope and the digest **before** deserialising, loads a fresh transformer with the adapter "
                "attached and overlays the tensors — a new object from files, not the in-memory model (VER2). The fresh "
                "pipeline adopts the prompt embeddings already encoded (so the text encoder is not loaded again), and the cell "
                "asserts that it reproduces the same held-out denoising loss and the same image for the same prompt and seed "
                "(VER4: a mean absolute pixel difference below 1 on the 0..255 scale — same device, same kernels).\n\n"
                "**Predict:** the reloaded pipeline is a new object built from files. Will its held-out denoising MSE differ from the in-memory one at all, and will the two 512 × 512 images for the same prompt and seed differ by exactly zero or merely by less than one grey level?"
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "new_generation = pipe.generate([NEW_PROMPT, NEW_PROMPT], seed=2000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "new_scores = score_generations(scorer, new_generation['images'])\n"
                "print({{'new_prompt': NEW_PROMPT, 'clip_prompt_similarity': new_scores['clip_prompt_similarity'], 'seconds': new_generation['seconds'], 'note': 'sanity check, not an evaluation'}})\n"
                "for i, entry in enumerate(new_generation['images']):\n"
                "    entry['image'].save(f'outputs/{stem}_new_prompt_{{i}}.png')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PixArtSigmaPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, base_dir=BASE_WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded.import_prompt_cache(pipe.export_prompt_cache())\n"
                "reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)\n"
                "before = pipe.generate([prompts[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "after = reloaded.generate([prompts[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "parity = {{'denoising_mse_diff': round(abs(reloaded_test['denoising_mse'] - adapted_test['denoising_mse']), 8), 'mean_abs_pixel_diff': round(float(np.abs(np.asarray(before, dtype=np.float32) - np.asarray(after, dtype=np.float32)).mean()), 4)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['denoising_mse_diff'] < 1e-6 and parity['mean_abs_pixel_diff'] < 1.0\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', ''), 'source': pipe.source}},\n"
                "    'components': {{**evaluation_report['components'], 'license': BASE_LICENSE}},\n"
                "    'scorer': {{**evaluation_report['scorer'], 'license': SCORER_LICENSE}},\n"
                "    'provenance': {{\n"
                "        'snapshots': {{'transformer': len(MANIFEST['files']), 'components': len(BASE_MANIFEST['files']), 'scorer': len(SCORER_MANIFEST['files'])}},\n"
                "        'safetensors_only': True,\n"
                "        'remote_code_executed': False,\n"
                "        'text_encoder_released_before_training': released,\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded run both parity numbers were exactly 0.0 — `denoising_mse_diff: 0.0` and `mean_abs_pixel_diff: 0.0` — because the reloaded pipeline ran on the same device with the same kernels and the same prompt embeddings. The check accepts anything below 1e-6 for the loss and below one grey level for the image, so a run on different hardware that passes with a small non-zero pixel difference is still parity.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A LoRA of four million parameters trained for a few minutes on 36 photographs lowers the held-out denoising loss on "
        "twelve photographs the model never saw and moves its generations towards the held-out real photographs of the same "
        "species. That is the claim: the adaptation contract teaches a diffusion transformer a narrow visual domain from a "
        "handful of captioned images, the held-out objective is measured on identical inputs before and after, and the artifact "
        "that carries the change is 16 MB.\n\n"
        "The numbers are sample-sanity evidence. A denoising loss is the training objective, not a quality score; CLIP "
        "similarity and CLIP's nearest-caption vote are a frozen model's opinion, not a human judgement, and CLIP itself has "
        "biases about what a species name looks like; twelve images per model from one seeded run give no dispersion "
        "estimate; and nothing here measures aesthetics, diversity, artefacts or prompt fidelity beyond the six captions. "
        "Fine-tuning on a narrow domain can also erode the model elsewhere — the new prompt in Section 9 is a sanity check on "
        "one composition, not a test of generality.\n\n"
        "Three things to carry to real data. **Captions are the contract:** the adapter learns the association between the "
        "caption text and the images; a caption that does not describe its image, or one caption for very different images, "
        "teaches noise. **Hold out by caption, not by image:** the split keeps every caption's images across sets so the "
        "held-out loss measures generalisation within the domain; a caption with one image cannot be evaluated. **Licences "
        "travel with the outputs:** the Σ weights are Open RAIL++-M, the training photographs here are CC0 — with your own "
        "data, the rights to the images and to what the adapter produces are yours to establish.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can stage and digest-verify three pinned safetensors snapshots, fetch and validate digest-pinned real "
        "photographs, encode prompts and release the encoder, execute bounded LoRA fine-tuning, evaluate the frozen and the "
        "adapted model on identical held-out inputs with a real-photo ceiling, and emit the shown machine-readable artifacts — "
        "without the repository being reachable. It does **not** establish benchmark superiority, production fitness, or image "
        "quality beyond the checks shown.\n\n"
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of host memory while the 19 GB T5-XXL shards are read in Section 5; restart the session, keep a GPU runtime and choose **Run all**.\n'
        '- **`CUDA out of memory` in Section 5, 6 or 7** — the runtime has less than 15 GB of GPU memory or another process holds some of it; restart the session on a T4 or larger and choose **Run all**. The order matters: Section 5 must release the encoder before Section 6 loads the scorer.\n'
        '- **`torch.cuda.is_available()` is False** — the runtime has no GPU; this notebook does not support CPU-only runtimes (the encoder alone needs 19 GB of RAM in float32).\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file and the snapshot. Delete the folder Section 3 prints for that snapshot and run Section 3 again; the three snapshots total about 22 GB.\n'
        '- **Section 4 cannot fetch a photograph** — `inaturalist-open-data.s3.amazonaws.com` is unreachable or a file changed; the message names the photo id and the rule (byte size or SHA-256).\n'
        '- **"Epoch 0 is not the Section 6 frozen model"** — Sections 6 and 7 were run out of order; run Section 6, then Section 7.\n'
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one"** — set `BYOD_PATH` to a zip in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **A `ValueError` from `validate_dataset` or `load_byod_dataset`** — it names the record and the rule: a missing caption, a shorter side below 256 px or above 4096 px, a duplicate id, a caption with fewer than three images, or fewer than four records.\n'
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        'Each of these changes one default and keeps the rest of the path; the frozen numbers of Section 6 are the fixed reference, because Sections 6 and 7 always start from the frozen LoRA snapshot. Raise `EPOCHS` or `LEARNING_RATE` and watch the validation loss for the epoch where it turns; change `GUIDANCE_SCALE` and read the prompt similarity against the reference similarity; set `IMAGES_PER_PROMPT = 4` for a less noisy label accuracy; or bring your own captioned photographs through BYOD and compare the real-photo ceiling with the adapted numbers.\n'
        '\n'
        '## Glossary\n'
        '\n'
        '- **Diffusion Transformer (DiT)** — a transformer that predicts the noise added to a latent at a given timestep; PixArt-Σ has 28 such blocks.\n'
        "- **Latent** — the SDXL VAE's 8× compressed, 4-channel representation of an image; generation and training happen there, and the VAE decodes it to pixels.\n"
        '- **Classifier-free guidance** — running the denoiser with and without the prompt and pushing the prediction away from the unconditional one by `GUIDANCE_SCALE`.\n'
        '- **Denoising loss (MSE)** — the training objective: the squared error between the predicted and the true noise; the held-out value is measured on photographs the model never trains on, with the same seed before and after, so it is a paired comparison.\n'
        '- **LoRA** — low-rank adapters (rank 8 here) added to the attention projections; B starts at zero, so the adapted model equals the pretrained one until training; 4,128,768 parameters, 0.7 % of the transformer.\n'
        '- **Frozen LoRA snapshot** — the untrained LoRA state Section 6 takes once; Sections 6 and 7 restore it before they score or train, so a re-run never builds on an earlier adaptation.\n'
        '- **T5-XXL prompt embeddings** — the (300, 4096) token embeddings the text encoder produces once per prompt; the encoder is released afterwards.\n'
        '- **DPM-Solver++ steps** — the number of denoising steps at generation (20 by default); more steps cost time, not necessarily quality.\n'
        "- **CLIP prompt similarity / label accuracy / reference similarity** — cosine × 100 between an image and its prompt; whether the nearest of the six captions is the right one; cosine × 100 to the mean embedding of the held-out real photographs of that species. All three are a frozen CLIP's opinion.\n"
        '- **Real-photo ceiling** — the same CLIP scores on the real test photographs: the values a perfect generator could reach.\n'
        '- **Reload parity** — the adapter written to disk, reloaded into a fresh pipeline, reproduces the held-out loss and the same image for the same prompt and seed.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: a zip with `captions.csv` beside the image files, read from `BYOD_PATH` or the Colab upload dialog.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        'Before you leave, write three lines in this cell: (1) the frozen and the adapted held-out denoising MSE and whether the paired change was in the direction you predicted; (2) which CLIP number moved most (prompt similarity, label accuracy or reference similarity) and how far it still is from the real-photo ceiling; (3) one thing you would change first with your own captioned photographs, and why the caption text is the contract.\n'
        '\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pixart-sigma-generation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pixart-sigma-generation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/pixart-sigma-generation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS (revision `{MODEL_REVISION}`)\n"
        "- Chen, J., Ge, C., Xie, E., et al. (2024). PixArt-Σ: Weak-to-strong training of diffusion transformer for 4K text-to-image generation. ECCV 2024: https://arxiv.org/abs/2403.04692\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
