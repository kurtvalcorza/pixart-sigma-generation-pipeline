# Release verification

`tutorials/pixart_sigma_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

> **2026-10-04 notebook review fixes (PX-M1..M4, PX-m1..m6; review PR #7).** The tutorial's blob changed from
> `99829b51`. It no longer installs into the kernel: Section 1 builds a uv isolated environment (managed CPython
> 3.12.12) from the hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`, the
> same pins as `pyproject.toml`) and routes every later cell to it, so no restart is part of the procedure; it runs on
> **Linux x86_64 only**. The real photographs are now a leave-one-out reference line (`real_photo_reference`), not a
> ceiling; Sections 6 and 7 reset the adapter to the pretrained base (`reset_adapter`) and Sections 5 and 9 release
> the reloaded pipeline and the scorer, so a BYOD re-run scores a frozen baseline and fits a 15 GB T4 (projected, not
> run); the T5-XXL encoder streams to the GPU (`device_map`) for a 12.7 GiB Colab VM; the guided layer, the BYOD fixes
> and an optional guidance-scale activity were added. The status is **Candidate**: the 2026-09-19 run below needed a
> restart after the install cell (2 passes, not a one-pass `Run all`), and the new blob passed in one pass on a Colab T4 on 2026-10-08 (Recorded executions).

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, a revision that contains the carried modules — never a
  `+working-tree` label — the module SHA-256 overall and per file, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST`, `BASE_MANIFEST` and `SCORER_MANIFEST` equal to the three
  committed snapshot manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook
  byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the uv isolated install from
  the carried hash lock (`--managed-python`, `--require-hashes`, `--only-binary :all:`, the uv wheel's size and SHA-256,
  Linux x86_64 only) and the router whose `google.colab` stubs carry a module spec — and every other cell routed to the
  isolated environment; the Infrastructure titles on the seven setup cells; `NOTEBOOK_SOURCE` with the per-module
  SHA-256 recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before staging), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the components
  revision `2c17b4e8…` and the scorer revision `1a25a446…` are the only other 40-hex commits the documents may name);
- the profile-specific public-API calls (`stage_missing_files` / `stage_missing_base_files` /
  `stage_missing_scorer_files` with `allow_download=True`, the three `verify_*_snapshot` calls,
  `PixArtSigmaPipeline.from_pretrained(weights_dir=..., base_dir=..., use_lora=True)`, `fetch_sample_dataset` from
  the pinned cache path, `load_byod_dataset`, `dataset_manifest`, `write_dataset_csv`, `validate_dataset` with the
  refusal probes, `pipe.encode_prompts` and `pipe.release_text_encoder`, `pipe.evaluate` and `pipe.generate` +
  `score_generations` + `real_photo_reference` (leave-one-out) on the frozen model, `pipe.adapt` with its explicit hyperparameters,
  `pipe.evaluate` after adaptation with the guaranteed assertions (kept-epoch validation loss ≤ frozen; re-scored validation loss matches the history), the new-prompt generation, `pipe.save_artifact`,
  `PixArtSigmaPipeline.from_artifact` + `import_prompt_cache` and the reload-parity assertion, and the provenance
  fields `safetensors_only: True`, `remote_code_executed: False` and the data base URL), the six expected `outputs/`
  paths, the learner-facing statements (three pinned snapshots, the encoder does not fit beside a training graph,
  generation has no ground truth, denoising loss, reference line, not a ceiling, leave-one-out, pretraining overlap,
  run-to-run variability, stratified within each caption, the split assumes independent photographs, not a human
  judgement, Open RAIL++-M, sample-sanity, CC0), the guided-layer markers, the absence of the stale text the review
  removed (no markdown calls the real photographs a ceiling) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct `huggingface_hub` /
  `safetensors` / `urllib` / `diffusers` / `transformers` / `peft` / `T5EncoderModel` / `CLIPModel` use,
  `torch.load(` / `pickle.load` / `Unpickler`, `torch.no_grad(` / `torch.inference_mode(` / `.backward(` /
  `pipe.transformer(` **outside the carried module cells**, `trust_remote_code=True`, `add_adapter(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub transformer, skipped without
torch), `tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`,
`tests/test_worker_colab_stubs.py`, `tests/test_pixart_sigma_generation_colab_review_fixes.py`; temporary
manifests, synthetic images, an injected fetcher, no weights and none of `diffusers`, `transformers` or `peft`).
These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better, ≥ 15 GB) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU (12 GB), sequential cell executor with a `google.colab` shim, pre-staged pins and snapshots | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshots `weights/pixart-sigma-xl-2-512-ms/`, `weights/pixart-sigma-t5-vae/`, `weights/clip-vit-b-32-laion2b/`
   or the data cache `weights/inat-birds/` (the standalone path writes the three manifests itself, stages all 24
   listed files from the Hub — about 22.4 GB — and fetches the 60 pinned photographs, so none of the directories may
   be seeded); the runtime needs about 25 GB of free disk and a GPU of at least 15 GB;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `BYOD_PATH = ''`, `STEPS = 20`, `GUIDANCE_SCALE = 4.5`, `IMAGES_PER_PROMPT = 2`,
   `MAX_GENERATION_PROMPTS = 12`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`, `RUN_ACTIVITY = False`); record
   the number of passes — a run that needed a restart or a second pass is not a one-pass `Run all` and does not promote
   the notebook;
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `diffusers==0.40.0`, `transformers==5.17.0`,
   `peft==0.21.0`, `torchao==0.18.0`, `accelerate==1.15.0`, `tokenizers==0.23.2`, `sentencepiece==0.2.2`, `protobuf==7.36.2`,
   `safetensors==0.8.0`, `huggingface-hub==1.32.0`, `numpy==2.5.3`, `pillow==11.3.0`, installed by Section 1 into the isolated
   environment from the hash lock (the kernel's own packages are not touched, so no restart is needed);
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash lock with no GitHub access, and every later cell routed to it;
   - the three carried module cells execute (defining `PixArtSigmaPipeline`, `build_transformer`,
     `lora_parameter_names`, the three `verify_*_snapshot` and `stage_missing_*` functions, `validate_inputs`,
     `validate_dataset`, `validate_prompts`, `preprocess_image`, `fetch_corpus`, `fetch_sample_dataset`,
     `build_sample_dataset`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `dataset_manifest`,
     `sample_prompts`, `ClipScorer`, `score_generations`, `real_photo_reference`) with no import of the repository
     package;
   - the inline manifests asserted against the module's constants, then the three staging calls reporting 3 + 12 + 9
     entries fetched at the immutable revisions and the three verifications reporting 3 / 12 / 9 verified files;
   - the model cell loading the transformer in float16 with the untrained LoRA attached (603 base tensors,
     610,856,096 parameters; 448 LoRA tensors, 4,128,768 parameters), the VAE in float32, the tokenizer and the
     scheduler config, with `source` "local-snapshot (three manifests verified; safetensors only)";
   - the dataset manifest with 36 / 12 / 12 records, six distinct captions, `outputs/pixart_sigma_generation_sample_captions.csv`
     written, and three refusals (missing caption, 200 px image, duplicate id);
   - `pipe.encode_prompts` reporting the encoder loaded in float16 on `cuda`, 8 prompts encoded (six captions, the
     new prompt, the empty negative prompt), 0 truncations, and `release_text_encoder` returning `True`;
   - the frozen model's held-out denoising MSE on the validation and test records, twelve 512² generations scored by
     CLIP with `outputs/pixart_sigma_generation_frozen_grid.jpg` written and displayed, and the leave-one-out
     real-photo reference on the test photographs;
   - `pipe.adapt` printing epoch 0 as the frozen model, 4,128,768 trainable of 614,984,864 parameters, 144 steps,
     and a four-epoch history with the validation denoising MSE at the kept epoch below the frozen model's;
   - `pipe.evaluate` on the validation and test records with the paired comparison, the assertions that the kept
     epoch's validation loss is no higher than the frozen model's and that the re-scored validation loss matches the
     history within 10⁻⁴ (the test change is printed as an observation), the adapted generations scored with
     `outputs/pixart_sigma_generation_adapted_grid.jpg` written, and `outputs/pixart_sigma_generation_evaluation_report.json`;
   - the new prompt rendered twice and CLIP-scored;
   - `pipe.save_artifact` writing `outputs/pixart_sigma_generation_adapter/{adapter.safetensors,manifest.json}`
     (448 tensors), and `PixArtSigmaPipeline.from_artifact` reloading it into a fresh pipeline that adopts the
     exported prompt cache, with the held-out MSE and a seeded generation matching the adapted pipeline (the cell
     asserts `denoising_mse_diff < 1e-6` and `mean_abs_pixel_diff < 1.0`), then the reloaded pipeline released;
   - `outputs/pixart_sigma_generation_result.json` written with `NOTEBOOK_SOURCE`, the three identities and
     licences, the provenance block (`safetensors_only: true`, `remote_code_executed: false`, the data base URL),
     the runtime versions, the comparison and the reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directories and the data cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `pixart_sigma_generation_colab.ipynb` (`E2E`) | `4862a6e` / `99829b51` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pixart-sigma-generation` v3; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` after, Python 3.12.13, `cuda`) | **PASSED in 2 passes — not a one-pass `Run all`**: pass 1 stopped at the install cell's stale-import guard (`RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings …, numpy …, protobuf …; Restart the runtime`), pass 2 after the restart: 11/11 code cells ok; 117 files, 22443 MB fetched into a clean runtime (the three Hub snapshots + the pinned photographs); comparison held-out denoising MSE (test) frozen 0.105983 → adapted 0.105838 (best epoch 3, validation 0.109093 → 0.108941; the guaranteed validation inequality asserted, the test change printed as an observation), CLIP prompt similarity / label accuracy / reference similarity of 12 generations frozen 26.79 / 0.50 / 65.57 → adapted 28.01 / 0.50 / 68.95 beside the real photographs' 30.25 / 0.917 / 88.66 (that version included each photo in its own reference; the leave-one-out reference similarity derived from the archived per-image values is 57.72, below both generated sets) (sample-sanity, not a quality claim); reload parity denoising_mse_diff: 0.0, mean_abs_pixel_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-pixart-sigma-generation/v3/evidence/` in the workspace |
| `pixart_sigma_generation_colab.ipynb` | generated, pre-commit | 2026-09-19 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti laptop, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pixart_sigma_generation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pixart_sigma_generation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-26 | `231e0f2` / `23463acb` (PR #9 head; final prose pass, regenerated from `db70ff9`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `231e0f239d6b4cbca13e5a81f084eba9ce0ae79e`, git blob SHA-1 `23463acb05028b54d4019f68d2ca2b3bef26cb8b` checked, input unchanged; captured executed-notebook SHA-256 `791db0f2bfad4809fe69eff30c946b5d2cc00eae082b951a921786154620cfdb` equals `run_summary.json` `executed_sha256` | 1058.9 s | **PASSED in one pass, no restart, 0 errors** — 14/14 code cells (`exec.log`); text encoder 13.13 GB loaded, 1.59 GB after release; frozen test denoising MSE 0.105983 → adapted 0.105842 (change −0.000141), lower at all five timesteps; validation 0.109093 → 0.108997 / 0.108953 / 0.10894 / 0.108942, epoch 3 kept; CLIP prompt similarity 26.785 → 28.127, label accuracy 0.5 → 0.5833, reference similarity 65.57 → 69.253 (real photos 30.25 / 0.9167 / 57.722); training 109.9 s; adapter 448 tensors, 16,570,800 bytes; reload parity denoising MSE diff 0.0, mean abs pixel diff 0.0. Drift against the prose's recorded values (recorded, not fixed): training 109.9 s, per-epoch validation loss, test MSE change −0.000141, Dark-eyed Junco 59.107 (see Current status). Evidence in `docs/execution-evidence/2026-10-26-231e0f2/` |
| 2026-10-25 | `17aae24` / `ce81cd57` (PR #9 head; historical wording, notebook commit `9083166`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `17aae245d4dcdea0ca83476c69f2f8d50b48f935`, git blob SHA-1 `ce81cd573cd0d183d3f522064bcb3af7a52ec915` checked, input unchanged; captured executed-notebook SHA-256 `a7ab51a5ebe59bd293043e57d57bfbb4f65631a78986cb0033717f41ecd02d84` equals `run_summary.json` `executed_sha256` | 1123.2 s | **PASSED in one pass, no restart, 0 errors** — 14/14 code cells (`exec.log`); encoder 13.13 GB, 1.59 GB after release; frozen test denoising MSE 0.105983 (validation 0.109093); frozen generations prompt similarity 26.785, label accuracy 0.5, reference similarity 65.57 against the leave-one-out real photos' 30.25 / 0.9167 / 57.722; four epochs (144 steps) in 116.9 s, validation MSE 0.109093 → 0.108997 / 0.108953 / 0.108934 / 0.108932, training loss 0.1151 / 0.1614 / 0.0699 / 0.0847, epoch 4 kept; adapted test MSE 0.105812 (change −0.000171), prompt similarity 26.785 → 28.212, label accuracy 0.5 → 0.5833, reference similarity 65.57 → 68.84; American Goldfinch 80.327 → 76.436 and Dark-eyed Junco 62.115 → 61.349 fell, the other four rose; adapter 448 tensors, 16,570,800 bytes; reload parity 0.0 / 0.0; 3.8 GB → 2.22 GB after release. Drift against the prose's recorded values (recorded, not fixed, per the 2026-10-26 ruling): the per-epoch validation-loss sequence matches none of the three named runs. Evidence in `docs/execution-evidence/2026-10-25-17aae24/` |
| 2026-10-25 | `6ee6e16` (PR #9 head) / `324e664e` (prose kept true across runs, notebook commit `afbb7b8`; downloaded from GitHub at the PR head and blob-verified before the session) | Colab CLI sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`. Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/14–14/14) | Default sample path, no overrides, input unchanged | 1028.9 s session wall | **PASSED — one pass, no restart, 0 errors, 14/14 code cells.** Isolated uv CPython 3.12.12 environment, 68 locked packages, 54 s. Encoder 13.13 GB → 1.59 GB after release. Frozen: validation denoising MSE 0.109093, test 0.105983; CLIP prompt similarity 26.785, label accuracy 0.5 (chance 0.167), reference similarity 65.57 (real photos 30.25 / 0.9167 / 57.722). Fine-tune (4,128,768 LoRA parameters, 144 steps, 111.5 s): validation 0.109093 → 0.108992 → 0.108945 → 0.108927 → 0.108929, training 0.1151 / 0.1614 / 0.0699 / 0.0847, best epoch 3. Adapted: test MSE 0.105832 (−0.000151, lower at every timestep), prompt similarity 28.351, label accuracy 0.5833, reference similarity 68.883; adapter 448 tensors, 16,570,800 bytes; reload parity exact (MSE and pixel difference 0.0), 3.8 GB with the reloaded pipeline. Prose check (confirming run): the training time (about two minutes), the mean reference similarity 68.88 (inside 68.22–69.27) and prompt similarity 28.35 (inside 28.02–28.40), the frozen readings and parity hold, and label accuracy 0.50 → 0.58 matches the prose's account of that reading as unstable. **Prose mismatch** (notebook not edited): (1) the Section 7 answer says the validation loss fell at every epoch and epoch 4 was kept in both Colab runs; this run's validation loss rose at epoch 4 (0.108927 → 0.108929) and epoch 3 was kept; (2) test MSE 0.105983 → 0.105832 (−0.000151, about 0.14 %) is outside the quoted 0.105809–0.105811 (−0.000172 / −0.000174, about 0.16 %); (3) the two falling species are the same (American Goldfinch, Dark-eyed Junco) but their values 80.33 → 79.74 and 62.12 → 60.74 lie outside the quoted 74.82–75.56 and 60.99–61.67; (4) the session wall 1028.9 s (about 17 minutes) is shorter than the stated "about 19 minutes" (1,140.3 / 1,118.9 s). Not exercised: the optional activity, BYOD on a hosted runtime. Evidence `docs/execution-evidence/2026-10-25-6ee6e16/pixart_sigma_generation_colab_6ee6e16_colab-cli-t4_default.ipynb` (sha256 `e43664568dce91ef3c6fb2ac95f286db1ed7dcbfcabb2df09cbf978bc181457a`; table under [Current status](#current-status)). Status stays **Candidate** |
| 2026-10-25 | `167ade7` (PR #9 head) / `adef085a` (tier C review fixes, generated at `b1f43b9`) | Colab CLI sequential execution on a fresh Colab Tesla T4 VM (session `suite-pixart-167ade7-cece`; isolated environment: Python 3.12.12, 68 locked packages, setup 44 s; kernel Python 3.13.16) — not a browser Run all | Default sample path, every form field at its default | 1118.9 s | **PASSED** — 14/14 code cells in order in `exec.log`, one pass, no restart, 0 error outputs, input notebook unchanged. T5 encoder 8 prompts encoded and cached (13.13 GB GPU with the encoder); frozen denoising MSE validation 0.109093 / test 0.105983 (t = 100 0.358295 … t = 900 0.000189); frozen generations prompt similarity 26.785, label accuracy 0.5, reference similarity 65.57 (real photos 30.25 / 0.9167 / 57.722); LoRA 4,128,768 of 614,984,864 parameters, 144 steps, 112.6 s, `best_epoch` 4; adapted test MSE 0.105809 (−0.000174, lower at all five timesteps), prompt similarity 28.022, label accuracy 0.5, reference similarity 68.224; adapter 448 tensors, 16,570,800 bytes; reload parity denoising-MSE and pixel differences 0.0. The guided answers quote the 2026-10-08 run of `a61c09b` (blob `4cf243c1`); this run printed different adapted numbers: validation MSE 0.109093 → 0.108994 → 0.108949 → 0.10893 → 0.108926 (quoted 0.108991 → 0.108946 → 0.108929 → 0.108927), training 112.6 s (quoted 119.9 s), test MSE → 0.105809, change −0.000174 (quoted 0.105811, −0.000172), reference similarity 65.57 → 68.224 (quoted 69.27), prompt similarity 26.785 → 28.022 (quoted 28.40), label accuracy 0.5 → 0.5 (quoted 0.50 → 0.58), American Goldfinch 80.327 → 75.558 (quoted 74.82), Dark-eyed Junco 62.115 → 60.994 (quoted 61.67). The frozen numbers, the real-photo reference, the training losses and the per-timestep losses match. The notebook is not edited by this record. Evidence `docs/execution-evidence/2026-10-25-167ade7/`: SHA-256 captured → committed: `pixart_sigma_generation_colab_167ade7_colab-cli-t4_default.ipynb` `a68b1363ed7fe61331944056f8957531bf34fbaec24d821bac324617f0c1290d` → unchanged (no local path); `exec.log` `beaf9755d51770215afa2a18bf07872201d19d8f4272badf1164e4a53d6832bf` → `4c5ee9a01e66e552346ff130fdb38435b2df0a690bc6168b4f65d139f8636745` (2 local-path replacements); `run_summary.json` `8fbc617465a6a473ea9086f18d23ed508514c37b0bbf31af31b85310f1b44136` → unchanged (no local path) |
| 2026-10-08 | `a61c09b` / `4cf243c1` | Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 VM (isolated environment: Python 3.12.12, 68 locked packages, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; kernel Python 3.13.15) | Default sample path; every code cell executed in order in one kernel, not a browser `Run all`; empty Hugging Face cache, no repository checkout; blob SHA-1 verified before execution; BYOD, an invalid BYOD input and the Section 10 activity not exercised | 1140.3 s | **PASSED in one pass** — 14/14 code cells, 0 error outputs, no restart; isolated environment built in 55 s; 24 snapshot files staged and verified (3 + 12 + 9); T5 encoder 13.13 GB → 1.59 GB after release; frozen held-out denoising MSE validation 0.109093 / test 0.105983 (`adapted: False`); adapted (best epoch 4, validation 0.108927) test 0.105811 (−0.000172, lower at all five timesteps; printed, not asserted); CLIP prompt similarity / label accuracy / reference similarity of 12 generations frozen 26.785 / 0.5 / 65.57 → adapted 28.398 / 0.5833 / 69.274, leave-one-out real-photo reference 30.25 / 0.9167 / 57.722 (0 photos without a reference); reload parity 0.0 / 0.0, reloaded pipeline released (3.8 → 2.22 GB); executed notebook SHA-256 `9a4ee5bd…`, `run_summary.json` `2fa0b5d6…`, `exec.log` `3b4c4a0b…` in `docs/execution-evidence/2026-10-08-a61c09b/` |
| 2026-09-19 | `4862a6e` / `99829b51` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pixart-sigma-generation` v3; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the three snapshots staged and digest-verified by the notebook, the pinned photographs fetched by the notebook | 1582.8 s | **PASSED in 2 passes — not a one-pass `Run all`**: pass 1 stopped at the install cell's stale-import guard (`RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings …, numpy …, protobuf …; Restart the runtime`), pass 2 after the restart: 11/11 code cells ok; 117 files, 22443 MB fetched into a clean runtime (the three Hub snapshots + the pinned photographs); comparison held-out denoising MSE (test) frozen 0.105983 → adapted 0.105838 (best epoch 3, validation 0.109093 → 0.108941; the guaranteed validation inequality asserted, the test change printed as an observation), CLIP prompt similarity / label accuracy / reference similarity of 12 generations frozen 26.79 / 0.50 / 65.57 → adapted 28.01 / 0.50 / 68.95 beside the real photographs' 30.25 / 0.917 / 88.66 (that version included each photo in its own reference; the leave-one-out reference similarity derived from the archived per-image values is 57.72, below both generated sets) (sample-sanity, not a quality claim); reload parity denoising_mse_diff: 0.0, mean_abs_pixel_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-pixart-sigma-generation/v3/evidence/` in the workspace |
| 2026-09-19 | generated, pre-commit | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti laptop 12 GB, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`) | Default sample path (stage → verify the three snapshots → load transformer + LoRA / VAE → pinned-photo fetch from the cache → validate → refusal probes → encode prompts + release encoder → frozen evaluation, generations and the real-photo reference (self-inclusive at the time) → LoRA adapt → paired evaluation → new prompt → export → reload); the three snapshots and the 60 photographs were pre-staged, so every staging call fetched 0 entries and the run verified 24 files by digest | 1994.8 s | **PASSED** — 11/11 code cells; probes refused; frozen test denoising MSE 0.105939 → adapted 0.105800 (best epoch 3; printed, not asserted); encoder peak on the 12 GB GPU; adapter 448 tensors; reload parity identical. Pre-flight; hosted clean-runtime run still required |

## Current status

**Candidate (2026-10-26, final prose pass).** The current notebook blob of `tutorials/pixart_sigma_generation_colab.ipynb` is `23463acb05028b54d4019f68d2ca2b3bef26cb8b` (commit `3f80e08`), regenerated from `db70ff9` under Kurt's 2026-10-26 ruling to fix only the prose-rule items flagged on the confirming run of `ce81cd57`: the Prerequisites clause saying this notebook has not yet been run on Colab is removed; the Section 6 'Look for' line no longer names a denoising MSE around 0.1..0.3; the Section 7 worked answer no longer quotes unlabelled per-timestep losses (about 0.36 and 0.0002); and the Interpretation no longer quotes the unlabelled 'recorded run of the previous version'. Markdown only; no code or output changed, and the drift recorded for that run below is kept, not fixed. This exact blob (PR #9 head `231e0f2`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-26, 14/14 code cells in one pass, no restart, 0 errors, 1058.9 s wall; frozen → adapted test denoising MSE 0.105983 → 0.105842, reference similarity 65.57 → 69.253, label accuracy 0.5 → 0.5833, reload parity 0.0; evidence in `docs/execution-evidence/2026-10-26-231e0f2/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The confirming run of `ce81cd57` and the earlier records below stay as history.

**Drift recorded, not fixed (2026-10-26, final prose pass; Kurt's ruling: drift is recorded here, never fixed).** The notebook prose of blob `23463acb` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Section 7 and Interpretation: four epochs 'took between 111.5 s and 119.9 s on a T4 in the recorded runs' / 'trained for 111.5 s to 119.9 s (the recorded Colab runs)'; this run printed 109.9 s.
- Section 7: validation loss per epoch in the three named Colab runs 0.108991–0.108994 → 0.108945–0.108949 → 0.108927–0.108930 → 0.108926–0.108929; this run printed 0.108997 → 0.108953 → 0.10894 → 0.108942 (it rose at epoch 4 and kept epoch 3, as the third named run did).
- Section 8: test denoising MSE 0.105983 → 0.105811, 0.105809 and 0.105832 (−0.000172, −0.000174 and −0.000151, about 0.14 % to 0.16 %) in the named runs; this run printed 0.105842 (−0.000141, about 0.13 %).
- Section 8: Dark-eyed Junco reference similarity 62.12 → 61.67, 60.99 and 60.74 in the named runs; this run printed 62.115 → 59.107 (American Goldfinch 80.327 → 76.035, inside the named values).

Evidence files for the 2026-10-26 default run of blob `23463acb`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-26-231e0f2/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pixart_sigma_generation_colab_231e0f2_colab-cli-t4_default.ipynb` | `791db0f2bfad4809fe69eff30c946b5d2cc00eae082b951a921786154620cfdb` | `791db0f2bfad4809fe69eff30c946b5d2cc00eae082b951a921786154620cfdb` (unchanged, no local path) |
| `exec.log` | `d49b8498d42a0d229b8327ff32bb43884beb432da23704eabdf3974a8f0e3425` | `d95ea7baff9e06b7c7e1f24de946d85db393a7960e1c5c5f6c63b45c66d927e8` (2 local-path replacements) |
| `run_summary.json` | `094b063ee94c2cd38d30615950704257ce55278de2f91f8a21cfafb91e918edc` | `094b063ee94c2cd38d30615950704257ce55278de2f91f8a21cfafb91e918edc` (unchanged, no local path) |

**Current status (2026-10-26, historical wording): Candidate, hosted run recorded.** The current notebook `tutorials/pixart_sigma_generation_colab.ipynb` is blob `ce81cd573cd0d183d3f522064bcb3af7a52ec915` (notebook commit `9083166`, regenerated from `38b2879`) under Kurt's 2026-10-26 ruling: every run-derived number in the prose names the hosted runs that printed it (Kaggle T4 of `99829b51` 2026-09-19; Colab T4 of `4cf243c1` 2026-10-08, `adef085a` and `324e664e` 2026-10-25), with no expectation ranges, timing promises or run-status sentences; it resolves the four prose mismatches recorded for the `324e664e` run (validation loss rose at epoch 4 and epoch 3 was kept; test MSE 0.105832; Goldfinch 79.74 and Junco 60.74; 1,028.9 s). Markdown only. This exact blob (PR #9 head `17aae24`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-25, 14/14 code cells in one pass, no restart, 0 errors, 1123.2 s wall; test MSE 0.105983 → 0.105812, reference similarity 65.57 → 68.84, label accuracy 0.5 → 0.5833, reload parity 0.0; evidence in `docs/execution-evidence/2026-10-25-17aae24/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The records below stay as history.

**Drift recorded, not fixed (2026-10-26, Kurt's 2026-10-26 "historical wording, then stop" ruling).** The notebook prose of blob `ce81cd57` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Section 7 worked answer: the validation loss went 0.109093 → 0.108997 → 0.108953 → 0.108934 → 0.108932, a sequence none of the three named Colab runs printed (it fell at every epoch and epoch 4 was kept, as in the first two).

Evidence files for the 2026-10-25 default run of blob `ce81cd57`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-17aae24/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pixart_sigma_generation_colab_17aae24_colab-cli-t4_default.ipynb` | `a7ab51a5ebe59bd293043e57d57bfbb4f65631a78986cb0033717f41ecd02d84` | `a7ab51a5ebe59bd293043e57d57bfbb4f65631a78986cb0033717f41ecd02d84` (unchanged, no local path) |
| `exec.log` | `8db2c3ede10cfdedde96fdcae025cc32deed36c4129979d030941ad7b4b86317` | `3c29ab0c429a03b3624c7eacf50f00bf234a3a388e5682ad2d71614932c5e038` (2 local-path replacements) |
| `run_summary.json` | `7bae2d526548c42b34428a60c27b790d64dff83c146237dc9b006839f8f7d9e2` | `7bae2d526548c42b34428a60c27b790d64dff83c146237dc9b006839f8f7d9e2` (unchanged, no local path) |

**Previous status (2026-10-25, prose kept true across runs): Candidate, hosted run recorded.** The previous notebook `tutorials/pixart_sigma_generation_colab.ipynb` was blob `324e664ea44ff974885101dbbc77ec12f09f6ed1` (PR #9 head `6ee6e16`; generator commits `858d8da` and `49bd10c`, notebook commit `afbb7b8`; markdown only). That exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 14/14 code cells, 1028.9 s; test MSE 0.105983 → 0.105832, reference similarity 65.57 → 68.88, label accuracy 0.50 → 0.58, reload parity exact; first row under "Recorded executions"). Part of the revised prose does not hold for this run: the validation loss rose at the last epoch and epoch 3 was kept, and the test MSE, the two falling species' values and the session time fall outside the quoted values (see the table row). This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser run of the interactive journeys). The 2026-10-25 run of the previous blob `adef085a` and the runs of earlier blobs are history (REL14).

Evidence files, redacted byte-wise (the local run directory became `<local-run-dir>`, the Colab CLI install path `<colab-cli-tool-dir>`; no JSON re-serialised):

| File (under `docs/execution-evidence/2026-10-25-6ee6e16/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pixart_sigma_generation_colab_6ee6e16_colab-cli-t4_default.ipynb` | `e43664568dce91ef3c6fb2ac95f286db1ed7dcbfcabb2df09cbf978bc181457a` | `e43664568dce91ef3c6fb2ac95f286db1ed7dcbfcabb2df09cbf978bc181457a` (unchanged, no local path) |
| `exec.log` | `c7eae2c655c3156c8dde239c194fb52f068522c91c293660260f068953552f5f` | `aaad5109c96677bfc7ff27085ab45c5466ae5a22e1ea32d46ca811e8e3ea21fb` (2 local-path replacements) |
| `run_summary.json` | `e903efb88ae734f9fe8f41fbcf2c9e470e8256b8718a08d52c97244fa30a788a` | `e903efb88ae734f9fe8f41fbcf2c9e470e8256b8718a08d52c97244fa30a788a` (unchanged, no local path) |

**Previous status (2026-10-25): Candidate.** The previous notebook `tutorials/pixart_sigma_generation_colab.ipynb` was blob `adef085a2ccde40477eb645e8e0fac57c8bb7c78` (PR #9 head `167ade7`, regenerated from generator commit `b1f43b9` for the tier C review findings of row 66). It passed a Colab CLI sequential execution on a fresh Colab Tesla T4 at `167ade7` on 2026-10-25 (14/14 code cells, one pass, no restart, 0 errors, 1118.9 s; first row under "Recorded executions"). That run covers the **default path** only. Its adapted numbers differ from the `a61c09b` run the guided answers quote (reference similarity 68.224 against 69.27, label accuracy 0.5 against 0.58; details in its row). Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). Every run of an earlier blob is history for that blob (REL14).

**Previous blob — Candidate.** The 2026-10-04 review fixes produced a new notebook blob: the review-fix blob `4cf243c1` (commit `a61c09b`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI 0.7.4 sequential execution, fresh VM — not a browser Run all; 14/14 code cells, 1140.3 s; evidence in `docs/execution-evidence/2026-10-08-a61c09b/`), recorded above. The
earlier `E2E` blob `99829b51` (committed at `4862a6e`) completed on a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11,
1582.8 s, 117 files / 22,443 MB fetched and digest-verified inside the notebook) only after a restart following the install
cell — 2 passes, not a one-pass `Run all` (REL11) — so it is not promotion evidence under NOTEBOOK_SPEC 2.2. The local
pre-flight rows above are history. What remains before release: a browser `Run all` of the new blob (or the maintainer's acceptance of the Colab CLI
record above), a BYOD run after the default path that prints `adapted: False` in
Section 6 and reaches reload parity, one invalid BYOD input, and the Section 10 activity, recorded here with the pass
count; promotion is the maintainer's decision.
