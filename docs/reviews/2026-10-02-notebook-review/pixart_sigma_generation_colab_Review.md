# PixArt-Σ 512-MS Text-to-Image LoRA E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/pixart-sigma-generation-pipeline`  
**Notebook:** `tutorials/pixart_sigma_generation_colab.ipynb`  
**Reviewed commit:** `00f3e04af9f7ace25057a6852408fc9f53f12226` (`main`, confirmed with `gh api repos/kurtvalcorza/pixart-sigma-generation-pipeline/commits/main`)  
**Notebook Git blob:** `99829b513f48613696b7736acd2bfe8962581560`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-19 (commit `4862a6e`); the notebook has not changed since.  
**Finding prefix:** `PX`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The engineering is careful. The notebook carries its three modules verbatim (parity-checked), stages three Hub snapshots (22.4 GB) at immutable revisions and re-hashes every file, fetches 60 digest-pinned CC0 photographs, encodes every prompt once and releases the 9.5 GB T5-XXL encoder before training, measures a paired held-out denoising loss with identical latents, noise and timesteps before and after a bounded LoRA, asserts only what the procedure guarantees, and reloads the exported adapter into a fresh pipeline with exact parity. The prose is honest about what a denoising loss and a CLIP score are not.

| Measure | This review (CPU, no weights) | Kaggle T4 record (blob `99829b51`) |
|---|---|---|
| Code cells completed | none of the notebook's model cells; carried modules exercised through the offline suite (34 passed, 1 module skipped: no `diffusers`) | 11/11 on pass 2; pass 1 stopped at the install guard |
| Real-photo reference similarity ("ceiling") | **57.72** with each photo excluded from its own reference (derived from the record) | printed **88.66** (each photo included in its own reference) |
| Frozen → adapted generations, reference similarity | not run | 65.57 → 68.95 |
| Label accuracy frozen → adapted / real photos | not run | 0.50 → 0.50 / 0.917 (prose expects 1.0) |
| BYOD smallest single-caption dataset accepted | **6 images** (stated minimum of 4 refused) | not run |
| LoRA state when BYOD is re-run as instructed | **not reset** (source): Section 6's "frozen" baseline is the sample-adapted model | not run |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (PX-M1).** The recorded run stopped at the install cell's stale-module guard and passed only after a restart. `docs/release-verification.md` calls the restart "expected", and the repository marks the blob `Release-grade` on that run.
2. **The "real-photo ceiling" is not a ceiling (PX-M2).** Its reference similarity compares each real photo with a mean that contains that same photo. With the photo excluded, the real photos score about 57.7, below both the frozen (65.57) and the adapted (68.95) generations. The notebook calls these numbers "the ceiling these numbers could reach".
3. **BYOD, followed as written, compares against a model that is not frozen (PX-M3).** `adapt` never resets the LoRA. The instruction to re-run from Section 4 therefore scores the sample-adapted model as the "frozen" baseline, then continues training from it while labelling epoch 0 "LoRA at initialisation: B = 0". The same rerun also reloads the T5 encoder while the second pipeline from Section 9 is still resident.
4. **Guided layer largely absent (PX-M4).** The notebook is declared `GUIDED`, but it has no audience statement, how-to-use section, roadmap, glossary, prediction prompt, checkpoint, troubleshooting or conclusion template. 2,009 lines of carried modules sit in three unlabelled, uncollapsed cells.

The defect pattern matches the sibling generator notebooks built from the same template (flux FS-M1/M2/M4, kandinsky-generation KGN-M1). Unlike those siblings, this notebook's real-photo **CLIP prompt similarity** (30.25) is above both generated sets (26.79 / 28.01), so only the reference-similarity ceiling is inverted here. No loss-trend error was found: the validation loss does fall monotonically in the record (0.109093 → 0.108941), as Section 7 says it will.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. The Prerequisites (cell 1, "Knowledge") assume the reader knows what a latent diffusion model does at inference, what classifier-free guidance is, what a LoRA adapter changes and why a training loss is not a quality score |
| Supported runtime | "a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU of at least 15 GB and Python 3.12)"; about 25 GB of disk; "CPU-only runtimes are not supported" |
| Promised outcomes | Pinned install; carried package; three snapshots staged and digest-verified; 60 digest-pinned CC0 photos validated and split 36 / 12 / 12 with three refusals; prompts encoded and the encoder released; frozen held-out denoising loss plus CLIP-scored generations "against the real-photo ceiling"; bounded LoRA (rank 8, 4,128,768 parameters); paired comparison on identical inputs; a new prompt; safetensors adapter export and fresh reload with parity; BYOD zip through the same contract |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `0f99b7b` (a PR-branch commit, reachable on GitHub; its three modules are byte-identical to `main`'s) |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code). Cells 5, 7 and 9 carry `pipeline.py` (980 lines), `metrics.py` (137 lines) and `samples.py` (892 lines). Also read: `pipeline.py` (`adapt`, `encode_prompts`, `release_text_encoder`, `generate`, `evaluate`), `metrics.py` (`score_generations`, `real_photo_baseline`), `samples.py` (`split_dataset`, `load_byod_dataset`, `dataset_manifest`), `tools/build_notebook.py`, `tools/notebook_template.py`, `tutorials/README.md`, `docs/release-verification.md` and `STATUS.md`. The repository has no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md`, row 2026-09-19, and the workspace archive it cites (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-pixart-sigma-generation/v3/evidence/`: `run_summary.json`, `executed.ipynb`, `executed-pass1.ipynb`, `outputs/…evaluation_report.json`). The run was on a Kaggle Tesla T4 with **the reviewed blob** (SHA-1 verified before execution) and a clean Hugging Face cache. Pass 1 raised `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings … numpy: loaded=2.0.2, installed=2.5.3; protobuf … Restart the runtime`. Pass 2 completed 11/11.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), the shared `eo-notebook-test` conda env (Python 3.12, torch 2.13.0+cpu, numpy 2.5.3, no `diffusers` / `transformers` / `peft`; nothing installed). No PixArt, T5, VAE or CLIP weights were downloaded and no image was generated: the model is far too large for this memory-tight CPU host.
  - **Probes (a few minutes in total):**
    - P1: the offline suite with `PYTHONPATH=src`: exit 0, 34 passed, `test_adaptation.py` skipped (no `diffusers`).
    - P2: `tools/build_notebook.py --check` (OK, byte-identical) and `tools/validate_release_assets.py` (PASS).
    - P3: the carried BYOD loader and splitter on nine synthetic zips.
    - P4: the real-photo reference similarity with and without self-inclusion, both from the recorded per-image values and with a stand-in scorer through the real `real_photo_baseline`.
    - P5: string and AST checks on the notebook and modules.
    - P6: identity and passes from the archived run summary.
- **Not verified:** any notebook cell executed end to end here; any Colab run (the stated runtime has no record); the real upload dialog; BYOD beyond the load/split stage; the BYOD rerun and the optional experiments on real weights; GPU memory on a rerun (PX-M3 memory is a projection).
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** the supply-chain handling is strong (immutable revisions, per-file SHA-256 re-hash before load, safetensors only, no remote code, a manifest-checked adapter with digest verification before deserialising), and the encoder-release design is sound. Defects:
  - the install pattern forces a restart (PX-M1);
  - `adapt` does not reset the LoRA, so every documented rerun (BYOD and the optional experiments) starts from adapted weights that are labelled frozen (PX-M3, PX-m6);
  - the BYOD rerun reloads the encoder beside the second pipeline (PX-M3).
- **Scientific validity:** these parts are sound:
  - the paired held-out denoising loss (same latents, noise and timesteps);
  - epoch selection on validation only, with the test split untouched;
  - one photo per observer per species;
  - the honest "printed, not asserted" test change.

  The weak point is the reference the generations are read against. The "ceiling" is inflated by self-inclusion (PX-M2). Independence, grouping and pretraining overlap are never stated (PX-m2, PX-m3).
- **Promise fulfilment:** the default-path promises are met on the documented run, except one-pass `Run all` (PX-M1) and the "real-photo ceiling" (PX-M2). BYOD's stated minimum is wrong (PX-m1), and the documented BYOD route does not deliver the promised "frozen baseline" (PX-M3). The learner is asked to compare grids "by eye", but the generated images are only written to files (PX-m5).
- **Learner experience:** the prose is precise and honest, with "Look for" notes in Sections 1 and 4–8 and a careful limits section. But there is no guided layer (PX-M4), two expected-output notes contradict the run, and the interpretation states as fact results that the run only partly shows (PX-m4).
- **Spec conformance:** see the table at the end of Section 5. The unresolved applicable MUSTs are:
  - RUN1, RUN10, ENV6, REL2, REL11 (PX-M1)
  - EVAL3 (PX-M2)
  - DAT14, REL12 (PX-M3)
  - DAT12, DAT19 (PX-m1)
  - SPL3 (PX-m2)
  - DAT9 (PX-m3)
  - ENV8 (PX-m4)

## 3. Promise and objective tracing

| Claim | Implementation | Observable result (record) | Verdict |
|---|---|---|---|
| Run all in a fresh GPU runtime, no restart (opening, §5 of spec 2.0) | cell 3 `pip install` into the kernel + stale-module guard | pass 1 `RuntimeError … Restart the runtime`; pass 2 11/11 | **Not met** (PX-M1) |
| Three snapshots staged and digest-verified | cell 11 `stage_missing_*` + `verify_*` | 3 + 12 + 9 files fetched and verified | Met (documented) |
| 60 photos, 36 / 12 / 12, three refusals | cell 13 | 36 / 12 / 12, six captions, three refusals | Met (documented) |
| Encoder released before training | cell 15 | 13.13 GB → 1.59 GB allocated; `released: True` | Met (documented) |
| Frozen model scored "against the real-photo ceiling … the ceiling these numbers could reach" | cell 17 `real_photo_baseline` | 30.25 / 0.917 / **88.66**; leave-one-out reference ≈ **57.72** < frozen 65.57 | **Misleading** (PX-M2) |
| "label accuracy well below the real photographs' 1.0" | cell 16 prose | real photos 0.917 | Inaccurate (PX-m4) |
| Validation denoising MSE falls from epoch 0 | cell 19 | 0.109093 → 0.108999 → 0.108956 → 0.108941 → 0.108941, best epoch 3 | Met |
| Paired comparison; "look for higher reference similarity and label accuracy" | cell 21 | reference 65.57 → 68.95 (2 of 6 captions fell); label accuracy 0.50 → 0.50 | Partly; prose pre-states it (PX-m4) |
| A second grid "to compare with the first by eye" | cell 21 `grid(...)` saves to file | never displayed | **Not met in-notebook** (PX-m5) |
| Export + fresh reload with parity | cell 23 | 448 tensors, 16.6 MB; parity 0.0 / 0.0 | Met (documented) |
| BYOD: "at least four images, and at least one caption with three or more images" | `split_dataset` | 4 images refused; 6 needed for one caption (P3) | **Not met as stated** (PX-m1) |
| BYOD "flows through the same contract — … frozen baseline, adaptation …" after a re-run from Section 4 | `adapt` (no LoRA reset), cells 15–23 | Section 6 scores the sample-adapted model (source) | **Not met** (PX-M3) |

| Learning objective | Learner activity | Evidence it is exercised |
|---|---|---|
| Install the pinned runtime; stage and verify snapshots | run cells 3, 11 | printed identity, fetched/verified counts |
| Fetch, validate and split a small dataset | run cell 13 | counts, digest, refusal probes |
| Encode prompts and release the encoder | run cell 15 | memory before and after |
| Read a held-out denoising loss and CLIP-scored generations against a real-photo ceiling | read cell 17 output | the "ceiling" is inflated (PX-M2); no checkpoint question (PX-M4) |
| Run a bounded LoRA with explicit hyperparameters | run cell 19 | history printed |
| Compare adapted and frozen on identical held-out inputs | read cell 21 output | the table is printed, but there is no prediction or interpretation prompt (PX-M4) |
| Render a new prompt; export and reload an adapter | run cell 23 | CLIP score, parity; images not shown (PX-m5) |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | The notebook gives a clear account of why the encoder is released and what each number is and is not, with "Look for" notes. But: no guided layer (PX-M4); the "ceiling" framing misleads (PX-M2); the "1.0" expectation and the stated conclusions disagree with the run (PX-m4); images are never shown (PX-m5); the data contract renders as `{{id, image, caption}}` (PX-m4). |
| Clean default | Documented execution evidence (Kaggle T4, this blob, 2026-09-19) | Completed only after a manual restart (PX-M1); otherwise 11/11, reload parity 0.0 / 0.0, 1582.8 s. No Colab record. Not re-run here. |
| Active learning | Source inspection only | The optional experiments give no rerun scope. Raising `EPOCHS` and rerunning Section 7 continues from the adapted LoRA with epoch 0 labelled frozen. Changing `GUIDANCE_SCALE` and rerunning Section 6 scores the adapted model as frozen (PX-m6). Not executed. |
| Reuse and recovery | Direct CPU execution of the carried loader/splitter (synthetic zips) + source inspection | The stated minimum is refused. Missing `captions.csv` and missing-column refusals are actionable. A non-image file raises a bare `UnidentifiedImageError` that names no file, and a non-zip raises `BadZipFile` (PX-m1). The documented "re-run from Section 4" route starts from the sample-adapted LoRA and reloads the encoder beside the resident `reloaded` pipeline (PX-M3, inferred). Downstream BYOD stages, the real upload dialog and a hosted BYOD run were not verified. |

## 5. Findings

### Major

#### PX-M1 — `Run all` needs a manual restart after the install cell, and the blob is marked `Release-grade` on that run

- **Cell/section:** Section 1 (cell 3); `docs/release-verification.md` step 4 and Current status; `STATUS.md`; `tutorials/README.md`.
- **Observed issue:** cell 3 `pip install`s 15 pins into the running kernel, then raises `RuntimeError(… Restart the runtime, then rerun from the top.)` when a loaded distribution changed. On the recorded Kaggle T4 run of this blob, pass 1 stopped there (cuda-bindings 12.9.4 → 13.4.2, numpy 2.0.2 → 2.5.3, protobuf 5.29.5 → 7.36.2); pass 2, after a restart, completed 11/11. The release procedure calls the restart "expected", and the blob is `Release-grade` on that run.
- **Consequence:** a learner choosing **Run all** in a fresh runtime hits an error in the first code cell. Spec 2.2 forbids exactly this, and the release status rests on a restart-dependent run.
- **Evidence:** documented execution (`run_summary.json` passes 1–2; P6) + source inspection (P5: `pip_install_into_kernel: true`, `uses_uv: false`).
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern. A carrier cell bootstraps uv and creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`. It installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:` and runs the workload in that env, so the kernel's preloaded torch and NumPy are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `origin/main`. Make the change in `tools/build_notebook.py` / `tools/notebook_template.py` and regenerate. Until a one-pass run is recorded, return the status to `Candidate` in `STATUS.md`, `README.md`, `tutorials/README.md` and `docs/release-verification.md`, and delete "an interpreter restart after the install is expected".
- **Acceptance check:** a fresh hosted GPU runtime (Colab preferred) executes the regenerated blob top to bottom in one pass with no error output in any cell and no restart. The release record names that blob and says "no restart".
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11 (MUST).

#### PX-M2 — The "real-photo ceiling" is not a ceiling: each photo is included in its own reference mean

- **Cell/section:** Section 6 prose (cell 16) and code (cell 17); Section 8 table (cell 21); objectives in cell 0; Interpretation (cell 24); `metrics.py` `real_photo_baseline`; `tutorials/README.md` ("the *real-photo ceiling*").
- **Observed issue:** `real_photo_baseline` calls `score_generations(scorer, generated, references=records)` with the **same** records as both images and references. Each test photo's reference similarity is therefore its cosine with a mean that contains itself. Its docstring says "references = the other records". Its own `note` field says "references include each photo itself", but that note is only written to the JSON and is never printed. With two test photos per species, self-inclusion gives 82.96–95.78 per caption (mean **88.66**). Excluding the photo gives 37.65–83.47 (mean **57.72**). That is below both the frozen (65.57) and the adapted (68.95) generations, so on this metric the generations "exceed the ceiling". The notebook nevertheless calls these numbers "the ceiling these numbers could reach" and lists "read … CLIP-scored generations against a real-photo ceiling" as a learning objective. The other two ceiling measures do sit above the generations (prompt similarity 30.25 vs 26.79 / 28.01; label accuracy 0.917 vs 0.50).
- **Consequence:** the learner reads a 20-point gap (68.95 vs 88.66) as headroom left by the adapter. On a like-for-like reference that gap does not exist, so the central comparison of Sections 6 and 8 teaches the wrong conclusion. In BYOD with one test photo per caption, the self-inclusive value is 100.0 by construction.
- **Evidence:** source inspection (`metrics.py`); documented execution (per-image values in the archived evaluation report); direct execution (P4: leave-one-out derived as `2v² − 1` for two unit vectors, mean 57.72; a stand-in scorer through the real `real_photo_baseline` reproduces the self-inclusive value 81.842 versus leave-one-out 33.961).
- **Recommended correction:** in `real_photo_baseline`, score each photo against the mean of the *other* same-caption references (leave-one-out). Report `None`, with a printed reason, when a caption has only one reference. Rename the row to "real photographs (leave-one-out reference)", and drop the word "ceiling" unless the numbers support it. Print the method next to the value. Fix the docstring. Add a unit test with two fixed embeddings that asserts the leave-one-out value. Regenerate.
- **Acceptance check:** for a caption with references {a, b}, the real-photo reference similarity equals cos(a, b) × 100 (unit test). The notebook prints how the real-photo reference is computed. No learner-facing text calls a number a "ceiling" when a generated score exceeds it in the recorded run.
- **Spec:** EVAL3 (MUST); EVAL10, GDL8 (SHOULD).

#### PX-M3 — The documented BYOD rerun scores the sample-adapted model as the "frozen" baseline, and reloads the encoder beside the resident second pipeline

- **Cell/section:** opening cell ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); Sections 5–9 (cells 15–23); `pipeline.py` `PixArtSigmaPipeline.adapt`.
- **Observed issue:**
  - **Adapted weights carried over.** `adapt` snapshots the current LoRA tensors as its restore point and trains from them. Nothing resets the LoRA to its initialisation (P5: `adapt_resets_lora: false`).
    - After the default run, `pipe` holds the sample-adapted adapter (best epoch 3). Re-running from Section 4 with BYOD therefore makes Section 6's "frozen model" loss and generations those of the sample-adapted model; `generate` even reports `adapted: True` there.
    - Section 7 then continues training from that adapter while labelling epoch 0 "frozen model (LoRA at initialisation: B = 0)". The Section 8 "frozen vs adapted" table compares two adapted models, and the exported adapter is trained on sample + BYOD data.
  - **Memory on the rerun (projected).** Section 5 reloads the T5 encoder (≈11.5 GB allocated in the record: 13.13 GB with the encoder vs 1.59 GB without). The Section 9 `reloaded` pipeline (≈1.6 GB) and the CLIP scorer (≈0.6 GB) are still resident beside `pipe` (2.22 GB after adaptation). That projects to ≈15.9 GB allocated on a T4 with 16.1 GB (15,360 MiB) of device memory.
- **Consequence:** BYOD is the notebook's transfer path, and the promised "frozen baseline" for the learner's own data is silently not frozen. Any conclusion about what the adapter did to their data is invalid, and the run may also fail on memory.
- **Evidence:** source inspection (`adapt`, `generate`, cells 13–23, opening BYOD instruction); documented execution for the memory figures (cells 15 and 19 outputs); the memory total is **inferred, not run**.
- **Recommended correction:**
  - Give BYOD a correct rerun route. Either have `adapt` refuse an already adapted pipeline with an actionable message, or add a `pipe.reset_adapter()` that restores the LoRA initialisation and call it at the top of Section 6 when `pipe.adapter is not None`.
  - Release `reloaded` (and empty the CUDA cache) before Section 5 re-encodes, or tell the learner to restart and run from the top with `USE_BYOD = True`.
  - State the chosen route in the opening cell and in Section 4. Regenerate.
- **Acceptance check:** a hosted run that completes the default path and then follows the documented BYOD route prints `adapted: False` in Section 6. Its Section 7 epoch-0 validation loss equals a fresh pipeline's on the same BYOD records, and the run reaches reload parity without OOM. A unit test on the stub transformer asserts that a second `adapt` call either refuses or starts from B = 0.
- **Spec:** DAT14, REL12 (MUST).

#### PX-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** whole notebook; metadata `notebook_mode: GUIDED`.
- **Observed issue:** none of these is present:
  - an intended-audience statement or a **How to use this notebook** section;
  - a roadmap or an Input → Model → Output task contract;
  - a glossary, although the notebook uses classifier-free guidance, latent, DPM-Solver++, LoRA rank, denoising MSE, zero-shot label accuracy and cosine × 100;
  - a prediction prompt before any principal result;
  - an interpretation checkpoint with a sample answer;
  - a Predict → Change → Run → Observe → Explain activity;
  - a troubleshooting section, although the hosted path has a restart, 22 GB of downloads and a 15 GB GPU floor;
  - a conclusion template.

  P5 found no "How to use", "Roadmap", "Glossary", "Check your reasoning", "Troubleshoot", "Infrastructure" or "conclusion template" text. Its "Predict" hit is the phrase "noise prediction", not a learner prompt. The three carried-module cells (2,009 lines) are not labelled Infrastructure and not collapsed (`cellView` absent).
- **Consequence:** the declared mode promises self-paced learning support that the notebook does not give. A first-time learner faces 2,000 lines of infrastructure before the lesson, and is never asked to predict, interpret or conclude.
- **Evidence:** source inspection (P5 `guided_markers`, `carried_cells_have_cellView_form`).
- **Recommended correction:** add the spec 2.2 §3.5 guided layer in `tools/notebook_template.py`:
  - audience and how-to-use text, a roadmap, an Input → Model → Output contract and a collapsible glossary;
  - a prediction before Sections 6 and 8, and "Check your reasoning" checkpoints with collapsible sample answers;
  - one bounded Predict → Change one thing → Run → Observe → Explain activity, for example `GUIDANCE_SCALE`, with its rerun scope;
  - a troubleshooting section covering the restart, downloads, GPU memory, digest mismatch and BYOD;
  - a conclusion template;
  - `# @title Infrastructure: …` with `cellView: form` on the carried and setup cells.
- **Acceptance check:** every item in the spec 2.2 §3.5 checklist (GDL1–GDL14) is present in the regenerated notebook. The carried cells are titled Infrastructure and collapsed. The validator checks for the new section headings.
- **Spec:** GDL1–GDL14, UX8 (SHOULD; required by the declared mode).

### Minor

#### PX-m1 — BYOD: the stated minimum is refused, and some refusals are not actionable

- **Cell/section:** opening cell and cell 13 (BYOD branch); `samples.py` `split_dataset`, `load_byod_dataset`.
- **Observed issue:**
  - The notebook states "at least four images, and at least one caption with three or more images". `split_dataset` refuses 4 images of one caption, 3 + 1 images, and 5 images of one caption ("split leaves 2/3 training records; at least 4 are required"). The smallest accepted single-caption set is 6 images, and 4 + 4 over two captions is accepted (P3).
  - A non-image file named in `captions.csv` raises a bare `UnidentifiedImageError` that names no file, and a non-zip upload raises `BadZipFile` (P3).
  - Cancelling the upload dialog raises `StopIteration` from `next(iter(uploaded.items()))` (source; not run).
  - In BYOD mode the test-split captions are still written to `outputs/pixart_sigma_generation_sample_captions.csv`.
- **Consequence:** a learner who follows the stated minimum is refused, and some bad inputs give no actionable message.
- **Evidence:** direct execution of the carried loader/splitter on synthetic zips (P3); source inspection.
- **Recommended correction:**
  - State the real rule: each caption with ≥ 3 images contributes 1 test + round(0.2 n) validation records, and at least 4 training records must remain (for example "at least 6 images of one caption").
  - Wrap the image decode and the zip open with messages that name the file and the expected format, and handle a cancelled upload.
  - Name the BYOD CSV for what it is.
- **Acceptance check:**
  - The stated minimum is accepted by `split_dataset`, and one image fewer is refused with the stated message.
  - A non-image entry and a non-zip upload each produce a message naming the file and the fix.
  - Cancelling the upload prints guidance instead of `StopIteration`.
- **Spec:** DAT12, DAT19 (MUST); UX10 (SHOULD).

#### PX-m2 — Split assumptions: no independence statement, and "hold out by caption" describes a within-caption split

- **Cell/section:** Section 4 (cell 12), Interpretation (cell 24); `tutorials/README.md` Conformance notes.
- **Observed issue:**
  - The notebook never says that the seeded random split assumes independent photographs (P5: `independence_statement: false`).
  - `tutorials/README.md` claims that "the notebook states … that photographs of the same individual or observer should be grouped in real data". The notebook contains no such statement; it says only that the sample has "one per observer per species".
  - The interpretation advises "**Hold out by caption, not by image**". But both the sample split and `split_dataset` stratify *within* each caption, so every caption appears in all three splits. The adjacent sentence describes that correctly, which makes the heading self-contradictory.
  - BYOD de-duplication is by exact pixel digest only.
- **Consequence:** a learner bringing burst photos or several photos of one subject gets an optimistic held-out loss with no warning, and the heading teaches the opposite of what the code does.
- **Evidence:** source inspection.
- **Recommended correction:** state the independence assumption in Section 4 and advise grouping near-duplicates (same individual, burst or observer) into one split. Rename the advice to "Stratify by caption; keep near-duplicates together". Make the README claim true or remove it.
- **Acceptance check:** Section 4 contains an explicit independence/grouping statement. The interpretation's split advice matches `split_dataset`'s behaviour. The README's conformance note quotes text that exists in the notebook.
- **Spec:** SPL3 (MUST); SPL10 (SHOULD).

#### PX-m3 — No pretraining-overlap statement for the public sample

- **Cell/section:** Section 4 (cell 12) / Interpretation (cell 24).
- **Observed issue:** the sample is 60 public iNaturalist photographs. PixArt-Σ's training corpus and the LAION-2B corpus behind the CLIP scorer are web-scale image–text collections that may contain these or near-identical photos. The notebook does not state this limitation (P5: `pretraining_overlap_statement: false`).
- **Consequence:** the learner may read the held-out loss and the CLIP reference similarity as evidence on unseen data when overlap cannot be ruled out.
- **Evidence:** source inspection.
- **Recommended correction:** add one sentence in Section 4 and in the limits saying that overlap between the public sample and the generator's or scorer's pretraining data cannot be ruled out, and what that does to the held-out numbers.
- **Acceptance check:** the regenerated notebook contains an explicit pretraining-overlap statement covering both the generator and the scorer.
- **Spec:** DAT9 (MUST).

#### PX-m4 — Expected outputs and conclusions that do not match the run; no variability statement; a template typo

- **Cell/section:** cells 1, 16, 20, 24.
- **Observed issue:**
  - **Cell 16** says to expect "label accuracy well below the real photographs' 1.0". The record shows 0.917 (one White-throated Sparrow photo is nearest the Junco caption).
  - **Cell 20** says to "look too for … higher reference similarity and label accuracy". Label accuracy was 0.50 → 0.50.
  - **The interpretation** states as *the claim* that the LoRA "lowers the held-out denoising loss" and "moves its generations towards the held-out real photographs". The recorded test change is −0.000145 (0.14 %), with no dispersion estimate, and the notebook itself prints it as "not asserted". Reference similarity fell for 2 of 6 captions (Goldfinch −4.0, Junco −0.13).
  - **Variability.** No statement says how much these numbers vary between runs or hardware (generation and fp16 kernels are seeded but not bit-reproducible across devices).
  - **Cell 1 typo.** The data contract renders as `{{id, image, caption}}`, a literal double brace from `tools/notebook_template.py` line 143.
- **Consequence:** a learner whose run matches the record sees an expected-output note contradicted, and the conclusion is pre-written rather than drawn from the evidence.
- **Evidence:** documented execution (archived evaluation report, P4 `per_caption_reference_change`) + source inspection (P5 `states_real_photos_1_0`, `double_brace_record`).
- **Recommended correction:**
  - Describe expected outputs without hard-coded values that the record contradicts, such as "near 1.0".
  - Phrase the interpretation conditionally ("if the test loss fell and reference similarity rose for most captions …") and point to the per-caption rows.
  - Add one sentence on run-to-run variability.
  - Fix the brace in the template.
- **Acceptance check:**
  - No "Look for" note is contradicted by the recorded run of the regenerated blob.
  - The interpretation contains no unconditional directional claim about the test loss or reference similarity.
  - A variability statement exists.
  - `{{` does not appear in rendered markdown.
- **Spec:** ENV8 (MUST); GDL8, GDL14 (SHOULD).

#### PX-m5 — The generated images are never displayed

- **Cell/section:** cells 17, 21, 23.
- **Observed issue:** `grid(...)` writes `outputs/…_frozen_grid.jpg` and `…_adapted_grid.jpg`, and cell 23 writes the new-prompt PNGs, but nothing displays them (P5: `display_calls: false`). Section 8 asks the learner to compare the second grid "with the first by eye".
- **Consequence:** in a text-to-image tutorial, the learner sees no images unless they open the file browser.
- **Evidence:** source inspection.
- **Recommended correction:** display both grids inline (`IPython.display.display(Image.open(path))`) and the two new-prompt images, with the prompt as a caption.
- **Acceptance check:** the executed notebook shows the frozen grid, the adapted grid and the new-prompt images as outputs of their cells.
- **Spec:** UX3, UX11 (SHOULD).

#### PX-m6 — The optional experiments give no rerun scope, and two of them score the adapted model as "frozen"

- **Cell/section:** Interpretation, "Optional experiments" (cell 24).
- **Observed issue:**
  - "Raise `EPOCHS` or `LEARNING_RATE`" reruns Section 7 on the already adapted LoRA (see PX-M3), with epoch 0 labelled "frozen model".
  - "Change `GUIDANCE_SCALE` and read the prompt similarity …" requires rerunning Section 6, which after Section 7 generates with the adapted model under the "frozen" label.
  - No experiment says which cells to rerun, and none asks for a prediction.
- **Consequence:** the documented experiments produce mislabelled comparisons.
- **Evidence:** source inspection (`adapt`, cells 17–21). Not executed.
- **Recommended correction:** give each experiment its rerun scope and a reset step (the reset from PX-M3). Frame one experiment as Predict → Change one thing → Run → Observe → Explain.
- **Acceptance check:** each optional experiment names the cells to rerun. Following it prints `adapted: False` wherever a frozen baseline is shown.
- **Spec:** GDL10 (SHOULD).

### Suggestions

- **PX-S1 — Regenerate against NOTEBOOK_SPEC 2.2.** The metadata, the opening cell, `NOTEBOOK_SOURCE` and `tutorials/README.md` declare 2.0.
- **PX-S2 — Print the chance level beside label accuracy** (1/6 for the sample, 1/k for BYOD with k captions), so 0.50 can be read against 0.17 (EVAL10, EVAL11).
- **PX-S3 — Record a Colab run.** Colab is the stated supported runtime, and the only clean-runtime record is Kaggle.
- **PX-S4 — Point Section 8 at the per-timestep paired change.** The t = 100 term (0.358) is about 68 % of the five-timestep mean (0.106), so the mean mostly reflects the lowest-noise step. A small frozen-vs-adapted per-timestep table or plot would show where the change is.
- **PX-S5 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`** in the install cell's markdown (EXE5).

### Spec conformance summary

| Requirement | Level | Status | Finding |
|---|---|---|---|
| RUN1, RUN10, ENV6, REL2, REL11 | MUST | Not met | PX-M1 |
| EVAL3 | MUST | Not met | PX-M2 |
| DAT14, REL12 | MUST | Not met on the documented BYOD route | PX-M3 |
| DAT12, DAT19 | MUST | Not met (stated minimum wrong; non-image / non-zip errors not actionable) | PX-m1 |
| SPL3 | MUST | Not met | PX-m2 |
| DAT9 | MUST | Not met | PX-m3 |
| ENV8 | MUST | Not met | PX-m4 |
| GDL1–GDL14, UX8 | SHOULD | Largely absent | PX-M4 |
| EVAL10, GDL8, GDL10, GDL14, UX3, UX10, UX11, SPL10, EXE5 | SHOULD | Deviations | PX-M2, PX-m1, PX-m2, PX-m4–m6, PX-S2, PX-S5 |
| Standalone carrier, parity, model acquisition, provenance, export and reload (ST, PAR, MOD, OUT, VER) | MUST | Met (static validator PASS, build `--check` OK, documented reload parity) | — |

## 6. Readiness

**Needs revision.** The open Majors are PX-M1 to PX-M4. The unresolved MUSTs are:

- RUN1, RUN10, ENV6, REL2, REL11 (PX-M1)
- EVAL3 (PX-M2)
- DAT14, REL12 (PX-M3)
- DAT12, DAT19 (PX-m1)
- SPL3 (PX-m2)
- DAT9 (PX-m3)
- ENV8 (PX-m4)

The repository's `Release-grade` status rests on a restart-dependent run and should return to `Candidate`. Gates remaining after the fixes:

- a one-pass hosted `Run all` of the regenerated blob (ideally on Colab, the stated runtime);
- re-recorded comparison numbers with a leave-one-out real-photo reference;
- a hosted BYOD run after the default path that shows an un-adapted Section 6 baseline and reaches reload parity, plus one invalid BYOD input.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, this review):**
  - the offline suite (34 passed, 1 module skipped), `build_notebook.py --check` and the validator;
  - the BYOD minimum and refusal behaviour of the carried loader and splitter;
  - the self-inclusion of `real_photo_baseline` (stand-in scorer).
- **Verified from documented execution (Kaggle T4, this blob):** the restart on pass 1; every recorded metric quoted above; reload parity.
- **Derived arithmetically:** the leave-one-out real-photo reference similarity, 57.72. It comes from the recorded self-inclusive per-caption values for two unit-norm embeddings per caption (`c = 2v² − 1`), not from recomputed embeddings.
- **Inferred from source, not run:**
  - the BYOD and experiment reruns start from the adapted LoRA (PX-M3, PX-m6);
  - the ≈15.9 GB projected memory on the BYOD rerun;
  - `StopIteration` on a cancelled upload.
- **Not verified:** any Colab run; the real upload dialog; BYOD beyond the split; learner understanding.
- **Most likely to be wrong:** the PX-M3 severity and memory projection. The LoRA carry-over is certain from the source, but a reader could treat "re-run from Section 4" as implicitly meaning "in a fresh runtime" and rate it Minor, as the sibling flux review did for the experiments (FS-m3). The ≈15.9 GB figure adds allocator totals from different cells and may be off by several hundred MB either way.
