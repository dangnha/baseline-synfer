# FER-Diffusion — AU-Conditioned Face Synthesis Pipeline

Reference implementation of the pipeline in [`docs/methodology.md`](docs/methodology.md): a **hybrid
DiffAE (semantic encoder) + Stable Diffusion (decoder)** model that edits faces along **disentangled
AU directions** to fill rare *emotion × AU-configuration* cells, with **text + AU conditioning** and an
**acceptance–rejection gate**. See [`docs/ideas.md`](docs/ideas.md) for the experiment backlog and
[`docs/already.md`](docs/already.md) for what the team has already validated.

## How it maps to the methodology

| Methodology | Code |
|---|---|
| §2.2 DDIM (deterministic, invertible) | `ferdiff/diffusion.py` |
| §2.3 DiffAE semantic + stochastic code | `ferdiff/encoder.py` (semantic) + DDIM inversion (stochastic) |
| §4.4 AU operator `W` + disentanglement | `ferdiff/directions.py` |
| §4.5 decoupled text/AU cross-attention | `ferdiff/adapter.py` |
| §5 objective functions | `ferdiff/losses.py` |
| §4.7 acceptance–rejection gates | `ferdiff/gates.py` |
| §4 / §7 end-to-end edit + synthesize | `ferdiff/pipeline.py` |

```
source x_0 ─► DiffAE encoder ─► z_sem            x_T = DDIM-invert(SD VAE(x_0))
                                 │
                                 ▼
               AU operator:  z_edit = z_sem + W·a_res   (dependency-aware + orthogonal-projected)
                                 │
              text "a photo of <emotion> face" ─► CLIP ─► c_text
              AU vector a ─► AU projector ─► c_au
                                 │
                                 ▼
              frozen SD U-Net + decoupled cross-attention (text + semantic + AU)
                                 │
                                 ▼
              DDIM decode ─► VAE decode ─► image ─► gates (AU, keep-A, emotion, identity)
```

## Repository layout

```
ferdiff/            # the package (all core logic)
  config.py         # dataclasses + YAML loader
  directions.py     # W operator: ridge fit, dependency-aware, orthogonal projection, neutralization
  losses.py         # L_diff, L_rec, L_AU, L_id, L_dis, L_text
  diffusion.py      # DDIM sampler + inversion
  adapter.py        # decoupled cross-attention + semantic/AU projectors
  encoder.py        # DiffAE semantic encoder wrapper
  interfaces.py     # pluggable loaders for DiffAE / OpenGraphAU / AUCANet / dlib / ArcFace / CLIP
  gates.py          # the 4 acceptance--rejection gates
  pipeline.py       # FerDiffusion orchestration (edit / synthesize)
scripts/
  learn_directions.py  # fit W from (z_sem, AU) pairs
  edit_cell.py         # A -> A+x editing for a source manifest
  synthesize.py        # sample identity -> neutralize -> apply AU -> decode -> gate
tests/
  test_directions.py   # pure-NumPy tests of the disentanglement math
  test_adapter.py      # torch tests of the decoupled cross-attention
configs/default.yaml   # all checkpoint/data paths + hyperparameters
```

## Install

```bash
# 1) create/activate an environment (conda or venv) with a CUDA torch
conda create -n ferdiff python=3.10 -y && conda activate ferdiff
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA

# 2) install the pipeline deps
pip install -r requirements.txt

# 3) optional, for the gates
pip install dlib          # identity gate (may need CMake)
pip install insightface   # ArcFace identity loss
```

The pure-math core (`directions.py`) and the adapters are unit-tested; run:

```bash
python -m pytest tests/ -q      # 11 tests, no GPU/weights required
```

## Before you run — wire your checkpoints

Everything in `interfaces.py` is a **stub** that raises a clear error until you plug in the models you
already have. This is intentional: the pipeline is a reference implementation and must not download or
retrain weights. Fill in `configs/default.yaml` and implement the matching loader:

| Component | Config key | What to return |
|---|---|---|
| DiffAE encoder | `diffae.checkpoint` | `encode(images) -> z_sem (n, 512)` |
| OpenGraphAU | (see `load_au_estimator`) | `au_estimator(images) -> (n, K)` scores |
| AUCANet | (see `load_emotion_classifier`) | `classifier(images) -> predicted label` |
| dlib identity | (see `load_identity_descriptor`) | object with `.descriptor(img)` and `.distance(img)` |
| ArcFace | (see `load_arcface`) | `arcface(images) -> embeddings` |
| CLIP text | (see `load_clip_text_encoder`) | `encode_text(texts) -> embeddings` |

The SD decoder is injected as a callable with the signature

```python
decoder(x_T, z_edit, c_text, a_target) -> image   # numpy array in [0, 1]
```

Wire it in `scripts/edit_cell.py::build_pipeline` using your diffusers Stable Diffusion pipeline
(VAE-encode for `x_T`, DDIM inversion from `ferdiff/diffusion.py`, and `ferdiff/adapter.py` for the
decoupled conditioning).

## Quickstart

```bash
# 1) Fit the AU direction operator W from precomputed semantic codes + AU scores
python scripts/learn_directions.py \
    --Z z_sem.npy --A au_scores.npy \
    --au-names AU4 AU5 AU17 \
    --dependency-aware --coactivating coactivating.npy \
    --nuisance nuisance_basis.npy \
    --out W.npz

# 2) Edit a cell A -> A+x (source images listed one per line)
python scripts/edit_cell.py \
    --config configs/default.yaml --W W.npz \
    --sources sources_anger_4plus17.txt --target-au AU5 --emotion anger \
    --out-dir outputs/edit

# 3) Synthesize new identities for a rare cell
python scripts/synthesize.py \
    --config configs/default.yaml --W W.npz \
    --target-au AU5 --emotion anger --num 8 --out-dir outputs/synthesize
```

`learn_directions.py` is fully runnable today (it only needs NumPy). The edit/synthesize scripts need
the external models wired first.

## Configuration reference

Key groups in `configs/default.yaml` (all editable):

- `diffae` — checkpoint path, `latent_dim=512`, nuisance labels.
- `sd` — SD model id/path, `dtype`, `num_inference_steps`, `guidance_scale`, and the decoupled strength
  dials `lambda_sem` / `lambda_au`.
- `au` — `au_names` (must match the column order of your AU scores), per-AU activation `thresholds`,
  and the `coactivation_graph` for the disentanglement loss.
- `gates` — the four thresholds from `docs/already.md`:
  - `target_au_delta` (target AU must rise ≥ +0.10 vs reconstruction),
  - `keep_a_delta` (each source AU may drop ≤ 0.10),
  - `identity_threshold` (dlib distance ≤ 0.6),
  - `demographic_bins` (synthesis-mode acceptance–rejection).
- `losses` — weights `alpha/beta/gamma/delta/eta` for the multi-task objective (§5).
- `train` — ridge `alpha`, neutralization `lambda/steps/lr`, and the `dose_*` sweep range.

## Extending

- **New adapter / conditioning** — add a branch to `DecoupledCrossAttention` or a new projector in
  `SemanticAUAdapter`; the strength dials are already per-channel.
- **SD vs DiffAE decoder** — the pipeline treats the decoder as a black box, so either backend plugs in
  behind the same `decoder(...)` callable.
- **`attach_to_unet`** (in `adapter.py`) is a documented scaffold for installing the decoupled
  cross-attention into a diffusers `UNet2DConditionModel`; adapt the `AttnProcessor` subclass to your
  installed diffusers version.
- **New experiments** — see `docs/ideas.md`; each idea maps to a module above (e.g. Idea 2 → `directions.project_out`, Idea 4 → `adapter`, Idea 9 → `gates`).

## Not implemented (deferred)

End-to-end training of the SD adapters (needs the data loaders + server GPU) and the 16 `ideas.md`
experiments. These are documented extension points, not part of this scaffold.
