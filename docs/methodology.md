# Methodology — AU-Conditioned Face Synthesis via a Hybrid DiffAE + Stable Diffusion Pipeline

> Proposed pipeline to synthesize faces that fill rare **emotion × AU-configuration** cells, using a **semantic latent edit (DiffAE)** + **text/AU conditioning (Stable Diffusion)** with a **disentangled AU direction operator W**.
>
> Grounded in: DiffAE (2111.15640), MagicFace (2501.02260), SynFER (2410.09865), FineFace (2407.20175), AUEditNet (2404.05063), Controlled-Aug (2602.19219), StyleGAN-AU-edit (2107.12143), IP-Adapter / Prompt-to-Prompt / InstructPix2Pix.

---

## 1. Problem statement

Let a dataset be indexed by images $x\in\mathbb{R}^{H\times W\times 3}$, an emotion label $e\in\mathcal{E}$, and an AU configuration $\mathbf{a}\in[0,1]^K$ (presence/intensity of $K$ action units). We define a **cell** as the pair $(e,\mathbf{a})$. The training set has a heavy tail: for rare emotions, a large fraction of configurations $\mathbf{a}$ are singletons, so the model never sees enough joint support $(e,\mathbf{a})$ to learn the AU branches.

**Goal:** given a real source image $x_0$ in a common cell $A$ of emotion $e$, and a target AU $x$ (so the target cell is $A+x$), produce a synthetic image $\hat{x}_0$ that

1. **adds** the target AU $x$ (AU-level control),
2. **keeps** the source configuration $A$ (no leakage),
3. **keeps** emotion $e$ (sample-level correctness),
4. **keeps** identity and non-target attributes (nuisance invariance).

We also want a **text** input — an *AU prompt* (structured AU request) plus a *global description* `"a photo of a <emotion> face"` — so the generator is steerable and interpretable.

---

## 2. Background: diffusion, and why DDIM was a real improvement

### 2.1 DDPM forward process

The forward (noising) process is a Markov chain with schedule $\beta_t\in(0,1)$:

$$
q(x_t\mid x_{t-1})=\mathcal{N}\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big).
$$

Defining $\bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s)$, we can jump directly from data to any timestep:

$$
q(x_t\mid x_0)=\mathcal{N}\big(x_t;\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\big)
\quad\Longleftrightarrow\quad
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad\epsilon\sim\mathcal{N}(0,I).
$$

A U-Net $\epsilon_\theta$ is trained to predict the added noise, giving the "simple" objective

$$
\mathcal{L}_{\text{simple}}=\mathbb{E}_{t,x_0,\epsilon}\Big[\lVert \epsilon-\epsilon_\theta(x_t,t)\rVert^2\Big],
$$

which is a re-weighted form of the variational lower bound (ELBO).

### 2.2 DDIM: from stochastic to deterministic, invertible sampling

DDPM's reverse step is stochastic (it samples from a Gaussian at every step). DDIM (Song et al.) observes that the same forward *marginals* can be shared by a **non-Markovian** family of reverse processes, indexed by a variance parameter $\sigma_t$. Writing the predicted clean image

$$
\hat{x}_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},
$$

the DDIM reverse step is

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat{x}_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t)+\sigma_t\,\epsilon_t.
$$

Setting $\sigma_t=0$ makes the mapping $x_t\mapsto x_{t-1}$ **deterministic**. Two consequences matter for us:

1. **Invertibility (encode–decode).** A deterministic sampler can be inverted approximately to obtain a *latent* for a real image, $x_0\mapsto x_T$, and decoded back $x_T\mapsto x_0$. This turns a diffusion model into an **autoencoder** — the backbone of editing.
2. **Semantic instability of $x_T$.** From the forward marginal,
   $$
   q(x_T\mid x_0)=\mathcal{N}\big(\sqrt{\bar\alpha_T}\,x_0,\,(1-\bar\alpha_T)I\big),
   $$
   the mean is a **scaled copy of the pixels**, so $x_T$ is dominated by pixel-level information plus noise. Interpolating two $x_T$ preserves layout/color but **not** identity/expression — *good reconstruction $\neq$ good semantics*.

> This is the precise "DDIM vs Diffusion" improvement we build on: DDIM gives us **deterministic, cheap, invertible** sampling, but its latent is still not *semantic*. The next ingredient fixes exactly that.

### 2.3 DiffAE: a semantic + stochastic two-part code

DiffAE (Preechakul et al.) splits the representation into

$$
x_0 \longmapsto (z_{\text{sem}},\, x_T),
$$

where $z_{\text{sem}}\in\mathbb{R}^d$ (here $d=512$) is a **low-dimensional, linear, semantically meaningful** code from a CNN encoder $E_\phi$, and $x_T$ is a **stochastic** code holding residual pixel detail. The decoder is a conditional DPM

$$
p(x_{t-1}\mid x_t, z_{\text{sem}}),
$$

which injects $z_{\text{sem}}$ through **Adaptive Group Normalization (AdaGN)**, so $z_{\text{sem}}$ plays the role of a StyleGAN "style vector". This is the basis for the "2-input" view in the survey: **semantic input** ($z_{\text{sem}}$) + **noise-step $T$ / stochastic input** ($x_T$).

---

## 3. Adding text: decoupled cross-attention (the IP-Adapter idea)

Stable Diffusion injects a text condition $c_t$ (CLIP text embeddings of the prompt) into the U-Net via cross-attention:

$$
Q=Z W_q,\quad K_t=c_t W_k,\quad V_t=c_t W_v,\qquad
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d}}\Big)V .
$$

IP-Adapter showed that naively concatenating a second condition $[c_t;c_i]$ is weak, and proposed a **decoupled** second cross-attention with the *same query*:

$$
Z_{\text{new}}=\underbrace{\mathrm{Attn}(Q,K_t,V_t)}_{\text{text}}
\;+\;\lambda\cdot\underbrace{\mathrm{Attn}(Q,K_i,V_i)}_{\text{image}},
$$

with $K_i=c_i W'_k$, $V_i=c_i W'_v$ initialized from the text weights ($W'_k\leftarrow W_k$, $W'_v\leftarrow W_v$). This gives an explicit, tunable **strength dial $\lambda$** per condition and is classifier-free-guidance (CFG)-compatible (drop the branch with probability during training).

We will reuse this machinery, but replace the *image* branch with an **AU branch**, and add a third **AU-prompt** pathway. This is the creative core of the proposal.

---

## 4. Proposed pipeline (Hybrid DiffAE + SD)

### 4.1 Data flow (input → output)

```
source x_0 ──► DiffAE encoder ──► z_sem (semantic)     x_T (stochastic)
                                        │
                                        ▼
        AU direction operator W:  z_edit = z_sem + W · a_res   (disentangled)
                                        │
        text: "a photo of a <emotion> face" ─► CLIP ─► c_text
        AU prompt: a ∈ R^K ─► AU projector ─► c_au (tokens)
                                        │
                                        ▼
        frozen SD U-Net + adapters (AU adapter, ID adapter)
          Z_new = Attn_text(Q,K_t,V_t) + λ_au·Attn_au(Q,K_a,V_a)
                                        │
                                        ▼
        DDIM decode of x_T conditioned on (z_edit, c_text, c_au)
                                        │
                                        ▼
        x̂_0  ──► acceptance–rejection gate (AU, emotion, identity, fidelity)
```

### 4.2 Inputs

- **Source image** $x_0$ (a real train image in cell $A$), used only in *edit* mode.
- **Target AU vector** $\mathbf{a}_{\text{target}}\in[0,1]^K$ — the requested AU configuration (or the residual $\mathbf{a}_{\text{res}}=\mathbf{a}_{\text{target}}-\mathrm{AU}(x_0)$).
- **Global text** $c_{\text{text}}$ = `"a photo of a <emotion> face"`, encoded by a frozen CLIP text encoder.
- **AU prompt** — a structured, learnable encoding of $\mathbf{a}$ (see §4.5).
- **Timestep** $t$ / stochastic code $x_T$.

### 4.3 Semantic encoder + stochastic channel (DiffAE)

A frozen DiffAE provides

$$
E_\phi(x_0)=z_{\text{sem}}\in\mathbb{R}^{512},\qquad x_T=\mathrm{DDIMInv}(x_0).
$$

$z_{\text{sem}}$ is **decodable** (near-exact reconstruction) and **linearly controllable** (empirically, linear probes from $z_{\text{sem}}$ reach AUROC 0.82–0.95 and 7-emotion accuracy 72.8%).

### 4.4 The AU edit operator $W$ (phase 1 of the project)

We learn a matrix $W\in\mathbb{R}^{d\times K}$ such that moving along column $k$ increases AU $k$. For AU $k$ we fit a linear predictor

$$
\hat{a}_k = z^\top w_k + w_{0k},
$$

and use $w_k$ as the **edit direction**. Editing is

$$
z_{\text{edit}} = z_{\text{sem}} + W\,\mathbf{a}_{\text{res}}
\qquad\text{or per-AU}\qquad
z'=z_{\text{sem}} + s\,w_k .
$$

**The disentanglement problem.** The semantic space is not axis-aligned; moving along $w_k$ leaks into co-activated AUs and nuisance attributes (glasses, gender, age). Three complementary fixes, taken from 2602.19219 / 2107.12143:

1. **Dependency-aware conditioning.** Fit the AU-$k$ predictor *conditioned* on the AUs that co-activate with it, blocking backdoor paths through a co-activation DAG:
   $$
   \hat{a}_k = f_k\big(z;\; \mathbf{a}_{\setminus k}\big).
   $$
2. **Orthogonal projection of nuisance directions.** For nuisance directions $u_1,\dots,u_m$ (glasses, beard, gender, age), build $U=[u_1,\dots,u_m]$ and project the edit direction onto the orthogonal complement:
   $$
   P_{\perp}=I-U(U^\top U)^{-1}U^\top,\qquad w_k^{\perp}=P_{\perp} w_k.
   $$
   (Projecting the *direction* $w_k$, not the code $z$.)
3. **Neutralization baseline.** Express every edit relative to a zero-AU neutral state $z_{\text{neutral}}$, so absolute AU levels are meaningful:
   $$
   \min_z\;\mathcal{L}\big(\mathrm{AU}(N(z)), \mathbf{a}^{\star}\big)+\lambda\,\lVert z-z_{\text{sample}}\rVert_2^2 .
   $$

Additionally we impose an **orthogonality regularizer** on $W$ (soft disentanglement), and optionally split the encoder into **identity vs expression branches** (dual-branch, from AUEditNet):

$$
\mathcal{L}_{\text{ortho}}=\lVert W^\top W - I\rVert_F^2 .
$$

### 4.5 Text injection — AU prompt + global description (the creative part)

We want the generator to be driven by *language* as well as a *structured AU vector*, and we want the two to compose. We do this with **three conditioning channels**:

1. **Global text** (frozen CLIP): $c_{\text{text}}$ from `"a photo of a <emotion> face"` — sets the coarse emotion.
2. **AU tokens** (learnable projector): map the AU vector $\mathbf{a}$ into a small set of tokens exactly like IP-Adapter maps a CLIP image embedding:
   $$
   c_{\text{au}} = \mathrm{LayerNorm}\big(\mathrm{Linear}(\mathbf{a})\big)\in\mathbb{R}^{N\times d},
   $$
   with $N\ll K$ tokens (e.g. $N=4$).
3. **AU-prompt text** (optional): an *AU sentence* such as `"brow lowerer strong, nose wrinkler active"`, encoded by CLIP and merged with $c_{\text{au}}$. This makes the control human-readable and allows a text-only fallback.

These enter the U-Net through **decoupled cross-attention** with a shared query $Q$:

$$
Z_{\text{new}}
=\mathrm{Attn}(Q,K_t,V_t)
+\lambda_{\text{au}}\,\mathrm{Attn}(Q,K_{\text{au}},V_{\text{au}}),
$$

where $K_{\text{au}}=c_{\text{au}}W'_k$, $V_{\text{au}}=c_{\text{au}}W'_v$ are the only newly-trained attention weights (initialized from the text weights). $\lambda_{\text{au}}$ is the **AU-strength dial**; $\lambda_{\text{au}}=0$ recovers the plain text-to-image model, and larger values push the image to follow the AU vector. This is the analogue of IP-Adapter's image-conditioning strength, repurposed for **structured AU control**.

Because the semantic edit $z_{\text{edit}}$ is injected via AdaGN *and* the AU vector is injected via cross-attention, we get **two complementary control paths**: a *precise, low-level* latent edit and a *coarse, language-like* conditioning — the "strongest per-component" combination the survey calls for.

### 4.6 Frozen SD decoder + adapters

The decoder is a pretrained Stable Diffusion U-Net, kept frozen except for:

- the **AU adapter** (the $W'_k,W'_v$ above),
- an **identity adapter** (self-attention merging of the source appearance, MagicFace-style) to preserve identity,
- an **attribute controller** (background/pose conditioning) to keep pose/background stable.

DDIM deterministically decodes $x_T$ conditioned on $(z_{\text{edit}}, c_{\text{text}}, c_{\text{au}})$.

### 4.7 Output + acceptance–rejection gate

Each candidate $\hat{x}_0$ is accepted only if it passes the frozen gates (same as `already.md`):

1. target AU active (score $\ge$ threshold **and** $\ge +0.10$ vs recon),
2. keep $A$ (each $A$-AU drops $\le 0.10$),
3. emotion not worse (AUCANet on edit $\ge$ recon, or $=$ label),
4. identity (dlib distance $\le 0.6$).

For **new identities** (synthesis mode), we add **acceptance–rejection sampling** over the demographic predictors: draw $z\sim p(z)$, predict gender/age, and keep only the desired bins (DiffRS-style). This gives balanced demographics without extra generative training.

---

## 5. Objective functions

The full multi-task objective is

$$
\mathcal{L}
=\mathcal{L}_{\text{diff}}
+\alpha\,\mathcal{L}_{\text{rec}}
+\beta\,\mathcal{L}_{\text{AU}}
+\gamma\,\mathcal{L}_{\text{id}}
+\delta\,\mathcal{L}_{\text{dis}}
+\eta\,\mathcal{L}_{\text{text}}.
$$

**Denoising loss** (the core generative signal):

$$
\mathcal{L}_{\text{diff}}
=\mathbb{E}_{t,x_0,\epsilon}\Big[\lVert \epsilon-\epsilon_\theta\big(x_t,\,t,\,z_{\text{sem}},\,c_{\text{text}},\,c_{\text{au}}\big)\rVert^2\Big].
$$

**Reconstruction loss** (keeps the semantic code decodable):

$$
\mathcal{L}_{\text{rec}}=\mathbb{E}_{x_0}\big[\lVert x_0-\hat{x}_0\rVert_2^2\big],
\quad \hat{x}_0=\mathrm{Decode}(z_{\text{sem}},x_T).
$$

**AU-consistency loss** (the edit actually moves the requested AU; frozen AU estimator $\mathrm{AU}(\cdot)$):

$$
\mathcal{L}_{\text{AU}}=\mathbb{E}\big[\lVert \mathrm{AU}(\hat{x}_0)-\mathbf{a}_{\text{target}}\rVert_2^2\big].
$$

**Identity loss** (ArcFace cosine):

$$
\mathcal{L}_{\text{id}}=\mathbb{E}\big[1-\cos\big(f_{\text{arc}}(x_0),\,f_{\text{arc}}(\hat{x}_0)\big)\big].
$$

**Disentanglement loss** (soft orthogonality + nuisance + co-activation):

$$
\mathcal{L}_{\text{dis}}
=\underbrace{\lVert W^\top W-I\rVert_F^2}_{\text{orthogonal directions}}
+\lambda_{\text{nuis}}\underbrace{\lVert P_{\text{nuis}} W\rVert_F^2}_{\text{nuisance leakage}}
+\lambda_{\text{graph}}\underbrace{\sum_{(k,l)\notin G}\big(w_k^\top w_l\big)^2}_{\text{off-graph co-activation}},
$$

where $G$ is the allowed AU co-activation graph (dependency-aware).

**Text-alignment loss** (optional, keeps output consistent with the global description):

$$
\mathcal{L}_{\text{text}}=-\mathbb{E}\Big[\cos\big(\mathrm{CLIP}_I(\hat{x}_0),\ \mathrm{CLIP}_T(c_{\text{text}})\big)\Big].
$$

---

## 6. Task-learning formulation

The pipeline is trained as a **joint of three tasks that share the semantic encoder $E_\phi$**:

| Task | Learns | Supervised by |
|---|---|---|
| **T1 — Autoencoding** | $E_\phi$ (semantic code decodable) | $\mathcal{L}_{\text{diff}}+\mathcal{L}_{\text{rec}}$ |
| **T2 — AU-conditioned editing** | $W$, AU adapter | $\mathcal{L}_{\text{AU}}+\mathcal{L}_{\text{dis}}$ |
| **T3 — Text/AU controllable synthesis** | text/AU/ID adapters | $\mathcal{L}_{\text{diff}}+\mathcal{L}_{\text{text}}$ |

The sharing is what makes it "one model": the semantic encoder must (i) reconstruct, (ii) linearly encode AUs for editing, and (iii) compose with text for generation — so each task acts as a regularizer on the others. The disentanglement losses ($\mathcal{L}_{\text{dis}}$) constrain the *geometry* of the edit directions $W$, while the AU-consistency loss constrains their *effect* on decoded images.

---

## 7. Training & inference recipe

**Training**
1. Pre-train / freeze DiffAE ($E_\phi$, decoder) and CLIP encoders.
2. Train $W$ by ridge regression of $\mathrm{AU}(x)$ on $z_{\text{sem}}$, with dependency-aware conditioning and orthogonal projection (closed-form; no gradient needed for the *direction*).
3. Train the AU adapter ($W'_k,W'_v$) with $\mathcal{L}_{\text{diff}}+\beta\mathcal{L}_{\text{AU}}+\delta\mathcal{L}_{\text{dis}}$, random-dropping the AU branch (for CFG).
4. Optionally train the ID adapter and attribute controller with $\mathcal{L}_{\text{id}}$.

**Inference (edit mode)**
1. Encode $x_0\to(z_{\text{sem}},x_T)$.
2. Compute $z_{\text{edit}}=z_{\text{sem}}+W\,\mathbf{a}_{\text{res}}$ (disentangled directions).
3. DDIM-decode $x_T$ with conditioning $(z_{\text{edit}},c_{\text{text}},c_{\text{au}})$, using CFG on the text/AU branches.
4. Run the acceptance–rejection gate; increase dose $\lVert\Delta z\rVert$ per source until first pass.

**Inference (synthesis mode)**
1. Sample $z\sim p(z)$; demographic acceptance–rejection.
2. Neutralize: $z_{\text{neutral}}=\arg\min_z\mathcal{L}(\mathrm{AU}(N(z)),0)+\lambda\lVert z-z_{\text{sample}}\rVert^2$.
3. Apply $W\mathbf{a}$ and decode; label as a controlled AU sample.

---

## 8. Theoretical notes (what is being claimed)

- **DDIM vs DDPM**, restated for our use: DDPM samples stochastically and has no clean inverse; DDIM trades a little diversity for a **deterministic, invertible** map, which is precisely what makes encode–edit–decode possible. Our contribution is to *augment* that deterministic map with a **semantic axis** ($z_{\text{sem}}$) and a **disentangled AU axis** ($W$), so the latent is no longer "pixel + noise" but "meaning + noise".
- **Orthogonal projection** removes the nuisance component of $w_k$ but can weaken the target if AU and nuisance genuinely overlap ($\lVert w_k^\perp\rVert\le\lVert w_k\rVert$). Projection sets are chosen empirically (§4.4).
- **Dependency-aware conditioning** handles entanglement *statistically* (blocking co-activation paths), while orthogonal projection handles it *geometrically*; the two are complementary, not redundant.
