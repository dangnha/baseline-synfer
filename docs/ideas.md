# Ideas — Experiment Backlog for the FER-Diffusion Pipeline

> How to improve the pipeline in `methodology.md`, organized as falsifiable experiments. Each idea = **Hypothesis → Method → Metric → Downstream task**. Grounded in `already.md` (what we have) and the surveyed papers.

## 0. Current baseline (what to beat)

The working editor is **DiffAE linear latent editing** (dependency-aware + orthogonal-projected direction, absolute dose `‖Δz‖`, 4 frozen gates). It passes 4/7 sources on Anger 4+17 → +AU5, but fails the emotion gate on mouth-region AUs and has **no text conditioning** yet. Every idea below should be measured against this baseline and against the E2 resampling/reweighting control.

## 1. Methods to borrow (map)

| Source | Key idea to steal | Cost / risk |
|---|---|---|
| DiffAE (2111.15640) | Two-part semantic+stochastic code, AdaGN decoding | High compute; already in use |
| DiffAE-V2 / DiffuseGAE | Stronger disentangled semantic latent | Retrain; medium |
| StyleGAN-AU-edit (2107.12143) | Detached regions-of-influence per AU | StyleGAN-only; low |
| AUEditNet (2404.05063) | Dual-branch identity vs attribute, implicit disentanglement | Retrain; medium |
| FineFace (2407.20175) | AU control via adapters, text + image prompts | Low; adapter training |
| SynFER (2410.09865) | Text + AU synthesis, semantic guidance + pseudo-label rectifier | Medium; needs SD |
| MagicFace (2501.02260) | SD + AU variation + ID encoder (self-attn) + attribute controller | Low-medium |
| Controlled-Aug (2602.19219) | Dependency-aware conditioning, orthogonal projection, neutralization | Low; closed-form |
| InstructPix2Pix | Instruction-based editing | Needs paired data |
| Prompt-to-Prompt | Cross-attention-map injection for localized edits | Low; inference-time |
| IP-Adapter | Decoupled cross-attention for a second condition | Low; adapter training |
| ControlNet | Structural/spatial conditioning | Medium |
| DiffRS (rejection sampling) | Output-level acceptance/rejection | Low |

## 2. Experiment ideas

### Idea 1 — Dependency-aware AU conditioning
- **Hypothesis:** AU leakage comes from learning through co-activation backdoor paths; conditioning the AU predictor on the co-activated AUs blocks these paths and produces purer edit directions.
- **Method:** Fit $w_k$ with $\hat a_k=f_k(z;\mathbf a_{\setminus k})$, using the AU co-activation DAG as the conditioning set (already partially done in W5-R8; extend to all AUs, and reason about colliders).
- **Metric:** Off-target AU change (per-AU Δ vs recon) at fixed target-AU Δ; disentanglement gap.
- **Downstream:** AU detection (BP4D/DISFA) on edited data.

### Idea 2 — Orthogonal projection of nuisance directions
- **Hypothesis:** Projecting $w_k$ onto the orthogonal complement of nuisance directions (glasses/beard/gender/age) removes attribute leakage at the cost of a weaker target, and improves identity/attribute preservation.
- **Method:** Build $U$ from CelebA-HQ nuisance predictors; $w_k^\perp=(I-U(U^\top U)^{-1}U^\top)w_k$; sweep projection sets empirically.
- **Metric:** Nuisance-attribute change (linear-probe score before/after), identity cosine, target-AU Δ.
- **Downstream:** Identity-preserving FER augmentation.

### Idea 3 — Expression neutralization as a learned module
- **Hypothesis:** A dedicated neutralizer $N$ (trained, not optimized per-sample) gives a stable zero-AU baseline, making absolute AU labels trustworthy for synthesized identities.
- **Method:** Train $N:z\mapsto z_{\text{neutral}}$ minimizing $\mathcal L(\mathrm{AU}(N(z)),0)+\lambda\lVert z-z_{\text{sample}}\rVert^2$, frozen detector, high AU recall threshold.
- **Metric:** AU recall, neutral-state AU error, identity preservation of $N$.
- **Downstream:** Controlled single-AU synthesis with trustworthy labels.

### Idea 4 — Decoupled AU cross-attention adapter (IP-Adapter style)
- **Hypothesis:** Injecting the AU vector through a *second* decoupled cross-attention branch gives continuous, tunable AU control that composes with text, outperforming prompt-embedded AU text.
- **Method:** AU projector $\mathrm{LayerNorm}(\mathrm{Linear}(\mathbf a))\to c_{\text{au}}$; $Z_{\text{new}}=\mathrm{Attn}_{\text{text}}+\lambda_{\text{au}}\mathrm{Attn}_{\text{au}}$; CFG via random branch dropout.
- **Metric:** AU control range (min→max Δ), monotonicity of AU score vs $\lambda_{\text{au}}$, off-target leakage.
- **Downstream:** AU-conditioned generation for rare cells.

### Idea 5 — Dual-branch semantic/stochastic + SD decoder (hybrid, the main bet)
- **Hypothesis:** Combining DiffAE's *semantic* edit (precise, low-level) with SD's *text* conditioning (coarse, language-like) is strictly stronger than either alone.
- **Method:** Full `methodology.md` pipeline (§4): frozen DiffAE encode → disentangled $W$ edit → frozen SD decode with AU + text + ID adapters.
- **Metric:** Pass rate through the 4 gates vs. DiffAE-only and MagicFace-only; FID/LPIPS/identity.
- **Downstream:** E3/E4 FER gain over E2.

### Idea 6 — AU prompt + global text description (text steering)
- **Hypothesis:** A structured AU prompt + a global `"a photo of a <emotion> face"` description gives interpretable, coarse-to-fine control and enables text-only fallback when the AU vector is unavailable.
- **Method:** Encode AU sentence with CLIP, merge with $c_{\text{au}}$; ablate "text only", "AU only", "both".
- **Metric:** FER accuracy of generated set, CLIP score, AU-vs-text conflict rate.
- **Downstream:** FER classification (AffectNet/RAF-DB).

### Idea 7 — Identity encoder via self-attention (MagicFace-style)
- **Hypothesis:** Merging source appearance features through self-attention preserves identity better than latent-space identity regularization alone.
- **Method:** Add a frozen/small ID encoder that injects appearance tokens into the SD decoder; combine with $\mathcal L_{\text{id}}$.
- **Metric:** ArcFace cosine, dlib distance, identity preservation at equal AU Δ.
- **Downstream:** Identity-consistent rare-cell augmentation.

### Idea 8 — Style-transfer of expression semantic features
- **Hypothesis:** Expression is a *style*; transferring the expression sub-vector of $z_{\text{sem}}$ from a donor identity to a target identity generalizes to identities unseen during AU-direction learning.
- **Method:** Split $z_{\text{sem}}$ into identity + expression subspaces; swap the expression component (AdaIN-style) and decode; compare vs. $W$-direction edits.
- **Metric:** Expression transfer accuracy (AU/FER), identity cosine, artifact rate.
- **Downstream:** Cross-identity expression augmentation (long-tail classes).

### Idea 9 — Acceptance–rejection sampling with an AU/FER discriminator
- **Hypothesis:** A post-hoc acceptance–rejection gate (AU estimator + FER classifier + identity) removes low-quality samples and raises the *effective* data quality without retraining the generator.
- **Method:** DiffRS-style: sample many candidates, accept only those passing frozen gates + thresholds; also demographic balancing via gender/age predictors.
- **Metric:** Rejection rate, downstream FER accuracy vs. accepted-set size, FID of accepted set.
- **Downstream:** Data augmentation for imbalanced FER/AU.

### Idea 10 — Instruction-based editing (InstructPix2Pix)
- **Hypothesis:** Human instructions (`"widen the eyes while keeping the brow furrowed"`) are a more flexible, semantically-grounded conditioning than a raw AU vector for rare expressions.
- **Method:** Fine-tune an instruction-editing SD variant on paired (source, AU-edit, instruction) triples; generate instructions from AU deltas with an LLM (SynFER-style).
- **Metric:** Edit faithfulness, AU Δ, instruction adherence (CLIP), artifact rate.
- **Downstream:** Rare-expression synthesis where AUs are unconventional.

### Idea 11 — Prompt-to-Prompt cross-attention injection for localized edits
- **Hypothesis:** Re-injecting the *cross-attention maps* of the source into the edited generation localizes the change and prevents background/identity drift.
- **Method:** Replace attention maps for the unchanged tokens during the AU edit, keeping the AU-token maps; sweep the injection timesteps.
- **Metric:** Background SSIM, identity cosine, target-AU Δ vs off-target Δ.
- **Downstream:** Localized AU editing without full-face re-render.

### Idea 12 — AU-occurrence balancing + demographic diversification
- **Hypothesis:** Balancing AU *occurrence* (not just emotion class) and diversifying demographics yields more disentangled AU detectors with fewer co-activation shortcuts.
- **Method:** Choose edit targets to equalize per-AU occurrence; synthesize new identities via demographic acceptance–rejection (2602.19219).
- **Metric:** AU detector accuracy, co-activation shortcut rate, learning-curve vs. more labeled data.
- **Downstream:** AU detection (BP4D/DISFA); FER long-tail.

### Idea 13 — Pseudo-label rectifier for synthetic FER labels
- **Hypothesis:** Synthetic samples carry noisy labels; a rectifier that re-labels generated images (SynFER's pseudo-label generator) improves the *effective* supervision they provide.
- **Method:** Train a label rectifier (FER + AU) to correct/soften labels; compare training with raw vs. rectified synthetic labels.
- **Metric:** Label agreement with human review, downstream FER accuracy.
- **Downstream:** FER classification with synthetic data.

### Idea 14 — Classifier-free / classifier guidance with a frozen FER model
- **Hypothesis:** Guiding DDIM sampling with a frozen FER/AU classifier concentrates generated mass on the target emotion/AU, improving tail-cell coverage.
- **Method:** Classifier guidance $\nabla_{x_t}\log p_\phi(e\mid x_t)$ or CFG with an emotion-conditional adapter; sweep guidance scale.
- **Metric:** Target-emotion purity, AU target hit rate, diversity (coverage/FID).
- **Downstream:** Rare-emotion / rare-cell synthesis.

### Idea 15 — Disentanglement via orthogonality + contrastive regularization
- **Hypothesis:** Explicitly penalizing off-graph direction overlap ($\sum_{(k,l)\notin G}(w_k^\top w_l)^2$) and pulling same-AU samples together produces a cleaner, more linear $W$.
- **Method:** Add $\mathcal L_{\text{dis}}$ terms (orthogonality + co-activation graph) to the $W$ learning objective; ablate each term.
- **Metric:** Disentanglement gap, linearity (AUROC of probe vs. edit monotonicity), leakage.
- **Downstream:** AU-controlled generation quality.

### Idea 16 — ControlNet-style structural AU conditioning
- **Hypothesis:** Conditioning on a spatial AU *heatmap* (rather than a global vector) localizes the edit to the correct facial region and avoids mouth/eye cross-talk.
- **Method:** Add a ControlNet branch driven by the AU control-region mask/heatmap; freeze the base SD.
- **Metric:** Region-localization (edit stays in mask), off-region SSIM, target-AU Δ.
- **Downstream:** Region-specific AU editing (mouth vs eye cells).

## 3. Evaluation metrics

**Fidelity / realism**
- FID, KID — distribution similarity to real data.
- LPIPS, SSIM — perceptual/pixel distance to source (edit mode).
- FaceScore, HPSv2 — facial quality / human preference.

**AU control & disentanglement**
- AU intensity MAE / ICC — edited vs. target AU.
- Target-AU Δ vs. off-target-AU Δ — the **disentanglement gap** (target should move, others stay).
- AU detection F1 / occurrence F1 — AU-level correctness.
- Direction linearity / monotonicity — AU score vs. dose `‖Δz‖`.

**Label quality**
- FER accuracy on the synthetic set (self-consistency).
- AU detection accuracy on synthetic vs. pseudo-labels.
- Human-review agreement.

**Downstream (the real test)**
- FER: Accuracy/WAR, UAR, Macro-F1, rare-pair accuracy (AffectNet-7/8, RAF-DB).
- AU: detection & intensity accuracy (BP4D, DISFA).

## 4. Downstream tasks

1. **FER classification** — AffectNet-7/8, RAF-DB (sample-level imbalance).
2. **AU detection & intensity estimation** — BP4D, DISFA (AU-level imbalance).
3. **Long-tail / few-shot FER** — does synthetic data close the tail gap better than resampling/reweighting (E2)?
4. **Expression editing faithfulness** — can a human tell the edit is localized, identity-preserving, and anatomically plausible?

## 5. Open aspects & risks (watch these)

- **AU-estimator leakage:** OpenGraphAU both *creates* pseudo-labels and *scores* the gate; a second independent detector (LibreFace) and human review are needed to break the loop.
- **Text–AU conflict:** when the global text ("angry") and the AU vector ("smile") disagree, the model may produce a semantically ambiguous face — needs a conflict-handling policy.
- **Identity drift:** strong AU edits or mouth-region AUs tend to drift identity; gate 4 (dlib) only catches it post-hoc.
- **Dose comparability:** `‖Δz‖` scale differs per AU (×10 = 18.3 for AU5 vs 9.8 for AU9) — must report absolute doses, not multipliers.
- **Detector-in-the-loop:** a gate using a detector must run in that detector's own config and geometry, cross-checked on real images.
- **Compute:** the hybrid pipeline adds a frozen SD decoder + adapters on top of DiffAE; worth an explicit cost/latency budget.
