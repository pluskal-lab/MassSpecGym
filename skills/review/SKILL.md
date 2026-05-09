---
name: review
description: Maintainer guide for reviewing MassSpecGym leaderboard submissions. Covers interpreting the automated review report, performing judgment calls the automation cannot make, and approving or rejecting PRs. The automated review (scripts/review_submission.py, triggered by the review_submission GH Action) handles deterministic checks; this guide covers the residual human review.
---

# MassSpecGym Submission Review — Maintainer Guide

## Role of this guide

Every leaderboard PR triggers `scripts/review_submission.py` automatically and posts a structured report as a PR comment. That report handles all deterministic checks (schema, CIs, tier integrity, MIST bug, pretraining filter, metric overrides). **Your job as maintainer is to:**

1. Read the automated report and triage any WARNINGs that require judgment
2. Fetch and read the paper if the automated LLM review flagged concerns or couldn't access the paper
3. Check items the automation explicitly cannot verify (listed below)
4. Approve or request changes

Hard failures in the automated report must be resolved by the author before you even look at the PR. Do not override hard failures without a documented reason.

---

## Submission requirements (what a valid PR must contain)

1. A new row in the correct `results/*.csv` with all required metrics and 95% bootstrap CIs
2. A `submissions/<method_name>/model_card.yaml` filled from the template

The method name in the CSV `Method` column must exactly match `method_name` in `model_card.yaml` (underscored folder name, spaced method name). See `submissions/SUBMISSION_GUIDE.md`.

---

## Step 1: Read the automated report

The PR comment from the review bot contains:

- **Hard failures** — must be fixed; CI blocks merge
- **Warnings** — require your sign-off; use a PR review comment to document your decision
- **LLM narrative review** — treat as a second opinion; verify any specific issues it flags

For warnings, document your reasoning inline on the PR before approving.

---

## Step 2: Things the automation cannot check — do these manually

### 2a. Paper methods section vs. model card

Read the paper's methods/data section and cross-check against `model_card.yaml`:

- Does the paper describe using MIST-CF at inference to pre-filter candidates in the **mass-based** (standard) challenge? If yes and the model is submitted to `results/retrieval.csv` (not bonus), reject.
- Does the paper describe pretraining on data sources *not listed* in the model card? Flag the discrepancy.
- Does the paper use ICEBERG-generated spectra for pretraining? Check which ICEBERG version (data-safe v1.5 or upstream). If unclear, ask the author.
- Does the paper's reported number match the CSV entry? Values that differ from the paper by >0.5 pp on the primary metric need explanation.

### 2b. Spectrum-blind shortcut check (S2)

If the model is a retrieval model, mentally check: could a model that ignores the spectrum entirely (ranking purely by PubChem deposition frequency or SMILES format) achieve similar results?

The PubChem frequency-prior baseline achieves >90% Recall@1 on non-corrected datasets. If the submission's Recall@1 is at or below this level, it may not be learning from spectra at all. Ask the author to compare against the frequency-prior baseline if not already included.

**Spectrum-blind classifier test (run if suspicious):** A rule-based RDKit format check (`smiles != Chem.MolToSmiles(Chem.MolFromSmiles(smiles))`) achieves >99% Recall@1 on non-canonicalized datasets. If the candidate set for this submission is non-standard, ask the author to confirm the v1.5 pre-canonicalized candidate set is used.

If you have access to compute: train or run a version of the model with the spectral input zeroed out or permuted. If performance stays substantially above random (~1/pool_size), the model likely exploits a spurious correlation and should be investigated further before acceptance.

### 2c. Pretraining data — if no parquet provided

If `pretraining.used=true` but no parquet URL is given, ask the author to either:
- Provide a public parquet for the InChIKey overlap check, or
- Run `python -m massspecgym.data.sanity_check --input their_data.parquet --inchikey-col inchikey_14` themselves and share the output

Do not accept claims of data safety without this check if pretraining is declared.

### 2d. Reproducibility spot-check

If the submission includes a checkpoint or inference script, spot-check one metric:

```bash
# In the massspecgym conda environment (Python 3.11)
git clone <author-fork-url>
cd MassSpecGym
pip install -e ".[dev]"
python scripts/run.py \
    --job_key review_run \
    --run_name <model_name>_review \
    --test_only \
    --no_wandb \
    --seed 42
```

A discrepancy >0.5 pp on the primary metric requires explanation. If the author cannot provide a reproducible checkpoint, note this in your review but it is not grounds for automatic rejection — reproducibility is aspirational for submissions without public checkpoints.

---

## Step 3: Metric specifications (reference)

All submitted metrics must match these pinned implementations:

| Metric | Specification |
|--------|--------------|
| InChIKey hit rate | First 14 characters (connectivity layer) only — **not** full 27-char |
| Tanimoto similarity | Morgan ECFP4, radius=2, 2048 bits |
| MCES distance | `threshold=15`, `always_stronger_bound=True` — see `massspecgym/utils.py:MyopicMCES` |
| Cosine similarity | Standard MS/MS cosine as in `massspecgym` |
| Jensen-Shannon similarity | As in `massspecgym` |

**Do not accept** metric implementations reimported from third-party codebases unless independently verified against the specifications above.

---

## Step 4: Required metrics per task

### De novo (standard and bonus)
`Top-1 Accuracy`, `Top-1 MCES`, `Top-1 Tanimoto`, `Top-10 Accuracy`, `Top-10 MCES`, `Top-10 Tanimoto` — all required, all with CIs.

### Retrieval (standard)
`Hit rate @ 1`, `Hit rate @ 5`, `Hit rate @ 20`, `MCES @ 1` — all required, all with CIs.

### Retrieval (bonus)
`Hit rate @ 1`, `Hit rate @ 5`, `Hit rate @ 20`, `MCES @ 1` — all required, all with CIs.

### Simulation (standard and bonus)
`Cosine Similarity`, `Jensen-Shannon Similarity`, `Hit rate @ 1`, `Hit rate @ 5`, `Hit rate @ 20` — all required, all with CIs.

Standard and bonus variants must not be compared in the same table row. A model submitted to both must have two separate CSV rows.

---

## Step 5: MIST batching bug (I1) — background

If a submission uses the MIST encoder from an external codebase (not `massspecgym/models/encoders/mist/`), verify the attention masking in batched inference. The original MIST encoder added `attn += attn_mask` without first converting the boolean mask to `-inf`, which allows padding tokens to contribute to the softmax. The v1.5 MassSpecGym MIST encoder has this fixed.

**Impact if not fixed:** Tanimoto similarity inflates from 0.37 to 0.52; de novo Top-1 inflates by ~17 pp; retrieval Top-1 inflates from 10.73% to 28.50%. This completely reorders the de novo leaderboard.

**What to look for:**
```python
# BUG (wrong):
attn += attn_mask

# SUBTLE BUG — fix code present but result discarded:
new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
new_attn_mask.masked_fill_(attn_mask, float("-inf"))
attn += attn_mask   # <-- still uses raw bool mask, new_attn_mask silently ignored

# FIX (correct):
new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
new_attn_mask.masked_fill_(attn_mask, float("-inf"))
attn += new_attn_mask
```

The v1.5 MassSpecGym MIST encoder (`massspecgym/models/encoders/mist/transformer_layer.py`) already applies the fix. If the submission imports from there, this is satisfied.

---

## Step 6: Data leakage — gradient of stringency

Data leakage is not binary. The steepest performance gradient appears at the transition from exact-match exclusion to Tanimoto ≥ 0.70–0.80 filtering — not at the exact-match boundary itself. This means a model may claim "clean" data (exact-match excluded) while still benefiting substantially from near-neighbor contamination.

**What to report in your review:**
- What filtering criterion is declared in the model card?
- Is this consistent with what the paper states?
- Does the paper compare against baselines at the same filtering level?

The leaderboard accepts exact-match exclusion as the minimum. If the criterion is weaker or unstated, request clarification before approval.

---

## Final approval checklist

Before approving the PR, confirm:

- [ ] Automated report has no hard failures (or failures have been resolved and re-reviewed)
- [ ] All warnings have been signed off with a documented reason
- [ ] Paper methods section consistent with model card
- [ ] No implicit formula use in standard-tier submission (from reading the paper)
- [ ] Pretraining data safety confirmed (parquet sanity check run or author confirmed)
- [ ] If MIST encoder used: v1.5 fix confirmed or external repo verified
- [ ] If MIST-CF or ICEBERG used: v1.5 data-safe version confirmed
- [ ] All required metrics present with CIs
- [ ] Method name in CSV matches model card exactly
- [ ] `df_test_path` configured so per-sample predictions are saved (recommended)
- [ ] Seed documented
