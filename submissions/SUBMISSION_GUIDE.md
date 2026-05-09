# MassSpecGym Leaderboard Submission Guide

To submit a new result to the MassSpecGym leaderboard, open a pull request against `main` that contains **one thing only**:

- A `submissions/<method_name>/model_card.yaml` with your metrics embedded

Do **not** edit `results/*.csv` directly. Those files are regenerated automatically from model cards on every merge to main.

An automated review runs immediately on your PR and posts a report as a comment. A maintainer performs final human review before merging.

---

## What belongs in this PR

### 1. Model card with metrics

Create `submissions/<method_name>/model_card.yaml`. Copy and fill in [`submissions/template/model_card.yaml`](template/model_card.yaml).

`<method_name>` must exactly match the `method_name` field in the card (spaces replaced with underscores in the folder name, e.g. `"My Model"` → `submissions/My_Model/model_card.yaml`).

Each `results` entry must include a `metrics` mapping with all required metric keys for its task. Use the exact key names below.

**De novo (standard and bonus) — required metric keys:**

```yaml
metrics:
  Top-1 Accuracy: 0.00
  Top-1 Accuracy CI Low: 0.00
  Top-1 Accuracy CI High: 0.00
  Top-1 MCES: 0.00
  Top-1 MCES CI Low: 0.00
  Top-1 MCES CI High: 0.00
  Top-1 Tanimoto: 0.00
  Top-1 Tanimoto CI Low: 0.00
  Top-1 Tanimoto CI High: 0.00
  Top-10 Accuracy: 0.00
  Top-10 Accuracy CI Low: 0.00
  Top-10 Accuracy CI High: 0.00
  Top-10 MCES: 0.00
  Top-10 MCES CI Low: 0.00
  Top-10 MCES CI High: 0.00
  Top-10 Tanimoto: 0.00
  Top-10 Tanimoto CI Low: 0.00
  Top-10 Tanimoto CI High: 0.00
```

**Retrieval (standard and bonus) — required metric keys:**

```yaml
metrics:
  Hit rate @ 1: 0.00
  Hit rate @ 1 CI Low: 0.00
  Hit rate @ 1 CI High: 0.00
  Hit rate @ 5: 0.00
  Hit rate @ 5 CI Low: 0.00
  Hit rate @ 5 CI High: 0.00
  Hit rate @ 20: 0.00
  Hit rate @ 20 CI Low: 0.00
  Hit rate @ 20 CI High: 0.00
  MCES @ 1: 0.00
  MCES @ 1 CI Low: 0.00
  MCES @ 1 CI High: 0.00
```

**Simulation (standard and bonus) — required metric keys:**

```yaml
metrics:
  Cosine Similarity: 0.00
  Cosine Similarity CI Low: 0.00
  Cosine Similarity CI High: 0.00
  Jensen-Shannon Similarity: 0.00
  Jensen-Shannon Similarity CI Low: 0.00
  Jensen-Shannon Similarity CI High: 0.00
  Hit rate @ 1: 0.00
  Hit rate @ 1 CI Low: 0.00
  Hit rate @ 1 CI High: 0.00
  Hit rate @ 5: 0.00
  Hit rate @ 5 CI Low: 0.00
  Hit rate @ 5 CI High: 0.00
  Hit rate @ 20: 0.00
  Hit rate @ 20 CI Low: 0.00
  Hit rate @ 20 CI High: 0.00
```

**All CIs are mandatory.** Use 95% bootstrap CIs computed with the standard MassSpecGym bootstrapper (`ReturnScalarBootStrapper` in `massspecgym/utils.py`). Results without CIs will be rejected.

**Tier integrity:** If your model uses any formula predictor (e.g., MIST-CF) at inference time to pre-filter candidates, set `uses_formula_at_inference: true` and `challenge: bonus`. Submitting a formula-assisted model to `challenge: standard` is grounds for rejection.

### 2. Code repository (optional but recommended)

You may include your model's source code directly in the PR by placing it under:

```
submissions/<method_name>/<repo_name>/
```

For example:

```
submissions/My_Model/
  model_card.yaml
  my-model-repo/        <- your code here
    train.py
    eval.py
    ...
```

When a local repository directory is present, the automated review will:
- Run all static code checks (MIST bug, metric overrides, formula leakage, split detection) against the local files
- Include the full source in the LLM narrative review instead of only the README

If no local directory is provided, the review falls back to cloning `code_url` from the model card. Including local code is strongly encouraged — it enables a more thorough automated review and reduces the burden on human maintainers.

---

## Leaderboard update on merge

When a PR is merged to `main`, the `update_leaderboard` workflow runs `scripts/generate_results_csvs.py`, which:

1. Reads all `submissions/*/model_card.yaml` files
2. Replaces or inserts the corresponding rows in `results/*.csv`
3. Commits the updated CSVs back to `main`

Baseline rows (Random, DeepSets, etc.) have no model card and are never touched by this process.

---

## Metric specifications

All metrics must be computed using the pinned MassSpecGym implementations:

| Metric | Specification |
|--------|--------------|
| InChIKey hit rate | First 14 characters (connectivity layer) only |
| Tanimoto similarity | Morgan ECFP4, radius=2, 2048 bits |
| MCES distance | `threshold=15`, `always_stronger_bound=True` (see `massspecgym/utils.py:MyopicMCES`) |
| Cosine similarity | Standard MS/MS cosine as implemented in `massspecgym` |
| Jensen-Shannon similarity | As implemented in `massspecgym` |

Do not re-implement these metrics. Use the parent ABC evaluation methods. Any override of `on_batch_end()` or custom metric computation outside the MassSpecGym parent classes must be explicitly justified.

---

## What the automated review checks

On every PR, a GitHub Action runs [`scripts/review_submission.py`](../scripts/review_submission.py) and posts a structured report. It checks:

- Model card present and all required fields filled
- All required metrics present in the card's `metrics` block, with non-null values and CIs
- Task/tier consistency (standard vs. bonus)
- SMILES canonicalization: 0 non-RDKit-canonical entries in the candidate set
- Pretraining InChIKey overlap (if pretraining data declared): runs `massspecgym.data.sanity_check`
- Oracle data safety (if MIST-CF or ICEBERG used): checks v1.5 data-safe versions referenced
- MIST batching bug: if MIST encoder used, scans linked code for the `-inf` mask fix
- Metric override: scans for custom `on_batch_end` / `evaluate_*` reimplementations
- Official data split used (no custom `split_pth`)
- LLM narrative review of paper and code (fetches arXiv/PDF if accessible)

**Hard failures block merge.** Warnings require explicit maintainer sign-off.

---

## Data safety requirements

### Pretraining data

If your model uses any external pretraining data (decoder pretraining, encoder pretraining, or fine-tuning on data outside MassSpecGym):

- Declare it in `model_card.yaml` under `pretraining`
- Specify the filtering criterion (`exact_match` or `tanimoto_0.70` recommended)
- Provide a publicly accessible parquet/CSV if possible

The MassSpecGym leaderboard accepts results trained with at least exact-match exclusion. We recommend Tanimoto ≥ 0.70 filtering for results claimed to reflect genuine generalization (see the MassSpecGym v1.5 paper for motivation).

### Oracle components

If you use MIST-CF or ICEBERG as auxiliary components, you must use the data-safe v1.5 versions provided in `massspecgym/models/oracles/`. Using externally trained versions of these models may introduce transitive data leakage.

---

## Checklist before opening PR

- [ ] `submissions/<method_name>/model_card.yaml` filled in completely
- [ ] All required metrics and CIs present in the `metrics` block
- [ ] `task` and `challenge` correct for each result entry
- [ ] `paper`, `doi`, and `publication_date` filled in each result entry
- [ ] Paper URL is accessible (arXiv, DOI, or preprint link)
- [ ] Code repository URL is public and accessible
- [ ] If pretraining data used: declared in model card with filtering criterion
- [ ] If MIST-CF or ICEBERG used: v1.5 data-safe version confirmed
- [ ] Evaluation run with official MassSpecGym data split (no custom `split_pth`)
- [ ] Random seed documented
- [ ] No changes to `results/*.csv` (regenerated automatically on merge)
