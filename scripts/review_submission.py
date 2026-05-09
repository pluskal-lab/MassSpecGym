"""
MassSpecGym submission review script.

Runs static analysis + LLM narrative review on a submissions/<method_name>/
folder and produces a structured JSON report plus a markdown summary.

Usage:
    python scripts/review_submission.py --submission submissions/MyModel
    python scripts/review_submission.py --submission submissions/MyModel --output review_report.json

Requires ANTHROPIC_API_KEY in environment for the LLM review step.
All other checks run without network access or GPU.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).parent.parent

REQUIRED_COLUMNS = {
    "de_novo": [
        "Method",
        "Top-1 Accuracy", "Top-1 Accuracy CI Low", "Top-1 Accuracy CI High",
        "Top-1 MCES", "Top-1 MCES CI Low", "Top-1 MCES CI High",
        "Top-1 Tanimoto", "Top-1 Tanimoto CI Low", "Top-1 Tanimoto CI High",
        "Top-10 Accuracy", "Top-10 Accuracy CI Low", "Top-10 Accuracy CI High",
        "Top-10 MCES", "Top-10 MCES CI Low", "Top-10 MCES CI High",
        "Top-10 Tanimoto", "Top-10 Tanimoto CI Low", "Top-10 Tanimoto CI High",
        "Paper", "DOI", "Comment", "Publication date",
    ],
    "retrieval": [
        "Method",
        "Hit rate @ 1", "Hit rate @ 1 CI Low", "Hit rate @ 1 CI High",
        "Hit rate @ 5", "Hit rate @ 5 CI Low", "Hit rate @ 5 CI High",
        "Hit rate @ 20", "Hit rate @ 20 CI Low", "Hit rate @ 20 CI High",
        "MCES @ 1", "MCES @ 1 CI Low", "MCES @ 1 CI High",
        "Paper", "DOI", "Comment", "Publication date",
    ],
    "simulation": [
        "Method",
        "Cosine Similarity", "Cosine Similarity CI Low", "Cosine Similarity CI High",
        "Jensen-Shannon Similarity", "Jensen-Shannon Similarity CI Low", "Jensen-Shannon Similarity CI High",
        "Hit rate @ 1", "Hit rate @ 1 CI Low", "Hit rate @ 1 CI High",
        "Hit rate @ 5", "Hit rate @ 5 CI Low", "Hit rate @ 5 CI High",
        "Hit rate @ 20", "Hit rate @ 20 CI Low", "Hit rate @ 20 CI High",
        "Paper", "DOI", "Comment", "Publication date",
    ],
}

METRIC_COLUMNS = {
    task: [c for c in cols if not any(x in c for x in ["CI Low", "CI High", "Method", "Paper", "DOI", "Comment", "Publication date"])]
    for task, cols in REQUIRED_COLUMNS.items()
}

TASK_CSV_MAP = {
    ("de_novo", "standard"): "results/de_novo.csv",
    ("de_novo", "bonus"): "results/de_novo_bonus.csv",
    ("retrieval", "standard"): "results/retrieval.csv",
    ("retrieval", "bonus"): "results/retrieval_bonus.csv",
    ("simulation", "standard"): "results/simulation.csv",
    ("simulation", "bonus"): "results/simulation_bonus.csv",
}

ORACLE_KEYWORDS = ["mist_cf", "mistcf", "mist-cf", "iceberg"]
MIST_ENCODER_KEYWORDS = ["mist", "MISTEncoder", "mist_retrieval", "MISTFingerprintRetrieval"]
MIST_BUG_PATTERN = re.compile(r"attn\s*\+=\s*attn_mask\b", re.IGNORECASE)
METRIC_OVERRIDE_PATTERN = re.compile(
    r"def\s+(on_batch_end|evaluate_retrieval_step|evaluate_de_novo_step|evaluate_mces_at_1)\s*\("
)
CUSTOM_SPLIT_PATTERN = re.compile(r"split_pth\s*=\s*['\"](?!None)[^'\"]+['\"]")
FORMULA_PREDICTOR_PATTERN = re.compile(
    r"(mist.cf|mistcf|formula.*predict|predict.*formula|mist_cf|MistCF|MIST.CF)", re.IGNORECASE
)


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    check_id: str
    status: Status
    message: str
    details: str = ""
    hard_fail: bool = False

    def is_blocking(self) -> bool:
        return self.hard_fail and self.status == Status.FAIL


@dataclass
class ReviewReport:
    submission_path: str
    method_name: str
    checks: list = field(default_factory=list)
    llm_review: Optional[str] = None
    llm_status: Optional[str] = None

    @property
    def blocked(self) -> bool:
        return any(c.is_blocking() for c in self.checks)

    @property
    def warnings(self) -> list:
        return [c for c in self.checks if c.status == Status.WARN]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blocked"] = self.blocked
        d["warning_count"] = len(self.warnings)
        return d

    def to_markdown(self) -> str:
        lines = []
        status_icon = "**BLOCKED**" if self.blocked else (
            "**WARNINGS**" if self.warnings else "**PASSED**"
        )
        lines.append(f"## MassSpecGym Submission Review: `{self.method_name}`")
        lines.append(f"\n**Overall status:** {status_icon}\n")

        hard_fails = [c for c in self.checks if c.is_blocking()]
        warnings = self.warnings
        passes = [c for c in self.checks if c.status == Status.PASS]
        skips = [c for c in self.checks if c.status == Status.SKIP]

        if hard_fails:
            lines.append("### Hard failures (must be resolved before merge)\n")
            for c in hard_fails:
                lines.append(f"- 🔴 **{c.check_id}**: {c.message}")
                if c.details:
                    lines.append(f"  ```\n{textwrap.indent(c.details, '  ')}\n  ```")

        if warnings:
            lines.append("\n### Warnings (require maintainer sign-off)\n")
            for c in warnings:
                lines.append(f"- 🟡 **{c.check_id}**: {c.message}")
                if c.details:
                    lines.append(f"  ```\n{textwrap.indent(c.details, '  ')}\n  ```")

        if passes:
            lines.append("\n### Passed checks\n")
            for c in passes:
                lines.append(f"- **{c.check_id}**: {c.message}")

        if skips:
            lines.append("\n### Skipped checks\n")
            for c in skips:
                lines.append(f"- **{c.check_id}**: {c.message}")

        if self.llm_review:
            lines.append(f"\n### LLM Narrative Review ({self.llm_status})\n")
            lines.append(self.llm_review)

        return "\n".join(lines)


def _check(report: ReviewReport, check_id: str, status: Status, message: str,
           details: str = "", hard_fail: bool = False) -> None:
    report.checks.append(CheckResult(check_id, status, message, details, hard_fail))


# ── Check implementations ────────────────────────────────────────────────────

def check_model_card_present(report: ReviewReport, submission_dir: Path) -> Optional[dict]:
    card_path = submission_dir / "model_card.yaml"
    if not card_path.exists():
        _check(report, "MC-PRESENT", Status.FAIL,
               "model_card.yaml not found in submission directory.",
               hard_fail=True)
        return None
    try:
        with open(card_path) as f:
            card = yaml.safe_load(f)
    except yaml.YAMLError as e:
        _check(report, "MC-PRESENT", Status.FAIL,
               "model_card.yaml could not be parsed.", str(e), hard_fail=True)
        return None
    _check(report, "MC-PRESENT", Status.PASS, "model_card.yaml found and parseable.")
    return card


def check_model_card_fields(report: ReviewReport, card: dict) -> None:
    required_top = ["method_name", "paper_url", "code_url", "results"]
    missing = [f for f in required_top if not card.get(f)]
    if missing:
        _check(report, "MC-FIELDS", Status.FAIL,
               f"Required model_card fields missing or empty: {missing}",
               hard_fail=True)
        return
    if not isinstance(card["results"], list) or len(card["results"]) == 0:
        _check(report, "MC-FIELDS", Status.FAIL,
               "model_card.yaml 'results' must be a non-empty list.", hard_fail=True)
        return
    for i, entry in enumerate(card["results"]):
        for f in ["task", "challenge"]:
            if not entry.get(f):
                _check(report, "MC-FIELDS", Status.FAIL,
                       f"results[{i}] missing required field '{f}'.", hard_fail=True)
                return
        if entry["task"] not in ("de_novo", "retrieval", "simulation"):
            _check(report, "MC-FIELDS", Status.FAIL,
                   f"results[{i}].task must be de_novo|retrieval|simulation, got '{entry['task']}'.",
                   hard_fail=True)
            return
        if entry["challenge"] not in ("standard", "bonus"):
            _check(report, "MC-FIELDS", Status.FAIL,
                   f"results[{i}].challenge must be standard|bonus, got '{entry['challenge']}'.",
                   hard_fail=True)
            return
        if not isinstance(entry.get("metrics"), dict) or not entry["metrics"]:
            _check(report, "MC-FIELDS", Status.FAIL,
                   f"results[{i}] missing required 'metrics' mapping.", hard_fail=True)
            return
        for required_meta in ["paper", "doi", "publication_date"]:
            if not entry.get(required_meta):
                _check(report, "MC-FIELDS", Status.FAIL,
                       f"results[{i}] missing required field '{required_meta}'.", hard_fail=True)
                return
    if card.get("random_seed") is None:
        _check(report, "MC-SEED", Status.WARN,
               "random_seed not specified in model_card.yaml. Document the seed used for reported results.")
    else:
        _check(report, "MC-SEED", Status.PASS, f"Random seed documented: {card['random_seed']}.")
    _check(report, "MC-FIELDS", Status.PASS, "All required model_card fields present.")


def check_metrics_in_card(report: ReviewReport, card: dict) -> None:
    method_name = card["method_name"]
    for i, entry in enumerate(card.get("results", [])):
        task = entry.get("task", "")
        challenge = entry.get("challenge", "")
        metrics = entry.get("metrics", {}) or {}
        check_id = f"METRICS-{i}"

        metric_cols = METRIC_COLUMNS.get(task, [])
        if not metric_cols:
            _check(report, check_id, Status.FAIL,
                   f"results[{i}]: unknown task '{task}'.", hard_fail=True)
            continue

        null_metrics = [c for c in metric_cols if metrics.get(c) is None]
        if null_metrics:
            _check(report, f"{check_id}-NULLS", Status.FAIL,
                   f"results[{i}]: missing or null metric values for Method='{method_name}': {null_metrics}",
                   hard_fail=True)
            continue

        missing_cis = []
        for mc in metric_cols:
            lo = metrics.get(f"{mc} CI Low")
            hi = metrics.get(f"{mc} CI High")
            if lo is None or hi is None:
                missing_cis.append(mc)
        if missing_cis:
            _check(report, f"{check_id}-CI", Status.WARN,
                   f"results[{i}]: missing 95% bootstrap CI values for: {missing_cis}. "
                   "CIs are required for leaderboard merge.")
            continue

        _check(report, check_id, Status.PASS,
               f"Metrics valid: task={task}, challenge={challenge}, method='{method_name}'.")


def check_tier_integrity(report: ReviewReport, card: dict) -> None:
    for i, entry in enumerate(card.get("results", [])):
        uses_formula = entry.get("uses_formula_at_inference", False)
        challenge = entry.get("challenge", "")
        task = entry.get("task", "")
        check_id = f"TIER-{i}"
        if uses_formula and challenge == "standard":
            _check(report, check_id, Status.FAIL,
                   f"results[{i}]: uses_formula_at_inference=true but challenge=standard. "
                   "Formula predictors used at inference must submit to the bonus tier.",
                   hard_fail=True)
        elif not uses_formula and challenge == "bonus":
            _check(report, check_id, Status.WARN,
                   f"results[{i}]: challenge=bonus but uses_formula_at_inference=false. "
                   "Confirm the bonus CSV is intentional.")
        else:
            _check(report, check_id, Status.PASS,
                   f"results[{i}]: tier integrity OK (task={task}, challenge={challenge}, "
                   f"uses_formula={uses_formula}).")


def check_pretraining(report: ReviewReport, card: dict) -> None:
    pre = card.get("pretraining", {}) or {}
    if not pre.get("used", False):
        _check(report, "PRE-DECLARED", Status.PASS, "No external pretraining declared.")
        return

    _check(report, "PRE-DECLARED", Status.PASS, "External pretraining declared.")

    criterion = pre.get("filtering_criterion", "")
    if not criterion:
        _check(report, "PRE-FILTER", Status.FAIL,
               "Pretraining declared but filtering_criterion not specified. "
               "Must be one of: exact_match | tanimoto_0.85 | tanimoto_0.70 | tanimoto_0.50 | none.",
               hard_fail=True)
    elif criterion == "none":
        _check(report, "PRE-FILTER", Status.FAIL,
               "Pretraining filtering_criterion=none. Results without any exclusion of test/val "
               "molecules cannot be accepted.", hard_fail=True)
    elif criterion == "exact_match":
        _check(report, "PRE-FILTER", Status.WARN,
               "Pretraining filtered by exact_match only. This is the minimum accepted criterion. "
               "Consider Tanimoto ≥ 0.70 for stronger generalization claims (see MSG v1.5 paper §4).")
    else:
        _check(report, "PRE-FILTER", Status.PASS,
               f"Pretraining filtering criterion: {criterion}.")

    inchikey_layer = pre.get("inchikey_layer_used", "")
    if inchikey_layer == "27char":
        _check(report, "PRE-INCHIKEY", Status.FAIL,
               "inchikey_layer_used=27char uses full InChIKey matching. If stereochemistry is stripped "
               "before InChI conversion (isomericSmiles=False), 27-char is functionally equivalent to "
               "14-char connectivity matching. Maintainer must verify stereo stripping is applied "
               "consistently in the pretraining pipeline.",
               hard_fail=True)
    elif inchikey_layer == "14char":
        _check(report, "PRE-INCHIKEY", Status.PASS, "InChIKey matching uses 14-char connectivity layer.")
    else:
        _check(report, "PRE-INCHIKEY", Status.WARN,
               "inchikey_layer_used not specified. Confirm 14-char (connectivity layer) was used.")

    parquet_url = pre.get("parquet_url", "")
    if parquet_url:
        _check(report, "PRE-SANITY", Status.WARN,
               f"Pretraining parquet declared at: {parquet_url}. "
               "Automated InChIKey overlap check requires local access; maintainer should run: "
               "python -m massspecgym.data.sanity_check --input <downloaded.parquet> --inchikey-col inchikey_14")
    else:
        _check(report, "PRE-SANITY", Status.WARN,
               "No pretraining parquet URL provided. Maintainer must verify data safety manually.")


def _clone_repo(url: str, tmpdir: str) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, tmpdir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return None
        return Path(tmpdir)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _collect_python_files(root: Path) -> list:
    return list(root.rglob("*.py"))


def _read_file_safe(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except Exception:
        return ""


def _find_local_repo(submission_dir: Path) -> Optional[Path]:
    """Return the first subdirectory of submission_dir that looks like a repo."""
    skip = {"__pycache__"}
    for child in sorted(submission_dir.iterdir()):
        if child.is_dir() and child.name not in skip and not child.name.startswith("."):
            return child
    return None


def check_code_repo(report: ReviewReport, card: dict, submission_dir: Path) -> Optional[Path]:
    # Prefer a local repo directory submitted alongside the model card.
    local_repo = _find_local_repo(submission_dir)
    if local_repo is not None:
        _check(report, "CODE-ACCESS", Status.PASS,
               f"Local repository found in submission: {local_repo.name}/")
        return local_repo

    code_url = card.get("code_url", "")
    if not code_url:
        _check(report, "CODE-ACCESS", Status.FAIL,
               "code_url not provided in model_card.yaml and no local repo directory found.",
               hard_fail=True)
        return None

    parsed = urlparse(code_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        _check(report, "CODE-ACCESS", Status.FAIL,
               f"code_url '{code_url}' does not look like a valid URL.", hard_fail=True)
        return None

    tmpdir = tempfile.mkdtemp(prefix="msgym_review_")
    repo_path = _clone_repo(code_url, tmpdir)
    if repo_path is None:
        _check(report, "CODE-ACCESS", Status.WARN,
               f"Could not clone code repository: {code_url}. "
               "Code-level checks skipped; maintainer must review manually.")
        return None

    _check(report, "CODE-ACCESS", Status.PASS, f"Code repository cloned: {code_url}.")
    return repo_path


def check_mist_batching_bug(report: ReviewReport, card: dict, repo_path: Optional[Path]) -> None:
    if not card.get("uses_mist_encoder", False):
        _check(report, "I1-MIST-BUG", Status.SKIP, "uses_mist_encoder=false, check skipped.")
        return
    if repo_path is None:
        _check(report, "I1-MIST-BUG", Status.FAIL,
               "uses_mist_encoder=true but repository not accessible. "
               "Cannot verify -inf attention mask fix. Provide accessible code repository.",
               hard_fail=True)
        return

    py_files = _collect_python_files(repo_path)
    buggy_files = []
    for p in py_files:
        src = _read_file_safe(p)
        if MIST_BUG_PATTERN.search(src):
            buggy_files.append(str(p.relative_to(repo_path)))

    if buggy_files:
        _check(report, "I1-MIST-BUG", Status.FAIL,
               "MIST batching bug detected: found 'attn += attn_mask' without -inf mask fill. "
               "This inflates Tanimoto similarity from 0.37 to 0.52 and de novo Top-1 by ~17 pp.",
               details="\n".join(buggy_files),
               hard_fail=True)
    else:
        _check(report, "I1-MIST-BUG", Status.PASS,
               "No MIST batching bug pattern found in repository.")


def check_metric_overrides(report: ReviewReport, repo_path: Optional[Path]) -> None:
    if repo_path is None:
        _check(report, "I2-METRIC-OVERRIDE", Status.SKIP, "Repository not accessible, check skipped.")
        return

    py_files = _collect_python_files(repo_path)
    flagged = []
    for p in py_files:
        src = _read_file_safe(p)
        for m in METRIC_OVERRIDE_PATTERN.finditer(src):
            rel = str(p.relative_to(repo_path))
            line_no = src[:m.start()].count("\n") + 1
            flagged.append(f"{rel}:{line_no}: {m.group(0)}")

    if flagged:
        _check(report, "I2-METRIC-OVERRIDE", Status.WARN,
               "Custom metric method overrides found. Verify these delegate to MassSpecGym parent ABCs "
               "and do not re-implement metric computation.",
               details="\n".join(flagged))
    else:
        _check(report, "I2-METRIC-OVERRIDE", Status.PASS,
               "No metric override patterns found.")


def check_official_split(report: ReviewReport, card: dict, repo_path: Optional[Path]) -> None:
    if not card.get("uses_official_split", True):
        _check(report, "I3-SPLIT", Status.WARN,
               "uses_official_split=false declared. Maintainer must verify custom split is scientifically justified.")
        return
    if repo_path is None:
        _check(report, "I3-SPLIT", Status.SKIP, "Repository not accessible, check skipped.")
        return

    py_files = _collect_python_files(repo_path)
    flagged = []
    for p in py_files:
        src = _read_file_safe(p)
        for m in CUSTOM_SPLIT_PATTERN.finditer(src):
            rel = str(p.relative_to(repo_path))
            line_no = src[:m.start()].count("\n") + 1
            flagged.append(f"{rel}:{line_no}: {m.group(0)}")

    if flagged:
        _check(report, "I3-SPLIT", Status.WARN,
               "Custom split_pth argument found. Confirm the official MassSpecGym split is used.",
               details="\n".join(flagged))
    else:
        _check(report, "I3-SPLIT", Status.PASS, "No custom split path detected.")


def check_smiles_canonicalization(report: ReviewReport) -> None:
    try:
        from rdkit import Chem
    except ImportError:
        _check(report, "S1-CANON", Status.SKIP, "RDKit not available, canonicalization check skipped.")
        return

    candidate_files = list((REPO_ROOT / "data").rglob("*candidates*")) + \
                      list((REPO_ROOT / "data").rglob("*retrieval*"))
    candidate_files = [f for f in candidate_files if f.suffix in (".csv", ".parquet", ".tsv")]

    if not candidate_files:
        _check(report, "S1-CANON", Status.SKIP,
               "No local candidate set files found; canonicalization check skipped. "
               "Maintainer should verify the submission uses the v1.5 pre-canonicalized candidate set.")
        return

    non_canonical_total = 0
    checked_total = 0
    for cf in candidate_files[:2]:
        try:
            df = pd.read_parquet(cf) if cf.suffix == ".parquet" else pd.read_csv(cf, sep=None, engine="python")
        except Exception:
            continue
        smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)
        if smiles_col is None:
            continue
        smiles_list = df[smiles_col].dropna().tolist()[:5000]
        for smi in smiles_list:
            checked_total += 1
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                if smi != Chem.MolToSmiles(mol):
                    non_canonical_total += 1
            except Exception:
                continue

    if checked_total == 0:
        _check(report, "S1-CANON", Status.SKIP,
               "Could not load candidate SMILES from local data files.")
        return

    if non_canonical_total > 0:
        _check(report, "S1-CANON", Status.FAIL,
               f"{non_canonical_total}/{checked_total} non-canonical SMILES found in candidate set. "
               "A spectrum-blind format classifier achieves >99% Recall@1 on non-canonicalized sets.",
               hard_fail=True)
    else:
        _check(report, "S1-CANON", Status.PASS,
               f"All {checked_total} sampled candidate SMILES are RDKit-canonical.")


def check_oracle_safety(report: ReviewReport, card: dict, repo_path: Optional[Path]) -> None:
    uses_mist_cf = card.get("uses_mist_cf", False)
    uses_iceberg = card.get("uses_iceberg", False)

    if not uses_mist_cf and not uses_iceberg:
        _check(report, "L3-ORACLE", Status.PASS, "No MIST-CF or ICEBERG oracles declared.")
        return

    issues = []
    if uses_mist_cf:
        ver = card.get("mist_cf_version", "")
        if "v1.5" not in str(ver) and "data_safe" not in str(ver):
            issues.append(
                f"uses_mist_cf=true but mist_cf_version='{ver}' does not confirm v1.5 data-safe version. "
                "The public MIST-CF checkpoint overlaps with the MSG test set."
            )
    if uses_iceberg:
        ver = card.get("iceberg_version", "")
        if "v1.5" not in str(ver) and "data_safe" not in str(ver):
            issues.append(
                f"uses_iceberg=true but iceberg_version='{ver}' does not confirm v1.5 data-safe version. "
                "ICEBERG trained on test molecules inflates downstream metrics (see MSG v1.5 §4.2)."
            )

    if repo_path is not None:
        py_files = _collect_python_files(repo_path)
        for p in py_files:
            src = _read_file_safe(p)
            for kw in ORACLE_KEYWORDS:
                if kw in src.lower():
                    issues.append(
                        f"Oracle keyword '{kw}' found in {p.relative_to(repo_path)}. "
                        "Confirm data-safe v1.5 version is used."
                    )
                    break

    if issues:
        _check(report, "L3-ORACLE", Status.WARN,
               "Oracle safety requires manual verification.",
               details="\n".join(issues))
    else:
        _check(report, "L3-ORACLE", Status.PASS, "Oracle versions declared as data-safe.")


def check_formula_leakage_in_code(report: ReviewReport, card: dict, repo_path: Optional[Path]) -> None:
    for i, entry in enumerate(card.get("results", [])):
        if entry.get("challenge") != "standard":
            continue
        if repo_path is None:
            _check(report, f"L4-FORMULA-{i}", Status.SKIP,
                   "Repository not accessible; formula leakage check skipped.")
            continue
        py_files = _collect_python_files(repo_path)
        flagged = []
        for p in py_files:
            src = _read_file_safe(p)
            if FORMULA_PREDICTOR_PATTERN.search(src):
                rel = str(p.relative_to(repo_path))
                flagged.append(rel)
        if flagged:
            _check(report, f"L4-FORMULA-{i}", Status.WARN,
                   f"results[{i}] (challenge=standard): formula predictor references found in code. "
                   "If formula is used to pre-filter candidates at inference, this must be submitted "
                   "to the bonus tier.",
                   details="\n".join(flagged))
        else:
            _check(report, f"L4-FORMULA-{i}", Status.PASS,
                   f"results[{i}]: no formula predictor usage detected for standard-tier submission.")


# ── LLM review ───────────────────────────────────────────────────────────────

def _fetch_arxiv_html(arxiv_url: str) -> Optional[str]:
    import urllib.request
    arxiv_id = None
    patterns = [
        r"arxiv\.org/abs/([0-9]{4}\.[0-9]+)",
        r"arxiv\.org/pdf/([0-9]{4}\.[0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, arxiv_url)
        if m:
            arxiv_id = m.group(1)
            break
    if arxiv_id is None:
        return None
    html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        req = urllib.request.Request(html_url, headers={"User-Agent": "MassSpecGym-review/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags crudely but keep text
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        return content[:60000]  # cap at 60k chars
    except Exception:
        return None


def _fetch_url_text(url: str) -> Optional[str]:
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MassSpecGym-review/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "pdf" in ct.lower():
                return None  # can't parse PDF here
            content = resp.read().decode("utf-8", errors="replace")
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        return content[:40000]
    except Exception:
        return None


def _fetch_github_readme(code_url: str) -> Optional[str]:
    import urllib.request
    m = re.match(r"https?://github\.com/([^/]+/[^/]+)", code_url)
    if not m:
        return None
    slug = m.group(1).rstrip("/")
    for branch in ("main", "master"):
        raw_url = f"https://raw.githubusercontent.com/{slug}/{branch}/README.md"
        try:
            req = urllib.request.Request(raw_url, headers={"User-Agent": "MassSpecGym-review/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="replace")[:10000]
        except Exception:
            continue
    return None


def _collect_repo_context(repo_path: Path, max_chars: int = 40000) -> str:
    """Read Python/config files from repo_path and return concatenated text, capped at max_chars."""
    # Prioritize files likely to contain model/eval logic
    PRIORITY_GLOBS = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.sh", "**/*.md"]
    SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}

    seen: set = set()
    files: list[tuple[int, Path]] = []  # (priority, path)

    for priority, glob in enumerate(PRIORITY_GLOBS):
        for p in repo_path.rglob(glob):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p in seen:
                continue
            seen.add(p)
            files.append((priority, p))

    files.sort(key=lambda x: (x[0], x[1]))

    parts = []
    total = 0
    for _, p in files:
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue
        rel = p.relative_to(repo_path)
        header = f"\n### {rel}\n"
        chunk = header + text
        if total + len(chunk) > max_chars:
            remaining = max_chars - total - len(header)
            if remaining > 200:
                parts.append(header + text[:remaining] + "\n[truncated]")
            break
        parts.append(chunk)
        total += len(chunk)

    return "".join(parts)


def run_llm_review(report: ReviewReport, card: dict, submission_dir: Path) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _check(report, "LLM-REVIEW", Status.SKIP,
               "ANTHROPIC_API_KEY not set; LLM review skipped.")
        return

    try:
        import anthropic
    except ImportError:
        _check(report, "LLM-REVIEW", Status.SKIP,
               "anthropic package not installed; LLM review skipped.")
        return

    sections = []
    sections.append("## Model card\n```yaml\n" + yaml.dump(card, default_flow_style=False) + "\n```")

    paper_url = card.get("paper_url", "")
    paper_text = None
    if paper_url:
        paper_text = _fetch_arxiv_html(paper_url) or _fetch_url_text(paper_url)
        if paper_text:
            sections.append(f"## Paper text (fetched from {paper_url}, truncated)\n{paper_text[:40000]}")
        else:
            sections.append(f"## Paper text\nCOULD NOT FETCH: {paper_url}")
            _check(report, "LLM-PAPER-ACCESS", Status.WARN,
                   f"Paper URL not accessible: {paper_url}. LLM review will proceed without paper text.")

    code_url = card.get("code_url", "")

    # Prefer local repo dir over fetching README from remote
    local_repo = _find_local_repo(submission_dir)
    if local_repo is not None:
        repo_context = _collect_repo_context(local_repo)
        sections.append(
            f"## Submitted repository code ({local_repo.name}/)\n{repo_context}"
        )
    else:
        readme_text = _fetch_github_readme(code_url) if code_url else None
        if readme_text:
            sections.append(f"## Code README (fetched from {code_url})\n{readme_text}")
        elif code_url:
            sections.append(f"## Code README\nCOULD NOT FETCH README from {code_url}")
            _check(report, "LLM-CODE-ACCESS", Status.WARN,
                   f"Could not fetch README from code repository: {code_url}.")

    auto_report_summary = "\n".join(
        f"- {c.check_id}: {c.status} — {c.message}" for c in report.checks
    )
    sections.append(f"## Automated check results so far\n{auto_report_summary}")

    context = "\n\n".join(sections)

    skill_md_path = REPO_ROOT / "skills" / "review" / "SKILL.md"
    if skill_md_path.exists():
        skill_md = skill_md_path.read_text()
        # Strip YAML frontmatter if present
        if skill_md.startswith("---"):
            end = skill_md.find("---", 3)
            skill_md = skill_md[end + 3:].lstrip() if end != -1 else skill_md
        system_prompt = (
            "You are an automated reviewer acting on behalf of the MassSpecGym maintainers. "
            "The following is the maintainer review guide — follow it as your instructions.\n\n"
            + skill_md
            + "\n\nYou have access to the submitted code (when provided). "
            "For every metric the submission reports, trace the actual computation path in the code "
            "and cite the specific file and line. If the code delegates to MassSpecGym parent ABCs "
            "without overriding, say so explicitly — that is the correct pattern. "
            "Do not use emoji in your response. Use plain text only."
        )
    else:
        system_prompt = (
            "You are an expert reviewer for the MassSpecGym benchmark. "
            "Identify data leakage, shortcut learning, metric implementation bugs, "
            "and tier integrity issues. Be specific and cite file:line when code is available. "
            "Do not use emoji in your response. Use plain text only."
        )

    user_prompt = f"""Review this MassSpecGym leaderboard submission for evaluation issues.

{context}

Provide a structured review with:
1. A one-sentence verdict (APPROVE / WARN / REJECT)
2. Metric implementation audit: for each reported metric, state whether the implementation matches the MassSpecGym spec (cite file:line if code is available), or flag the deviation
3. Other specific issues found, if any, with references to the paper/code evidence
4. Items that require human maintainer judgment
4. Items that look clean

Be concise. Flag real issues only — do not invent problems that are not evidenced. Do not use emoji."""

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )
        llm_text = response.content[0].text
        verdict_line = llm_text.split("\n")[0].upper()
        if "REJECT" in verdict_line:
            llm_status = "REJECT"
        elif "WARN" in verdict_line:
            llm_status = "WARN"
        else:
            llm_status = "APPROVE"
        report.llm_review = llm_text
        report.llm_status = llm_status
    except Exception as e:
        _check(report, "LLM-REVIEW", Status.WARN,
               f"LLM review call failed: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def review(submission_dir: Path) -> ReviewReport:
    method_name = submission_dir.name.replace("_", " ")
    report = ReviewReport(
        submission_path=str(submission_dir),
        method_name=method_name,
    )

    card = check_model_card_present(report, submission_dir)
    if card is None:
        return report

    if card.get("method_name"):
        report.method_name = card["method_name"]

    check_model_card_fields(report, card)
    check_metrics_in_card(report, card)
    check_tier_integrity(report, card)
    check_pretraining(report, card)
    check_smiles_canonicalization(report)

    repo_path = check_code_repo(report, card, submission_dir)

    check_mist_batching_bug(report, card, repo_path)
    check_metric_overrides(report, repo_path)
    check_official_split(report, card, repo_path)
    check_oracle_safety(report, card, repo_path)
    check_formula_leakage_in_code(report, card, repo_path)

    run_llm_review(report, card, submission_dir)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a MassSpecGym leaderboard submission.")
    parser.add_argument("--submission", required=True,
                        help="Path to submissions/<method_name>/ directory")
    parser.add_argument("--output", default="review_report.json",
                        help="Output path for JSON report (default: review_report.json)")
    parser.add_argument("--markdown", default=None,
                        help="Output path for markdown summary (default: stdout)")
    args = parser.parse_args()

    submission_dir = Path(args.submission)
    if not submission_dir.exists():
        print(f"ERROR: submission directory not found: {submission_dir}", file=sys.stderr)
        sys.exit(1)

    report = review(submission_dir)

    with open(args.output, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    md = report.to_markdown()
    if args.markdown:
        Path(args.markdown).write_text(md)
    else:
        print(md)

    if report.blocked:
        print(f"\nREVIEW BLOCKED: {sum(1 for c in report.checks if c.is_blocking())} hard failure(s).",
              file=sys.stderr)
        sys.exit(1)
    elif report.warnings:
        print(f"\nREVIEW PASSED WITH {len(report.warnings)} WARNING(S). Maintainer sign-off required.",
              file=sys.stderr)
        sys.exit(0)
    else:
        print("\nREVIEW PASSED.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
