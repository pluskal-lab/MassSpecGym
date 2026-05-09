"""
Regenerate results/*.csv from submissions/*/model_card.yaml.

Baseline rows (those with no model card) are preserved unchanged.
Rows for methods that have a model card are replaced by the card's metrics.
Run on every merge to main via the update_leaderboard workflow.

Usage:
    python scripts/generate_results_csvs.py [--dry-run]
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

REPO_ROOT = Path(__file__).parent.parent

TASK_CSV_MAP = {
    ("de_novo", "standard"): REPO_ROOT / "results" / "de_novo.csv",
    ("de_novo", "bonus"):    REPO_ROOT / "results" / "de_novo_bonus.csv",
    ("retrieval", "standard"): REPO_ROOT / "results" / "retrieval.csv",
    ("retrieval", "bonus"):    REPO_ROOT / "results" / "retrieval_bonus.csv",
    ("simulation", "standard"): REPO_ROOT / "results" / "simulation.csv",
    ("simulation", "bonus"):    REPO_ROOT / "results" / "simulation_bonus.csv",
}

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

KNOWN_BASELINES = {
    "Random", "Random chemical generation", "Precursor m/z",
    "DeepSets", "Fingerprint FFN", "DeepSets + Fourier features",
    "FFN Fingerprint", "SMILES Transformer", "SELFIES Transformer",
    "GNN", "FraGNNet",
}


def load_model_cards() -> list[dict]:
    """Return list of (card, card_path) for all non-template model cards."""
    cards = []
    submissions_dir = REPO_ROOT / "submissions"
    for card_path in sorted(submissions_dir.rglob("model_card.yaml")):
        if "template" in card_path.parts:
            continue
        try:
            with open(card_path) as f:
                card = yaml.safe_load(f)
            if card and card.get("method_name") and card.get("results"):
                cards.append((card, card_path))
        except Exception as e:
            print(f"WARNING: could not parse {card_path}: {e}", file=sys.stderr)
    return cards


def card_to_csv_row(method_name: str, entry: dict, task: str) -> dict:
    """Build a CSV row dict from a model card result entry."""
    metrics = entry.get("metrics", {}) or {}
    row = {"Method": method_name}
    for col in REQUIRED_COLUMNS[task]:
        if col == "Method":
            continue
        if col in ("Paper", "DOI", "Comment", "Publication date"):
            key_map = {
                "Paper": "paper",
                "DOI": "doi",
                "Comment": "comment",
                "Publication date": "publication_date",
            }
            row[col] = entry.get(key_map[col], "") or ""
        else:
            row[col] = metrics.get(col, "")
    return row


def generate(dry_run: bool = False) -> bool:
    cards = load_model_cards()

    # Build index: (task, challenge) -> {method_name -> row_dict}
    new_rows: dict[tuple, dict[str, dict]] = {k: {} for k in TASK_CSV_MAP}
    errors = []

    for card, card_path in cards:
        method_name = card["method_name"]
        for i, entry in enumerate(card.get("results", [])):
            task = entry.get("task", "")
            challenge = entry.get("challenge", "")
            key = (task, challenge)
            if key not in TASK_CSV_MAP:
                errors.append(f"{card_path}: results[{i}] unknown task/challenge '{task}/{challenge}'")
                continue
            new_rows[key][method_name] = card_to_csv_row(method_name, entry, task)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return False

    ok = True
    for (task, challenge), csv_path in TASK_CSV_MAP.items():
        if not csv_path.exists():
            print(f"WARNING: {csv_path} does not exist, skipping.", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)
        columns = REQUIRED_COLUMNS[task]

        # Ensure all required columns exist (add missing ones as empty)
        for col in columns:
            if col not in df.columns:
                df[col] = ""

        # Baseline rows: keep as-is (by method name in KNOWN_BASELINES or no card)
        carded_methods = set(new_rows[(task, challenge)].keys())
        baseline_mask = df["Method"].apply(
            lambda m: str(m) in KNOWN_BASELINES or str(m) not in carded_methods
        )
        # Drop existing rows for carded methods (they will be replaced)
        df_baselines = df[baseline_mask].copy()

        # Build new rows dataframe
        new_rows_list = list(new_rows[(task, challenge)].values())
        if new_rows_list:
            df_new = pd.DataFrame(new_rows_list, columns=columns)
        else:
            df_new = pd.DataFrame(columns=columns)

        df_out = pd.concat([df_baselines[columns], df_new], ignore_index=True)

        if dry_run:
            print(f"[dry-run] {csv_path.name}: {len(df_baselines)} baseline rows + {len(df_new)} card rows")
        else:
            df_out.to_csv(csv_path, index=False)
            print(f"Written {csv_path.name}: {len(df_baselines)} baseline + {len(df_new)} card rows")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate results CSVs from model cards.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying files.")
    args = parser.parse_args()
    success = generate(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
