"""SIRIUS candidate formula enumeration for MIST-CF."""

import math
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np
import pandas as pd


EL_STR_DEFAULT = "C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]"
EL_STR_EXPANDED = "C[0-]N[0-]O[0-]H[0-]S[0-27]P[0-30]I[0-8]Cl[0-33]F[0-54]Br[0-15]"

_ROUND = 4
_MAX_BATCH = 10000


def _get_sirius_path() -> str:
    path = os.environ.get("SIRIUS_PATH")
    if not path:
        raise RuntimeError(
            "SIRIUS_PATH environment variable is not set. Set it to the SIRIUS binary path."
        )
    return path


def _run_sirius_decomp(
    masses: str,
    output_path: str,
    adduct: Optional[str],
    ppm_tol: int,
    el_str: str,
    filter_: Optional[str],
    cores: int,
    verbose: bool,
) -> None:
    cmd = [
        _get_sirius_path(),
        "--cores",
        str(cores),
        "--log",
        "WARNING",
        "decomp",
        "--mass",
        masses,
        "--output",
        output_path,
        "--elements",
        el_str,
        "--ppm",
        str(ppm_tol),
    ]
    if adduct is not None:
        cmd.extend(["--ion", adduct])
    if filter_ is not None:
        cmd.extend(["--filter", filter_])

    result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr if result.stderr else ""
        raise RuntimeError(f"SIRIUS decomposition failed with code {result.returncode}: {stderr}")


def enumerate_candidates_sirius(
    precursor_mz: float,
    adduct: Optional[str] = None,
    ppm_tol: int = 15,
    el_str: str = EL_STR_DEFAULT,
    filter_: Optional[str] = "RDBE",
    cores: int = 1,
    verbose: bool = False,
) -> List[str]:
    """Enumerate candidate molecular formulas via SIRIUS decomposition."""
    mass = round(precursor_mz, _ROUND)

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _run_sirius_decomp(
            masses=str(mass),
            output_path=tmp_path,
            adduct=adduct,
            ppm_tol=ppm_tol,
            el_str=el_str,
            filter_=filter_,
            cores=cores,
            verbose=verbose,
        )
        df = pd.read_csv(tmp_path, sep="\t")
    finally:
        os.unlink(tmp_path)

    if "decompositions" not in df.columns or df.empty:
        return []
    decomps = df["decompositions"].values
    if len(decomps) == 0 or not isinstance(decomps[0], str):
        return []
    return [c for c in decomps[0].strip().split(",") if c]


def enumerate_candidates_sirius_batch(
    masses: List[float],
    adduct: Optional[str] = None,
    ppm_tol: int = 15,
    el_str: str = EL_STR_DEFAULT,
    filter_: Optional[str] = "RDBE",
    cores: int = 16,
    verbose: bool = False,
) -> dict:
    """Enumerate formulas for many precursor masses in batched SIRIUS calls."""
    unique_masses = sorted(set(round(m, _ROUND) for m in masses))
    mass_to_forms: dict = {}
    num_batches = math.ceil(len(unique_masses) / _MAX_BATCH)

    for batch_masses in np.array_split(unique_masses, num_batches):
        mass_list = ",".join(str(m) for m in batch_masses)
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            _run_sirius_decomp(
                masses=mass_list,
                output_path=tmp_path,
                adduct=adduct,
                ppm_tol=ppm_tol,
                el_str=el_str,
                filter_=filter_,
                cores=cores,
                verbose=verbose,
            )
            df = pd.read_csv(tmp_path, sep="\t")
        finally:
            os.unlink(tmp_path)

        if "m/z" not in df.columns or "decompositions" not in df.columns:
            continue
        for _, row in df.iterrows():
            mz = round(float(row["m/z"]), _ROUND)
            decomp = row["decompositions"]
            if isinstance(decomp, str) and decomp.strip():
                mass_to_forms[mz] = decomp.strip().split(",")
            else:
                mass_to_forms[mz] = []

    return mass_to_forms
