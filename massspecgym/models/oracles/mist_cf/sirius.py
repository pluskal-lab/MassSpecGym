"""
SIRIUS-based formula candidate enumeration for MIST-CF.

Wraps the SIRIUS CLI ``decomp`` command to enumerate candidate molecular
formulas from a precursor mass. Requires the SIRIUS binary to be available
at the path given by the ``$SIRIUS_PATH`` environment variable.
"""

import math
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np
import pandas as pd

# Default element constraints matching ~/mist-cf EL_STR_DEFAULT
EL_STR_DEFAULT = "C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]"
EL_STR_EXPANDED = "C[0-]N[0-]O[0-]H[0-]S[0-27]P[0-30]I[0-8]Cl[0-33]F[0-54]Br[0-15]"

_ROUND = 4
_MAX_BATCH = 10000


def _get_sirius_path() -> str:
    path = os.environ.get("SIRIUS_PATH")
    if not path:
        raise RuntimeError(
            "SIRIUS_PATH environment variable is not set. "
            "Set it to the path of the SIRIUS binary, e.g.:\n"
            "  export SIRIUS_PATH=/path/to/sirius/bin/sirius"
        )
    return path


def enumerate_candidates_sirius(
    precursor_mz: float,
    adduct: Optional[str] = None,
    ppm_tol: int = 15,
    el_str: str = EL_STR_DEFAULT,
    filter_: str = "RDBE",
    cores: int = 1,
    verbose: bool = False,
) -> List[str]:
    """Enumerate candidate molecular formulas via SIRIUS decomposition.

    Args:
        precursor_mz: Precursor m/z value (neutral mass if adduct is None).
        adduct: Adduct type string (e.g. "[M+H]+"). If None, treats precursor_mz
            as neutral mass.
        ppm_tol: PPM tolerance for mass matching.
        el_str: SIRIUS element constraint string.
        filter_: SIRIUS filter mode ("RDBE", "NONE", etc.).
        cores: Number of CPU cores for SIRIUS.
        verbose: Print SIRIUS command.

    Returns:
        List of formula strings sorted by closeness to the precursor mass.

    Raises:
        RuntimeError: If SIRIUS_PATH is not set.
    """
    sirius = _get_sirius_path()
    mass = round(precursor_mz, _ROUND)

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = (
            f"{sirius} --cores {cores} --log WARNING decomp"
            f" --mass {mass}"
            f" --output {tmp_path}"
            f" --elements {el_str}"
            f" --ppm {ppm_tol}"
        )
        if adduct is not None:
            cmd += f" --ion {adduct}"
        if filter_ is not None:
            cmd += f" --filter {filter_}"

        if verbose:
            print(f"SIRIUS cmd: {cmd}")

        result = subprocess.run(cmd, shell=True, capture_output=not verbose)
        if result.returncode != 0 and verbose:
            print(f"SIRIUS stderr: {result.stderr.decode()}")

        df = pd.read_csv(tmp_path, sep="\t")
    finally:
        os.unlink(tmp_path)

    decomps = df["decompositions"].values
    if len(decomps) == 0 or not isinstance(decomps[0], str):
        return []

    candidates = decomps[0].strip().split(",") if isinstance(decomps[0], str) else []
    candidates = [c for c in candidates if c]
    return candidates


def enumerate_candidates_sirius_batch(
    masses: List[float],
    adduct: Optional[str] = None,
    ppm_tol: int = 15,
    el_str: str = EL_STR_DEFAULT,
    filter_: str = "RDBE",
    cores: int = 16,
    verbose: bool = False,
) -> dict:
    """Enumerate candidates for a batch of masses in a single SIRIUS call.

    More efficient than calling enumerate_candidates_sirius per spectrum.

    Returns:
        Dict mapping rounded mass → list of formula strings.
    """
    sirius = _get_sirius_path()
    unique_masses = sorted(set(round(m, _ROUND) for m in masses))

    mass_to_forms: dict = {}
    num_batches = math.ceil(len(unique_masses) / _MAX_BATCH)
    for batch_masses in np.array_split(unique_masses, num_batches):
        mass_list = ",".join(str(m) for m in batch_masses)

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = (
                f"{sirius} --cores {cores} --log WARNING decomp"
                f" --mass {mass_list}"
                f" --output {tmp_path}"
                f" --elements {el_str}"
                f" --ppm {ppm_tol}"
            )
            if adduct is not None:
                cmd += f" --ion {adduct}"
            if filter_ is not None:
                cmd += f" --filter {filter_}"

            if verbose:
                print(f"SIRIUS batch cmd: {cmd}")

            result = subprocess.run(cmd, shell=True, capture_output=not verbose)
            if result.returncode != 0 and verbose:
                print(f"SIRIUS stderr: {result.stderr.decode()}")

            df = pd.read_csv(tmp_path, sep="\t")
        finally:
            os.unlink(tmp_path)

        for _, row in df.iterrows():
            mz = round(float(row["m/z"]), _ROUND)
            decomp = row["decompositions"]
            if isinstance(decomp, str) and decomp.strip():
                mass_to_forms[mz] = decomp.strip().split(",")
            else:
                mass_to_forms[mz] = []

    return mass_to_forms
