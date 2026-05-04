"""MIST-CF formula prediction API."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from massspecgym.models.encoders.mist.chem_constants import (
    ELEMENT_TO_MASS,
    ELECTRON_MASS,
    ion_to_mass,
    VALID_MONO_MASSES,
    formula_to_dense,
)
from massspecgym.data.subformulae import assign_subformulae_single

logger = logging.getLogger(__name__)

_CLS_TYPE = 1
_FRAG_TYPE = 0


@dataclass
class FormulaCandidate:
    """A ranked formula candidate from MIST-CF prediction."""
    formula: str
    adduct: str
    score: float
    parentmass: float


def _neutral_mass_from_precursor(precursor_mz: float, adduct: str) -> float:
    adduct_mass = ion_to_mass.get(adduct, ELEMENT_TO_MASS["H"] - ELECTRON_MASS)
    return precursor_mz - adduct_mass


def _formula_mass(formula: str) -> float:
    return float(formula_to_dense(formula).dot(VALID_MONO_MASSES))


def _norm_ppm(ppm: float) -> float:
    """Normalize ppm to the scale used by the MIST-CF model."""
    return ppm / 10.0


def _cls_ppm(formula: str, neutral_mass: float) -> float:
    """Normalized ppm between candidate neutral mass and query neutral mass."""
    cand_mass = _formula_mass(formula)
    raw_ppm = abs(cand_mass - neutral_mass) / max(neutral_mass, 200.0) * 1e6
    return _norm_ppm(raw_ppm)


def enumerate_candidate_formulas(
    precursor_mz: float,
    adduct: str = "[M+H]+",
    ppm_tol: float = 10.0,
    max_candidates: int = 500,
) -> List[str]:
    """Enumerate candidate molecular formulas from a precursor mass.

    Uses a simple combinatorial enumeration over common organic elements
    (C, H, N, O, S, P, F, Cl, Br) filtered by mass tolerance.

    This is a pure-Python fallback for SIRIUS decomposition.

    Args:
        precursor_mz: Precursor m/z value.
        adduct: Adduct type string.
        ppm_tol: PPM tolerance for mass matching.
        max_candidates: Maximum number of candidates to return.

    Returns:
        List of molecular formula strings.
    """
    neutral_mass = _neutral_mass_from_precursor(precursor_mz, adduct)

    if neutral_mass <= 0 or neutral_mass > 2000:
        return []

    mass_tol = neutral_mass * ppm_tol * 1e-6

    element_masses = {
        "C": VALID_MONO_MASSES[0],   # 12.0
        "H": VALID_MONO_MASSES[1],   # 1.008
        "N": VALID_MONO_MASSES[11],  # 14.003
        "O": VALID_MONO_MASSES[13],  # 15.995
        "S": VALID_MONO_MASSES[15],  # 31.972
        "P": VALID_MONO_MASSES[14],  # 30.974
    }

    candidates = []
    max_c = min(int(neutral_mass / element_masses["C"]) + 1, 100)

    for nc in range(0, max_c + 1):
        mass_c = nc * element_masses["C"]
        if mass_c > neutral_mass + mass_tol:
            break
        remaining = neutral_mass - mass_c
        max_n = min(int(remaining / element_masses["N"]) + 1, 20)

        for nn in range(0, max_n + 1):
            mass_cn = mass_c + nn * element_masses["N"]
            if mass_cn > neutral_mass + mass_tol:
                break
            remaining2 = neutral_mass - mass_cn
            max_o = min(int(remaining2 / element_masses["O"]) + 1, 30)

            for no in range(0, max_o + 1):
                mass_cno = mass_cn + no * element_masses["O"]
                if mass_cno > neutral_mass + mass_tol:
                    break
                remaining3 = neutral_mass - mass_cno

                for ns in range(0, min(3, int(remaining3 / element_masses["S"]) + 1)):
                    mass_cnos = mass_cno + ns * element_masses["S"]
                    if mass_cnos > neutral_mass + mass_tol:
                        break
                    remaining4 = neutral_mass - mass_cnos

                    nh_approx = remaining4 / element_masses["H"]
                    nh = round(nh_approx)
                    if nh < 0:
                        continue

                    total_mass = mass_cnos + nh * element_masses["H"]
                    ppm_diff = abs(total_mass - neutral_mass) / neutral_mass * 1e6

                    if ppm_diff <= ppm_tol:
                        rdbe = nc - nh / 2 + nn / 2 + 1
                        if rdbe >= -0.5 and nh <= 2 * nc + nn + 2:
                            parts = []
                            if nc > 0: parts.append(f"C{nc}" if nc > 1 else "C")
                            if nh > 0: parts.append(f"H{nh}" if nh > 1 else "H")
                            if nn > 0: parts.append(f"N{nn}" if nn > 1 else "N")
                            if no > 0: parts.append(f"O{no}" if no > 1 else "O")
                            if ns > 0: parts.append(f"S{ns}" if ns > 1 else "S")
                            formula = "".join(parts)
                            if formula:
                                candidates.append(formula)

                    if len(candidates) >= max_candidates:
                        return candidates

    return candidates


def _build_model_inputs(
    formulas: List[str],
    spectrum: np.ndarray,
    adduct: str,
    neutral_mass: float,
    max_subpeak: int = 10,
) -> dict:
    """Build MistCFNet inputs for spectrum-conditioned candidate scoring."""
    entries = []
    for formula in formulas:
        cls_vec = formula_to_dense(formula)
        cls_ppm = _cls_ppm(formula, neutral_mass)

        subform = assign_subformulae_single(
            formula, spectrum, adduct, mass_diff_thresh=15.0
        )
        tbl = subform.get("output_tbl")

        form_vecs = [cls_vec]
        peak_types = [_CLS_TYPE]
        intens = [1.0]
        rel_diffs = [cls_ppm]

        if tbl and tbl.get("formula"):
            for frag_formula, frag_inten, frag_ppm in zip(
                tbl["formula"][:max_subpeak],
                tbl["ms2_inten"][:max_subpeak],
                tbl["mass_diff"][:max_subpeak],
            ):
                form_vecs.append(formula_to_dense(frag_formula))
                peak_types.append(_FRAG_TYPE)
                intens.append(float(frag_inten))
                rel_diffs.append(_norm_ppm(float(frag_ppm)))

        entries.append({
            "form_vecs": np.asarray(form_vecs, dtype=np.float32),
            "peak_types": np.asarray(peak_types, dtype=np.int64),
            "intens": np.asarray(intens, dtype=np.float32),
            "rel_diffs": np.asarray(rel_diffs, dtype=np.float32),
            "n": len(form_vecs),
        })

    batch_size = len(entries)
    max_len = max(e["n"] for e in entries)
    formula_dim = entries[0]["form_vecs"].shape[1]

    form_vec_t = torch.zeros(batch_size, max_len, formula_dim)
    peak_types_t = torch.zeros(batch_size, max_len, dtype=torch.long)
    intens_t = torch.zeros(batch_size, max_len)
    rel_diffs_t = torch.zeros(batch_size, max_len)
    num_peaks_t = torch.zeros(batch_size, dtype=torch.long)

    for i, entry in enumerate(entries):
        n_peaks = entry["n"]
        form_vec_t[i, :n_peaks] = torch.from_numpy(entry["form_vecs"])
        peak_types_t[i, :n_peaks] = torch.from_numpy(entry["peak_types"])
        intens_t[i, :n_peaks] = torch.from_numpy(entry["intens"])
        rel_diffs_t[i, :n_peaks] = torch.from_numpy(entry["rel_diffs"])
        num_peaks_t[i] = n_peaks

    return {
        "num_peaks": num_peaks_t,
        "peak_types": peak_types_t,
        "form_vec": form_vec_t,
        "intens": intens_t,
        "rel_mass_diffs": rel_diffs_t,
    }


def _enumerate_candidates(
    precursor_mz: float,
    adduct: str,
    ppm_tol: float,
    max_candidates: int,
    use_sirius: bool,
    el_str: Optional[str],
) -> List[str]:
    if use_sirius:
        try:
            from massspecgym.models.oracles.mist_cf.sirius import (
                EL_STR_DEFAULT,
                enumerate_candidates_sirius,
            )

            candidates = enumerate_candidates_sirius(
                precursor_mz=precursor_mz,
                adduct=adduct,
                ppm_tol=int(ppm_tol),
                el_str=el_str or EL_STR_DEFAULT,
            )
            if candidates:
                return candidates[:max_candidates]
        except Exception as exc:
            logger.warning("SIRIUS formula enumeration failed; using Python fallback: %s", exc)

    return enumerate_candidate_formulas(
        precursor_mz=precursor_mz,
        adduct=adduct,
        ppm_tol=ppm_tol,
        max_candidates=max_candidates,
    )


def predict_formulas(
    spectrum_mzs: Union[np.ndarray, list],
    spectrum_intensities: Union[np.ndarray, list],
    precursor_mz: float,
    adduct: str = "[M+H]+",
    top_k: int = 10,
    checkpoint: Optional[str] = None,
    instrument: str = "unknown",
    ppm_tol: float = 10.0,
    model: Optional["MistCFNet"] = None,
    candidates: Optional[List[str]] = None,
    fast_filter_model: Optional["FastFFN"] = None,
    fast_filter_max_k: int = 256,
    max_candidates: int = 500,
    max_subpeak: int = 10,
    el_str: Optional[str] = None,
    use_sirius: bool = True,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> List[FormulaCandidate]:
    """Predict chemical formulas from an MS/MS spectrum using MIST-CF.

    Pipeline:
    1. Enumerate candidate formulas from precursor mass.
    2. Assign subformulae to the spectrum for each candidate.
    3. Score with MistCFNet (if model/checkpoint provided).
    4. Return top-k candidates ranked by score.

    Args:
        spectrum_mzs: Array of m/z values.
        spectrum_intensities: Array of intensity values.
        precursor_mz: Precursor m/z value.
        adduct: Adduct type.
        top_k: Number of top candidates to return.
        checkpoint: Path to MIST-CF checkpoint (for model loading).
        instrument: Instrument type string. Reserved for checkpoint compatibility.
        ppm_tol: PPM tolerance for formula enumeration.
        model: Pre-loaded MistCFNet (skips checkpoint loading if provided).
        candidates: Optional pre-enumerated candidate formula list.
        fast_filter_model: Optional FastFFN model for formula-only pre-filtering.
        fast_filter_max_k: Maximum candidate count after FastFFN filtering.
        max_candidates: Maximum count for fallback Python enumeration.
        max_subpeak: Maximum assigned fragment peaks per candidate.
        el_str: Optional SIRIUS element constraint string.
        use_sirius: Try SIRIUS enumeration before the Python fallback.
        device: Torch device for model inference.
        batch_size: Candidate scoring batch size.

    Returns:
        List of FormulaCandidate objects, sorted by score (descending).
    """
    from massspecgym.models.oracles.mist_cf.model import MistCFNet

    mzs = np.asarray(spectrum_mzs, dtype=np.float64)
    intensities = np.asarray(spectrum_intensities, dtype=np.float64)
    spectrum = np.column_stack([mzs, intensities])

    if intensities.max() > 0:
        spectrum[:, 1] = spectrum[:, 1] / spectrum[:, 1].max()

    neutral_mass = _neutral_mass_from_precursor(precursor_mz, adduct)

    if candidates is None:
        candidates = _enumerate_candidates(
            precursor_mz=precursor_mz,
            adduct=adduct,
            ppm_tol=ppm_tol,
            max_candidates=max_candidates,
            use_sirius=use_sirius,
            el_str=el_str,
        )
    else:
        candidates = list(candidates)

    if not candidates:
        return []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fast_filter_model is not None and len(candidates) > fast_filter_max_k:
        from massspecgym.models.oracles.mist_cf.fast_filter import fast_filter_candidates

        keep_idx = fast_filter_candidates(
            candidates,
            fast_filter_model,
            fast_filter_max_k,
            device=device,
        )
        candidates = [candidates[i] for i in keep_idx]

    if model is None and checkpoint is not None:
        model = MistCFNet.from_pretrained(checkpoint)

    if model is not None:
        model = model.to(device).eval()
        scores = []
        for start in range(0, len(candidates), batch_size):
            batch_candidates = candidates[start:start + batch_size]
            inputs = _build_model_inputs(
                batch_candidates,
                spectrum,
                adduct,
                neutral_mass,
                max_subpeak=max_subpeak,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                batch_scores = model(
                    num_peaks=inputs["num_peaks"],
                    peak_types=inputs["peak_types"],
                    form_vec=inputs["form_vec"],
                    ion_vec=None,
                    instrument_vec=None,
                    intens=inputs["intens"],
                    rel_mass_diffs=inputs["rel_mass_diffs"],
                ).squeeze(-1)
            scores.extend(float(score) for score in batch_scores.cpu())
    else:
        scores = []
        for formula in candidates:
            subform = assign_subformulae_single(
                formula, spectrum, adduct, mass_diff_thresh=15.0
            )
            tbl = subform.get("output_tbl")
            n_assigned = len(tbl["formula"]) if tbl and tbl.get("formula") else 0
            scores.append(n_assigned - _cls_ppm(formula, neutral_mass))

    results = []
    for formula, score in zip(candidates, scores):
        mass = _formula_mass(formula)
        results.append(FormulaCandidate(
            formula=formula,
            adduct=adduct,
            score=float(score),
            parentmass=mass,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]
