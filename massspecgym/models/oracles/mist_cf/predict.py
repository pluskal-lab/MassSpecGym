"""
MIST-CF formula prediction API.

High-level interface for predicting chemical formulas from MS/MS spectra.

Pipeline:
1. Enumerate candidate formulas via SIRIUS (or accept a custom list).
2. Optionally pre-filter with FastFFN to reduce candidates.
3. Assign subformulae to spectrum peaks for each candidate.
4. Score each candidate with MistCFNet.
5. Return ranked FormulaCandidate results.

Usage:
    from massspecgym.models.oracles.mist_cf import predict_formulas

    results = predict_formulas(
        spectrum_mzs=[91.05, 125.02, 246.11],
        spectrum_intensities=[0.25, 1.0, 0.73],
        precursor_mz=288.12,
        adduct="[M+H]+",
        checkpoint="path/to/mist_cf.ckpt",
        top_k=10,
    )
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from massspecgym.models.encoders.mist.chem_constants import (
    formula_to_dense,
    ion_to_mass,
    VALID_MONO_MASSES,
)
from massspecgym.data.subformulae import assign_subformulae_single

logger = logging.getLogger(__name__)

# Peak type constants — must match mist_cf_data.py in ~/mist-cf
_CLS_TYPE = 1
_FRAG_TYPE = 0


@dataclass
class FormulaCandidate:
    """A ranked formula candidate from MIST-CF prediction."""
    formula: str
    adduct: str
    score: float
    parentmass: float


def _norm_ppm(ppm: float) -> float:
    """Normalize PPM to the scale the model was trained on (divide by 10)."""
    return ppm / 10.0


def _cls_ppm(formula: str, adduct: str, parentmass: float) -> float:
    """Normalized PPM between candidate mass (with adduct) and precursor mass."""
    dense = formula_to_dense(formula)
    cand_mass = float(dense.dot(VALID_MONO_MASSES)) + ion_to_mass.get(adduct, 0.0)
    raw_ppm = abs(cand_mass - parentmass) / max(parentmass, 200.0) * 1e6
    return _norm_ppm(raw_ppm)


def _build_model_inputs(
    formulas: List[str],
    spectrum: np.ndarray,
    adduct: str,
    parentmass: float,
    max_subpeak: int = 10,
) -> dict:
    """Build MistCFNet input tensors from a list of candidate formulas.

    Args:
        formulas: Candidate formula strings.
        spectrum: (N, 2) array of [mz, intensity], normalized to max=1.
        adduct: Adduct string.
        parentmass: Neutral mass (precursor m/z minus adduct mass offset).
        max_subpeak: Max fragment subpeaks per candidate.

    Returns:
        Dict with keys: num_peaks, peak_types, form_vec, intens, rel_mass_diffs.
        ion_vec and instrument_vec are omitted (not used by default model config).
    """
    entries = []
    for formula in formulas:
        cls_vec = formula_to_dense(formula)
        cls_ppm = _cls_ppm(formula, adduct, parentmass)

        subform = assign_subformulae_single(formula, spectrum, adduct, mass_diff_thresh=15.0)
        tbl = subform.get("output_tbl")

        form_vecs = [cls_vec]
        peak_types = [_CLS_TYPE]
        intens = [1.0]
        rel_diffs = [cls_ppm]

        if tbl and tbl.get("formula"):
            for f, inten, ppm in zip(
                tbl["formula"][:max_subpeak],
                tbl["ms2_inten"][:max_subpeak],
                tbl["mass_diff"][:max_subpeak],
            ):
                form_vecs.append(formula_to_dense(f))
                peak_types.append(_FRAG_TYPE)
                intens.append(float(inten))
                rel_diffs.append(_norm_ppm(float(ppm)))

        entries.append({
            "form_vecs": np.array(form_vecs, dtype=np.float32),
            "peak_types": np.array(peak_types, dtype=np.int64),
            "intens": np.array(intens, dtype=np.float32),
            "rel_diffs": np.array(rel_diffs, dtype=np.float32),
            "n": len(form_vecs),
        })

    N = len(entries)
    max_len = max(e["n"] for e in entries)
    formula_dim = entries[0]["form_vecs"].shape[1]

    form_vec_t = torch.zeros(N, max_len, formula_dim)
    peak_types_t = torch.zeros(N, max_len, dtype=torch.long)
    intens_t = torch.zeros(N, max_len)
    rel_diffs_t = torch.zeros(N, max_len)
    num_peaks_t = torch.zeros(N, dtype=torch.long)

    for i, e in enumerate(entries):
        n = e["n"]
        form_vec_t[i, :n] = torch.from_numpy(e["form_vecs"])
        peak_types_t[i, :n] = torch.from_numpy(e["peak_types"])
        intens_t[i, :n] = torch.from_numpy(e["intens"])
        rel_diffs_t[i, :n] = torch.from_numpy(e["rel_diffs"])
        num_peaks_t[i] = n

    return {
        "num_peaks": num_peaks_t,
        "peak_types": peak_types_t,
        "form_vec": form_vec_t,
        "intens": intens_t,
        "rel_mass_diffs": rel_diffs_t,
    }


def predict_formulas(
    spectrum_mzs: Union[np.ndarray, list],
    spectrum_intensities: Union[np.ndarray, list],
    precursor_mz: float,
    adduct: str = "[M+H]+",
    top_k: int = 10,
    checkpoint: Optional[str] = None,
    model: Optional["MistCFNet"] = None,
    candidates: Optional[List[str]] = None,
    fast_filter_model: Optional["FastFFN"] = None,
    fast_filter_max_k: int = 256,
    ppm_tol: int = 15,
    el_str: Optional[str] = None,
    max_subpeak: int = 10,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> List[FormulaCandidate]:
    """Predict chemical formulas from an MS/MS spectrum.

    Args:
        spectrum_mzs: Array of m/z values.
        spectrum_intensities: Array of intensity values.
        precursor_mz: Precursor m/z.
        adduct: Adduct type.
        top_k: Number of top results to return.
        checkpoint: Path to MistCFNet checkpoint. Ignored if ``model`` is provided.
        model: Pre-loaded MistCFNet. If neither model nor checkpoint is given,
            candidates are ranked by number of assigned subformulae only.
        candidates: Custom candidate formula list. If provided, skips SIRIUS.
            Supports SIRIUS-generated, BUDDY, or any other candidate set.
        fast_filter_model: Loaded FastFFN for pre-filtering. Applied before
            neural scoring when provided.
        fast_filter_max_k: Max candidates to keep after fast filter.
        ppm_tol: PPM tolerance for SIRIUS (ignored when candidates is given).
        el_str: SIRIUS element constraint string (uses default if None).
        max_subpeak: Max subpeaks per candidate for the transformer.
        device: Torch device for inference.
        batch_size: Batch size for neural scoring.

    Returns:
        List of FormulaCandidate sorted by score descending.

    Raises:
        RuntimeError: If SIRIUS_PATH is not set and candidates is None.
    """
    from massspecgym.models.oracles.mist_cf.model import MistCFNet
    from massspecgym.models.oracles.mist_cf.fast_filter import fast_filter_candidates
    from massspecgym.models.oracles.mist_cf.sirius import (
        enumerate_candidates_sirius, EL_STR_DEFAULT,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mzs = np.asarray(spectrum_mzs, dtype=np.float64)
    intensities = np.asarray(spectrum_intensities, dtype=np.float64)
    if intensities.max() > 0:
        intensities = intensities / intensities.max()
    spectrum = np.column_stack([mzs, intensities])

    adduct_offset = ion_to_mass.get(adduct, 0.0)
    parentmass = precursor_mz - adduct_offset

    # --- Candidate enumeration ---
    if candidates is not None:
        cands = list(candidates)
        logger.info(f"Using {len(cands)} provided candidates.")
    else:
        el = el_str or EL_STR_DEFAULT
        logger.info(f"Running SIRIUS decomposition (ppm={ppm_tol}, elements={el}).")
        cands = enumerate_candidates_sirius(
            precursor_mz=precursor_mz,
            adduct=adduct,
            ppm_tol=ppm_tol,
            el_str=el,
        )
        logger.info(f"SIRIUS returned {len(cands)} candidates.")

    if not cands:
        return []

    # --- Fast filter ---
    if fast_filter_model is not None and len(cands) > fast_filter_max_k:
        logger.info(f"Fast-filtering {len(cands)} → {fast_filter_max_k} candidates.")
        keep_idx = fast_filter_candidates(
            cands, fast_filter_model, fast_filter_max_k, device=device
        )
        cands = [cands[i] for i in keep_idx]
        logger.info(f"After fast filter: {len(cands)} candidates.")

    # --- Load model if checkpoint given ---
    if model is None and checkpoint is not None:
        model = MistCFNet.from_pretrained(checkpoint)
        logger.info(f"Loaded MistCFNet from {checkpoint}")

    # --- Neural scoring ---
    if model is not None:
        model = model.to(device).eval()
        scores = []
        for start in range(0, len(cands), batch_size):
            batch_cands = cands[start: start + batch_size]
            inputs = _build_model_inputs(
                batch_cands, spectrum, adduct, parentmass, max_subpeak
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_scores = model(
                    num_peaks=inputs["num_peaks"],
                    peak_types=inputs["peak_types"],
                    form_vec=inputs["form_vec"],
                    ion_vec=None,
                    instrument_vec=None,
                    intens=inputs["intens"],
                    rel_mass_diffs=inputs["rel_mass_diffs"],
                )
            scores.extend(batch_scores.cpu().tolist())
    else:
        # Fallback: rank by number of assigned subformulae minus normalized ppm
        scores = []
        for formula in cands:
            subform = assign_subformulae_single(
                formula, spectrum, adduct, mass_diff_thresh=15.0
            )
            tbl = subform.get("output_tbl")
            n_assigned = len(tbl["formula"]) if tbl and tbl.get("formula") else 0
            cls_p = _cls_ppm(formula, adduct, parentmass)
            scores.append(n_assigned - cls_p)

    results = []
    for formula, score in zip(cands, scores):
        dense = formula_to_dense(formula)
        pmass = float(dense.dot(VALID_MONO_MASSES))
        results.append(FormulaCandidate(
            formula=formula,
            adduct=adduct,
            score=float(score),
            parentmass=pmass,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]
