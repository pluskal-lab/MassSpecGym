"""
MIST-CF oracle: Chemical formula prediction from MS/MS spectra.

Components:
- MistCFNet: Neural scoring model (FormulaTransformer + Linear head).
- FastFFN: Formula pre-filter model for candidate pruning.
- predict_formulas(): High-level API for formula prediction.
- enumerate_candidate_formulas(): Pure-Python fallback formula enumeration.
- enumerate_candidates_sirius(): Optional SIRIUS formula enumeration.

Adapted from external MIST-CF implementations.
"""

from .predict import predict_formulas, enumerate_candidate_formulas, FormulaCandidate
from .model import MistCFNet, MistCFFormulaTransformer
from .fast_filter import FastFFN, fast_filter_candidates
from .sirius import enumerate_candidates_sirius, EL_STR_DEFAULT, EL_STR_EXPANDED
