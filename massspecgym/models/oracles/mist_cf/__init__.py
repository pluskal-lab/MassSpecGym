"""
MIST-CF oracle: Chemical formula prediction from MS/MS spectra.

Components:
- MistCFNet: Neural scoring model (FormulaTransformer + Linear head).
- FastFFN: Formula pre-filter model for fast candidate pruning.
- predict_formulas(): High-level API for formula prediction.
- enumerate_candidates_sirius(): SIRIUS-based candidate enumeration.
- fast_filter_candidates(): Pre-filter a candidate list with FastFFN.

Adapted from https://github.com/samgoldman97/mist-cf.
"""

from .model import MistCFNet, MistCFFormulaTransformer
from .fast_filter import FastFFN, fast_filter_candidates
from .predict import predict_formulas, FormulaCandidate
from .sirius import enumerate_candidates_sirius, EL_STR_DEFAULT, EL_STR_EXPANDED
