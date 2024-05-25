import numpy as np
import matchms
from massspecgym.transforms import SpecTokenizer


def test_spec_tokenizer():
    spec = matchms.Spectrum(
        mz=np.array(
            [
                45.134823,
                45.13699,
                110.096245,
                130.064972,
                136.111862,
                277.16571,
                289.165924,
                307.177856,
                406.223083,
            ]
        ),
        intensities=np.array(
            [
                0.01082,
                0.01064,
                0.17184,
                0.1397,
                0.00874,
                1.0,
                0.52842,
                0.00793,
                0.43696
            ]
        ),
        metadata={"precursor_mz": 406.22},
    )

    tokenizer = SpecTokenizer(n_peaks=60)
    spec_np = tokenizer(spec)
    assert spec_np.shape == (60, 2)
    assert (spec_np[:spec.peaks.mz.shape[0], 0] == spec.peaks.mz).all()
    assert (spec_np[:spec.peaks.intensities.shape[0], 1] == spec.peaks.intensities).all()
