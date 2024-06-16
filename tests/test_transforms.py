import numpy as np
import torch
import matchms
from massspecgym.data.transforms import SpecTokenizer, SpecBinner


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
            [0.01082, 0.01064, 0.17184, 0.1397, 0.00874, 1.0, 0.52842, 0.00793, 0.43696]
        ),
        metadata={"precursor_mz": 406.22},
    )

    tokenizer = SpecTokenizer(n_peaks=60)
    spec_t = tokenizer(spec)
    assert spec_t.shape == (60, 2)
    print(spec_t[: spec.peaks.mz.shape[0], 0])
    print(torch.from_numpy(spec.peaks.mz))
    assert (
        spec_t[: spec.peaks.mz.shape[0], 0] == torch.from_numpy(spec.peaks.mz)
    ).all()
    assert (
        spec_t[: spec.peaks.intensities.shape[0], 1]
        == torch.from_numpy(spec.peaks.intensities)
    ).all()


def test_spec_binner():
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
        intensities=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.1, 0.2]),
        metadata={"precursor_mz": 406.22},
    )
    binner = SpecBinner(max_mz=1000, bin_width=100, to_rel_intensities=False)
    spec_t = binner(spec)
    assert spec_t.shape == (1000 // 100,)
    assert torch.allclose(
        spec_t,
        torch.tensor([0.2, 0.3, 1.2, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]).double(),
    )
