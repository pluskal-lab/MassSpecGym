import argparse

import matchms
import numpy as np
import pytest
import torch

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class _MetricProbeRetrieval(RetrievalMassSpecGymModel):
    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        raise NotImplementedError

    def _update_metric(self, *args, **kwargs):
        return None


def test_retrieval_metrics_include_mrr_and_filtered_samples():
    model = _MetricProbeRetrieval(myopic_mces_kwargs={"threshold": 15})
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.1])
    labels = torch.tensor([False, True, False, True, False])
    batch_ptr = torch.tensor([3, 2])

    vals = model.evaluate_retrieval_step(
        scores,
        labels,
        batch_ptr,
        stage=Stage.TEST,
        processable_mask=torch.tensor([True, False]),
    )

    assert vals["test_hit_rate@1"].tolist() == [1.0, 0.0]
    assert vals["test_mrr"].tolist() == [1.0, 0.0]


def test_retrieval_mces_filtered_samples_are_thresholded():
    model = _MetricProbeRetrieval(myopic_mces_kwargs={"threshold": 15})
    model.myopic_mces = lambda true, pred: 0

    vals = model.evaluate_mces_at_1(
        scores=torch.tensor([0.1, 0.9, 0.8, 0.1]),
        labels=torch.tensor([False, True, True, False]),
        smiles=["CCO", "CCN"],
        candidates_smiles=["CCC", "CCO", "CCN", "CCC"],
        batch_ptr=torch.tensor([2, 2]),
        stage=Stage.TEST,
        processable_mask=torch.tensor([True, False]),
    )

    assert vals["test_mces@1"].tolist() == [0, 15]


def test_mist_peak_formula_tokenizer_marks_unprocessable_without_dropping():
    from massspecgym.data.transforms import MISTPeakFormulaTokenizer

    spec = matchms.Spectrum(
        mz=np.array([50.0, 75.0]),
        intensities=np.array([0.4, 1.0]),
        metadata={"precursor_mz": 100.0, "identifier": "x"},
    )
    item = MISTPeakFormulaTokenizer(n_peaks=4)(spec)

    assert item["processable_mask"].item() is False
    assert item["form_vec"].shape == (5, 18)
    assert item["types"].shape == (5,)
    assert item["num_peaks"].item() == 1


def test_fp2mol_dataset_emits_diffms_graph_tensors():
    from massspecgym.data.fp2mol_dataset import FP2MolDataset

    ds = FP2MolDataset(["CCO"], exclude_inchikeys=False, fp_bits=2048, graph_max_nodes=8)
    item = ds[0]
    batch = FP2MolDataset.collate_fn([item])

    assert item["fingerprint"].shape == (2048,)
    assert item["X"].shape == (8, 8)
    assert item["E"].shape == (8, 8, 5)
    assert item["node_mask"].sum().item() == 3
    assert batch["X"].shape == (1, 8, 8)


def test_run_challenge_validation_formula_only_and_iceberg():
    import scripts.run as run_script

    formula_only = argparse.Namespace(
        task="retrieval",
        model="mist_fingerprint",
        challenge="mass",
        training_mode="spec2mol",
        candidates_pth=None,
    )
    with pytest.raises(ValueError):
        run_script._validate_challenge(formula_only)

    iceberg = argparse.Namespace(
        task="retrieval",
        model="iceberg_retrieval",
        challenge="mass",
        training_mode="spec2mol",
        candidates_pth=None,
    )
    run_script._validate_challenge(iceberg)


def test_iceberg_retrieval_uses_spectrum_cosine(monkeypatch):
    from massspecgym.models.retrieval.iceberg_retrieval import IcebergRetrieval

    class DummyIceberg(torch.nn.Module):
        def predict_mol(self, smi, **kwargs):
            if smi == "match":
                return {"spec": [{"mz": 100.0, "intensity": 1.0}]}
            return {"spec": [{"mz": 300.0, "intensity": 1.0}]}

    model = IcebergRetrieval(num_bins=1000, mz_max=1000, myopic_mces_kwargs={"threshold": 15})
    monkeypatch.setattr(model, "_get_iceberg", lambda: DummyIceberg())

    batch = {
        "spec": torch.tensor([[[100.0, 1.0], [0.0, 0.0]]]),
        "candidates_smiles": ["match", "miss"],
        "batch_ptr": torch.tensor([2]),
        "adduct": ["[M+H]+"],
        "precursor_mz": torch.tensor([101.0]),
    }
    out = model.step(batch)

    assert out["scores"][0] > out["scores"][1]
