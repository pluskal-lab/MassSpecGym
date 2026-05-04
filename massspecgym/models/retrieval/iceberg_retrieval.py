"""
ICEBERG cosine similarity retrieval.

Simulates MS/MS spectra for each candidate molecule using ICEBERG,
then ranks candidates by cosine similarity between simulated and
query experimental spectra.

This matches the retrieval approach in external/ms-pred/src/ms_pred/retrieval/.
This is a bonus-task retrieval strategy.
"""

import typing as T

import numpy as np
import torch

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class IcebergRetrieval(RetrievalMassSpecGymModel):
    """ICEBERG-based retrieval via simulated spectrum cosine similarity.

    For each candidate molecule, simulates an MS/MS spectrum using ICEBERG,
    then computes cosine similarity between simulated and query spectra.
    Candidates are ranked by this similarity.

    Args:
        gen_checkpoint: Path to ICEBERG FragGNN checkpoint.
        inten_checkpoint: Path to ICEBERG IntenGNN checkpoint.
        num_bins: Number of bins for spectrum comparison.
        mz_max: Maximum m/z for binning.
    """

    def __init__(
        self,
        gen_checkpoint: T.Optional[str] = None,
        inten_checkpoint: T.Optional[str] = None,
        num_bins: int = 15000,
        mz_max: float = 1500.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gen_checkpoint = gen_checkpoint
        self._inten_checkpoint = inten_checkpoint
        self.num_bins = num_bins
        self.mz_max = mz_max
        self._iceberg_model = None

    def _get_iceberg(self):
        if self._iceberg_model is not None:
            return self._iceberg_model
        from massspecgym.models.simulation.iceberg.joint_model import JointModel
        from massspecgym.models.simulation.iceberg.gen_model import FragGNN
        from massspecgym.models.simulation.iceberg.inten_model import IntenGNN

        gen = FragGNN(hidden_size=256)
        inten = IntenGNN(hidden_size=256)
        self._iceberg_model = JointModel(gen, inten)

        if self._gen_checkpoint and self._inten_checkpoint:
            gen_ckpt = torch.load(self._gen_checkpoint, map_location="cpu")
            gen.load_state_dict(gen_ckpt.get("state_dict", gen_ckpt), strict=False)
            inten_ckpt = torch.load(self._inten_checkpoint, map_location="cpu")
            inten.load_state_dict(inten_ckpt.get("state_dict", inten_ckpt), strict=False)

        return self._iceberg_model

    def _bin_spectrum(self, mzs, intensities):
        """Bin a spectrum into fixed-size vector."""
        bins = np.linspace(0, self.mz_max, self.num_bins)
        binned = np.zeros(self.num_bins, dtype=np.float32)
        if len(mzs) > 0:
            indices = np.digitize(mzs, bins) - 1
            valid = (indices >= 0) & (indices < self.num_bins)
            for idx, inten in zip(indices[valid], intensities[valid]):
                binned[idx] += inten
        norm = np.linalg.norm(binned)
        if norm > 0:
            binned /= norm
        return binned

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        loss = torch.tensor(0.0, device=self.device)

        query_spec = batch.get("spec", None)
        cands_smiles = batch.get("candidates_smiles", [])
        batch_ptr = batch["batch_ptr"]

        iceberg = self._get_iceberg()
        iceberg = iceberg.to(self.device)

        query_bins = []
        if query_spec is not None:
            for spec in query_spec.detach().cpu().numpy():
                nonzero = spec[:, 1] > 0
                mzs = spec[nonzero, 0]
                intens = spec[nonzero, 1]
                query_bins.append(self._bin_spectrum(mzs, intens))
        else:
            query_bins = [np.zeros(self.num_bins, dtype=np.float32) for _ in range(batch_ptr.size(0))]

        sample_index = torch.repeat_interleave(
            torch.arange(batch_ptr.size(0), device=batch_ptr.device),
            batch_ptr,
        ).cpu().numpy()
        adducts = batch.get("adduct", ["[M+H]+"] * batch_ptr.size(0))
        precursor_mzs = batch.get("precursor_mz", [None] * batch_ptr.size(0))

        all_scores = []
        for cand_idx, smiles in enumerate(cands_smiles):
            sample_idx = int(sample_index[cand_idx])
            try:
                result = iceberg.predict_mol(
                    smi=smiles,
                    adduct=adducts[sample_idx] if isinstance(adducts, list) else str(adducts[sample_idx]),
                    precursor_mz=(
                        float(precursor_mzs[sample_idx])
                        if torch.is_tensor(precursor_mzs) else precursor_mzs[sample_idx]
                    ),
                    device=str(self.device),
                )
                spec = result.get("spec", [])
                if spec:
                    sim_mzs = np.array([s["mz"] for s in spec])
                    sim_ints = np.array([s["intensity"] for s in spec])
                else:
                    sim_mzs, sim_ints = np.array([]), np.array([])
            except Exception:
                sim_mzs, sim_ints = np.array([]), np.array([])

            sim_bin = self._bin_spectrum(sim_mzs, sim_ints)
            score = float(np.dot(query_bins[sample_idx], sim_bin))
            all_scores.append(score)

        scores = torch.tensor(all_scores, dtype=torch.float32, device=self.device)
        return dict(loss=loss, scores=scores, processable_mask=batch.get("processable_mask", None))
