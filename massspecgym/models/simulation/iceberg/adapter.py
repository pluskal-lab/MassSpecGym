"""
ICEBERG adapter for MassSpecGym SimulationMassSpecGymModel interface.

Wraps the ICEBERG JointModel (FragGNN + IntenGNN) to follow the same
interface as GNNSimulationMassSpecGymModel and FPSimulationMassSpecGymModel.
"""

import typing as T

import torch

from massspecgym.models.simulation.base import SimulationMassSpecGymModel
from massspecgym.simulation_utils.misc_utils import safelog


class IcebergSimulationMassSpecGymModel(SimulationMassSpecGymModel):
    """ICEBERG spectrum simulation model for MassSpecGym.

    Predicts MS/MS spectra from molecular structures using a two-stage
    DAG-based approach via FragGNN + IntenGNN, adapted to the
    SimulationMassSpecGymModel interface.

    Args:
        gen_checkpoint: Path to pretrained FragGNN checkpoint.
        inten_checkpoint: Path to pretrained IntenGNN checkpoint.
        sparse_k: Number of top peaks to retain.
        max_nodes: Maximum number of DAG nodes.
        threshold: Minimum intensity threshold.
    """

    def __init__(
        self,
        gen_checkpoint: T.Optional[str] = None,
        inten_checkpoint: T.Optional[str] = None,
        sparse_k: int = 128,
        max_nodes: int = 100,
        threshold: float = 0.001,
        **kwargs,
    ):
        self._gen_checkpoint = gen_checkpoint
        self._inten_checkpoint = inten_checkpoint
        self._sparse_k = sparse_k
        self._max_nodes = max_nodes
        self._threshold = threshold
        super().__init__(**kwargs)

    def _setup_model(self):
        """Set up the ICEBERG JointModel from checkpoints."""
        from .gen_model import FragGNN
        from .inten_model import IntenGNN
        from .joint_model import JointModel

        if self._gen_checkpoint and self._inten_checkpoint:
            gen_obj = FragGNN.load_from_checkpoint(self._gen_checkpoint, map_location="cpu")
            inten_obj = IntenGNN.load_from_checkpoint(self._inten_checkpoint, map_location="cpu")
            self.model = JointModel(gen_obj, inten_obj)
        else:
            self.model = JointModel(
                FragGNN(hidden_size=256),
                IntenGNN(hidden_size=256),
            )

    def configure_optimizers(self):
        return None

    def forward(self, **kwargs) -> dict:
        smiles_list = kwargs.get("smiles")
        if smiles_list is None:
            raise ValueError("IcebergSimulationMassSpecGymModel requires batch['smiles'].")

        adducts = kwargs.get("adduct", ["[M+H]+"] * len(smiles_list))
        collision_energies = kwargs.get("collision_energy", [40.0] * len(smiles_list))
        precursor_mzs = kwargs.get("precursor_mz", [None] * len(smiles_list))

        pred_mzs, pred_ints, pred_batch_idxs = [], [], []
        for batch_idx, smiles in enumerate(smiles_list):
            adduct = adducts[batch_idx] if isinstance(adducts, list) else str(adducts[batch_idx])
            ce = float(collision_energies[batch_idx]) if torch.is_tensor(collision_energies) else float(collision_energies[batch_idx])
            precursor_mz = (
                float(precursor_mzs[batch_idx])
                if torch.is_tensor(precursor_mzs) else precursor_mzs[batch_idx]
            )
            result = self.model.predict_mol(
                smi=smiles,
                collision_eng=ce,
                precursor_mz=precursor_mz,
                adduct=adduct,
                threshold=self._threshold,
                device=str(self.device),
                max_nodes=self._max_nodes,
            )
            spec = result.get("spec", [])[: self._sparse_k]
            if not spec:
                spec = [{"mz": float(precursor_mz or 0.0), "intensity": 1.0}]
            for peak in spec:
                pred_mzs.append(float(peak["mz"]))
                pred_ints.append(float(peak["intensity"]))
                pred_batch_idxs.append(batch_idx)

        if not pred_mzs:
            pred_mzs = [0.0]
            pred_ints = [1.0]
            pred_batch_idxs = [0]

        pred_mzs = torch.tensor(pred_mzs, dtype=torch.float32, device=self.device)
        pred_ints = torch.tensor(pred_ints, dtype=torch.float32, device=self.device)
        pred_batch_idxs = torch.tensor(pred_batch_idxs, dtype=torch.long, device=self.device)
        denom = torch.zeros(len(smiles_list), dtype=torch.float32, device=self.device)
        denom.scatter_add_(0, pred_batch_idxs, pred_ints)
        pred_logprobs = safelog(pred_ints / denom[pred_batch_idxs].clamp(min=1e-12))
        return {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
        }
