"""
DiffMS decoder: Discrete graph diffusion for fingerprint-to-molecule generation.

Ported from Bohde et al., "DiffMS: Diffusion generation of molecules conditioned
on mass spectra", 2025.

The model performs discrete diffusion over molecular graphs, where:
- Node features: one-hot atom types (C, O, P, N, S, Cl, F, H)
- Edge features: one-hot bond types (no-bond, single, double, triple, aromatic)
- Global features: Morgan fingerprint as conditioning signal

Training uses discrete diffusion loss (CE on edges, optionally nodes/global).
Sampling reverses the diffusion process via posterior sampling.
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from massspecgym.models.base import Stage
from massspecgym.models.de_novo.fp2mol.base import FP2MolDeNovoModel
from .graph_transformer import GraphTransformer
from .diffusion_utils import (
    PlaceHolder,
    PredefinedNoiseScheduleDiscrete,
    MarginalUniformTransition,
    DiscreteUniformTransition,
    compute_batched_over0_posterior_distribution,
    sample_discrete_features,
    sample_discrete_feature_noise,
)

ATOM_DECODER = ["C", "O", "P", "N", "S", "Cl", "F", "H"]
BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def mol_from_graphs(node_list, adjacency_matrix, atom_decoder=ATOM_DECODER):
    """Convert graph representation to RDKit molecule."""
    mol = Chem.RWMol()
    node_to_idx = {}
    for i in range(len(node_list)):
        if node_list[i] == -1:
            continue
        a = Chem.Atom(atom_decoder[int(node_list[i])])
        node_to_idx[i] = mol.AddAtom(a)

    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            if iy <= ix:
                continue
            if 1 <= bond <= 4 and ix in node_to_idx and iy in node_to_idx:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], BOND_TYPES[bond])

    try:
        mol = mol.GetMol()
        return mol
    except Exception:
        return None


class DiffMSDecoder(FP2MolDeNovoModel):
    """DiffMS discrete graph diffusion decoder.

    Args:
        num_atom_types: Number of atom types (8).
        num_bond_types: Number of bond types including no-bond (5).
        diffusion_steps: Number of diffusion steps (500).
        noise_schedule: Noise schedule type ('cosine').
        transition: Transition type ('marginal' or 'uniform').
        n_layers: Number of GraphTransformer layers.
        hidden_dims: Hidden dimensions for the GraphTransformer.
        lambda_train: Loss weights [node_CE, edge_CE, global_CE].
        max_nodes: Maximum number of nodes in generated graphs.
        fingerprint_bits: Number of fingerprint bits (2048 for DiffMS).
    """

    def __init__(
        self,
        num_atom_types: int = 8,
        num_bond_types: int = 5,
        diffusion_steps: int = 500,
        noise_schedule: str = "cosine",
        transition: str = "marginal",
        n_layers: int = 6,
        hidden_dims: Optional[dict] = None,
        lambda_train: Optional[list] = None,
        max_nodes: int = 50,
        fingerprint_bits: int = 2048,
        *args,
        **kwargs,
    ):
        super().__init__(fingerprint_bits=fingerprint_bits, use_formula=False, *args, **kwargs)

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.T = diffusion_steps
        self.max_nodes = max_nodes
        self.lambda_train = lambda_train or [0.0, 1.0, 0.0]

        if hidden_dims is None:
            hidden_dims = {
                "dx": 256, "de": 64, "dy": 256, "n_head": 8,
                "dim_ffX": 256, "dim_ffE": 128,
            }

        extra_X_feat = 1  # timestep
        extra_E_feat = 1
        extra_y_feat = 1

        input_dims = {
            "X": num_atom_types + extra_X_feat,
            "E": num_bond_types + extra_E_feat,
            "y": fingerprint_bits + extra_y_feat,
        }
        output_dims = {"X": num_atom_types, "E": num_bond_types, "y": 0}
        hidden_mlp_dims = {"X": 256, "E": 128, "y": 256}

        self.model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
        )

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule, diffusion_steps)

        x_marginals = torch.ones(num_atom_types) / num_atom_types
        e_marginals = torch.zeros(num_bond_types)
        e_marginals[0] = 0.9
        e_marginals[1:] = 0.1 / (num_bond_types - 1)

        if transition == "marginal":
            self.transition_model = MarginalUniformTransition(x_marginals, e_marginals, 0)
        else:
            self.transition_model = DiscreteUniformTransition(num_atom_types, num_bond_types, 0)

        self.limit_dist = PlaceHolder(X=x_marginals, E=e_marginals, y=torch.zeros(0))

    def apply_noise(self, X, E, y, node_mask):
        """Apply forward diffusion noise to graph features."""
        t_int = torch.randint(1, self.T + 1, size=(X.size(0), 1), device=X.device)
        s_int = t_int - 1
        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, X.device)
        prob_X = torch.einsum("bnc,bcd->bnd", X, Qtb.X).clamp(min=1e-5)
        prob_E = torch.einsum("bijc,bcd->bijd", E, Qtb.E).clamp(min=1e-5)

        sampled = sample_discrete_features(prob_X, prob_E, node_mask)
        X_t = F.one_hot(sampled.X, self.num_atom_types).float()
        E_t = F.one_hot(sampled.E, self.num_bond_types).float()

        x_mask = node_mask.unsqueeze(-1)
        X_t = X_t * x_mask
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)
        E_t = E_t * e_mask

        return {
            "t_int": t_int, "t": t_float,
            "beta_t": beta_t, "alpha_s_bar": alpha_s_bar, "alpha_t_bar": alpha_t_bar,
            "X_t": X_t, "E_t": E_t, "y_t": y,
        }

    def compute_decoder_loss(self, batch: dict) -> torch.Tensor:
        """Compute discrete diffusion training loss."""
        X = batch.get("X")
        E = batch.get("E")
        y = batch.get("fingerprint")
        node_mask = batch.get("node_mask")

        if X is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        noisy = self.apply_noise(X, E, y, node_mask)

        t_emb = noisy["t"]
        X_in = torch.cat([noisy["X_t"], t_emb[:, None, :].expand(-1, noisy["X_t"].size(1), -1)], dim=-1)
        E_in = torch.cat([noisy["E_t"], t_emb[:, None, None, :].expand(-1, noisy["E_t"].size(1), noisy["E_t"].size(2), -1)], dim=-1)
        y_in = torch.cat([noisy["y_t"], t_emb], dim=-1)

        pred = self.model(X_in, E_in, y_in, node_mask)

        true_X = X.argmax(dim=-1) if X.dim() == 3 else X
        true_E = E.argmax(dim=-1) if E.dim() == 4 else E

        loss_X = F.cross_entropy(pred.X.reshape(-1, self.num_atom_types), true_X.reshape(-1), reduction="mean")
        loss_E = F.cross_entropy(pred.E.reshape(-1, self.num_bond_types), true_E.reshape(-1), reduction="mean")

        loss = self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E
        return loss

    @torch.no_grad()
    def decode_from_fingerprint(
        self,
        fingerprint: torch.Tensor,
        formula=None,
        num_samples: int = 1,
    ) -> list:
        """Generate molecules from fingerprint via reverse diffusion."""
        self.eval()
        batch_size = fingerprint.shape[0]
        all_preds = []

        for b_idx in range(batch_size):
            fp = fingerprint[b_idx:b_idx + 1]
            sample_mols = []

            for _ in range(num_samples):
                n_nodes = torch.randint(5, self.max_nodes, (1,)).item()
                node_mask = torch.ones(1, n_nodes, device=self.device).bool()

                z_T = sample_discrete_feature_noise(self.limit_dist, node_mask)
                X = z_T.X.to(self.device)
                E = z_T.E.to(self.device)
                y = fp.to(self.device)

                for s_int in reversed(range(0, self.T)):
                    t_int = s_int + 1
                    t = torch.tensor([t_int / self.T], device=self.device)

                    t_emb = t.unsqueeze(-1)
                    X_in = torch.cat([X, t_emb.expand(-1, X.size(1), -1)], dim=-1)
                    E_in = torch.cat([E, t_emb.unsqueeze(1).expand(-1, E.size(1), E.size(2), -1)], dim=-1)
                    y_in = torch.cat([y, t_emb], dim=-1)

                    pred = self.model(X_in, E_in, y_in, node_mask.float())

                    pred_X = F.softmax(pred.X, dim=-1)
                    pred_E = F.softmax(pred.E, dim=-1)

                    sampled = sample_discrete_features(pred_X, pred_E, node_mask)
                    X = F.one_hot(sampled.X, self.num_atom_types).float()
                    E = F.one_hot(sampled.E, self.num_bond_types).float()

                node_types = X.argmax(dim=-1)[0].cpu().numpy()
                adj = E.argmax(dim=-1)[0].cpu().numpy()
                mol = mol_from_graphs(node_types, adj)
                smi = Chem.MolToSmiles(mol) if mol is not None else None
                sample_mols.append(smi)

            all_preds.append(sample_mols)
        return all_preds
