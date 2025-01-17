from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from massspecgym.simulation_utils.formula_embedder import get_embedder
from massspecgym.simulation_utils.nn_utils import SpecFFN, GNN, build_pool_module


class BaseModel(nn.Module, ABC):

    def __init__(
        self,
        metadata_insert_location: str,
        collision_energy_input_size: int,
        collision_energy_insert_size: int,
        adduct_input_size: int,
        adduct_insert_size: int,
        instrument_type_input_size: int,
        instrument_type_insert_size: int
    ):

        super().__init__()
        # init metadata
        self.metadata_insert_size = 0
        self.metadata_insert_location = metadata_insert_location
        self._check_metadata_insert_location(metadata_insert_location)
        self._collision_energy_init(
            collision_energy_input_size=collision_energy_input_size,
            collision_energy_insert_size=collision_energy_insert_size
        )
        self._adduct_init(
            adduct_input_size=adduct_input_size,
            adduct_insert_size=adduct_insert_size
        )
        self._instrument_type_init(
            instrument_type_input_size=instrument_type_input_size,
            instrument_type_insert_size=instrument_type_insert_size
        )

    @abstractmethod
    def _check_metadata_insert_location(self, metadata_insert_location: str):

        pass

    def _collision_energy_init(
        self,
        collision_energy_input_size: int,
        collision_energy_insert_size: int):

        # embedder
        if collision_energy_insert_size > 0:
            embedder = get_embedder("abs-sines", max_count_int=collision_energy_input_size)
            collision_energy_embedder = nn.Sequential(
                embedder,
                nn.Linear(embedder.num_dim,collision_energy_insert_size)
            )	
            self.collision_energy_embedder = collision_energy_embedder
            self.metadata_insert_size += collision_energy_insert_size

    def _adduct_init(
        self,
        adduct_input_size: int,
        adduct_insert_size: int):

        if adduct_insert_size > 0:
            self.adduct_embedder = nn.Embedding(adduct_input_size+1, adduct_insert_size)	
            self.metadata_insert_size += adduct_insert_size

    def _instrument_type_init(
        self,
        instrument_type_input_size: int,
        instrument_type_insert_size: int):

        if instrument_type_insert_size > 0:
            self.instrument_type_embedder = nn.Embedding(instrument_type_input_size+1, instrument_type_insert_size)
            self.metadata_insert_size += instrument_type_insert_size

    def embed_metadata(
        self, 
        collision_energy: Optional[torch.Tensor] = None, 
        adduct: Optional[torch.Tensor] = None, 
        instrument_type: Optional[torch.Tensor] = None
    ):

        metadata_embeds = []
        if hasattr(self,"collision_energy_embedder"):
            collision_energy_embed = self.collision_energy_embedder(collision_energy.reshape(-1,1))
            metadata_embeds.append(collision_energy_embed)
        if hasattr(self,"adduct_embedder"):
            adduct_embed = self.adduct_embedder(adduct)
            metadata_embeds.append(adduct_embed)
        if hasattr(self,"instrument_type_embedder"):
            instrument_type_embed = self.instrument_type_embedder(instrument_type)
            metadata_embeds.append(instrument_type_embed)
        metadata_embeds = torch.cat(metadata_embeds, dim=1)
        return metadata_embeds


class FPModel(BaseModel):

    def __init__(
        self,
        # input
        fps_input_size: int,
        metadata_insert_location: str,
        collision_energy_input_size: int,
        collision_energy_insert_size: int,
        adduct_input_size: int,
        adduct_insert_size: int,
        instrument_type_input_size: int,
        instrument_type_insert_size: int,
        # output
        mz_max: int,
        mz_bin_res: float,
        # model
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_num_layers: int,
        mlp_use_residuals: bool,
        ff_prec_mz_offset: int,
        ff_bidirectional: bool,
        ff_output_map_size: int
    ):

        super().__init__(
            metadata_insert_location=metadata_insert_location,
            collision_energy_input_size=collision_energy_input_size,
            collision_energy_insert_size=collision_energy_insert_size,
            adduct_input_size=adduct_input_size,
            adduct_insert_size=adduct_insert_size,
            instrument_type_input_size=instrument_type_input_size,
            instrument_type_insert_size=instrument_type_insert_size
        )

        self.mlp_input_dim = fps_input_size + self.metadata_insert_size

        self.ffn = SpecFFN(
            input_size=self.mlp_input_dim,
            hidden_size=mlp_hidden_size,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout,
            prec_mz_offset=ff_prec_mz_offset,
            bidirectional=ff_bidirectional,
            use_residuals=mlp_use_residuals,
            output_map_size=ff_output_map_size
        )

    def _check_metadata_insert_location(self, location):

        assert location in ["mlp","none"], f"metadata_insert_location={location} not supported"

    def forward(
        self,
        fps: torch.Tensor, 
        precursor_mz: torch.Tensor,
        collision_energy: torch.Tensor,
        adduct: torch.Tensor,
        instrument_type: torch.Tensor,
        **kwargs
    ):

        # embed metadata
        metadata_embeds = self.embed_metadata(
            collision_energy=collision_energy,
            adduct=adduct,
            instrument_type=instrument_type
        )

        # combine with fingerprint
        fh = torch.cat([fps, metadata_embeds], dim=1)

        # apply ffn
        pred_mzs, pred_logprobs, pred_batch_idxs = self.ffn(fh,precursor_mz)
        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs
        }
        return out_d


class PrecOnlyModel(nn.Module):

    def __init__(self):

        super().__init__()
        self.dummy_params = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def forward(
        self, 
        precursor_mz: torch.Tensor,
        **kwargs):

        pred_mzs = precursor_mz
        pred_logprobs = 0.*self.dummy_params + torch.zeros_like(pred_mzs)
        pred_batch_idxs = torch.arange(pred_mzs.shape[0],device=pred_mzs.device)

        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs
        }
        return out_d


class GNNModel(BaseModel):

    def __init__(
        self,
        # input
        mol_node_feats_size: int,
        mol_edge_feats_size: int,
        metadata_insert_location: str,
        collision_energy_input_size: int,
        collision_energy_insert_size: int,
        adduct_input_size: int,
        adduct_insert_size: int,
        instrument_type_input_size: int,
        instrument_type_insert_size: int,
        # output
        mz_max: int,
        mz_bin_res: float,
        # model
        mol_hidden_size: int,
        mol_num_layers: int,
        mol_gnn_type: str,
        mol_normalization: str,
        mol_dropout: float,
        mol_pool_type: str,
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_num_layers: int,
        mlp_use_residuals: bool,
        ff_prec_mz_offset: int,
        ff_bidirectional: bool,
        ff_output_map_size: int,
    ):
        super().__init__(
            metadata_insert_location=metadata_insert_location,
            collision_energy_input_size=collision_energy_input_size,
            collision_energy_insert_size=collision_energy_insert_size,
            adduct_input_size=adduct_input_size,
            adduct_insert_size=adduct_insert_size,
            instrument_type_input_size=instrument_type_input_size,
            instrument_type_insert_size=instrument_type_insert_size
        )

        # setup mol gnn
        self.mol_node_feats_size = mol_node_feats_size
        self.mol_edge_feats_size = mol_edge_feats_size
        if self.metadata_insert_location == "mol":
            self.mol_node_feats_size += self.metadata_insert_size
        mol_kwargs = {
            "node_feats_size": self.mol_node_feats_size,
            "edge_feats_size": self.mol_edge_feats_size,
            "hidden_size": mol_hidden_size,
            "num_layers": mol_num_layers,
            "gnn_type": mol_gnn_type,
            "dropout": mol_dropout,
            "normalization": mol_normalization,
        }
        # Mol GNN
        self.mol_embedder = GNN(**mol_kwargs)
        self.mol_pool_type = mol_pool_type
        self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)

        # MLP input = GNN output
        self.mlp_input_dim = mol_hidden_size
        # metadata
        if self.metadata_insert_location == "mlp":
            self.mlp_input_dim += self.metadata_insert_size

        self.ffn = SpecFFN(
            input_size=self.mlp_input_dim,
            hidden_size=mlp_hidden_size,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout,
            prec_mz_offset=ff_prec_mz_offset,
            bidirectional=ff_bidirectional,
            use_residuals=mlp_use_residuals,
            output_map_size=ff_output_map_size
        )

    def _check_metadata_insert_location(self, location):

        assert location in ["mlp","mol","none"], f"metadata_insert_location={location} not supported"

    def forward(
        self, 
        mol_pyg: Batch, 
        precursor_mz: torch.Tensor,
        collision_energy: torch.Tensor,
        adduct: torch.Tensor,
        instrument_type: torch.Tensor,
        **kwargs
    ):
        # mol features
        # mol_x: mol level node feature matrix
        # mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
        # edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
        # batch: sample idx repsect to current batch
        mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch

        # int_dtype = mol_edge_index.dtype
        batch_size = mol_batch[-1]+1

        # embed metadata
        metadata_embeds = self.embed_metadata(
            collision_energy=collision_energy,
            adduct=adduct,
            instrument_type=instrument_type
        )		

        # metadata embeddings at the node feature level
        if self.metadata_insert_location == "mol":
            mol_metadata_embeds = torch.repeat_interleave(mol_metadata_embeds,torch.unique(mol_batch,return_counts=True)[1],dim=0)
            mol_x = torch.cat([mol_x,mol_metadata_embeds],dim=1)
        
        # get per-atom embeddings
        mol_embed_gnn = self.mol_embedder(
            mol_x,
            mol_batch,
            mol_edge_index,
            mol_edge_attr
        )
        mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)
        ffn_input = mol_embed_gnn_pool

        if self.metadata_insert_location == "mlp":
            ffn_input = torch.cat([ffn_input,metadata_embeds],dim=1)

        # apply ffn
        pred_mzs, pred_logprobs, pred_batch_idxs = self.ffn(ffn_input,precursor_mz)
        out_d = {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs
        }
        return out_d
