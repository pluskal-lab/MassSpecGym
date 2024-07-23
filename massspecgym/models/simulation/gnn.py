import torch

from massspecgym.transforms import MolToPyG, StandardMeta
from massspecgym.models.simulation.base import SimulationMassSpecGymModel
from massspecgym.simulation_utils.model_utils import GNNModel


class GNNSimulationMassSpecGymModel(SimulationMassSpecGymModel):

    def _get_input_sizes(self):

        mol_transform = MolToPyG()
        meta_transform = StandardMeta(
            adducts=self.hparams.adducts,
            instrument_types=self.hparams.instrument_types,
            max_collision_energy=self.hparams.max_collision_energy)
        input_d = {
            **mol_transform.get_input_sizes(),
            **meta_transform.get_input_sizes()
        }
        return input_d

    def _setup_model(self):

        input_d = self._get_input_sizes()
        self.model = GNNModel(
            # input
            mol_node_feats_size=input_d["mol_node_feats_size"],
            mol_edge_feats_size=input_d["mol_edge_feats_size"],
            metadata_insert_location=self.hparams.metadata_insert_location,
            collision_energy_input_size=input_d["collision_energy_input_size"],
            collision_energy_insert_size=self.hparams.collision_energy_insert_size,
            adduct_input_size=input_d["adduct_input_size"],
            adduct_insert_size=self.hparams.adduct_insert_size,
            instrument_type_input_size=input_d["instrument_type_input_size"],
            instrument_type_insert_size=self.hparams.instrument_type_insert_size,
            # output
            mz_max=self.hparams.mz_max,
            mz_bin_res=self.hparams.mz_bin_res,
            # model
            mol_hidden_size=self.hparams.mol_hidden_size,
            mol_num_layers=self.hparams.mol_num_layers,
            mol_gnn_type=self.hparams.mol_gnn_type,
            mol_normalization=self.hparams.mol_normalization,
            mol_dropout=self.hparams.mol_dropout,
            mol_pool_type=self.hparams.mol_pool_type,
            mlp_hidden_size=self.hparams.mlp_hidden_size,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
            ff_bidirectional=self.hparams.ff_bidirectional,
            ff_output_map_size=self.hparams.ff_output_map_size
        )
