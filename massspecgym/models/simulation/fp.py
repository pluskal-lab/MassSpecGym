import torch

from massspecgym.data.transforms import MolToFingerprints, StandardMeta
from massspecgym.models.simulation.base import SimulationMassSpecGymModel
from massspecgym.simulation_utils.model_utils import FPModel


class FPSimulationMassSpecGymModel(SimulationMassSpecGymModel):

    def __init__(
        self,
        fp_types,
        adducts,
        instrument_types,
        max_collision_energy,
        metadata_insert_location,
        adduct_insert_size,
        instrument_type_insert_size,
        collision_energy_insert_size,
        mlp_hidden_size,
        mlp_dropout,
        mlp_num_layers,
        mlp_use_residuals,
        ff_prec_mz_offset,
        ff_bidirectional,
        ff_output_map_size,
        **kwargs
    ):
        self.fp_types = fp_types
        self.adducts = adducts
        self.instrument_types = instrument_types
        self.max_collision_energy = max_collision_energy
        self.metadata_insert_location = metadata_insert_location
        self.adduct_insert_size = adduct_insert_size
        self.instrument_type_insert_size = instrument_type_insert_size
        self.collision_energy_insert_size = collision_energy_insert_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout
        self.mlp_num_layers = mlp_num_layers
        self.mlp_use_residuals = mlp_use_residuals
        self.ff_prec_mz_offset = ff_prec_mz_offset
        self.ff_bidirectional = ff_bidirectional
        self.ff_output_map_size = ff_output_map_size
        super().__init__(**kwargs)

    def _get_input_sizes(self):

        mol_transform = MolToFingerprints(
            fp_types=self.fp_types
        )
        meta_transform = StandardMeta(
            adducts=self.adducts,
            instrument_types=self.instrument_types,
            max_collision_energy=self.max_collision_energy)
        input_d = {
            **mol_transform.get_input_sizes(),
            **meta_transform.get_input_sizes()
        }
        return input_d

    def _setup_model(self):

        input_d = self._get_input_sizes()
        self.model = FPModel(
            # input
            fps_input_size=input_d["fps_input_size"],
            metadata_insert_location=self.metadata_insert_location,
            collision_energy_input_size=input_d["collision_energy_input_size"],
            collision_energy_insert_size=self.collision_energy_insert_size,
            adduct_input_size=input_d["adduct_input_size"],
            adduct_insert_size=self.adduct_insert_size,
            instrument_type_input_size=input_d["instrument_type_input_size"],
            instrument_type_insert_size=self.instrument_type_insert_size,
            # output
            mz_max=self.mz_max, ###
            mz_bin_res=self.mz_bin_res,
            # model
            mlp_hidden_size=self.mlp_hidden_size,
            mlp_dropout=self.mlp_dropout,
            mlp_num_layers=self.mlp_num_layers,
            mlp_use_residuals=self.mlp_use_residuals,
            ff_prec_mz_offset=self.ff_prec_mz_offset,
            ff_bidirectional=self.ff_bidirectional,
            ff_output_map_size=self.ff_output_map_size
        )
