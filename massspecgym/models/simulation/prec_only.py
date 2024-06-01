import torch

from massspecgym.transforms import MolToFingerprints, StandardMeta
from massspecgym.models.simulation.base import SimulationMassSpecGymModel
from massspecgym.simulation_utils.model_utils import PrecOnlyModel
from massspecgym.simulation_utils.spec_utils import sparse_cosine_distance


class PrecOnlySimulationMassSpecGymModel(SimulationMassSpecGymModel):

    def _setup_model(self):

        self.model = PrecOnlyModel()
