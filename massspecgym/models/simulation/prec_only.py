import torch

from massspecgym.data.transforms import MolToFingerprints, StandardMeta
from massspecgym.models.simulation.base import SimulationMassSpecGymModel
from massspecgym.simulation_utils.model_utils import PrecOnlyModel
from massspecgym.simulation_utils.spec_utils import sparse_cosine_distance


class PrecOnlySimulationMassSpecGymModel(SimulationMassSpecGymModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._setup_model()
        self._setup_loss_fn()
        self._setup_spec_fns()
        self._setup_metric_kwargs()

    def _setup_model(self):

        self.model = PrecOnlyModel()
