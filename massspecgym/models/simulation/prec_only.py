from massspecgym.models.simulation.base import SimulationMassSpecGymModel
from massspecgym.simulation_utils.model_utils import PrecOnlyModel


class PrecOnlySimulationMassSpecGymModel(SimulationMassSpecGymModel):

    def _setup_model(self):

        self.model = PrecOnlyModel()
