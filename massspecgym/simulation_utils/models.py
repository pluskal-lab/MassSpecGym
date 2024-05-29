from abc import ABC
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):

    def _ce_init(
        self,
        int_embedder,
        ce_insert_location: str,
        ce_insert_size: int):

        # ce stuff
        assert ce_insert_location in ["none","mol","frag","mlp"]
        self.ce_insert_location = ce_insert_location
        self.ce_insert_size = ce_insert_size
        self.int_embedder = int_embedder
        self._ce_location_check()
        # embedder
        embedder = get_embedder(self.int_embedder, max_count_int=int(NCE_MAX))
        ce_embedder = nn.Sequential(
            embedder,
            nn.Linear(embedder.num_dim,self.ce_insert_size)
        )	
        ce_input_dim = self.ce_insert_size
        # location
        if self.ce_insert_location == "mol":
            ce_mol_input_dim = ce_input_dim
            ce_mlp_input_dim = 0
        elif self.ce_insert_location == "mlp":
            ce_mol_input_dim = 0
            ce_mlp_input_dim = ce_input_dim
        else:
            assert self.ce_insert_location == "none"
            ce_mol_input_dim = 0
            ce_mlp_input_dim = 0
        self.ce_transform = ce_transform
        self.ce_embedder = ce_embedder
        self.ce_mol_input_dim = ce_mol_input_dim
        self.ce_mlp_input_dim = ce_mlp_input_dim

    @abstractmethod
    def _ce_location_check(self):

        pass

    def _adduct_init(self):

        pass

    @abstractmethod
    def _adduct_location_check(self):

        pass

    def _instrument_type_init(self):

        pass

    @abstractmethod
    def _instrument_type_location_check(self):

        pass

        