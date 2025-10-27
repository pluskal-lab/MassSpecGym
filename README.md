# MassSpecGym: A benchmark for the discovery and identification of molecules

<p>
  <a href="https://huggingface.co/datasets/roman-bushuiev/MassSpecGym"><img alt="Dataset on Hugging Face" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" height="22px"></a>
  <a href="https://doi.org/10.48550/arXiv.2410.23326"><img alt="arXiv badge" src="https://img.shields.io/badge/arXiv-2410.23326-b31b1b.svg" height="22px"></a>
  <a href="https://massspecgym.onrender.com/"><img src="https://img.shields.io/badge/Leaderboard-gold.svg" height="22px"></a>
  <a href="https://pypi.org/project/massspecgym"><img alt="Dataset on Hugging Face" src="https://img.shields.io/pypi/v/massspecgym" height="22px"></a>
  <a href="https://github.com/pytorch/pytorch"> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" height="22px"></a>
  <a href="https://github.com/Lightning-AI/pytorch-lightning"> <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" height="22px"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg" height="22px"></a>
<p>

<p align="center">
  <img src="https://raw.githubusercontent.com/pluskal-lab/MassSpecGym/5d7d58af99947988f947eeb5bd5c6a472c2938b7/assets/MassSpecGym_abstract.svg" width="80%"/>
</p>

MassSpecGym provides three challenges for benchmarking the discovery and identification of new molecules from MS/MS spectra:

- 💥 ***De novo* molecule generation** (MS/MS spectrum → molecular structure)
    - ✨ **Bonus chemical formulae challenge** (MS/MS spectrum + chemical formula → molecular structure)
- 💥 **Molecule retrieval** (MS/MS spectrum → ranked list of candidate molecular structures)
    - ✨ **Bonus chemical formulae challenge** (MS/MS spectrum → ranked list of candidate molecular structures with ground-truth chemical formulae)
- 💥 **Spectrum simulation** (molecular structure → MS/MS spectrum)
    - ✨ **Bonus chemical formulae challenge** (molecular structure → MS/MS spectrum; evaluated on the retrieval of molecular structures with ground-truth chemical formulae)

The provided challenges abstract the process of scientific discovery from biological and environmental samples into well-defined machine learning problems with pre-defined datasets, data splits, and evaluation metrics.

<!-- [![Dataset on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym)   -->

📚 Please see more details in our [NeurIPS 2024 Spotlight paper](https://arxiv.org/abs/2410.23326).

## 🏅 MassSpecGym leaderboard

The MassSpecGym leaderboard is available at [https://massspecgym.onrender.com](https://massspecgym.onrender.com), providing an interactive web to track state-of-the-art results across the MassSpecGym challenges. To submit new results from your paper, please open a pull request that updates the results tables in the `results` folder.

Update 10/27/2025: For the spectrum simulation challenge, the latest state-of-the-art model, ICEBERG, now has MassSpecGym-compatible [weights](https://www.dropbox.com/scl/fo/d73o0o4u5ymr9ubtp3m7j/AL4r7e3p9ElV0ewBwDCScbM?rlkey=tr99zkzy208ol8aw0pfsdsf5v&st=2zg9n01y&dl=0) and [codebase](https://github.com/coleygroup/ms-pred) available. If you have any questions regarding usage, please open an issue in the [`ms-pred`](https://github.com/coleygroup/ms-pred) repository. 

## 📦 Installation

Installation is available via [pip](https://pypi.org/project/massspecgym):

```bash
pip install massspecgym
```

If you use conda, we recommend creating and activating a new environment before installing MassSpecGym:

```bash
conda create -n massspecgym python==3.11
conda activate massspecgym
```

If you are planning to run Jupyter notebooks provided in the repository or contribute to the project, we recommend installing the optional dependencies:

```bash
pip install massspecgym[notebooks, dev]
```

<!-- For AMD GPUs, you may need to install PyTorch for ROCm:

```bash
pip install -U torch==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
``` -->

## 🍩 Getting started with MassSpecGym

<p align="center">
  <img src="https://raw.githubusercontent.com/pluskal-lab/MassSpecGym/5d7d58af99947988f947eeb5bd5c6a472c2938b7/assets/MassSpecGym_infrastructure.svg" width="80%"/>
</p>

MassSpecGym’s infrastructure consists of predefined components that serve as building blocks for the implementation and evaluation of new models.

First of all, the MassSpecGym dataset is available as a [Hugging Face dataset](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) and can be downloaded within the code into a pandas DataFrame as follows.

```python
from massspecgym.utils import load_massspecgym
df = load_massspecgym()
```

Second, MassSpecGym provides [a set of transforms](https://github.com/pluskal-lab/MassSpecGym/blob/main/massspecgym/data/transforms.py) for spectra and molecules, which can be used to preprocess data for machine learning models. These transforms can be used in conjunction with the `MassSpecDataset` class (or its subclasses), resulting in a PyTorch `Dataset` object that implicitly applies the specified transforms to each data point. Note that `MassSpecDataset` also automatically downloads the dataset from the Hugging Face repository as needed.

```python
from massspecgym.data import MassSpecDataset
from massspecgym.transforms import SpecTokenizer, MolFingerprinter

dataset = MassSpecDataset(
    spec_transform=SpecTokenizer(n_peaks=60),
    mol_transform=MolFingerprinter(),
)
```

Third, MassSpecGym provides a `MassSpecDataModule`, a PyTorch Lightning [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) that automatically handles data splitting into training, validation, and testing folds, as well as loading data into batches.

```python
from massspecgym.data import MassSpecDataModule

data_module = MassSpecDataModule(
    dataset=dataset,
    batch_size=32
)
```

Finally, MassSpecGym defines evaluation metrics by implementing abstract subclasses of `LightningModule` for each of the MassSpecGym challenges: [`DeNovoMassSpecGymModel`](https://github.com/pluskal-lab/MassSpecGym/blob/df2ff567ed5ad60244b4106a180aaebc3c787b7e/massspecgym/models/de_novo/base.py#L14), [`RetrievalMassSpecGymModel`](https://github.com/pluskal-lab/MassSpecGym/blob/df2ff567ed5ad60244b4106a180aaebc3c787b7e/massspecgym/models/retrieval/base.py#L14), and [`SimulationMassSpecGymModel`](https://github.com/pluskal-lab/MassSpecGym/blob/df2ff567ed5ad60244b4106a180aaebc3c787b7e/massspecgym/models/simulation/base.py#L12). To implement a custom model, you should inherit from the appropriate abstract class and implement the `forward` and `step` methods. This procedure is described in the next section. If you looking for more examples, please see the [`massspecgym/models`](https://github.com/pluskal-lab/MassSpecGym/tree/df2ff567ed5ad60244b4106a180aaebc3c787b7e/massspecgym/models) folder.

## 🚀 Train and evaluate your model

MassSpecGym allows you to implement, train, validate, and test your model with a few lines of code. Built on top of PyTorch Lightning, MassSpecGym abstracts data preparation and splitting while eliminating boilerplate code for training and evaluation loops. To train and evaluate your model, you only need to implement your custom architecture and prediction logic.

Below is an example of how to implement a simple model based on [DeepSets](https://arxiv.org/abs/1703.06114) for the molecule retrieval task. The model is trained to predict the fingerprint of a molecule from its spectrum and then retrieves the most similar molecules from a set of candidates based on fingerprint similarity. For more examples, please see [`notebooks/demo.ipynb`](https://github.com/pluskal-lab/MassSpecGym/blob/df2ff567ed5ad60244b4106a180aaebc3c787b7e/notebooks/demo.ipynb).

1. Import necessary modules:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from massspecgym.data import RetrievalDataset, MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter
from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
```

2. Implement your model:

```python
class MyDeepSetsRetrievalModel(RetrievalMassSpecGymModel):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 4096,  # fingerprint size
        *args,
        **kwargs
    ):
        """Implement your architecture."""
        super().__init__(*args, **kwargs)

        self.phi = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implement your prediction logic."""
        x = self.phi(x)
        x = x.sum(dim=-2)  # sum over peaks
        x = self.rho(x)
        return x

    def step(
        self, batch: dict, stage: Stage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implement your custom logic of using predictions for training and inference."""
        # Unpack inputs
        x = batch["spec"]  # input spectra
        fp_true = batch["mol"]  # true fingerprints
        cands = batch["candidates"]  # candidate fingerprints concatenated for a batch
        batch_ptr = batch["batch_ptr"]  # number of candidates per sample in a batch

        # Predict fingerprint
        fp_pred = self.forward(x)

        # Calculate loss
        loss = nn.functional.mse_loss(fp_true, fp_pred)

        # Calculate final similarity scores between predicted fingerprints and retrieval candidates
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)

        return dict(loss=loss, scores=scores)
```

3. Train and validate your model:

```python
# Init hyperparameters
n_peaks = 60
fp_size = 4096
batch_size = 32

# Load dataset
dataset = RetrievalDataset(
    spec_transform=SpecTokenizer(n_peaks=n_peaks),
    mol_transform=MolFingerprinter(fp_size=fp_size),
)

# Init data module
data_module = MassSpecDataModule(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=4
)

# Init model
model = MyDeepSetsRetrievalModel(out_channels=fp_size)

# Init trainer
trainer = Trainer(accelerator="cpu", devices=1, max_epochs=5)

# Train
trainer.fit(model, datamodule=data_module)
```

4. Test your model:

```python
# Test
trainer.test(model, datamodule=data_module)
```

## 🔗 References

If you use MassSpecGym in your work, please cite the following paper:

```bibtex
@inproceedings{bushuiev2024massspecgym,
 author = {Bushuiev, Roman and Bushuiev, Anton and de Jonge, Niek F. and Young, Adamo and Kretschmer, Fleming and Samusevich, Raman and Heirman, Janne and Wang, Fei and Zhang, Luke and D\"{u}hrkop, Kai and Ludwig, Marcus and Haupt, Nils A. and Kalia, Apurva and Brungs, Corinna and Schmid, Robin and Greiner, Russell and Wang, Bo and Wishart, David S. and Liu, Li-Ping and Rousu, Juho and Bittremieux, Wout and Rost, Hannes and Mak, Tytus D. and Hassoun, Soha and Huber, Florian and van der Hooft, Justin J.J. and Stravs, Michael A. and B\"{o}cker, Sebastian and Sivic, Josef and Pluskal, Tom\'{a}\v{s}},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {110010--110027},
 publisher = {Curran Associates, Inc.},
 title = {MassSpecGym: A benchmark for the discovery and identification of molecules},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {37},
 year = {2024}
}
```
