import typing as T
from abc import ABC

import pulp
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from myopic_mces.myopic_mces import MCES
from torchmetrics.aggregation import MeanMetric

from massspecgym.models.base import MassSpecGymModel, Stage
from massspecgym.utils import morgan_fp, mol_to_inchi_key


class DeNovoMassSpecGymModel(MassSpecGymModel, ABC):

    def __init__(
        self,
        top_ks: T.Iterable[int] = (1, 10),
        myopic_mces_kwargs: T.Optional[T.Mapping] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.top_ks = top_ks

        # TODO Replace with utils.MyopicMCES class
        self.myopic_mces_kwargs = dict(
            ind=0,  # dummy index
            solver=pulp.listSolvers(onlyAvailable=True)[0],  # Use the first available solver
            threshold=15,  # MCES threshold
            solver_options=dict(msg=0)  # make ILP solver silent
        )
        self.myopic_mces_kwargs |= myopic_mces_kwargs or {}
        self.mol_pred_kind: T.Literal["smiles", "rdkit"] = "smiles"
        # caches of already computed results to avoid expensive re-computations
        self.mces_cache = dict()
        self.mol_2_morgan_fp = dict()

    def on_batch_end(
        self,
        outputs: T.Any,
        batch: dict,
        batch_idx: int,
        stage: Stage
    ) -> None:
        self.log(
            f"{stage.to_pref()}loss",
            outputs['loss'],
            batch_size=batch['spec'].size(0),
            sync_dist=True,
            prog_bar=True,
        )

        if stage in self.log_only_loss_at_stages:
            return
        
        self.evaluate_de_novo_step(
            outputs["mols_pred"],  # (bs, k) list of generated rdkit molecules or SMILES strings
            batch["mol"],  # (bs) list of ground truth SMILES strings
            stage=stage
        )

    def evaluate_de_novo_step(
        self,
        mols_pred: list[list[T.Optional[Chem.Mol | str]]],
        mol_true: list[str],
        stage: Stage,
    ) -> None:
        """
        # TODO: refactor to compute only for max(k) and then use the result to obtain the rest by
        subsetting.

        Main evaluation method for the models for de novo molecule generation from mass spectra.

        Args:
            mols_pred (list[list[Mol | str]]): (bs, k) list of generated rdkit molecules or SMILES
                strings with possible Nones if no molecule was generated
            mol_true (list[str]): (bs) list of ground-truth SMILES strings
        """
        # Get SMILES and RDKit molecule objects for all predictions
        if self.mol_pred_kind == "smiles":
            smiles_pred_valid, mols_pred_valid = [], []
            for mols_pred_sample in mols_pred:
                smiles_pred_valid_sample, mols_pred_valid_sample = [], []
                for s in mols_pred_sample:
                    m = Chem.MolFromSmiles(s) if s is not None else None
                    # If SMILES cannot be converted to RDKit molecule, the molecule is set to None
                    smiles_pred_valid_sample.append(s if m is not None else None)
                    mols_pred_valid_sample.append(m)
                smiles_pred_valid.append(smiles_pred_valid_sample)
                mols_pred_valid.append(mols_pred_valid_sample)
            smiles_pred, mols_pred = smiles_pred_valid, mols_pred_valid
        elif self.mol_pred_kind == "rdkit":
            smiles_pred = [
                [Chem.MolToSmiles(m) if m is not None else None for m in ms]
                for ms in mols_pred
            ]
        else:
            raise ValueError(f"Invalid mol_pred_kind: {self.mol_pred_kind}")

        # Auxiliary metric: number of valid molecules
        self._update_metric(
            stage.to_pref() + f"num_valid_mols",
            MeanMetric,
            ([sum([m is not None for m in ms]) for ms in mols_pred],),
            batch_size=len(mols_pred),
        )

        # Get RDKit molecule objects for ground truth
        smile_true = mol_true
        mol_true = [Chem.MolFromSmiles(sm) for sm in mol_true]

        def _get_morgan_fp_with_cache(mol):
            """
            A helper function to retrieve either cached Morgan Fingerprint value, or to compute and cache it
            @param mol: RDKit molecule object
            @return:
            """
            if mol not in self.mol_2_morgan_fp:
                self.mol_2_morgan_fp[mol] = morgan_fp(mol, to_np=False)
            return self.mol_2_morgan_fp[mol]



        # Evaluate top-k metrics
        for top_k in self.top_ks:
            # Get top-k predicted molecules for each ground-truth sample
            smiles_pred_top_k = [smiles_pred_sample[:top_k] for smiles_pred_sample in smiles_pred]
            mols_pred_top_k = [mols_pred_sample[:top_k] for mols_pred_sample in mols_pred]

            # 1. Evaluate minimum common edge subgraph:
            # Calculate MCES distance between top-k predicted molecules and ground truth and
            # report the minimum distance. The minimum distances for each sample in the batch are
            # averaged across the epoch.
            min_mces_dists = []
            mces_thld = 100
            # Iterate over batch
            for preds, true in zip(smiles_pred_top_k, smile_true):
                # Iterate over top-k predicted molecule samples
                dists = []
                for pred in preds:
                    if pred is None:
                        dists.append(mces_thld)
                    else:
                        if (true, pred) not in self.mces_cache:
                            mce_val = MCES(s1=true, s2=pred, **self.myopic_mces_kwargs)[1]
                            self.mces_cache[(true, pred)] = mce_val
                        dists.append(self.mces_cache[(true, pred)])
                min_mces_dists.append(min(min(dists), mces_thld))
            self._update_metric(
                stage.to_pref() + f"top_{top_k}_min_mces_dist",
                MeanMetric,
                (min_mces_dists,),
                batch_size=len(min_mces_dists),
            )

            # 2. Evaluate Tanimoto similarity:
            # Calculate Tanimoto similarity between top-k predicted molecules and ground truth and
            # report the maximum similarity. The maximum similarities for each sample in the batch
            # are averaged across the epoch.
            fps_pred_top_k = [
                [_get_morgan_fp_with_cache(m) if m is not None else None for m in ms]
                for ms in mols_pred_top_k
            ]
            fp_true = [_get_morgan_fp_with_cache(m) for m in mol_true]

            max_tanimoto_sims = []
            # Iterate over batch
            for preds, true in zip(fps_pred_top_k, fp_true):
                # Iterate over top-k predicted molecule samples
                sims = [
                    TanimotoSimilarity(true, pred)
                    if pred is not None else 0
                    for pred in preds
                ]
                max_tanimoto_sims.append(max(sims))
            self._update_metric(
                stage.to_pref() + f"top_{top_k}_max_tanimoto_sim",
                MeanMetric,
                (max_tanimoto_sims,),
                batch_size=len(max_tanimoto_sims),
            )

            # 3. Evaluate exact match (accuracy):
            # Calculate if the ground truth molecule is in the top-k predicted molecules and report
            # the average across the epoch.
            in_top_k = [
                mol_to_inchi_key(true) in [
                    mol_to_inchi_key(pred)
                    if pred is not None else None
                    for pred in preds
                ]
                for true, preds in zip(mol_true, mols_pred_top_k)
            ]
            self._update_metric(
                stage.to_pref() + f"top_{top_k}_accuracy",
                MeanMetric,
                (in_top_k,),
                batch_size=len(in_top_k)
            )
