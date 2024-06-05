import time
import unittest
from collections.abc import Generator

import torch
from rdkit import Chem
from rdkit.Chem import Draw
import pickle

from massspecgym.models.de_novo import RandomDeNovo
from massspecgym.models.de_novo.random import AtomWithValence, ValenceAndCharge


class RandomDeNovoTestcase(unittest.TestCase):
    def setUp(self, draw_molecules: bool = False) -> None:
        self.generator_with_formula = RandomDeNovo(formula_known=True)
        self.generator_without_formula = RandomDeNovo(formula_known=True)
        self.draw_molecules = draw_molecules

    def test_generator_of_element_atoms_split_into_valence_groups_1(self):
        self.assertTrue(
            isinstance(
                self.generator_with_formula.generator_for_splits_of_chem_element_atoms_by_possible_valences(
                    atom_type="N",
                    possible_valences=[3, 5],
                    atom_count=22,
                    already_assigned_groups_of_atoms=dict(),
                ),
                Generator,
            )
        )

    def test_generator_of_element_atoms_split_into_valence_groups_2(self):
        self.assertEqual(
            len(
                list(
                    self.generator_with_formula.generator_for_splits_of_chem_element_atoms_by_possible_valences(
                        atom_type="N",
                        possible_valences=[3, 5],
                        atom_count=22,
                        already_assigned_groups_of_atoms=dict(),
                    )
                )
            ),
            23,
        )

    def test_generator_of_element_atoms_split_into_valence_groups_3(self):
        self.assertEqual(
            len(
                list(
                    self.generator_with_formula.generator_for_splits_of_chem_element_atoms_by_possible_valences(
                        atom_type="Mn",
                        possible_valences=[7, 4, 2],
                        atom_count=22,
                        already_assigned_groups_of_atoms=dict(),
                    )
                )
            ),
            276,
        )

    def test_valence_assignment_1(self):
        valence_assignment_generator = (
            self.generator_with_formula.assigner_of_valences_to_all_atoms(
                unassigned_molecule_elements_with_counts={"C": 1, "O": 2},
                already_assigned_atoms_with_valences={},
                common_valences_only=True,
            )
        )
        all_common_valence_possibilities = list(valence_assignment_generator)
        self.assertEqual(3, len(all_common_valence_possibilities))

    def test_valence_assignment_2(self):
        valence_assignment_generator = (
            self.generator_with_formula.assigner_of_valences_to_all_atoms(
                unassigned_molecule_elements_with_counts={"C": 1, "O": 2},
                already_assigned_atoms_with_valences={},
                common_valences_only=True,
            )
        )
        valence_assignment = list(valence_assignment_generator)[0]
        self.assertEqual(len(valence_assignment), 2)

    def test_valence_assignment_3(self):
        valence_assignment_generator = (
            self.generator_with_formula.assigner_of_valences_to_all_atoms(
                unassigned_molecule_elements_with_counts={"C": 1, "O": 2},
                already_assigned_atoms_with_valences={},
                common_valences_only=True,
            )
        )
        valence_assignment = list(valence_assignment_generator)[0]
        atom_types = {atom.atom_type for atom in valence_assignment.keys()}
        self.assertEqual({"C", "O"}, atom_types)

    def test_valence_assignment_4(self):
        valence_assignment_generator = (
            self.generator_with_formula.assigner_of_valences_to_all_atoms(
                unassigned_molecule_elements_with_counts={
                    "C": 7,
                    "H": 5,
                    "N": 3,
                    "O": 6,
                },
                already_assigned_atoms_with_valences={},
                common_valences_only=True,
            )
        )
        self.assertEqual(28, len(list(valence_assignment_generator)))

    def test_valence_assignment_5(self):
        valence_assignment_generator = (
            self.generator_with_formula.assigner_of_valences_to_all_atoms(
                unassigned_molecule_elements_with_counts={"C": 17, "H": 14, "O": 4},
                already_assigned_atoms_with_valences={},
                common_valences_only=True,
            )
        )
        self.assertEqual(5, len(list(valence_assignment_generator)))

    def test_valence_feasibility_check_too_few_edges(self):
        valence_assignment = {
            AtomWithValence(
                atom_type="C", atom_valence_and_charge=ValenceAndCharge(1, 0)
            ): 7,
            AtomWithValence(
                atom_type="H", atom_valence_and_charge=ValenceAndCharge(1, 0)
            ): 5,
            AtomWithValence(
                atom_type="N", atom_valence_and_charge=ValenceAndCharge(1, 0)
            ): 3,
            AtomWithValence(
                atom_type="O", atom_valence_and_charge=ValenceAndCharge(1, 0)
            ): 6,
        }
        is_assignment_feasible = (
            self.generator_with_formula.is_valence_assignment_feasible(
                valence_assignment
            )
        )
        self.assertFalse(is_assignment_feasible)

    def test_valence_feasibility_check_odd_degrees_sum(self):
        valence_assignment = {
            AtomWithValence(
                atom_type="C", atom_valence_and_charge=ValenceAndCharge(3, -1)
            ): 1,
            AtomWithValence(
                atom_type="C", atom_valence_and_charge=ValenceAndCharge(4, 0)
            ): 1,
            AtomWithValence(
                atom_type="H", atom_valence_and_charge=ValenceAndCharge(1, 0)
            ): 6,
        }
        is_assignment_feasible = (
            self.generator_with_formula.is_valence_assignment_feasible(
                valence_assignment
            )
        )
        self.assertFalse(is_assignment_feasible)

    def test_sampling_of_atoms_by_valence_partition(self):
        self.assertEqual(
            len(
                self.generator_with_formula.get_feasible_atom_valence_assignments(
                    "C23H27N5O2"
                )
            ),
            3,
        )

    def test_random_molecule_generation_hard(self):
        start_time = time.time()
        molecule = (
            self.generator_with_formula.generate_random_molecule_graphs_via_traversal(
                chemical_formula="C13H24Cl6O8P2"
            )
        )
        total_secs = time.time() - start_time
        self.assertLess(total_secs, 1)

    def test_random_molecule_generation_for_ion_1(self):
        molecules = (
            self.generator_with_formula.generate_random_molecule_graphs_via_traversal(
                chemical_formula="C8H18OP+"
            )
        )
        if self.draw_molecules:
            for mol_i, molecule in enumerate(molecules):
                img = Draw.MolToImage(molecule)
                img.save(f"molecule_charged_{mol_i}.png")

    def test_random_molecule_generation_for_ion_2(self):
        molecules = (
            self.generator_with_formula.generate_random_molecule_graphs_via_traversal(
                chemical_formula="C3H6NO4S-"
            )
        )
        if self.draw_molecules:
            for mol_i, molecule in enumerate(molecules):
                img = Draw.MolToImage(molecule)
                img.save(f"molecule_charged_{mol_i}.png")

    def test_random_molecule_generation(self):
        for gen_i in range(100):
            molecules = self.generator_with_formula.generate_random_molecule_graphs_via_traversal(
                chemical_formula="C23H17Cl2N5O4"
            )
            if self.draw_molecules:
                for mol_i, molecule in enumerate(molecules):
                    img = Draw.MolToImage(molecule)
                    img.save(f"molecule_{gen_i}_{mol_i}.png")

    def test_step_function(self):
        batch = {
            "mol": [
                "C/C1=C/CC[C@@]2(C)O[C@@H]2[C@H]2OC(=O)[C@H](CN(C)C)[C@@H]2CC1",
                "COc1ncc2cc(C(=O)Nc3c(Cl)ccc(C(=O)NCc4cc(Cl)ccc4)c3)c(=O)[nH]c2n1",
                "CNC(=O)O[C@H]1COc2c(cc(N3CCN(C4COC4)CC3)cc2)[C@@H]1NC(=O)c1ccc(F)cc1",
                "COc1nc(N2CCC3(CCCN(Cc4c[nH]c5ccccc45)C3=O)CC2)ncc1",
                "Cc1c(C)c2c(cc1)c(=O)c1cccc(CC(=O)O)c1o2",
            ]
        }
        for batch_i in range(100):
            mol_preds = self.generator_with_formula.step(batch)["mols_pred"]
            if self.draw_molecules:
                for input_mol_i, molecule_smiles in enumerate(mol_preds):
                    for mol_i, _smiles in enumerate(molecule_smiles):
                        molecule = Chem.MolFromSmiles(_smiles)
                        img = Draw.MolToImage(molecule)
                        img.save(f"step_molecule_{input_mol_i}_{mol_i}.png")

    def test_single_candidate_generation(self):
        generator = RandomDeNovo(formula_known=True, max_top_k=1)
        batch = {
            "mol": [
                "CCCC[C@@H](C)[C@H]([C@H](C[C@@H](C)C[C@@H](CCCCCC[C@@H]([C@@H](C)N)O)O)OC(=O)CC(CC(=O)O)C(=O)O)OC(=O)CC(CC(=O)O)C(=O)O",
            ]
        }
        for _ in range(100):
            self.assertEqual(1, len(generator.step(batch)["mols_pred"]))

    def test_slow_eval_molecule(self):
        molecules = (
            self.generator_with_formula.generate_random_molecule_graphs_via_traversal(
                chemical_formula="C23H17Cl2N5O4"
            )
        )

    def test_weight_recording(self):
        batch = {
            "mol": [
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=CNC7=CC=CC(=C67)C5=C4)C)O",
            ],
        }
        self.generator_without_formula.training_step(batch, batch_idx=torch.Tensor())
        self.assertEqual(
            {575.3107694120001: ["C32H41N5O5"], 908.6085741279999: ["C48H84N4O12"]},
            self.generator_without_formula.mol_weight_2_formulas,
        )

    def test_weight_recording_on_train_end(self):
        batch = {
            "mol": [
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "C",  # dummy for the test
                "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=CNC7=CC=CC(=C67)C5=C4)C)O",
            ],
        }
        self.generator_without_formula.training_step(batch, batch_idx=torch.Tensor())
        self.generator_without_formula.on_train_end()
        self.assertEqual(
            {
                16.031300127999998: [["CH4"], [1.0]],
                575.3107694120001: [["C32H41N5O5"], [1.0]],
                908.6085741279999: [["C48H84N4O12"], [1.0]],
            },
            self.generator_without_formula.mol_weight_2_formulas,
        )

    def test_sample_formula_with_the_closest_molecular_weight_1(self):
        batch = {
            "mol": [
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "C",  # dummy for the test
                "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=CNC7=CC=CC(=C67)C5=C4)C)O",
            ],
        }
        self.generator_without_formula.training_step(batch, batch_idx=torch.Tensor())
        self.generator_without_formula.on_train_end()

        self.assertEqual(
            "CH4",
            self.generator_without_formula.sample_formula_with_the_closest_molecular_weight(
                20
            ),
        )

    def test_sample_formula_with_the_closest_molecular_weight_2(self):
        batch = {
            "mol": [
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "C",  # dummy for the test
                "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=CNC7=CC=CC(=C67)C5=C4)C)O",
            ],
        }
        self.generator_without_formula.training_step(batch, batch_idx=torch.Tensor())
        self.generator_without_formula.on_train_end()

        self.assertEqual(
            "C32H41N5O5",
            self.generator_without_formula.sample_formula_with_the_closest_molecular_weight(
                300
            ),
        )

    def test_sample_formula_with_the_closest_molecular_weight_3(self):
        batch = {
            "mol": [
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "C",  # dummy for the test
                "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=CNC7=CC=CC(=C67)C5=C4)C)O",
            ],
        }
        self.generator_without_formula.training_step(batch, batch_idx=torch.Tensor())
        self.generator_without_formula.on_train_end()

        self.assertEqual(
            "C48H84N4O12",
            self.generator_without_formula.sample_formula_with_the_closest_molecular_weight(
                908.6085741279999
            ),
        )

    def test_chemical_elements_stats_computation_train_step(self):
        batch = {
            "mol": [
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "C",  # dummy for the test
                "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=CNC7=CC=CC(=C67)C5=C4)C)O",
            ],
        }
        generator_with_stats = RandomDeNovo(estimate_chem_element_stats=True)
        generator_with_stats.training_step(batch, batch_idx=torch.Tensor())

        self.assertEqual(
            118,
            generator_with_stats.element_2_bond_stats["C"][
                ValenceAndCharge(valence=4, charge=0)
            ][0][()][("C", 4, 0, 1.0)],
        )

    def test_chemical_elements_stats_computation_post_train(self):
        batch = {
            "mol": [
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "CC(C)C[C@H]1C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N1C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C)CC(C)C)C)C(C)C",
                "C",  # dummy for the test
                "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=CNC7=CC=CC(=C67)C5=C4)C)O",
            ],
        }
        generator_with_stats = RandomDeNovo(estimate_chem_element_stats=True)
        generator_with_stats.training_step(batch, batch_idx=torch.Tensor())
        generator_with_stats.on_train_end()

        self.assertEqual(
            ([(1.0, 118), (2.0, 10)], 128),
            generator_with_stats.element_2_bond_stats["C"][
                ValenceAndCharge(valence=4, charge=0)
            ][0][()][
                AtomWithValence(
                    atom_type="C",
                    atom_valence_and_charge=ValenceAndCharge(valence=4, charge=0),
                )
            ],
        )

    def test_random_molecule_generation_with_stats_for_ion_1(self):
        generator_with_stats = RandomDeNovo(estimate_chem_element_stats=True)
        with open("element_stats_from_trn.pkl", "rb") as file:
            generator_with_stats.element_2_bond_stats = pickle.load(file)
        molecules = (
            self.generator_with_formula.generate_random_molecule_graphs_via_traversal(
                chemical_formula="C8H18OP+"
            )
        )
        if self.draw_molecules:
            for mol_i, molecule in enumerate(molecules):
                img = Draw.MolToImage(molecule)
                img.save(f"molecule_stats_charged_{mol_i}.png")

    def test_random_molecule_generation_with_stats(self):
        generator_with_stats = RandomDeNovo(estimate_chem_element_stats=True)
        with open("element_stats_from_trn.pkl", "rb") as file:
            generator_with_stats.element_2_bond_stats = pickle.load(file)
        for gen_i in range(50):
            molecules = (
                generator_with_stats.generate_random_molecule_graphs_via_traversal(
                    chemical_formula="C23H17Cl2N5O4"
                )
            )
            if self.draw_molecules:
                for mol_i, molecule in enumerate(molecules):
                    img = Draw.MolToImage(molecule)
                    img.save(f"molecule_with_stats_{gen_i}_{mol_i}.png")


if __name__ == "__main__":
    unittest.main()
