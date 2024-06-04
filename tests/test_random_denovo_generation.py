import time
import unittest
from collections.abc import Generator

from rdkit import Chem
from rdkit.Chem import Draw

from massspecgym.models.de_novo import RandomDeNovo
from massspecgym.models.de_novo.random import AtomWithValence


class RandomDeNovoTestcase(unittest.TestCase):
    def setUp(self, draw_molecules: bool = True) -> None:
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
        self.assertEqual(len(all_common_valence_possibilities), 1)

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
        self.assertEqual(len(list(valence_assignment_generator)), 4)

    def test_valence_assignment_5(self):
        valence_assignment_generator = (
            self.generator_with_formula.assigner_of_valences_to_all_atoms(
                unassigned_molecule_elements_with_counts={"C": 17, "H": 14, "O": 4},
                already_assigned_atoms_with_valences={},
                common_valences_only=True,
            )
        )
        self.assertEqual(len(list(valence_assignment_generator)), 1)

    def test_valence_feasibility_check_too_few_edges(self):
        valence_assignment = {
            AtomWithValence(atom_type="C", atom_valence=1): 7,
            AtomWithValence(atom_type="H", atom_valence=1): 5,
            AtomWithValence(atom_type="N", atom_valence=1): 3,
            AtomWithValence(atom_type="O", atom_valence=1): 6,
        }
        is_assignment_feasible = (
            self.generator_with_formula.is_valence_assignment_feasible(
                valence_assignment
            )
        )
        self.assertFalse(is_assignment_feasible)

    def test_valence_feasibility_check_odd_degrees_sum(self):
        valence_assignment = {
            AtomWithValence(atom_type="C", atom_valence=3): 1,
            AtomWithValence(atom_type="C", atom_valence=4): 1,
            AtomWithValence(atom_type="H", atom_valence=1): 6,
        }
        is_assignment_feasible = (
            self.generator_with_formula.is_valence_assignment_feasible(
                valence_assignment
            )
        )
        self.assertFalse(is_assignment_feasible)

    def test_sampling_of_atoms_by_valence_partition(self):
        print(self.generator_with_formula.get_feasible_atom_valence_assignments(
                    "C23H27N5O2"
                ))
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
        molecule = self.generator_with_formula.generate_random_molecule_graphs_via_traversal(chemical_formula='C13H24Cl6O8P2')
        total_secs = time.time() - start_time
        self.assertLess(total_secs, 1)

    def test_random_molecule_generation_for_ion_1(self):
        molecules = self.generator_with_formula.generate_random_molecule_graphs_via_traversal(chemical_formula='C8H18OP+')
        if self.draw_molecules:
            for mol_i, molecule in enumerate(molecules):
                img = Draw.MolToImage(molecule)
                img.save(f'molecule_charged_{mol_i}.png')

    def test_random_molecule_generation_for_ion_2(self):
        molecules = self.generator_with_formula.generate_random_molecule_graphs_via_traversal(chemical_formula='C3H6NO4S-')
        if self.draw_molecules:
            for mol_i, molecule in enumerate(molecules):
                img = Draw.MolToImage(molecule)
                img.save(f'molecule_charged_{mol_i}.png')

    def test_random_molecule_generation(self):
        for gen_i in range(100):
            molecules = self.generator_with_formula.generate_random_molecule_graphs_via_traversal(chemical_formula='C23H17Cl2N5O4')
            if self.draw_molecules:
                for mol_i, molecule in enumerate(molecules):
                    img = Draw.MolToImage(molecule)
                    img.save(f'molecule_{gen_i}_{mol_i}.png')

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
        for batch_i in range(200):
            mol_preds = self.generator_with_formula.step(batch)["mols_pred"]
            print('mol_preds: ', mol_preds)
            if self.draw_molecules:
                for input_mol_i, molecule_smiles in enumerate(mol_preds):
                    for mol_i, _smiles in enumerate(molecule_smiles):
                        print('_smiles: ', _smiles)
                        molecule = Chem.MolFromSmiles(_smiles)
                        img = Draw.MolToImage(molecule)
                        img.save(f'step_molecule_{input_mol_i}_{mol_i}.png')


if __name__ == "__main__":
    unittest.main()
