from collections import deque, defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from random import choice, shuffle

import chemparse
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdchem import Mol, BondType

from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel

# type aliases for code readability
chem_element = str
number_of_atoms = int


@dataclass(frozen=True)
class AtomWithValence:
    """
    A data class to store atom info including the computed valence
    """

    atom_type: chem_element
    atom_valence: int


@dataclass
class AtomNodeForRandomTraversal:
    """
    A data class to store atom info including the computed valence
    """

    atom_type: chem_element
    _remaining_node_degree: int

    @property
    def remaining_node_degree(self):
        """remaining_node_degree variable getter"""
        return self._remaining_node_degree

    @remaining_node_degree.setter
    def remaining_node_degree(self, value: int):
        """remaining_node_degree variable setter"""
        self._remaining_node_degree = value


def create_rdkit_molecule_from_edge_list(
    edge_list: list[tuple[int, int]], all_graph_nodes: list[AtomNodeForRandomTraversal]
) -> Mol:
    """
    A helper function converting a randomly generated edge list into rdkit.Chem.rdchem.Mol object
    @param edge_list: a list of edges, where each edge is specified by the index of its nodes
    @param all_graph_nodes: a list of all atomic nodes in the molecular graph
    """
    # first we traverse all randomly generated edges and compute bond types between each pair of atoms
    edge_2_bondtype = defaultdict(int)
    for edge_node_i, edge_node_j in edge_list:
        edge_2_bondtype[
            (min(edge_node_i, edge_node_j), max(edge_node_i, edge_node_j))
        ] += 1

    # helper routine to get the rdking enum bondtype
    def _get_rdkit_bondtype(bondtype: int) -> BondType:
        int_bondtype_2_enum = {
            1: BondType.SINGLE,
            2: BondType.DOUBLE,
            3: BondType.TRIPLE,
            4: BondType.QUADRUPLE,
            5: BondType.QUINTUPLE,
            6: BondType.HEXTUPLE,
        }
        try:
            return int_bondtype_2_enum[bondtype]
        except KeyError:
            raise NotImplementedError(f"Bond type {bondtype} is not supported")

    edge_list_rdkit = [
        (node_i, node_j, _get_rdkit_bondtype(bondtype))
        for (node_i, node_j), bondtype in edge_2_bondtype.items()
    ]
    # creating an empty editable molecule
    mol = Chem.RWMol()
    # adding the atoms to the molecule object
    for atom in all_graph_nodes:
        mol.AddAtom(Chem.Atom(atom.atom_type))
    # adding bonds
    for (edge_node_i, edge_node_j, bond_type) in edge_list_rdkit:
        mol.AddBond(edge_node_i, edge_node_j, bond_type)
    # returning the rdkit.Chem.rdchem.Mol object
    return mol.GetMol()


class RandomDeNovo(DeNovoMassSpecGymModel):
    def __init__(
        self, formula_known: bool = False, count_of_valid_valence_assignments: int = 3
    ):
        """

        @param formula_known: a boolean flag about the information available prior to generation
                              If formula_known is True, we should generate molecules with the specified formula
                              If formula_known is False, we should generate any molecule with the specified mass
        @param count_of_valid_valence_assignments: an integer controlling process of selecting valence assignment
                                                   to each atom in the generated molecule.
                                                   `count_of_valid_valence_assignments` of assignment corresponding to
                                                    the formula are generated, then one assignment is is picked at random.
                                                    The default is set to 3 for the computational speed purposes.
                                                    When setting to 1, the first feasible valence assignment will be used.
        """
        super(RandomDeNovo, self).__init__()
        self.formula_known = formula_known
        self.count_of_valid_valence_assignments = count_of_valid_valence_assignments

    def generator_for_splits_of_chem_element_atoms_by_possible_valences(
        self,
        atom_type: chem_element,
        possible_valences: list[int],
        atom_count: int,
        already_assigned_groups_of_atoms: dict[AtomWithValence, number_of_atoms],
    ) -> Generator[dict[AtomWithValence, number_of_atoms]]:
        """
        A recursive generator function to iterate over all possible partitions of element atoms
        into groups with different valid valences.
        Each allowed valence value can have any number from atoms, from zero up to total `atom_count`
        @param atom_type: chemical element
        @param possible_valences: a list of allowed valences
        @param atom_count: a total number of element atoms to split into valence groups
        @param already_assigned_groups_of_atoms: partial results to pass into the subsequent recursive calls

        @return A generator for lazy enumeration over all possible splits of `atom_count` atoms into subgroups
                of valid valences specified in `possible valences` parameters.
                Each return value is a dictionary, mapping atom with fixed valence to a total count of such instances
                in the molecule.

        @note In the future the method can be made into a function in a separate utils module,
        for the simplicity of codebase organization and testing purposes it's kept as the method for now
        """
        # the check for a base case of the recursion
        if atom_count == 0:
            yield already_assigned_groups_of_atoms
        elif len(possible_valences):
            # taking the first valence value from the possible ones
            next_valence = possible_valences[0]
            # iterating over possible sizes for a group of atoms with `next_valence` value of the valence
            for size_of_group in range(atom_count, -1, -1):
                # recording the assigned size of the group
                already_assigned_groups_of_atoms_next = (
                    already_assigned_groups_of_atoms.copy()
                )
                atom_with_valence = AtomWithValence(
                    atom_type=atom_type, atom_valence=next_valence
                )
                already_assigned_groups_of_atoms_next[atom_with_valence] = size_of_group
                yield from self.generator_for_splits_of_chem_element_atoms_by_possible_valences(
                    atom_type=atom_type,
                    possible_valences=possible_valences[1:],
                    atom_count=atom_count - size_of_group,
                    already_assigned_groups_of_atoms=already_assigned_groups_of_atoms_next,
                )

    def assigner_of_valences_to_all_atoms(
        self,
        unassigned_molecule_elements_with_counts: dict[chem_element, number_of_atoms],
        already_assigned_atoms_with_valences: dict[AtomWithValence, number_of_atoms],
        common_valences_only: bool = True,
    ) -> Generator[dict[AtomWithValence, number_of_atoms]]:
        """
        A recursive function to iterate over all possible valid assignments of valences for each atom in the molecule
        @param unassigned_molecule_elements_with_counts: a dictionary representation of a molecule,
                                                         mapping each present element to a corresponding number of atoms.
                                                         The function is recursive, in the subsequence calls
                                                         the dictionary represents an yet-unprocessed submolecule
        @param already_assigned_atoms_with_valences: partial results to pass into the subsequent recursive calls,
                                                     stored as a dictionary, mapping atom with fixed valence
                                                     to a total count of such atoms in the molecule
        @param common_valences_only: a flag for using the common valence values for each element

        @return A generator for lazy enumeration over all possible assignments of all molecule atoms into subgroups
                defined by valences. Valence values are the valid ones for the corresponding chemical element.
                Each return value is a dictionary, mapping atom of specified chemical element with a fixed valence
                to a total count of such atoms in the molecule.

        @note In the future the method can be made into a function in a separate utils module,
        for the simplicity of codebase organization and testing purposes it's kept as the method for now
        """
        # the check for a base case of the recursion
        if len(unassigned_molecule_elements_with_counts) == 0:
            yield already_assigned_atoms_with_valences
        else:
            # processing the next chemical element in the molecule
            chem_element_type, atom_count = list(
                unassigned_molecule_elements_with_counts.items()
            )[0]
            # for the subsequence recursive calls the picked atom will be removed from the yet-to-be-processed
            remaining_unassigned_atoms_with_counts = (
                unassigned_molecule_elements_with_counts.copy()
            )
            del remaining_unassigned_atoms_with_counts[chem_element_type]
            # generating splits of the element count into groups with possible valences
            valences_common, valences_others = ELEMENT_VALENCES[
                chem_element_type.capitalize()
            ]
            possible_element_valences = (
                valences_common
                if common_valences_only
                else valences_common + valences_others
            )
            # we ignore "the direction" of ionic bonds, therefore we work with absolute values of valences
            possible_element_valences = map(
                lambda x: np.abs(x), possible_element_valences
            )
            # we require a connected molecule graph, so we ignore possible 0 values of valences
            possible_element_valences = list(
                set(filter(lambda x: x > 0, possible_element_valences))
            )
            # creating a generator for lazy enumeration over all possible splits of element atoms
            # into subgroups of possible valid valences
            valence_split_generator = (
                self.generator_for_splits_of_chem_element_atoms_by_possible_valences(
                    atom_type=chem_element_type,
                    possible_valences=possible_element_valences,
                    atom_count=atom_count,
                    already_assigned_groups_of_atoms=dict(),
                )
            )
            # iterating over splits of the element count into groups with possible valences
            for element_atoms_with_valence_2_count in valence_split_generator:
                already_assigned_atoms_with_valences_new = (
                    already_assigned_atoms_with_valences.copy()
                )
                already_assigned_atoms_with_valences_new.update(
                    element_atoms_with_valence_2_count
                )
                yield from self.assigner_of_valences_to_all_atoms(
                    unassigned_molecule_elements_with_counts=remaining_unassigned_atoms_with_counts,
                    already_assigned_atoms_with_valences=already_assigned_atoms_with_valences_new,
                    common_valences_only=common_valences_only,
                )

    def is_valence_assignment_feasible(
        self, valence_assignment: dict[AtomWithValence, number_of_atoms]
    ) -> bool:
        """
        A function for checking if the valence assignment to all molecule atoms can be feasible

        @param valence_assignment: an assignment of all molecule atoms into subgroups of plausible valences

        @note In the future the method can be made into a function in a separate utils module,
        for the simplicity of codebase organization and testing purposes it's kept as the method for now
        """
        # considering a molecule as a graph with atom being nodes and chemical bonds being edges
        # computing sum of all node degrees
        sum_of_all_node_degrees = sum(
            [
                atom.atom_valence * count_of_atoms
                for atom, count_of_atoms in valence_assignment.items()
            ]
        )
        if sum_of_all_node_degrees % 2 == 1:
            # the valence assignment is infeasible as in the graph the number of edges is half of the total degrees sum
            # therefore the sum_of_all_node_degrees must be an even number
            return False
        total_number_of_bonds = sum_of_all_node_degrees / 2
        # the total number of all atoms in the whole molecule
        total_number_of_atoms_in_molecule = sum(valence_assignment.values())
        if total_number_of_bonds < total_number_of_atoms_in_molecule - 1:
            # the valence assignment is infeasible as the molecule graph cannot be connected
            return False
        return True

    def get_feasible_atom_valence_assignments(
        self, chemical_formula: str
    ) -> list[dict[AtomWithValence, number_of_atoms]]:
        """
        A function generating candidate assignments of valences to individual atoms in the molecule.
        Candidates are returned in a random order.
        @param chemical_formula: a string containing the chemical formula of the molecule

        @note In the future the method can be made into a function in a separate utils module,
        for the simplicity of codebase organization and testing purposes it's kept as the method for now
        """
        # parsing chemical formula into a dictionary of elements with corresponding counts
        element_2_count = {
            element: int(count)
            for element, count in chemparse.parse_formula(chemical_formula).items()
        }
        # checking that all input elements are valid
        for element in element_2_count.keys():
            if element.capitalize() not in ELEMENT_VALENCES:
                raise ValueError(
                    f"Found an unknown element {element.capitalize()} in the formula {chemical_formula}"
                )

        # estimate the total number of all atoms in the whole molecule
        # it will be used to check validity of the valence assignments
        total_number_of_atoms_in_molecule = sum(element_2_count.values())
        generated_candidate_valence_assignments = []
        valence_assignment_generator = self.assigner_of_valences_to_all_atoms(
            unassigned_molecule_elements_with_counts=element_2_count,
            already_assigned_atoms_with_valences=dict(),
            common_valences_only=True,
        )
        termination_assignment_value = {AtomWithValence("No more assignments", -1): -1}
        next_valence_assignment = next(
            valence_assignment_generator, termination_assignment_value
        )
        while (
            len(generated_candidate_valence_assignments)
            < self.count_of_valid_valence_assignments
            and next_valence_assignment != termination_assignment_value
        ):
            if self.is_valence_assignment_feasible(next_valence_assignment):
                generated_candidate_valence_assignments.append(next_valence_assignment)
            next_valence_assignment = next(
                valence_assignment_generator, termination_assignment_value
            )
        # if no valence assignment was found with common valences,
        # then try generating assignments including not-common valences
        if len(generated_candidate_valence_assignments) == 0:
            valence_assignment_generator = self.assigner_of_valences_to_all_atoms(
                unassigned_molecule_elements_with_counts=element_2_count,
                already_assigned_atoms_with_valences=dict(),
                common_valences_only=False,
            )
            next_valence_assignment = next(
                valence_assignment_generator, termination_assignment_value
            )
            while (
                len(generated_candidate_valence_assignments)
                < self.count_of_valid_valence_assignments
                and next_valence_assignment != termination_assignment_value
            ):
                if self.is_valence_assignment_feasible(next_valence_assignment):
                    generated_candidate_valence_assignments.append(
                        next_valence_assignment
                    )
                next_valence_assignment = next(
                    valence_assignment_generator, termination_assignment_value
                )

        if len(generated_candidate_valence_assignments) == 0:
            raise ValueError(
                f"No valence assignments can be generated for the formula {chemical_formula}"
            )
        shuffle(generated_candidate_valence_assignments)
        return generated_candidate_valence_assignments

    def generate_random_molecule_graph_via_traversal(self, chemical_formula: str):
        """
        A function generating a random molecule graph.
        The generation process ensures that the graph is connected.
        If any of the `self.count_of_valid_valence_assignments` enables it,
        the function returns a graphs without self-loops.

        @param chemical_formula: a string containing the chemical formula of the molecule

        @note In the future the method can be made into a function in a separate utils module,
        for the simplicity of codebase organization and testing purposes it's kept as the method for now
        """
        # get candidate partitions of all molecule atoms into valences
        candidate_valence_assignments = self.get_feasible_atom_valence_assignments(
            chemical_formula
        )
        # iterate over each valence assignment to all atoms, the order is random
        assert (
            len(candidate_valence_assignments) > 0
        ), f"No potentially feasible atom valence assignment for {chemical_formula}"
        for valence_assignment in candidate_valence_assignments:
            # first randomly create a spanning tree of the molecule graph, to ensure the connectivity of molecule.
            # The feasibility check `self.is_valence_assignment_feasible` inside the
            # `self.get_feasible_atom_valence_assignments` function should ensure the possibility to create the tree.
            spanning_tree_was_generated = False
            while not spanning_tree_was_generated:
                # we optimistically set the value of `spanning_tree_was_generated` to True,
                # If the current traversal do not lead to a spanning tree,
                # then `spanning_tree_was_generated` is set to False in the code below
                spanning_tree_was_generated = True

                # prepare node list for a random edges generation
                all_graph_nodes = []
                for (
                    atom_with_valence,
                    num_of_atoms_in_molecule,
                ) in valence_assignment.items():
                    for _ in range(num_of_atoms_in_molecule):
                        all_graph_nodes.append(
                            AtomNodeForRandomTraversal(
                                atom_with_valence.atom_type,
                                atom_with_valence.atom_valence,
                            )
                        )

                # a set of nodes which still have unpaired electrons
                # (as indicated by the `remaining_node_degree` attribute)
                nodes_open_to_traversal = set(range(len(all_graph_nodes)))
                # the final edge list will be stored into the variable below.
                # An edge is defined by a pair of position indices in the `all_graph_nodes` list
                edge_list = []

                # the nodes already included into the spanning tree
                # the set is used for quick blacklisting, while the list is used for possible backtracking when
                spanning_tree_visited_nodes_set, spanning_tree_traversal_list = (
                    set(),
                    deque(),
                )
                # sample a random start of spanning tree generation
                edge_start_node_i = choice(list(nodes_open_to_traversal))
                spanning_tree_visited_nodes_set.add(edge_start_node_i)
                spanning_tree_traversal_list.append(edge_start_node_i)
                while len(spanning_tree_visited_nodes_set) < len(all_graph_nodes):
                    possible_candidates_for_end_node = (
                        nodes_open_to_traversal.difference(
                            spanning_tree_visited_nodes_set
                        )
                    )
                    # sample the next node for the spanning tree
                    edge_end_node_i = choice(list(possible_candidates_for_end_node))
                    # note that the graph is undirected, start-end node refers to the random traversal only
                    edge_list.append((edge_start_node_i, edge_end_node_i))
                    # recording the new node added to the random spanning tree
                    spanning_tree_visited_nodes_set.add(edge_end_node_i)
                    spanning_tree_traversal_list.append(edge_end_node_i)
                    # decrease the node degrees correspondingly
                    for node_of_a_new_edge_i in [edge_start_node_i, edge_end_node_i]:
                        all_graph_nodes[node_of_a_new_edge_i].remaining_node_degree -= 1
                        # if all bonds are created for the particular atom, it is no more open for traversal
                        if (
                            all_graph_nodes[node_of_a_new_edge_i].remaining_node_degree
                            == 0
                        ):
                            nodes_open_to_traversal.remove(node_of_a_new_edge_i)
                    # finding a start node for the next sampled edge.
                    # We have to ensure that such a node still has some degree not covered by sampling nodes.
                    # For that, we might need to backtrack.
                    candidate_for_start_node_i = edge_end_node_i
                    try:
                        while (
                            all_graph_nodes[
                                candidate_for_start_node_i
                            ].remaining_node_degree
                            == 0
                        ):
                            spanning_tree_traversal_list.pop()
                            candidate_for_start_node_i = spanning_tree_traversal_list[
                                -1
                            ]
                    except IndexError:
                        spanning_tree_was_generated = False
                        break
                    edge_start_node_i = candidate_for_start_node_i

            # after the spanning tree edges were sampled,
            # now we randomly connect nodes with remaining degrees yet uncovered by sampled bonds
            while len(nodes_open_to_traversal) >= 2:
                # sample edge nodes
                edge_start_node_i = choice(list(nodes_open_to_traversal))
                possible_candidates_for_end_node = nodes_open_to_traversal.difference(
                    {edge_start_node_i}
                )
                edge_end_node_i = choice(list(possible_candidates_for_end_node))
                edge_list.append((edge_start_node_i, edge_end_node_i))
                # decrease the node degrees correspondingly
                for node_of_a_new_edge_i in [edge_start_node_i, edge_end_node_i]:
                    all_graph_nodes[node_of_a_new_edge_i].remaining_node_degree -= 1
                    # if all bonds are created for the particular atom, it is no more open for traversal
                    if all_graph_nodes[node_of_a_new_edge_i].remaining_node_degree == 0:
                        nodes_open_to_traversal.remove(node_of_a_new_edge_i)
            # if all nodes were covered by edges without self-loops, then we return the molecule
            if len(nodes_open_to_traversal) == 0:
                return create_rdkit_molecule_from_edge_list(edge_list, all_graph_nodes)
        return create_rdkit_molecule_from_edge_list(edge_list, all_graph_nodes)

    def step(
        self, batch: dict, metric_pref: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mols = batch["mol"]  # List of SMILES of length batch_size

        # If formula_known is True, we should generate molecules with the same formula as label (`mols` above)
        # If formula_known is False, we should generate any molecule with the same mass as label (`mols` above)

        # getting the formula from SMILES
        if self.formula_known:
            formulas = []
            for smiles in mols:
                molecule = Chem.MolFromSmiles(smiles)
                formulas.append(CalcMolFormula(molecule))
        else:
            raise NotImplementedError
        # (bs, k) list of rdkit molecules
        mols_pred = [
            self.generate_random_molecule_graph_via_traversal(formula)
            for formula in formulas
        ]

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)
        return dict(loss=loss, mols_pred=mols_pred)

    def configure_optimizers(self):
        # No optimizer needed for a random baseline
        return None


# element valences taken from https://sciencenotes.org/element-valency-pdf
# the first list contains the typical valences
ELEMENT_VALENCES = {
    "H": ([1], [0, -1]),
    "He": ([0], []),
    "Li": ([1], [-1]),
    "Be": ([2], []),
    "B": ([3], [2, 1]),
    "C": ([4, -4], [2, -2, 1, -1, 3]),
    "N": ([-3, 5], [3, 2, 4, 1, 0, -1, -2]),
    "O": ([-2], [-1, 1, 2, 0]),
    "F": ([-1], [0]),
    "Ne": ([0], []),
    "Na": ([1], [-1]),
    "Mg": ([2], []),
    "Al": ([3], [1]),
    "Si": ([4], [-4, 2, -2, 3, 1, -1]),
    "P": ([3, 5, -3], [4, 2, 1, 0, -1, -2]),
    "S": ([-2, 6], [2, 4, 3, 5, 1, 0, -1]),
    "Cl": ([-1], [-2, 0, 1, 2, 3, 4, 5, 6]),
    "Ar": ([0], []),
    "K": ([1], [-1]),
    "Ca": ([2], []),
    "Sc": ([3], [2, 1]),
    "Ti": ([4], [3, 2, 0, -1, -2]),
    "V": ([5, 4, 3], [2, 1, 0, -1, -2]),
    "Cr": ([6, 3, 2], [5, 4, 1, 0, -1, -2, -3, -4]),
    "Mn": ([7, 4, 2], [6, 5, 3, 1, 0, -1, -2, -3]),
    "Fe": ([2, 3], [6, 5, 4, 1, 0, -1, -2]),
    "Co": ([2, 3], [5, 4, 1, 0, -1]),
    "Ni": ([2], [6, 4, 3, 1, 0, -1]),
    "Cu": ([2, 1], [4, 3, 0]),
    "Zn": ([2], [1, 0]),
    "Ga": ([3], [2, 1]),
    "Ge": ([4], [3, 2, 1]),
    "As": ([3, 5], [-3, 2]),
    "Se": ([-2, 4], [6, 2, 1]),
    "Br": ([-1], [7, 5, 4, 3, 1, 0]),
    "Kr": ([0], [2]),
    "Rb": ([1], [-1]),
    "Sr": ([2], []),
    "Y": ([3], [2]),
    "Zr": ([4], [3, 2, 1, 0, -2]),
    "Nb": ([5], [4, 3, 2, 1, 0, -1, -3]),
    "Mo": ([6, 4], [5, 3, 2, 1, 0, -1, -2]),
    "Tc": ([7, 4], [6, 5, 3, 2, 1, 0, -1, -3]),
    "Ru": ([4, 3], [8, 7, 6, 5, 2, 1, 0, -2]),
    "Rh": ([3], [6, 5, 4, 2, 1, 0, -1]),
    "Pd": ([4, 2], [0]),
    "Ag": ([1], [3, 2, 0]),
    "Cd": ([2], [1]),
    "In": ([3], [2, 1]),
    "Sn": ([2, -4], [4]),
    "Sb": ([3], [5, -3]),
    "Te": ([4], [-2, 6]),
    "I": ([-1, 5], [1, 3, 7, 0]),
    "Xe": ([0], [2, 4, 6, 8]),
    "Cs": ([1], [-1]),
    "Ba": ([2], []),
    "La": ([3], []),
    "Ce": ([3], [4]),
    "Pr": ([3], [4]),
    "Nd": ([3], []),
    "Pm": ([3], []),
    "Sm": ([3], []),
    "Eu": ([3], [2]),
    "Gd": ([3], []),
    "Tb": ([3], [4]),
    "Dy": ([3], []),
    "Ho": ([3], []),
    "Er": ([3], []),
    "Tm": ([3], []),
    "Yb": ([3], [2]),
    "Lu": ([3], []),
    "Hf": ([4], []),
    "Ta": ([5], []),
    "W": ([6, 4], [5, 3, 2]),
    "Re": ([5, 4, 3], [7, 6, 2, 1, 0, -1]),
    "Os": ([4], [8, 6, 2]),
    "Ir": ([4, 3], [6, 4]),
    "Pt": ([2], [4]),
    "Au": ([3], [1]),
    "Hg": ([2], [1]),
    "Tl": ([3], [1]),
    "Pb": ([4], [2]),
    "Bi": ([3, 1], [5, -3]),
    "Po": ([4], [2]),
    "At": ([-1], [5, 3, 1, 7]),
    "Rn": ([0], [2]),
    "Fr": ([1], []),
    "Ra": ([2], []),
    "Ac": ([3], []),
    "Th": ([4], []),
    "Pa": ([5], [4]),
    "U": ([6], [5, 4, 3]),
    "Np": ([7], [6, 5, 4, 3]),
    "Pu": ([7, 4], [6, 5, 3]),
    "Am": ([3], [5, 4]),
    "Cm": ([6, 5, 3], []),
    "Bk": ([3], [4]),
    "Cf": ([3], []),
    "Es": ([3], []),
    "Fm": ([3], []),
    "Md": ([3], []),
    "No": ([3], [2]),
    "Lr": ([3], []),
    "Rf": ([4], []),
    "Db": ([5], []),
    "Sg": ([6], []),
    "Bh": ([7], []),
    "Hs": ([8], []),
    "Mt": ([8], []),
    "Ds": ([8], []),
    "Rg": ([8], []),
    "Cn": ([2], []),
    "Nh": ([3], []),
    "Fl": ([4], []),
    "Mc": ([3], []),
    "Lv": ([4], []),
    "Ts": ([7], []),
    "Og": ([0], []),
}
