from collections import deque, defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from random import choice, shuffle

import chemparse
import numpy as np
import torch
from massspecgym.models.base import Stage
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdchem import Mol, BondType
from copy import deepcopy
from collections import Counter
import bisect
from itertools import combinations

# type aliases for code readability
chem_element = str
number_of_atoms = int


@dataclass(frozen=True, order=True)
class ValenceAndCharge:
    """
    A data class to store valence value with the corresponding charge
    """

    valence: int
    charge: int


@dataclass(frozen=True, order=True)
class AtomWithValence:
    """
    A data class to store atom info including the computed valence
    """

    atom_type: chem_element
    atom_valence_and_charge: ValenceAndCharge


@dataclass(frozen=True, order=True)
class BondToNeighbouringAtom:
    """
    A data class to store info about the adjacent atom
    """

    adjacent_atom: AtomWithValence
    bond_type: int


@dataclass
class AtomNodeForRandomTraversal:
    """
    A data class to store atom info including the computed valence
    """

    atom_with_valence: AtomWithValence
    _remaining_node_degree: int = None
    _remaining_node_charge: int = None

    def __post_init__(self):
        """Setting up remaining node degree and charge for random traversal"""
        self._remaining_node_degree = (
            self.atom_with_valence.atom_valence_and_charge.valence
        )
        self._remaining_node_charge = (
            self.atom_with_valence.atom_valence_and_charge.charge
        )

    @property
    def remaining_node_degree(self):
        """remaining_node_degree variable getter"""
        return self._remaining_node_degree

    @remaining_node_degree.setter
    def remaining_node_degree(self, value: int):
        """remaining_node_degree variable setter"""
        self._remaining_node_degree = value

    @property
    def remaining_node_charge(self):
        """remaining_node_charge variable getter"""
        return self._remaining_node_charge

    @remaining_node_charge.setter
    def remaining_node_charge(self, value: int):
        """remaining_node_charge variable setter"""
        self._remaining_node_charge = value


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

    # as some all_graph nodes can represent charges, we have to remember mapping of molecular atom index to
    # the corresponding atom index in all_graph_nodes
    all_graph_atom_idx_2_mol_atom_idx = {}
    for all_graph_atom_idx, atom in enumerate(all_graph_nodes):
        # ignoring charge-related graph nodes
        if atom.atom_with_valence.atom_type not in {"+", "-"}:
            all_graph_atom_idx_2_mol_atom_idx[all_graph_atom_idx] = mol.GetNumAtoms()
            next_atom = Chem.Atom(atom.atom_with_valence.atom_type)
            next_atom.SetFormalCharge(
                atom.atom_with_valence.atom_valence_and_charge.charge
            )
            mol.AddAtom(next_atom)

    # adding bonds
    for (edge_node_i, edge_node_j, bond_type) in edge_list_rdkit:
        # checking if the edge represents a charge of connected atom
        the_edge_represents_charge = len(
            {
                all_graph_nodes[node_i].atom_with_valence.atom_type
                for node_i in [edge_node_i, edge_node_j]
            }.intersection({"+", "-"})
        )
        if the_edge_represents_charge:
            # setting a charge to the corresponding atom
            for node_i in [edge_node_i, edge_node_j]:
                if all_graph_nodes[node_i].atom_with_valence.atom_type in {"+", "-"}:
                    charge_value = (
                        1
                        if all_graph_nodes[node_i].atom_with_valence.atom_type == "+"
                        else -1
                    )
                else:
                    atom_node_i = node_i
            mol.GetAtomWithIdx(
                all_graph_atom_idx_2_mol_atom_idx[atom_node_i]
            ).SetFormalCharge(charge_value)
        else:
            mol.AddBond(
                all_graph_atom_idx_2_mol_atom_idx[edge_node_i],
                all_graph_atom_idx_2_mol_atom_idx[edge_node_j],
                bond_type,
            )
    # returning the rdkit.Chem.rdchem.Mol object
    return mol.GetMol()


class RandomDeNovo(DeNovoMassSpecGymModel):
    def __init__(
        self,
        formula_known: bool = True,
        count_of_valid_valence_assignments: int = 10,
        estimate_chem_element_stats: bool = False,
        max_top_k: int = 10,
        enforce_connectivity: bool = True,
        cache_results: bool = True,
        **kwargs
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
        @param estimate_chem_element_stats: a boolean flag controlling if prior information about elements' valences
                                            and bond type distributions is estimated from training data
        @param max_top_k: a maximum number of candidates to generate. If the count of valid valence assignments do
                          not allow generation of max_top_k, then less candidates are returned
        @param enforce_connectivity: a boolean flag controlling connectivity of randomly generated molecules.
                                     When it is set to True, first a random spanning tree is sampled
        @param cache_results: a boolean flag controlling caching of already generated structures.
                              When set to True, for each unique formula the set of random molecules is cached to avoid
                              recomputation.
        """
        super(RandomDeNovo, self).__init__(**kwargs)
        self.formula_known = formula_known
        self.count_of_valid_valence_assignments = count_of_valid_valence_assignments
        self.estimate_chem_element_stats = estimate_chem_element_stats
        self.max_top_k = min(max(self.top_ks), max_top_k)
        self.enforce_connectivity = enforce_connectivity
        # prior chemical knownledge about element valences
        self.element_2_valences = ELEMENT_VALENCES
        # a dictionary structure to record molecular weights with corresponding formulas from training data
        # during training steps, for each molecular weight we record all encountered formulas
        # then on training end we compute proportions of the formulas and record it as a mapping
        # mol_weight -> [[formula_1, formula_2], [proportion_of_formula_1, proportion_of_formula_2]]
        self.mol_weight_2_formulas = defaultdict(list)
        # a helper array to store sorted list of train molecular weights.
        # It will be used for the O(logn) lookup of the closest mol weight
        self.mol_weight_trn_values: list[float] = None
        # a dictionary structure for statistics about bond type distributions
        # the dictionary has the following mapping:
        # chem_element ->
        #   ValenceAndCharge ->
        #     number of already bonded atoms ->
        #       [already created BondToNeighbouringAtom] ->
        #                                           AtomWithValence ->
        #                                                          list of (bond_type, count) + total_count
        self.element_2_bond_stats = None
        # a cache with already precomputed sets of randomly generated molecules for the given formula
        self.formula_2_random_smiles = {}
        self.cache_results = cache_results

    def generator_for_splits_of_chem_element_atoms_by_possible_valences(
        self,
        atom_type: chem_element,
        possible_valences: list[ValenceAndCharge],
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
                    atom_type=atom_type, atom_valence_and_charge=next_valence
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
            valences_common, valences_others = self.element_2_valences[
                chem_element_type.capitalize()
            ]
            possible_element_valences = (
                valences_common
                if common_valences_only
                else valences_common + valences_others
            )
            # we ignore "the direction" of ionic bonds, therefore we work with absolute values of valences
            possible_element_valences = map(
                lambda x: ValenceAndCharge(valence=np.abs(x.valence), charge=x.charge),
                possible_element_valences,
            )
            # we require a connected molecule graph, so we ignore possible 0 values of valences
            possible_element_valences = list(
                set(filter(lambda x: x.valence > 0, possible_element_valences))
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
                atom.atom_valence_and_charge.valence * count_of_atoms
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
        # check that charges add up to zero
        total_charge = 0
        for atom, count_of_atoms in valence_assignment.items():
            # we do not take virtual nodes for the charged molecules, we force the remaining submolecule to be neutral
            if atom.atom_type not in {"+", "-"}:
                total_charge += atom.atom_valence_and_charge.charge * count_of_atoms
        if total_charge != 0:
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
            if element.capitalize() not in self.element_2_valences:
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

    def sample_second_edgenode_at_random(
        self,
        edge_start_node_i: int,
        all_graph_nodes: list[AtomNodeForRandomTraversal],
        open_nodes_for_sampling: dict[str, set[int]],
        possible_candidates_type: str,
        closed_set: set[int],
        use_chem_element_stats: bool = False,
        already_connected_neighbours: list[BondToNeighbouringAtom] = None,
    ):
        """
        A function randomly sampling the second node for an edge
        @param edge_start_node_i: index of the first edge node
        @param all_graph_nodes: a list of all nodes in the molecule graph
        @param open_nodes_for_sampling: dictionary with sets of node indices which
                                                 can be considered for closing the edge.
                                                 Each set is specified by the dictionary key:
                                                     "coordinate_bond_negatively_charged_targets",
                                                     "coordinate_bond_positively_charged_targets",
                                                     "covalent_bond_targets"
        @param possible_candidates_type: the `open_nodes_for_sampling` dictionary key
        @param closed_set: closed set for traversal
        @param use_chem_element_stats: a boolean flag setting up usage of per chem. elements statistics about its bonds
        @param already_connected_neighbours: an adjacency list of already sampled neighbours
        """
        if not use_chem_element_stats:
            edge_end_node_j = choice(
                [
                    candidate_node_j
                    for candidate_node_j in open_nodes_for_sampling[
                        possible_candidates_type
                    ]
                    if candidate_node_j not in closed_set
                ]
            )
            bond_degree = 1
        else:
            # checking the current state of the atom and gathering the corresponding stats
            number_of_already_sampled_neighbours = len(already_connected_neighbours)
            # note that the graph is undirected, start-end node refers to the random traversal only
            start_atom = all_graph_nodes[edge_start_node_i]
            if self.element_2_bond_stats is None:
                raise RuntimeError(
                    "To use chem. element stats, the model has to be trained first,"
                    "to record training molecular weights with corresponding formulas."
                )
            # the structure of `self.element_2_bond_stats` is
            # chem_element ->
            #   ValenceAndCharge ->
            #     number of already bonded atoms ->
            #       [already created BondToNeighbouringAtom] ->
            #                                           AtomWithValence ->
            #                                                          list of (bond_type, count)+ total_count

            # if we don't have stats  -> fall back to sampling from all candidates
            try:
                element_stats = self.element_2_bond_stats[
                    start_atom.atom_with_valence.atom_type
                ][
                    ValenceAndCharge(
                        start_atom.atom_with_valence.atom_valence_and_charge.valence,
                        start_atom.atom_with_valence.atom_valence_and_charge.charge,
                    )
                ][
                    number_of_already_sampled_neighbours
                ][
                    tuple(sorted(already_connected_neighbours))
                ]

                full_candidates_list = []
                neighb_with_stats_candidates_list = []
                neighb_with_stats_bondcounts = []
                neighb_with_stats_bondlists = []
                # iterating over open nodes of the corresponding bond type
                for candidate_node_j in open_nodes_for_sampling[
                    possible_candidates_type
                ]:
                    if candidate_node_j not in closed_set:
                        # remembering all candidates in case no statistic-based option is there
                        full_candidates_list.append(candidate_node_j)
                        # checking if the candidate is present in element-specific bond stats
                        candidate_neighb_atom = all_graph_nodes[
                            candidate_node_j
                        ].atom_with_valence
                        if candidate_neighb_atom in element_stats:
                            neighb_with_stats_candidates_list.append(candidate_node_j)
                            bondslist, total_bond_count = element_stats[
                                candidate_neighb_atom
                            ]
                            neighb_with_stats_bondcounts.append(total_bond_count)
                            neighb_with_stats_bondlists.append(bondslist)
                # when no stats-based neighbour remain (e.g. hydrogens are not recorded in the stats)
                if len(neighb_with_stats_candidates_list) == 0:
                    edge_end_node_j = choice(full_candidates_list)
                    bond_degree = 1
                else:
                    # sampling based on frequences in bond stats
                    total_bondcount_sum = sum(neighb_with_stats_bondcounts)
                    proportions = [
                        val / total_bondcount_sum
                        for val in neighb_with_stats_bondcounts
                    ]
                    edge_end_node_j = np.random.choice(
                        neighb_with_stats_candidates_list, p=proportions
                    )
                    # getting i of the sampled neighbour to access its bond-stats
                    neighb_i = neighb_with_stats_candidates_list.index(edge_end_node_j)
                    # for the sampled end node, we sample the type of the bond based on the stats
                    bondtypes_possible = []
                    counts_of_possible_bondtypes = []
                    total_possible_bondtype_count = 0
                    # we leave only the bonds which current state of random generation allows
                    # i.e., we cannot sample a bond violating the current remaining degree of `edge_start_node_i`
                    start_node_remaining_degree = all_graph_nodes[
                        edge_start_node_i
                    ].remaining_node_degree
                    for bondtype, count in neighb_with_stats_bondlists[neighb_i]:
                        if bondtype <= start_node_remaining_degree:
                            bondtypes_possible.append(bondtype)
                            counts_of_possible_bondtypes.append(count)
                            total_possible_bondtype_count += count
                    # if no bonds can be closed for the sampled element, fall back to sampling from full candidates list
                    if len(bondtypes_possible) == 0:
                        edge_end_node_j = choice(full_candidates_list)
                        bond_degree = 1
                    else:
                        bond_degree_proportions = [
                            num / total_possible_bondtype_count
                            for num in counts_of_possible_bondtypes
                        ]
                        bond_degree = np.random.choice(
                            bondtypes_possible, p=bond_degree_proportions
                        )
                        already_connected_neighbours.append(
                            BondToNeighbouringAtom(
                                adjacent_atom=all_graph_nodes[
                                    edge_end_node_j
                                ].atom_with_valence,
                                bond_type=bond_degree,
                            )
                        )
            except:
                edge_end_node_j = choice(
                    [
                        candidate_node_j
                        for candidate_node_j in open_nodes_for_sampling[
                            possible_candidates_type
                        ]
                        if candidate_node_j not in closed_set
                    ]
                )
                bond_degree = 1
        return edge_end_node_j, bond_degree, already_connected_neighbours

    def sample_edge_at_random(
        self,
        all_graph_nodes: list[AtomNodeForRandomTraversal],
        open_nodes_for_sampling: dict[str, set[int]],
        edge_start_node_i: int = None,
        closed_set: set[int] = None,
        use_chem_element_stats: bool = False,
        atom_2_already_connected_neighbours: list[list[BondToNeighbouringAtom]] = None,
    ) -> tuple[tuple[int, int], list[AtomNodeForRandomTraversal], set[int]]:
        """
        Helper function to filter atoms suitable for generation of a random bond with `edge_start_node_i`
        and sampling a random edge
        @param all_graph_nodes: a list of all nodes in the molecule graph
        @param edge_start_node_i: index of the first edge node
        @param open_nodes_for_sampling: dictionary with sets of node indices which
                                                 can be considered for closing the edge.
                                                 Each set is specified by the dictionary key:
                                                     "coordinate_bond_negatively_charged_targets",
                                                     "coordinate_bond_positively_charged_targets",
                                                     "covalent_bond_targets"
        @param use_chem_element_stats: a boolean flag setting up usage of per chem. elements statistics about its bonds
        @param closed_set: closed set for traversal
        @param atom_2_already_connected_neighbours: a mapping from atom to its adjacency list of already sampled neighbours
        @return: a sampled edge and updated structures `all_graph_nodes`, `open_nodes_for_sampling`
        """
        # sample the start node for the edge if it's not specified
        if edge_start_node_i is None:
            edge_start_node_i = choice(
                sum(map(list, open_nodes_for_sampling.values()), [])
            )
        if closed_set is None:
            closed_set = {edge_start_node_i}
        # check if the start edge atom has the charge and therefore can form coordinate bond
        can_form_coordinate_bond = (
            all_graph_nodes[edge_start_node_i].remaining_node_charge != 0
        )
        # if possible, create coordinate bond at random
        is_bond_coordinate = can_form_coordinate_bond and np.random.rand() < 0.5
        if is_bond_coordinate:
            start_node_charge_sign = np.sign(
                all_graph_nodes[edge_start_node_i].remaining_node_charge
            )
            # if for the coordinate bond one atom is positively charged, then another must be charged negatively
            if start_node_charge_sign > 0:
                possible_candidates_type = "coordinate_bond_neg_charged_targets"
            else:
                possible_candidates_type = "coordinate_bond_pos_charged_targets"
        else:
            possible_candidates_type = "covalent_bond_targets"

        (
            edge_end_node_j,
            node_degree_reduction,
            atom_2_already_connected_neighbours[edge_start_node_i],
        ) = self.sample_second_edgenode_at_random(
            edge_start_node_i,
            all_graph_nodes,
            open_nodes_for_sampling,
            possible_candidates_type,
            closed_set,
            use_chem_element_stats,
            atom_2_already_connected_neighbours[edge_start_node_i],
        )

        # decrease the node degrees correspondingly
        for node_of_a_new_edge_i in [edge_start_node_i, edge_end_node_j]:
            all_graph_nodes[
                node_of_a_new_edge_i
            ].remaining_node_degree -= node_degree_reduction
            # if all bonds are created for the particular atom, it is no more open for traversal
            if all_graph_nodes[node_of_a_new_edge_i].remaining_node_degree <= 0:
                for candidates_type in open_nodes_for_sampling.keys():
                    if node_of_a_new_edge_i in open_nodes_for_sampling[candidates_type]:
                        open_nodes_for_sampling[candidates_type].remove(
                            node_of_a_new_edge_i
                        )
            # if the added bond was coordinate, modify the remaining charges correspondingly
            elif is_bond_coordinate:
                new_charge_abs_value = (
                    np.abs(all_graph_nodes[node_of_a_new_edge_i].remaining_node_charge)
                    - 1
                )
                # check if the node still can form coordinate bonds
                if new_charge_abs_value == 0:
                    for candidates_type in [
                        "coordinate_bond_neg_charged_targets",
                        "coordinate_bond_pos_charged_targets",
                    ]:
                        if (
                            node_of_a_new_edge_i
                            in open_nodes_for_sampling[candidates_type]
                        ):
                            open_nodes_for_sampling[candidates_type].remove(
                                node_of_a_new_edge_i
                            )
                else:
                    charge_sign = np.sign(
                        all_graph_nodes[node_of_a_new_edge_i].remaining_node_charge
                    )
                    all_graph_nodes[node_of_a_new_edge_i].remaining_node_charge = (
                        charge_sign * new_charge_abs_value
                    )
        return (
            (edge_start_node_i, edge_end_node_j),
            all_graph_nodes,
            open_nodes_for_sampling,
        )

    def generate_random_molecule_graphs_via_traversal(
        self,
        chemical_formula: str,
        max_number_of_retries_per_valence_assignment: int = 100,
    ) -> list[Mol]:
        """
        A function generating random molecule graph(s).
        The generation process ensures that each graph is connected.
        If any of the `self.count_of_valid_valence_assignments` enables it,
        the function returns graph(s) without self-loops.

        @param chemical_formula: a string containing the chemical formula of the molecule
        @param max_number_of_retries_per_valence_assignment: a max count of attempts to generate a random spanning tree
                                                             for a given potentially feasible valence assignment

        @note In the future the method can be made into a function in a separate utils module,
        for the simplicity of codebase organization and testing purposes it's kept as the method for now
        """
        # check if for the input formula the random structures have been already generated
        if self.cache_results and chemical_formula in self.formula_2_random_smiles:
            return self.formula_2_random_smiles[chemical_formula]

        # get candidate partitions of all molecule atoms into valences
        candidate_valence_assignments = self.get_feasible_atom_valence_assignments(
            chemical_formula
        )
        # iterate over each valence assignment to all atoms, the order is random
        assert (
            len(candidate_valence_assignments) > 0
        ), f"No potentially feasible atom valence assignment for {chemical_formula}"
        # number of iteration over feasible valence assignments
        num_of_iterations_over_splits_into_valences = int(
            np.ceil(self.max_top_k / len(candidate_valence_assignments))
        )
        generated_molecules = []

        # we request to generate self.max_top_k molecule(s)
        while len(generated_molecules) < self.max_top_k:
            for _ in range(num_of_iterations_over_splits_into_valences):
                for valence_assignment in candidate_valence_assignments:
                    # first randomly create a spanning tree of the molecule graph, to ensure the connectivity of molecule.
                    # The feasibility check `self.is_valence_assignment_feasible` inside the
                    # `self.get_feasible_atom_valence_assignments` function should ensure the possibility to create the tree.
                    spanning_tree_was_generated = False
                    spanning_tree_generation_attempts = 0
                    while (
                        not spanning_tree_was_generated
                        and spanning_tree_generation_attempts
                        < max_number_of_retries_per_valence_assignment
                    ):
                        spanning_tree_generation_attempts += 1
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
                                        atom_with_valence=atom_with_valence
                                    )
                                )

                        # a helper structure to record already sampled bonds
                        # it is used only if we use estimated chem elements stats in the generation process
                        atom_2_already_connected_neighbours = [
                            [] for _ in range(len(all_graph_nodes))
                        ]

                        # recording sets of nodes available for random sampling of covalent and coordinate bonds
                        coordinate_bond_neg_charged_targets = {
                            node_i
                            for node_i, node in enumerate(all_graph_nodes)
                            if np.sign(node.remaining_node_charge) == -1
                        }
                        coordinate_bond_pos_charged_targets = {
                            node_i
                            for node_i, node in enumerate(all_graph_nodes)
                            if np.sign(node.remaining_node_charge) == 1
                        }
                        covalent_bond_targets = {
                            node_i
                            for node_i, node in enumerate(all_graph_nodes)
                            if node.remaining_node_charge == 0
                            or node.remaining_node_degree
                            > np.abs(node.remaining_node_charge)
                        }

                        open_nodes_for_sampling = {
                            "coordinate_bond_neg_charged_targets": coordinate_bond_neg_charged_targets,
                            "coordinate_bond_pos_charged_targets": coordinate_bond_pos_charged_targets,
                            "covalent_bond_targets": covalent_bond_targets,
                        }

                        # the final edge list will be stored into the variable below.
                        # An edge is defined by a pair of position indices in the `all_graph_nodes` list
                        edge_list = []

                        # the nodes already included into the spanning tree
                        # the set is used for quick blacklisting, while the list is used for possible backtracking when
                        (
                            spanning_tree_visited_nodes_set,
                            spanning_tree_traversal_list,
                        ) = (
                            set(),
                            deque(),
                        )
                        # sample a random start of spanning tree generation
                        edge_start_node_i = choice(list(range(len(all_graph_nodes))))
                        spanning_tree_visited_nodes_set.add(edge_start_node_i)
                        spanning_tree_traversal_list.append(edge_start_node_i)
                        while self.enforce_connectivity and len(
                            spanning_tree_visited_nodes_set
                        ) < len(all_graph_nodes):
                            # check if the start edge atom has the charge and therefore can form coordinate bond
                            try:
                                (
                                    (edge_start_node_i, edge_end_node_i),
                                    all_graph_nodes,
                                    open_nodes_for_sampling,
                                ) = self.sample_edge_at_random(
                                    all_graph_nodes,
                                    open_nodes_for_sampling,
                                    edge_start_node_i=edge_start_node_i,
                                    closed_set=spanning_tree_visited_nodes_set,
                                    use_chem_element_stats=self.estimate_chem_element_stats,
                                    atom_2_already_connected_neighbours=atom_2_already_connected_neighbours,
                                )
                            except IndexError:
                                spanning_tree_was_generated = False
                                break
                            # note that the graph is undirected, start-end node refers to the random traversal only
                            edge_list.append((edge_start_node_i, edge_end_node_i))
                            # recording the node added to the random spanning tree
                            spanning_tree_visited_nodes_set.add(edge_end_node_i)
                            spanning_tree_traversal_list.append(edge_end_node_i)

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
                                    candidate_for_start_node_i = (
                                        spanning_tree_traversal_list[-1]
                                    )
                            except IndexError:
                                spanning_tree_was_generated = False
                                break
                            edge_start_node_i = candidate_for_start_node_i

                    # after the spanning tree edges were sampled,
                    # now we randomly connect nodes with remaining degrees yet uncovered by sampled bonds
                    while sum(map(len, open_nodes_for_sampling.values())) >= 2:
                        try:
                            (
                                (edge_start_node_i, edge_end_node_i),
                                all_graph_nodes,
                                open_nodes_for_sampling,
                            ) = self.sample_edge_at_random(
                                all_graph_nodes,
                                open_nodes_for_sampling,
                                use_chem_element_stats=self.estimate_chem_element_stats,
                                atom_2_already_connected_neighbours=atom_2_already_connected_neighbours,
                            )
                        except IndexError:
                            break
                        edge_list.append((edge_start_node_i, edge_end_node_i))

                    # if all nodes were covered by edges without self-loops, then we remember the generated molecule
                    if sum(map(len, open_nodes_for_sampling.values())) == 0:
                        generated_molecules.append(
                            create_rdkit_molecule_from_edge_list(
                                edge_list, all_graph_nodes
                            )
                        )
                        if len(generated_molecules) == self.max_top_k:
                            if self.cache_results:
                                self.formula_2_random_smiles[
                                    chemical_formula
                                ] = generated_molecules
                            return generated_molecules
        if self.cache_results:
            self.formula_2_random_smiles[chemical_formula] = generated_molecules
        return generated_molecules

    def training_step(
        self, batch: dict, batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # recording statistics about chemical elements
        if self.estimate_chem_element_stats:
            if self.element_2_bond_stats is None:
                self.element_2_bond_stats = defaultdict(dict)
            for mol_smiles in batch["mol"]:
                molecule = Chem.MolFromSmiles(mol_smiles)
                # in order to work with double and single bonds instead of aromatic
                Chem.Kekulize(molecule, clearAromaticFlags=True)
                # we add hydrogen atoms (which are ommited by default)
                molecule = Chem.AddHs(molecule)
                formula = CalcMolFormula(molecule)
                for atom in molecule.GetAtoms():
                    valence = atom.GetTotalValence()
                    charge = atom.GetFormalCharge()
                    valence_charge = ValenceAndCharge(valence, charge)
                    chem_element_type = atom.GetSymbol()
                    atom_bonds = atom.GetBonds()
                    if (
                        valence_charge
                        not in self.element_2_bond_stats[chem_element_type]
                    ):
                        # for each value of atom's valence, number of neighbours we will count types of neighbouring atoms
                        self.element_2_bond_stats[chem_element_type][
                            valence_charge
                        ] = dict()

                    all_atom_neighbours = set()
                    for bond in atom_bonds:
                        start_atom_idx = atom.GetIdx()
                        end_atom = [
                            _atom
                            for _atom in [bond.GetBeginAtom(), bond.GetEndAtom()]
                            if _atom.GetIdx() != start_atom_idx
                        ][0]
                        all_atom_neighbours.add(
                            (
                                end_atom.GetSymbol(),
                                end_atom.GetTotalValence(),
                                end_atom.GetFormalCharge(),
                                bond.GetBondTypeAsDouble(),
                            )
                        )

                    all_neighbour_subsets = [
                        [set(subset) for subset in combinations(all_atom_neighbours, r)]
                        for r in range(len(all_atom_neighbours))
                    ]
                    for neighbour_subsets_of_fixed_size in all_neighbour_subsets:
                        subset_size = len(neighbour_subsets_of_fixed_size[0])
                        # for each number of already connected atoms we record the neighbours and then possible bonds yet to be closed
                        if (
                            subset_size
                            not in self.element_2_bond_stats[chem_element_type][
                                valence_charge
                            ]
                        ):
                            self.element_2_bond_stats[chem_element_type][
                                valence_charge
                            ][subset_size] = dict()
                        for neighbours in neighbour_subsets_of_fixed_size:
                            neighbours_tuple = tuple(sorted(neighbours))
                            if (
                                neighbours_tuple
                                not in self.element_2_bond_stats[chem_element_type][
                                    valence_charge
                                ][subset_size]
                            ):
                                self.element_2_bond_stats[chem_element_type][
                                    valence_charge
                                ][subset_size][neighbours_tuple] = defaultdict(int)
                            remaining_bonds = all_atom_neighbours.difference(neighbours)
                            for remaining_bonded_neighbour in remaining_bonds:
                                self.element_2_bond_stats[chem_element_type][
                                    valence_charge
                                ][subset_size][neighbours_tuple][
                                    remaining_bonded_neighbour
                                ] += 1

        # recording molecular weight
        for mol_smiles in batch["mol"]:
            molecule = Chem.MolFromSmiles(mol_smiles)
            formula = CalcMolFormula(molecule)
            weight = ExactMolWt(molecule)
            self.mol_weight_2_formulas[weight].append(formula)
        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)
        return dict(loss=loss, mols_pred=["C"])

    def on_train_end(self) -> None:
        # for each molecular weight we compute proportions of recorded molecular formulas
        molecular_weight_2_formula_counts = {
            weight: Counter(formulas)
            for weight, formulas in self.mol_weight_2_formulas.items()
        }
        weight_2_formula_proportions = {}
        for weight, formula_2_count in molecular_weight_2_formula_counts.items():
            total_count = sum(formula_2_count.values())
            weight_2_formula_proportions[weight] = {
                formula: count / total_count
                for formula, count in formula_2_count.items()
            }
        # for consequent sampling using numpy.random.choice function, we store the results in the format
        # weight -> [[formula_1, formula_2], [proportion_of_formula_1, proportion_of_formula_2]]
        self.mol_weight_2_formulas = {
            weight: [
                list(formula_2_proportions.keys()),
                list(formula_2_proportions.values()),
            ]
            for weight, formula_2_proportions in weight_2_formula_proportions.items()
        }
        # storing weights in the sorted list for the logarithmic time look-up of the closest weight value
        self.mol_weight_trn_values = sorted(self.mol_weight_2_formulas.keys())

        # if chem element stats are used, then the corresponding data structure is reformated in accordance with the
        # description from docstring to the class __init__:
        # chem_element ->
        #   ValenceAndCharge ->
        #     number of already bonded atoms ->
        #       [already created BondToNeighbouringAtom] ->
        #                                           AtomWithValence ->
        #                                                          list of (bond_type, count) + total_count
        if self.estimate_chem_element_stats:
            element_2_bond_stats = defaultdict(dict)
            for (
                chem_element,
                valence_charge_2_stats,
            ) in self.element_2_bond_stats.items():
                for valence_charge, num_bonds_2_stats in valence_charge_2_stats.items():
                    element_2_bond_stats[chem_element][valence_charge] = dict()
                    for num_bonds, bonds_2_stats in num_bonds_2_stats.items():
                        element_2_bond_stats[chem_element][valence_charge][
                            num_bonds
                        ] = dict()
                        for (
                            bonds,
                            neighb_atom_with_valence_2_stats,
                        ) in bonds_2_stats.items():
                            present_bonds_sorted = tuple(
                                sorted(
                                    [
                                        BondToNeighbouringAtom(
                                            adjacent_atom=AtomWithValence(
                                                atom_type=bond[0],
                                                atom_valence_and_charge=ValenceAndCharge(
                                                    valence=bond[1], charge=bond[2]
                                                ),
                                            ),
                                            bond_type=bond[3],
                                        )
                                        for bond in bonds
                                    ]
                                )
                            )
                            element_2_bond_stats[chem_element][valence_charge][
                                num_bonds
                            ][present_bonds_sorted] = defaultdict(list)
                            for (
                                neighb_atom_type,
                                neighb_atom_valence,
                                neighb_atom_charge,
                                bondtype,
                            ), count in neighb_atom_with_valence_2_stats.items():
                                neighbouring_atom = AtomWithValence(
                                    atom_type=neighb_atom_type,
                                    atom_valence_and_charge=ValenceAndCharge(
                                        valence=neighb_atom_valence,
                                        charge=neighb_atom_charge,
                                    ),
                                )
                                element_2_bond_stats[chem_element][valence_charge][
                                    num_bonds
                                ][present_bonds_sorted][neighbouring_atom].append(
                                    (bondtype, count)
                                )
                            # computing total count of all bound per neighbouring atom
                            for (
                                neighbouring_atom,
                                list_of_bondtype_counts,
                            ) in element_2_bond_stats[chem_element][valence_charge][
                                num_bonds
                            ][
                                present_bonds_sorted
                            ].items():
                                total_count_of_bonds = sum(
                                    map(lambda x: x[1], list_of_bondtype_counts)
                                )
                                element_2_bond_stats[chem_element][valence_charge][
                                    num_bonds
                                ][present_bonds_sorted][neighbouring_atom] = (
                                    list_of_bondtype_counts,
                                    total_count_of_bonds,
                                )
            self.element_2_bond_stats = element_2_bond_stats

    def sample_formula_with_the_closest_molecular_weight(
        self, molecular_weight: float
    ) -> str:
        """
        A method sampling chemical formula observed in training data with the closest weight to `molecular_weight`
        @param molecular_weight: Molecular weight of a structure to be generated
        """
        if self.mol_weight_trn_values is None:
            raise RuntimeError(
                "For random denovo generation without known formula, the model has to be trained first,"
                "to record training molecular weights with corresponding formulas."
            )
        # finding a place in the sorted array for insertion of the `molecular_weight`, while preserving sorted order
        idx_of_closest_larger = bisect.bisect_left(
            self.mol_weight_trn_values, molecular_weight
        )
        # check if the exact same molecular weight was observed in training data, otherwise select the closest weight
        if molecular_weight == self.mol_weight_trn_values[idx_of_closest_larger]:
            idx_of_closest = idx_of_closest_larger
        elif idx_of_closest_larger > 0:
            # determining the closest molecular weight out of both neighbours
            idx_of_closest_smaller = idx_of_closest_larger - 1
            weight_difference_with_smaller_neighbour = (
                molecular_weight - self.mol_weight_trn_values[idx_of_closest_smaller]
            )
            weight_difference_with_larger_neighbour = (
                self.mol_weight_trn_values[idx_of_closest_larger] - molecular_weight
            )
            if (
                weight_difference_with_larger_neighbour
                < weight_difference_with_smaller_neighbour
            ):
                idx_of_closest = idx_of_closest_larger
            else:
                idx_of_closest = idx_of_closest_smaller
        else:
            idx_of_closest = 0
        # the value of the molecular weight observed in training labels, which is the closest to `molecular_weight`
        closest_observed_molecular_weight = self.mol_weight_trn_values[idx_of_closest]
        # getting chemical formulas observed for this molecular weight
        # self.mol_weight_2_formulas is a dictionary containing the following mapping
        #  weight -> [[formula_1, formula_2], [proportion_of_formula_1, proportion_of_formula_2]]
        feasible_formulas, formula_proportions = self.mol_weight_2_formulas[
            closest_observed_molecular_weight
        ]
        # if just one formula is known, it is returned directly
        if len(feasible_formulas) == 1:
            return feasible_formulas[0]
        # otherwise we randomly sample in accordance with proportions
        return np.random.choice(feasible_formulas, p=formula_proportions)

    def step(
        self, batch: dict, stage: Stage = Stage.NONE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mols = batch["mol"]  # List of SMILES of length batch_size

        # If formula_known is True, we should generate molecules with the same formula as label (`mols` above)
        # If formula_known is False, we should generate any molecule with the same mass as label

        # obtaining molecule objects from SMILES
        molecules = [Chem.MolFromSmiles(smiles) for smiles in mols]
        # getting the formulas
        if self.formula_known:
            formulas = [CalcMolFormula(molecule) for molecule in molecules]
        else:
            molecular_weights = [ExactMolWt(molecule) for molecule in molecules]
            formulas = [
                self.sample_formula_with_the_closest_molecular_weight(mol_weight)
                for mol_weight in molecular_weights
            ]
        # (bs, k) list of rdkit molecules
        mols_pred = [
            self.generate_random_molecule_graphs_via_traversal(formula)
            for formula in formulas
        ]

        for predicted_mol_group in mols_pred:
            for mol in predicted_mol_group:
                Chem.RemoveHs(mol)

        # list of predicted smiles
        smiles_pred = [
            [
                Chem.MolToSmiles(mol_candidate)
                for mol_candidate in candidates_per_input_mol
            ]
            for candidates_per_input_mol in mols_pred
        ]

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)
        return dict(loss=loss, mols_pred=smiles_pred)

    def configure_optimizers(self):
        # No optimizer needed for a random baseline
        return None


# element valences taken from sources like https://sciencenotes.org/element-valency-pdf
# the first list contains the typical valences, each tuple is a valence value with the corresponding charge
ELEMENT_VALENCES = {
    "H": (
        [ValenceAndCharge(valence=1, charge=0)],
        [ValenceAndCharge(valence=0, charge=0), ValenceAndCharge(valence=1, charge=-1)],
    ),
    "He": ([ValenceAndCharge(valence=0, charge=0)], []),
    "Li": (
        [ValenceAndCharge(valence=1, charge=0)],
        [ValenceAndCharge(valence=1, charge=-1)],
    ),
    "Be": ([ValenceAndCharge(valence=2, charge=0)], []),
    "B": (
        [ValenceAndCharge(valence=3, charge=0), ValenceAndCharge(valence=4, charge=-1)],
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=1, charge=0)],
    ),
    "C": (
        [ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=3, charge=-1),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=2, charge=-1),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=1, charge=-1),
        ],
    ),
    "N": (
        [ValenceAndCharge(valence=3, charge=0), ValenceAndCharge(valence=4, charge=1)],
        [
            ValenceAndCharge(valence=2, charge=-1),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
            ValenceAndCharge(valence=1, charge=-1),
        ],
    ),
    "O": (
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=1, charge=-1)],
        [ValenceAndCharge(valence=3, charge=1)],
    ),
    "F": ([ValenceAndCharge(valence=1, charge=0)], []),
    "Ne": ([ValenceAndCharge(valence=0, charge=0)], []),
    "Na": (
        [ValenceAndCharge(valence=1, charge=0)],
        [],
    ),
    "Mg": ([ValenceAndCharge(valence=2, charge=0)], []),
    "Al": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=1, charge=0)],
    ),
    "Si": (
        [ValenceAndCharge(valence=4, charge=0)],
        [],
    ),
    "P": (
        [ValenceAndCharge(valence=5, charge=0)],
        [
            ValenceAndCharge(valence=4, charge=1),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
        ],
    ),
    "S": (
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=6, charge=0)],
        [
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=1, charge=-1),
            ValenceAndCharge(valence=3, charge=1),
        ],
    ),
    "Cl": (
        [ValenceAndCharge(valence=1, charge=0)],
        [],
    ),
    "Ar": ([ValenceAndCharge(valence=0, charge=0)], []),
    "K": (
        [ValenceAndCharge(valence=1, charge=0)],
        [],
    ),
    "Ca": ([ValenceAndCharge(valence=2, charge=0)], []),
    "Sc": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=1, charge=0)],
    ),
    "Ti": (
        [ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "V": (
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=3, charge=0),
        ],
        [
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Cr": (
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
        ],
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Mn": (
        [
            ValenceAndCharge(valence=7, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=2, charge=0),
        ],
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Fe": (
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=3, charge=0)],
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Co": (
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=3, charge=0)],
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Ni": (
        [ValenceAndCharge(valence=2, charge=0)],
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Cu": (
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=1, charge=0)],
        [
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Zn": (
        [ValenceAndCharge(valence=2, charge=0)],
        [ValenceAndCharge(valence=1, charge=0), ValenceAndCharge(valence=0, charge=0)],
    ),
    "Ga": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=1, charge=0)],
    ),
    "Ge": (
        [ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
        ],
    ),
    "As": (
        [ValenceAndCharge(valence=5, charge=0), ValenceAndCharge(valence=4, charge=1)],
        [],
    ),
    "Se": (
        [ValenceAndCharge(valence=2, charge=0)],
        [],
    ),
    "Br": (
        [ValenceAndCharge(valence=1, charge=0)],
        [],
    ),
    "Kr": (
        [ValenceAndCharge(valence=0, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "Rb": (
        [ValenceAndCharge(valence=1, charge=0)],
        [],
    ),
    "Sr": ([ValenceAndCharge(valence=2, charge=0)], []),
    "Y": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "Zr": (
        [ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Nb": (
        [ValenceAndCharge(valence=5, charge=0)],
        [
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Mo": (
        [ValenceAndCharge(valence=6, charge=0), ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Tc": (
        [ValenceAndCharge(valence=7, charge=0), ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Ru": (
        [ValenceAndCharge(valence=4, charge=0), ValenceAndCharge(valence=3, charge=0)],
        [
            ValenceAndCharge(valence=8, charge=0),
            ValenceAndCharge(valence=7, charge=0),
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Rh": (
        [ValenceAndCharge(valence=3, charge=0)],
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Pd": (
        [ValenceAndCharge(valence=4, charge=0), ValenceAndCharge(valence=2, charge=0)],
        [ValenceAndCharge(valence=0, charge=0)],
    ),
    "Ag": (
        [ValenceAndCharge(valence=1, charge=0)],
        [
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Cd": (
        [ValenceAndCharge(valence=2, charge=0)],
        [ValenceAndCharge(valence=1, charge=0)],
    ),
    "In": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=1, charge=0)],
    ),
    "Sn": (
        [ValenceAndCharge(valence=2, charge=0)],
        [ValenceAndCharge(valence=4, charge=0)],
    ),
    "Sb": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=5, charge=0), ValenceAndCharge(valence=3, charge=-1)],
    ),
    "Te": (
        [ValenceAndCharge(valence=4, charge=0)],
        [ValenceAndCharge(valence=2, charge=0), ValenceAndCharge(valence=6, charge=0)],
    ),
    "I": (
        [ValenceAndCharge(valence=1, charge=0)],
        [
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=7, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Xe": (
        [ValenceAndCharge(valence=0, charge=0)],
        [
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=8, charge=0),
        ],
    ),
    "Cs": (
        [ValenceAndCharge(valence=1, charge=0)],
        [],
    ),
    "Ba": ([ValenceAndCharge(valence=2, charge=0)], []),
    "La": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Ce": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=4, charge=0)],
    ),
    "Pr": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=4, charge=0)],
    ),
    "Nd": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Pm": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Sm": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Eu": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "Gd": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Tb": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=4, charge=0)],
    ),
    "Dy": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Ho": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Er": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Tm": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Yb": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "Lu": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Hf": ([ValenceAndCharge(valence=4, charge=0)], []),
    "Ta": ([ValenceAndCharge(valence=5, charge=0)], []),
    "W": (
        [ValenceAndCharge(valence=6, charge=0), ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=2, charge=0),
        ],
    ),
    "Re": (
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=3, charge=0),
        ],
        [
            ValenceAndCharge(valence=7, charge=0),
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=2, charge=0),
            ValenceAndCharge(valence=1, charge=0),
            ValenceAndCharge(valence=0, charge=0),
        ],
    ),
    "Os": (
        [ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=8, charge=0),
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=2, charge=0),
        ],
    ),
    "Ir": (
        [ValenceAndCharge(valence=4, charge=0), ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=6, charge=0), ValenceAndCharge(valence=4, charge=0)],
    ),
    "Pt": (
        [ValenceAndCharge(valence=2, charge=0)],
        [ValenceAndCharge(valence=4, charge=0)],
    ),
    "Au": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=1, charge=0)],
    ),
    "Hg": (
        [ValenceAndCharge(valence=2, charge=0)],
        [ValenceAndCharge(valence=1, charge=0)],
    ),
    "Tl": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=1, charge=0)],
    ),
    "Pb": (
        [ValenceAndCharge(valence=4, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "Bi": (
        [ValenceAndCharge(valence=3, charge=0), ValenceAndCharge(valence=1, charge=0)],
        [ValenceAndCharge(valence=5, charge=0)],
    ),
    "Po": (
        [ValenceAndCharge(valence=4, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "At": (
        [ValenceAndCharge(valence=1, charge=0)],
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=3, charge=0),
            ValenceAndCharge(valence=7, charge=0),
        ],
    ),
    "Rn": (
        [ValenceAndCharge(valence=0, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "Fr": ([ValenceAndCharge(valence=1, charge=0)], []),
    "Ra": ([ValenceAndCharge(valence=2, charge=0)], []),
    "Ac": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Th": ([ValenceAndCharge(valence=4, charge=0)], []),
    "Pa": (
        [ValenceAndCharge(valence=5, charge=0)],
        [ValenceAndCharge(valence=4, charge=0)],
    ),
    "U": (
        [ValenceAndCharge(valence=6, charge=0)],
        [
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=3, charge=0),
        ],
    ),
    "Np": (
        [ValenceAndCharge(valence=7, charge=0)],
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=4, charge=0),
            ValenceAndCharge(valence=3, charge=0),
        ],
    ),
    "Pu": (
        [ValenceAndCharge(valence=7, charge=0), ValenceAndCharge(valence=4, charge=0)],
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=3, charge=0),
        ],
    ),
    "Am": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=5, charge=0), ValenceAndCharge(valence=4, charge=0)],
    ),
    "Cm": (
        [
            ValenceAndCharge(valence=6, charge=0),
            ValenceAndCharge(valence=5, charge=0),
            ValenceAndCharge(valence=3, charge=0),
        ],
        [],
    ),
    "Bk": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=4, charge=0)],
    ),
    "Cf": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Es": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Fm": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Md": ([ValenceAndCharge(valence=3, charge=0)], []),
    "No": (
        [ValenceAndCharge(valence=3, charge=0)],
        [ValenceAndCharge(valence=2, charge=0)],
    ),
    "Lr": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Rf": ([ValenceAndCharge(valence=4, charge=0)], []),
    "Db": ([ValenceAndCharge(valence=5, charge=0)], []),
    "Sg": ([ValenceAndCharge(valence=6, charge=0)], []),
    "Bh": ([ValenceAndCharge(valence=7, charge=0)], []),
    "Hs": ([ValenceAndCharge(valence=8, charge=0)], []),
    "Mt": ([ValenceAndCharge(valence=8, charge=0)], []),
    "Ds": ([ValenceAndCharge(valence=8, charge=0)], []),
    "Rg": ([ValenceAndCharge(valence=8, charge=0)], []),
    "Cn": ([ValenceAndCharge(valence=2, charge=0)], []),
    "Nh": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Fl": ([ValenceAndCharge(valence=4, charge=0)], []),
    "Mc": ([ValenceAndCharge(valence=3, charge=0)], []),
    "Lv": ([ValenceAndCharge(valence=4, charge=0)], []),
    "Ts": ([ValenceAndCharge(valence=7, charge=0)], []),
    "Og": ([ValenceAndCharge(valence=0, charge=0)], []),
    "+": ([ValenceAndCharge(valence=1, charge=1)], []),
    "-": ([ValenceAndCharge(valence=1, charge=-1)], []),
}
