import math
import numpy

from typing import List
from enum import Enum
from dataclasses import field, dataclass
from rnapolis.tertiary import Atom, Structure3D
from rnapolis.common import BasePair, Residue, Stacking
from eltetrado.analysis import Tetrad, has_tetrad, center_of_mass
from functools import cmp_to_key

'''
def center_of_mass(atoms: List[Atom]) -> numpy.typing.NDArray[numpy.floating]:
    coords = [atom.coordinates for atom in atoms]
    xs = (coord[0] for coord in coords)
    ys = (coord[1] for coord in coords)
    zs = (coord[2] for coord in coords)
    return numpy.array(
        (sum(xs) / len(coords), sum(ys) / len(coords), sum(zs) / len(coords))
    )

def oai_atoms(residue) -> numpy.typing.NDArray[numpy.floating]:
    return center_of_mass([residue.outermost_atom, residue.innermost_atom])


def nucleobase_atoms(residue) -> numpy.typing.NDArray[numpy.floating]:
    atoms = []
    upper = residue.one_letter_name.upper()
    if upper == "T":
        upper = "U"
    if upper in residue.nucleobase_heavy_atoms:
        base_atoms = residue.nucleobase_heavy_atoms[upper]
        for base_atom in base_atoms:
            atom = residue.find_atom(base_atom)
            if atom is not None:
                atoms.append(atom)
        return center_of_mass(atoms)
    else:
        return residue.outermost_atom.coordinates
'''

def in_tetrad(nt, tetrad):
    for nucl in tetrad.nucleotides:
        if nucl == nt:
            return True
    return False

def in_tetrads(nt, tetrads):
    for tetrad in tetrads:
        if in_tetrad(nt, tetrad):
            return True
    return False

def filter_out_tetrad_residues(analysis):
    # Filter out resi that are already part of any of the tetrads
    filtered_residues = []
    for residue in analysis.structure3d.residues:
        if residue.base_normal_vector is None:
            continue
        if not in_tetrads(residue, analysis.tetrads):
            filtered_residues.append(residue)
    return filtered_residues

class CONN(Enum):
    NONE = "None"
    SEQ = "Seq"
    SEQ_SEQ = "Seq+Seq"
    BP ="BP"
    BP_SEQ = "BP+Seq"
    BP_BP = "BP+BP"
    UNK = "Unknown"

    def score(self):
        if self == CONN.NONE:
            return 0
        elif self == CONN.SEQ:
            return 1
        elif self == CONN.BP:
            return 1
        elif self == CONN.SEQ_SEQ:
            return 2
        elif self == CONN.BP_SEQ:
            return 3
        elif self == CONN.BP_BP:
            return 4
        return 4

def get_conn_type(bp, seq):
    if bp == 0 and seq == 0:
        return CONN.NONE
    elif seq == 1 and bp == 0:
        return CONN.SEQ
    elif seq == 2 and bp == 0:
        return CONN.SEQ_SEQ
    elif seq == 0 and bp == 1:
        return CONN.BP
    elif seq == 1 and bp == 1:
        return CONN.BP_SEQ
    elif seq == 0 and bp == 2:
        return CONN.BP_BP
    else:
        return CONN.UNK

def is_between(x ,lhs, rhs):
    return ((x - lhs) * (rhs - x)) >= 0

@dataclass(order=True)
class Candidate:
    resi: Residue
    tetrad: Tetrad

    # Connections
    connection: CONN
    connected_resi: List[Residue]

    # Stage 2 - Tilt
    TILT_DEGREE_THRESHOLD_MAX = 55
    TILT_DEGREE_THRESHOLD_AVG = 45
    # Tilt - normalized to degrees form 0
    tilt_avg: float = field(init=False)
    tilt_max: float = field(init=False)
    # List of tilts in relation to nt1 ... nt4
    tilts: List[float] = field(default_factory=list)
    tilts_true: List[float] = field(default_factory=list)

    # Stage 3 - Distance
    DISTANCE_A_THRESHOLD_INN = 14.00
    DISTANCE_A_THRESHOLD_INN_CLOSE = 12.50
    DISTANCE_A_THRESHOLD_OUT = 12.50
    DISTANCE_A_THRESHOLD_OUT_CLOSE = 10.00
    # Distance of candidate to center of tetrad
    dist_outer: float = field(default_factory=list)
    dist_inner: float = field(default_factory=list)
    # Distance of candidate to each nucleotide
    dists_inner: List[float] = field(default_factory=list)
    dists_outer: List[float] = field(default_factory=list)
    # Distance of nucleotides within tetrad
    tetrad_dists_inner: List[float] = field(default_factory=list)
    tetrad_dists_outer: List[float] = field(default_factory=list)

    # Stage 4 - Level plane height
    HEIGHT_DIFF_THRESHOLD = 3.6
    #HEIGHT_DIFF_THRESHOLD_CENTRAL = 3.0
    height_inner: List[float] = field(default_factory=list)
    height_outer: List[float] = field(default_factory=list)


    def __post_init__(self):
        # Stage 2
        # Calculate tilt of the candidate residue in relation to the tetrad
        self.calculate_tilt()
        self.calculate_distances()
        self.calculate_height()

    def calculate_tilt(self):
        avg = 0
        for nt in self.tetrad.nucleotides:
            dot = numpy.dot(self.resi.base_normal_vector, nt.base_normal_vector)
            deg = math.degrees(math.acos(dot))
            self.tilts_true.append(deg)
            if deg > 90:
                deg -= 180
            self.tilts.append(deg)
        self.tilt_avg = sum(self.tilts) / len(self.tilts)
        self.tilt_max = max(self.tilts, key=abs)

    def calculate_distances(self):
        # Distance of candidate from center of tetrad
        self.dist_inner = numpy.linalg.norm(self.tetrad.center() - self.resi.innermost_atom.coordinates)
        self.dist_outer = numpy.linalg.norm(self.tetrad.center() - self.resi.outermost_atom.coordinates)
        # Distances of nucleotides WITHIN tetrad from center
        for nt in self.tetrad.nucleotides:
            self.tetrad_dists_inner.append(numpy.linalg.norm(self.tetrad.center() - nt.innermost_atom.coordinates))
            self.tetrad_dists_outer.append(numpy.linalg.norm(self.tetrad.center() - nt.outermost_atom.coordinates))
        # Distances of candidate and each nucleotide in the tetrad
        for nt in self.tetrad.nucleotides:
            self.dists_inner.append(numpy.linalg.norm(self.resi.innermost_atom.coordinates - nt.innermost_atom.coordinates))
            self.dists_outer.append(numpy.linalg.norm(self.resi.outermost_atom.coordinates - nt.outermost_atom.coordinates))

    def calculate_height(self):
        for nt in self.tetrad.nucleotides:
            self.height_inner.append(numpy.dot(self.tetrad.center() - self.resi.innermost_atom.coordinates, nt.base_normal_vector))
            self.height_outer.append(numpy.dot(self.tetrad.center() - self.resi.outermost_atom.coordinates, nt.base_normal_vector))



    def location(self):
        # Between nt1 & nt2
        if is_between(self.tetrad.global_index[self.resi],
                self.tetrad.global_index[self.tetrad.nt1],
                self.tetrad.global_index[self.tetrad.nt2]):
            return 1
        if is_between(self.tetrad.global_index[self.resi],
                self.tetrad.global_index[self.tetrad.nt2],
                self.tetrad.global_index[self.tetrad.nt3]):
            return 2
        if is_between(self.tetrad.global_index[self.resi],
                self.tetrad.global_index[self.tetrad.nt3],
                self.tetrad.global_index[self.tetrad.nt4]):
            return -1
        if self.tetrad.global_index[self.resi] > max(self.tetrad.global_index[self.tetrad.nt4],
                                                     self.tetrad.global_index[self.tetrad.nt1]) or \
           self.tetrad.global_index[self.resi] < min(self.tetrad.global_index[self.tetrad.nt4],
                                                     self.tetrad.global_index[self.tetrad.nt1]):
            return -2


    # Give yourself a score acording to how far off the thresholds we are.
    # Simple weighted exponential scoring.
    # Lower = Better
    # Weights
    # Tilt avg = 0.75
    # Tilt max = 0.50
    # Distance: - none, hard to determine 
    # Height_inn_avg = 1.00
    # Height_inn_max = 0.75
    # Height_out_avg = 1.00
    # Height_out_max = 0.75
    def score(self):
        # Sanity check
        if not self.is_valid():
            return 99999.
        score = 0
        score += 1.0 * \
                math.pow(((self.TILT_DEGREE_THRESHOLD_AVG - math.fabs(self.tilt_avg))
                    / self.TILT_DEGREE_THRESHOLD_AVG)
                    * 100.,
                    2.0)
        #print("1", score)
        score += 0.5 * \
                math.pow(((self.TILT_DEGREE_THRESHOLD_MAX - math.fabs(self.tilt_max))
                    / self.TILT_DEGREE_THRESHOLD_MAX)
                    * 100.,
                    2.0)
        #print("2", score)
        score += 1.00 * \
                math.pow(((self.HEIGHT_DIFF_THRESHOLD - (sum(self.height_inner) / len(self.height_inner)))
                    / self.HEIGHT_DIFF_THRESHOLD)
                    * 100.,
                    1.75)
        #print("3", score)
        score += 0.75 * \
                math.pow(((self.HEIGHT_DIFF_THRESHOLD - max(self.height_inner, key=abs))
                    /  self.HEIGHT_DIFF_THRESHOLD)
                    * 100.,
                    1.75)
        #print("4", score)
        score += 1.00 * \
                math.pow(((self.HEIGHT_DIFF_THRESHOLD - (sum(self.height_outer) / len(self.height_outer)))
                    / self.HEIGHT_DIFF_THRESHOLD)
                    * 100.,
                    1.75)
        #print("5", score)
        score += 0.75 * \
                math.pow(((self.HEIGHT_DIFF_THRESHOLD - max(self.height_outer, key=abs))
                    /  self.HEIGHT_DIFF_THRESHOLD)
                    * 100.,
                    1.75)
        #print("6", score)
        score /= self.connection.score()
        #print("7", score)

        return score

    def is_valid(self):
        # Stage 1 - Check Connection
        # Skip, we know all are connected properly

        # Stage 2 - Check tilt
        if math.fabs(self.tilt_avg) > self.TILT_DEGREE_THRESHOLD_AVG:
            return False
        if math.fabs(self.tilt_max) > self.TILT_DEGREE_THRESHOLD_MAX:
            return False

        # Stage 3 - Distances
        if self.dist_inner >= self.DISTANCE_A_THRESHOLD_INN:
            return False
        if self.dist_outer >= self.DISTANCE_A_THRESHOLD_OUT:
            return False
        # Candidate cannot be same distance away as current tetrad nucleotides
        for dist in self.tetrad_dists_inner:
            if self.dist_inner <= dist + 1.75:
                return False
        for dist in self.tetrad_dists_outer:
            if self.dist_outer <= dist + 2.00:
                return False
        # Check if candidate is close to 2 nucleotides
        close_inn = 0
        for dist in self.dists_inner:
            if dist < self.DISTANCE_A_THRESHOLD_INN_CLOSE:
                close_inn += 1
        if close_inn < 2:
            return False
        close_out = 0
        for dist in self.dists_outer:
            if dist < self.DISTANCE_A_THRESHOLD_OUT_CLOSE:
                close_out += 1
        if close_out < 2:
            return False

        # Stage 4 - Height Level
        for height in self.height_inner:
            if math.fabs(height) >= self.HEIGHT_DIFF_THRESHOLD:
                return False
        for height in self.height_outer:
            if math.fabs(height) >= self.HEIGHT_DIFF_THRESHOLD:
                return False

        # Everything passed
        return True

def connected2tetrad(residues, analysis):
    tetrad_candidates = {}
    for tetrad in analysis.tetrads:
        candidates = []
        for residue in residues:
            connected_resi = []
            seq_ct = 0
            # Is next/prev in sequence
            for nt in tetrad.nucleotides:
                if residue.chain == nt.chain and math.fabs(residue.number - nt.number) == 1:
                    seq_ct += 1
                    connected_resi.append(nt)

            bp_ct = 0
            # Is base pair
            for bp in analysis.mapping.base_pairs:
                if bp.nt1.full_name == residue.full_name:
                    for nt in tetrad.nucleotides:
                        if bp.nt2.full_name == nt.full_name:
                            bp_ct += 1
                            connected_resi.append(nt)

            if len(connected_resi) > 0:
                candidates.append(Candidate(residue, tetrad, get_conn_type(bp_ct, seq_ct), connected_resi))
        if len(candidates) > 0:
            tetrad_candidates[tetrad] = candidates

    return tetrad_candidates

class MultimerClass(Enum):
    PENTAD = "Pentad"
    HEXAD = "Hexad"
    HEPTAD = "Heptad"
    OCTAD = "Octad"
    UNKNOWN = "Unknown"


class Multimer:
    tetrad: Tetrad
    candidates: List[Candidate]
    multimer_class: MultimerClass
    nts: List[Residue]

    def __init__(self, tetrad, candidates):
        self.tetrad = tetrad
        self.candidates = candidates
        match len(candidates):
            case 1:
                self.multimer_class = MultimerClass.PENTAD
            case 2:
                self.multimer_class = MultimerClass.HEXAD
            case 3:
                self.multimer_class = MultimerClass.HEPTAD
            case 4:
                self.multimer_class = MultimerClass.OCTAD
            case _:
                self.multimer_class = MultimerClass.UNKNOWN

        self.nts = []
        self.nts.append(self.tetrad.nt1)
        for c in self.candidates:
            if c.location() == 1:
                self.nts.append(c.resi)
                break
        self.nts.append(self.tetrad.nt2)
        for c in self.candidates:
            if c.location() == 2:
                self.nts.append(c.resi)
                break
        self.nts.append(self.tetrad.nt3)
        for c in self.candidates:
            if c.location() == -1:
                self.nts.append(c.resi)
                break
        self.nts.append(self.tetrad.nt4)
        for c in self.candidates:
            if c.location() == -2:
                self.nts.append(c.resi)
                break

    def __repr__(self):
        return f"{repr(self.nts)} - {repr(self.multimer_class.value)}"

    def __str__(self):
        return (
            f"    "
            f"{self.nts} "
            f"- {self.multimer_class.value}"
        )

def ntads(analysis):
    # Stage 0
    # Filter out all residues that already are a part of any tetrad.
    filtered_residues = filter_out_tetrad_residues(analysis)

    # Stage 1
    # The residue should be connected to tetrad in some way
    # a) BasePair connection
    # b) Sequence continuity
    tetrad_candidates = connected2tetrad(filtered_residues, analysis)
    #print(tetrad_candidates)

    # In practice connected_tetrad_candidates already have all needed parameters calculated here
    # and we need to remove all unneeded / not valid candidates with incorrect:
    # 1. Tilt
    # 2. Distance
    # 3. Level Height
    # TODO Make it automatic when creating candidate and make everything at once.
    valid_tetrad_candidates = {}
    for tetrad, candidates in tetrad_candidates.items():
        valid_candidates = [c for c in candidates if c.is_valid()]
        valid_tetrad_candidates[tetrad] = valid_candidates

    # Even though we  thing every candidate is valid addition it might not be true.
    # 1. Some residues can be candidates in multiple tetrads
    # 2. The number of final candidates should be equal for all multimers
    # Stage 1 - Make sure all are "Unique"
    candidate_to_tetrads = {}
    for tetrad, candidates in valid_tetrad_candidates.items():
        for candidate in candidates:
            if candidate.resi in candidate_to_tetrads:
                candidate_to_tetrads[candidate.resi] = [*candidate_to_tetrads[candidate.resi], candidate]
            else:
                candidate_to_tetrads[candidate.resi] = [candidate]
    problematic = {}
    for resi, candidates in candidate_to_tetrads.items():
        if len(candidates) > 1:
            problematic[resi] = candidates
    # Problematic containes residues that are problematic, aka. non unique and exist
    # as a candidate for multiple tetrads
    for resi, problematic in problematic.items():
        # Only keep in valid_tetrad_candidates best from problematic
        best_score = 99999999999.
        best_candi = None

        for candidate in problematic:
            if best_score > candidate.score():
                best_score = candidate.score()
                best_candi = candidate

        for candidate in problematic:
            if best_candi != candidate:
                valid_tetrad_candidates[candidate.tetrad].remove(candidate)

    # Now valid_tetrad_candidates have unique residues. Number can still differ.
    # Construct multimers from candidates
    for tetrad, candidates in valid_tetrad_candidates.items():
        if len(candidates) < 1:
            continue
        left_candidates = candidates
        left_candidates.sort(key=lambda x: x.score())

        # Pentad
        add_order = []
        add_order.append(left_candidates[0])
        left_candidates.remove(add_order[-1])
        # Remove all from the same potential location as curent additions.
        to_remove = []
        for candidate in left_candidates:
            if add_order[-1].location() == candidate.location():
                to_remove.append(candidate)
        for rm in to_remove:
            left_candidates.remove(rm)
        # First addition - anchor.
        # Dictates next addition to the multimer as it has to be on the opposite side.
        if (len(left_candidates) >= 1):
            # Hexad
            # Find best for the next oposite location.
            for candidate in left_candidates:
                if add_order[-1].location() != candidate.location() and \
                   add_order[-1].location() + candidate.location() == 0:
                       # 
                       add_order.append(candidate)
                       left_candidates.remove(candidate)
                       break
            # Remove all from the same potential location as curent additions.
            to_remove = []
            for candidate in left_candidates:
                if math.fabs(add_order[-1].location()) == math.fabs(candidate.location()):
                    to_remove.append(candidate)
            for rm in to_remove:
                left_candidates.remove(rm)

        if (len(left_candidates) >= 1):
            # Now take best, with heptad the next one can be on either side but NOT in the same
            # location as current picks
            add_order.append(left_candidates[0])
            left_candidates.remove(add_order[-1])
            to_remove = []
            for candidate in left_candidates:
                if add_order[-1].location() == candidate.location():
                    to_remove.append(candidate)
            for rm in to_remove:
                left_candidates.remove(rm)

        # Now try to add to octad
        if (len(left_candidates) >= 1):
            # Hexad
            # Find best for the next oposite location.
            for candidate in left_candidates:
                if add_order[-1].location() != candidate.location() and \
                   add_order[-1].location() + candidate.location() == 0:
                       # 
                       add_order.append(candidate)
                       left_candidates.remove(candidate)
                       break
            # Remove all from the same potential location as curent additions.
            to_remove = []
            for candidate in left_candidates:
                if math.fabs(add_order[-1].location()) == math.fabs(candidate.location()):
                    to_remove.append(candidate)
            for rm in to_remove:
                left_candidates.remove(rm)

            for candidate in left_candidates:
                if add_order[-1].location() != candidate.location() and \
                   add_order[-1].location() + candidate.location() == 0:
                       add_order.append(candidate)
                       left_candidates.remove(candidate)
                       break


        # Clean unwanted heptads
        # Heptads are more unstable and the move from 6 -> 7 should be as stable as possible.
        # Check if score difference of hexad and added heptead candidate is not too much.
        # Do not check if we have made an octad.
        if len(add_order) == 3:
            if add_order[0].score() + add_order[1].score() < add_order[2].score():
                add_order.remove(add_order[2])

        valid_tetrad_candidates[tetrad] = add_order

    # Now we have made multimers from available candidates.
    # Step 2 - Shrink down all multimers to match numbers and positions.
    # They should fill similar positions for each multimer and have same number of nucleotides.
    # Match to the lowest possible.
    desired_len = 999
    desired_positions = []
    for tetrad, candidates in valid_tetrad_candidates.items():
        if len(candidates) < desired_len and len(candidates) > 0:
            desired_len = len(candidates)
            desired_positions = []
            for candidate in candidates:
                desired_positions.append(candidate.location())

    for tetrad, candidates in valid_tetrad_candidates.items():
        if len(candidates) > 0:
            to_remove = []
            for candidate in candidates:
                if candidate.location() not in desired_positions:
                    to_remove.append(candidate)
            for rm in to_remove:
                candidates.remove(rm)

    multimers = []
    for tetrad, candidates in valid_tetrad_candidates.items():
        if len(candidates) > 0:
            multimers.append(Multimer(tetrad, candidates))

    return multimers


def linktetrado(analysis):
    return ntads(analysis)
