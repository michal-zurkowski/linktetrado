import sys
from argparse import ArgumentParser

from linktetrado.multimers import linktetrado, Multimer

# For eltetrado analysis
import rnapolis.annotator
import rnapolis.parser
from eltetrado.analysis import eltetrado
from eltetrado.cli import handle_input_file


def eltetrado_analysis(args):
    cif_or_pdb = handle_input_file(args.input)
    structure3d = rnapolis.parser.read_3d_structure(cif_or_pdb, args.model, nucleic_acid_only=False)

    base_interactions = rnapolis.annotator.extract_base_interactions(structure3d, args.model)

    analysis = eltetrado(
        base_interactions,
        structure3d,
        args.strict,
        args.no_reorder,
        args.stacking_mismatch,
    )

    return analysis

def main():
    parser = ArgumentParser('linktetrado',
        epilog='')
    parser.add_argument('-i', '--input', help='path to input PDB, PDBx/mmCIF file.')
    parser.add_argument('--print-eltetrado',
                        action='store_true',
                        help='(optional) should ElTetrado analysis output also be provided alongside multimer analysis.')
    # Local cutoff params
    parser.add_argument('--tilt-max',
                        help='(optional) maximum tilt in degrees between potential polyad candidate nucleotide and all tetrad nucleotides [default=55]',
                        default=55,
                        type=float)
    parser.add_argument('--tilt-avg',
                        help='(optional) average tilt in degrees between potential polyad candidate nucleotide and all tetrad nucleotides [default=45]',
                        default=45,
                        type=float)
    parser.add_argument('--height-diff-max',
                        help='(optional) maximum height difference in Angstrem between between potential polyad candidate nucleotide and all tetrad nucleotides [default=3.7]',
                        default=3.7,
                        type=float)
    parser.add_argument('--height-diff-avg',
                        help='(optional) average height difference in Angstrem between between potential polyad candidate nucleotide and all tetrad nucleotides [default=3.15]',
                        default=3.15,
                        type=float)

    parser.add_argument('--distance-inner-max',
                        help='(optional) maximum distance in Angstrem between inner atoms between potential polyad candidate nucleotide and all tetrad nucleotides [default=13.75]',
                        default=13.75,
                        type=float)
    parser.add_argument('--distance-outer-max',
                        help='(optional) maximum distance in Angstrem between outer atoms between potential polyad candidate nucleotide and all tetrad nucleotides [default=12.5]',
                        default=12.5,
                        type=float)

    # ElTetrado options.
    parser.add_argument('-m', '--model', help='(optional, ElTetrado) model number to process', default=1, type=int)
    parser.add_argument(
        "--stacking-mismatch",
        help="a perfect tetrad stacking covers 4 nucleotides; this option can be used with value 1 or "
        "2 to allow this number of nucleotides to be non-stacked with otherwise well aligned "
        "tetrad [default=2]",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="nucleotides in tetrad are found when linked only by cWH pairing",
    )
    parser.add_argument(
        "--no-reorder",
        action="store_true",
        help="chains of bi- and tetramolecular quadruplexes should be reordered to be able to have "
        "them classified; when this is set, chains will be processed in original order, which for "
        "bi-/tetramolecular means that they will likely be misclassified; use with care!",
    )
    args = parser.parse_args()

    if not args.input:
        print(parser.print_help())
        sys.exit(1)

    analysis = eltetrado_analysis(args)

    multimers = linktetrado(analysis, args)

    if args.print_eltetrado:
        print(analysis)
        print()

    for multimer in multimers:
        print(multimer)
