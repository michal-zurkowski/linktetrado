import sys
from argparse import ArgumentParser
from linktetrado.multimers import linktetrado, Multimer

# For eltetrado analysis
import eltetrado.analysis
import eltetrado.cli
import rnapolis.annotator
import rnapolis.parser

def eltetrado_analysis(args):
    cif_or_pdb = eltetrado.cli.handle_input_file(args.input)
    structure3d = rnapolis.parser.read_3d_structure(cif_or_pdb, args.model, nucleic_acid_only=False)
    structure2d = (
        rnapolis.annotator.extract_base_interactions(structure3d, args.model)
        if args.dssr_json is None
        else eltetrado.cli.read_secondary_structure_from_dssr(structure3d, args.model, args.dssr_json)
    )

    return eltetrado.analysis.eltetrado(structure2d, structure3d, args.strict, args.no_reorder, args.stacking_mismatch)

def main():
    parser = ArgumentParser('linktetrado',
        epilog='')
    parser.add_argument('-i', '--input', help='path to input PDB, PDBx/mmCIF file.')
    parser.add_argument('--print-eltetrado',
                        action='store_true',
                        help='(optional) should ElTetrado analysis output also be provided alongside multimer analysis.')
    # ElTetrado options.
    parser.add_argument('-m', '--model', help='(optional, ElTetrado) model number to process', default=1, type=int)
    parser.add_argument('--stacking-mismatch',
                        help='(optional, ElTetrado) a perfect tetrad stacking covers 4 nucleotides; this option can be used with values 1 or '
                             '2 to allow this number of nucleotides to be non-stacked with otherwise well aligned '
                             'tetrad [default=2]',
                        default=2,
                        type=int)
    parser.add_argument('--strict',
                        action='store_true',
                        help='(optional, ElTetrado) nucleotides in tetrad are found when linked only by cWH pairing')
    parser.add_argument('--no-reorder',
                        action='store_true',
                        help='(optional, ElTetrado) chains of bi- and tetramolecular quadruplexes should be reordered to be able to have '
                             'them classified; when this is set, chains will be processed in original order, which for '
                             'bi-/tetramolecular means that they will likely be misclassified; use with care!')
    parser.add_argument(
        "--dssr-json",
        help="(optional, ElTetrado) provide a JSON file generated by DSSR to read the secondary structure information from (use --nmr and --json switches)",
    )

    args = parser.parse_args()

    if not args.input:
        print(parser.print_help())
        sys.exit(1)

    analysis = eltetrado_analysis(args)

    multimers = linktetrado(analysis)

    if args.print_eltetrado:
        print(analysis)
        print()

    for multimer in multimers:
        print(multimer)
