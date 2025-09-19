# Project description

LinkTetrado is algorithm designed for the identification and classification
of multimeric nucleotide assemblies in nucleic acid structures.
LinkTetrado automatically identifies nucleotides interacting with tetrads in
a planar arrangement, allowing for the detection of pentads, hexads, heptads, octads, and beyond.
It analyzes nucleic acid 3D structures, accepting both PDB and mmCIF file formats.
It leverages the ElTetrado engine to extract detailed structural information about
base pairs and tetrads in an input structure. Next, it searches the space in the vicinity
of tetrads for possible nucleotides that could interact with the tetrads.

# Installation

Please download repository and run:

    python -m pip install .

<!-- TODO prepare proper pip install package

Please run:

    pip install linktetrado

-->

# Dependencies

The project is written in Python 3.8+ and requires
[NumPy](https://numpy.org/), and 
[ElTetrado](https://github.com/tzok/eltetrado) (Zok _et al._, 2022;
Popenda _et al._, 2020; Zok _et al._, 2020).

LinkTetrado parses the output of
[ElTetrado](https://github.com/tzok/eltetrado). It can also process PDB or
PDBx/mmCIF files which will be first analyzed internally with ElTetrado.

# Usage

    usage: linktetrado [-h] [-i INPUT] [--print-eltetrado] [--tilt-max TILT_MAX]
                       [--tilt-avg TILT_AVG] [--height-diff-max HEIGHT_DIFF_MAX]
                       [--height-diff-avg HEIGHT_DIFF_AVG]
                       [--distance-inner-max DISTANCE_INNER_MAX]
                       [--distance-outer-max DISTANCE_OUTER_MAX] [-m MODEL]
                       [--stacking-mismatch STACKING_MISMATCH] [--strict]
                       [--no-reorder]

    options:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            path to input PDB, PDBx/mmCIF file.
      --print-eltetrado     (optional) should ElTetrado analysis output also be
                            provided alongside multimer analysis.
      --tilt-max TILT_MAX   (optional) maximum tilt in degrees between potential
                            polyad candidate nucleotide and all tetrad nucleotides
                            [default=55]
      --tilt-avg TILT_AVG   (optional) average tilt in degrees between potential
                            polyad candidate nucleotide and all tetrad nucleotides
                            [default=45]
      --height-diff-max HEIGHT_DIFF_MAX
                            (optional) maximum height difference in Angstrem between
                            between potential polyad candidate nucleotide and all
                            tetrad nucleotides [default=3.7]
      --height-diff-avg HEIGHT_DIFF_AVG
                            (optional) average height difference in Angstrem between
                            between potential polyad candidate nucleotide and all
                            tetrad nucleotides [default=3.15]
      --distance-inner-max DISTANCE_INNER_MAX
                            (optional) maximum distance in Angstrem between inner
                            atoms between potential polyad candidate nucleotide and
                            all tetrad nucleotides [default=13.75]
      --distance-outer-max DISTANCE_OUTER_MAX
                            (optional) maximum distance in Angstrem between outer
                            atoms between potential polyad candidate nucleotide and
                            all tetrad nucleotides [default=12.5]
      -m MODEL, --model MODEL
                            (optional, ElTetrado) model number to process
      --stacking-mismatch STACKING_MISMATCH
                            a perfect tetrad stacking covers 4 nucleotides; this
                            option can be used with value 1 or 2 to allow this
                            number of nucleotides to be non-stacked with otherwise
                            well aligned tetrad [default=2]
      --strict              nucleotides in tetrad are found when linked only by cWH
                            pairing
      --no-reorder          chains of bi- and tetramolecular quadruplexes should be
                            reordered to be able to have them classified; when this
                            is set, chains will be processed in original order,
                            which for bi-/tetramolecular means that they will likely
                            be misclassified; use with care!


<!-- TODO

# Examples

-->

# Bibliography

<div id="refs" class="references csl-bib-body">

1.  Zok T, Popenda M, Szachniuk M (2020) ElTetrado: a tool for
    identification and classification of tetrads and quadruplexes, BMC
    Bioinformatics 21:40 (doi:10.1186/s12859-020-3385-1).

2.  Popenda M, Miskiewicz J, Sarzynska J, Zok T, Szachniuk M (2020)
    Topology-based classification of tetrads and quadruplex structures,
    Bioinformatics 36(4):1129-1134 (doi:10.1093/bioinformatics/btz738).

3.  Zok T, Kraszewska N, Miskiewicz J, Pielacinska P, Zurkowski M,
    Szachniuk M (2022) ONQUADRO: a database of experimentally determined
    quadruplex structures, Nucleic Acids Research 50(D1):D253-D258
    (doi:10.1093/nar/gkab1118).

</div>
