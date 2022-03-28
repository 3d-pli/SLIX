from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, SUPPRESS
import glob
import os
import sys
from SLIX import io, classification
from SLIX._logging import get_logger


def load_parameter_maps(input_folder) -> {}:
    """
    Loads all parameter maps in the input folder.

    Args:
        input_folder: Folder containing the parameter maps.

    Returns:
        parameter_maps: Dictionary containing all parameter maps.

    """
    valid_parameter_map_names = [
        'high_prominence_peaks',
        'low_prominence_peaks',
        'peakprominence',
        'peakdistance',
        'peakwidth',
        'dir_1',
        'dir_2',
        'dir_3',
        'avg',
        'min',
        'max'
    ]

    basename = None

    loaded_parameter_maps = {}
    list_of_files = glob.glob(input_folder + "/*")
    for file in list_of_files:
        for param in valid_parameter_map_names:
            if param not in file:
                continue
            if basename is None:
                basename = os.path.splitext(os.path.basename(file))[0]
                basename = basename.replace(param, 'basename')
            try:
                image_in_file = io.imread(file)
                loaded_parameter_maps[param] = image_in_file
            except ...:
                pass

    return loaded_parameter_maps, basename


def create_argparse():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Creation of feature set from '
                                        'scattering image.',
                            add_help=False
                            )
    # Required parameters
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i',
                          '--input',
                          help='Input files (.nii or .tiff/.tif).',
                          required=True)
    required.add_argument('-o',
                          '--output',
                          help='Output folder where images will be saved to',
                          required=True)
    # Optional parameters
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--output_type',
                          required=False,
                          default='tiff',
                          help='Define the output data type of the parameter'
                               ' images. Default = tiff. Supported types:'
                               ' nii, h5, tiff.')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )
    # Parameters to select which images will be generated
    image = parser.add_argument_group('output choice (none = all except '
                                      'optional)')
    image.add_argument('--all',
                       action='store_true')
    image.add_argument('--inclination',
                       action='store_true')
    image.add_argument('--crossing',
                       action='store_true')
    image.add_argument('--flat',
                       action='store_true')
    # Return generated parser
    return parser


def main():
    parser = create_argparse()
    arguments = parser.parse_args()
    args = vars(arguments)
    logger = get_logger("SLIXCluster")

    output_data_type = '.' + args['output_type']

    if output_data_type not in ['.nii', '.nii.gz', '.h5', '.tiff', '.tif']:
        logger.error('Output data type is not supported. Please choose a valid '
                     'datatype!')
        exit(1)

    if not io.check_output_dir(args['output']):
        exit(1)

    all = False
    inclination = False
    crossing = False
    flat = False

    if args['all']:
        all = True
    if args['inclination']:
        inclination = True
    if args['crossing']:
        crossing = True
    if args['flat']:
        flat = True

    # If no parameter map needs to be generated
    if not all and not inclination and not crossing and not flat:
        parser.print_help()
        sys.exit(0)

    # Load all parameter maps from the user given folder
    loaded_parameter_maps, basename = load_parameter_maps(args['input'])

    if flat or inclination or crossing:
        loaded_parameter_maps['flat_mask'] = classification.flat_mask(
            loaded_parameter_maps['high_prominence_peaks'],
            loaded_parameter_maps['low_prominence_peaks'],
            loaded_parameter_maps['peakdistance']
        )

        if flat:
            flat_name = basename.replace('basename', 'flat_mask')
            io.imwrite(f'{args["output"]}/{flat_name}{output_data_type}',
                       loaded_parameter_maps['flat_mask'])

    if inclination:
        inclination_mask = classification.inclinated_mask(
            loaded_parameter_maps['high_prominence_peaks'],
            loaded_parameter_maps['peakdistance'],
            loaded_parameter_maps['max'],
            loaded_parameter_maps['flat_mask']
        )
        inclination_name = basename.replace('basename', 'inclination_mask')
        io.imwrite(f'{args["output"]}/{inclination_name}{output_data_type}',
                   inclination_mask)

    if crossing:
        crossing_mask = classification.crossing_mask(
            loaded_parameter_maps['high_prominence_peaks'],
            loaded_parameter_maps['max'],
        )
        crossing_name = basename.replace('basename', 'crossing_mask')
        io.imwrite(f'{args["output"]}/{crossing_name}{output_data_type}',
                   crossing_mask)

    if all:
        full_mask = classification.full_mask(
            loaded_parameter_maps['high_prominence_peaks'],
            loaded_parameter_maps['low_prominence_peaks'],
            loaded_parameter_maps['peakdistance'],
            loaded_parameter_maps['max']
        )
        full_name = basename.replace('basename', 'classification_mask')
        io.imwrite(f'{args["output"]}/{full_name}{output_data_type}',
                   full_mask)


if __name__ == "__main__":
    main()
