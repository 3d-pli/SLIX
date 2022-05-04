from typing import Optional
import numpy
import os
import re
import SLIX
from SLIX._logging import get_logger
import matplotlib
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS

available_colormaps = {
    'rgb': SLIX.visualization.Colormap.rgb,
    'hsvBlack': SLIX.visualization.Colormap.hsv_black,
    'hsvWhite': SLIX.visualization.Colormap.hsv_white,
    'rgb_r': SLIX.visualization.Colormap.rgb_reverse,
    'hsvBlack_r': SLIX.visualization.Colormap.hsv_black_reverse,
    'hsvWhite_r': SLIX.visualization.Colormap.hsv_white_reverse
}


def create_argument_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Creation of feature set from '
                                        'scattering image.',
                            add_help=False
                            )
    # Required parameters
    parser.add_argument('-i', '--input',
                        '--direction',
                        dest='input',
                        nargs='*',
                        help='Input direction (.tiff) images. '
                             'Please put all directions in the right order.',
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        help='Output folder where images will be saved to.',
                        required=True)
    parser.add_argument('-c', '--colormap',
                        dest='colormap',
                        required=False,
                        choices=available_colormaps.keys(),
                        default='hsvBlack',
                        help='Colormap to use for the images.')
    parser.add_argument('--disable_colorbubble',
                        required=False,
                        action='store_true',
                        help='Disable the colorbubble and only write the visualization image.')
    parser.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )

    subparser = parser.add_subparsers(dest='command')
    fom_parser = subparser.add_parser('fom', help="Write approximate "
                                                  "fiber orientation map from"
                                                  " direction images.")
    fom_parser.add_argument('--inclination',
                            nargs='*',
                            help='Input inclination (.tiff) images. '
                                 'Please put all inclinations in the right order.',
                            required=False)
    fom_parser.add_argument('--output_type',
                            required=False,
                            default='tiff',
                            help='Define the output data type of the parameter'
                                 ' images. Default = tiff. Supported types:'
                                 ' h5, tiff.')
    fom_parser.add_argument('--saturation',
                            required=False,
                            default="",
                            type=str,
                            help='Change the saturation of the FOM based on another parameter map. The parameter'
                                 ' map will be normed to 0-1.')
    fom_parser.add_argument('--value',
                            required=False,
                            default="",
                            type=str,
                            help='Change the value of the FOM based on another parameter map. The parameter'
                                 ' map will be normed to 0-1.')

    vector_parser = subparser.add_parser('vector', help="Write vector "
                                                        "orientation "
                                                        "map from direction"
                                                        " images. Please add "
                                                        "the corresponding "
                                                        "measurement image for"
                                                        " the background.")
    vector_parser.add_argument('--slimeasurement', type=str, required=True,
                               help='Add measurement to the background'
                                    'of the visualized image.')
    vector_parser.add_argument('--distribution', action='store_true',
                               help='Print vector distribution instead of '
                                    'just a single vector. With this,'
                                    ' it is easier to check if vectors are '
                                    'similar to each other when reducing'
                                    ' the number of vectors.')
    vector_parser.add_argument('--alpha', default=0.8, type=float,
                               help='Factor for the vectors which will be used'
                                    ' during visualization. A higher value'
                                    ' means that the vectors will be more'
                                    ' visible.')
    vector_parser.add_argument('--scale', default=-1, type=float,
                               help='Increase the vector length by the given '
                                    'scale. Vectors will be longer and might '
                                    'overlap if the scale is too high. If no '
                                    'scale is used the scale will match the '
                                    'thinout option.')
    vector_parser.add_argument('--thinout', default=20, type=int,
                               help='Thin out vectors by an integer value. '
                                    'A thinout of 20 means that both the '
                                    'x-axis and y-axis are thinned by '
                                    'a value of 20.')
    vector_parser.add_argument('--threshold', default=0.5, type=float,
                               help='When using the thinout option, you might'
                                    ' not want to get a vector for a lonely'
                                    ' vector in the base image. This parameter'
                                    ' defines a percentage which will be used '
                                    ' to reduce the number of shown vectors.'
                                    ' The percentage defines the number of '
                                    ' vectors present in the area which are '
                                    'not zero.')
    vector_parser.add_argument('--vector_width', default=-1, type=float,
                               help='Change the default width of the shown '
                                    'vectors. A larger value might help'
                                    ' to see the vectors better when using'
                                    ' a large thinout.')
    vector_parser.add_argument('--weight_map', type=str, required=False,
                               help="Weight the unit vectors shown by the "
                                    "given parameter map. Lower values "
                                    "will reduce the length of the vectors.",
                               default=None)
    vector_parser.add_argument('--dpi',
                               default=1000,
                               type=int,
                               help='Set the Matplotlib dpi vale for the vector plots. Higher dpi images'
                                    ' will be more clear but also consume more disk space and will take '
                                    'significantly longer to render completely.')

    # Return generated parser
    return parser


def main():
    parser = create_argument_parser()
    arguments = parser.parse_args()
    args = vars(arguments)

    if not SLIX.io.check_output_dir(args['output']):
        exit(1)

    filename_without_extension = \
        os.path.splitext(os.path.basename(args['input'][0]))[0]
    filename_without_extension = re.sub(r"dir_[0-9]+", "",
                                        filename_without_extension)
    output_path_name = args['output'] + '/' + filename_without_extension

    direction_image = None
    for direction_file in args['input']:
        single_direction_image = SLIX.io.imread(direction_file)
        if direction_image is None:
            direction_image = single_direction_image
        else:
            if len(direction_image.shape) == 2:
                direction_image = numpy.stack((direction_image,
                                               single_direction_image),
                                              axis=-1)
            else:
                direction_image = numpy.concatenate((direction_image,
                                                     single_direction_image
                                                     [:, :, numpy.newaxis]),
                                                    axis=-1)
    if args['command'] == "fom":
        write_fom(args, direction_image, output_path_name)
    if args['command'] == "vector":
        write_vector(args, direction_image, output_path_name)


def read_weight_map(path: Optional[str]) -> Optional[numpy.ndarray]:
    if path is not None:
        weight_map = SLIX.io.imread(path).astype(float)
        weight_map = (weight_map - weight_map.min()) / \
                     (numpy.percentile(weight_map, 95) - weight_map.min())
    else:
        weight_map = None

    return weight_map


def write_vector(args, direction_image, output_path_name):
    logger = get_logger(__name__)
    image = SLIX.io.imread(args['slimeasurement'])
    UnitX, UnitY = SLIX.toolbox.unit_vectors(direction_image, use_gpu=False)

    # Try to fix image shape if the two axes are swapped
    if image.shape[:2] != UnitX.shape[:2] and \
            image.shape[:2][::-1] == UnitX.shape[:2]:
        image = image.T
    if image.shape[:2] != UnitX.shape[:2]:
        logger.warning("Direction and SLI measurement are not correctly aligned."
                       " The program will still run but the results might not represent"
                       " the expected result. Please check your input!")
    weight_map = read_weight_map(args['weight_map'])

    thinout = args['thinout']
    scale = args['scale']
    alpha = args['alpha']
    background_threshold = args['threshold']
    vector_width = args['vector_width']
    fig, ax = plt.subplots(dpi=args['dpi'])
    fig.subplots_adjust(0, 0, 1, 1)
    if vector_width < 0:
        vector_width = numpy.ceil(thinout / 3)
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(numpy.max(image, axis=-1), cmap='gray')
    if args['distribution']:
        write_vector_distribution(UnitX, UnitY, alpha, args, output_path_name, scale, thinout, vector_width, weight_map,
                                  fig, ax)
    else:
        write_vector_field(UnitX, UnitY, alpha, args, background_threshold, output_path_name, scale, thinout,
                           vector_width, weight_map, fig, ax)
    plt.clf()


def write_vector_field(UnitX, UnitY, alpha, args, background_threshold, output_path_name, scale, thinout, vector_width,
                       weight_map, fig, ax):

    SLIX.visualization.unit_vectors(UnitX, UnitY,
                                    ax=ax,
                                    thinout=thinout,
                                    scale=scale,
                                    alpha=alpha,
                                    vector_width=vector_width,
                                    background_threshold=
                                    background_threshold,
                                    weighting=weight_map,
                                    colormap=available_colormaps[args['colormap']])
    fig.canvas.draw()
    vector_image = numpy.array(fig.canvas.buffer_rgba(), dtype=numpy.uint8)
    vector_image = vector_image[:, :, :3]
    SLIX.io.imwrite_rgb(f"{output_path_name}vector_{args['colormap']}.tiff", vector_image)
    if not args['disable_colorbubble']:
        SLIX.io.imwrite_rgb(f"{args['output']}/color_bubble_{args['colormap']}.tiff",
                            SLIX.visualization.color_bubble(available_colormaps[args['colormap']]))


def write_vector_distribution(UnitX, UnitY, alpha, args, output_path_name, scale,
                              thinout, vector_width, weight_map, fig, ax):
    SLIX.visualization.unit_vector_distribution(UnitX, UnitY,
                                                ax=ax,
                                                thinout=thinout,
                                                scale=scale,
                                                alpha=alpha,
                                                vector_width=
                                                vector_width,
                                                weighting=weight_map,
                                                colormap=available_colormaps[args['colormap']])
    fig.canvas.draw()
    vector_image = numpy.array(fig.canvas.buffer_rgba(), dtype=numpy.uint8)
    vector_image = vector_image[:, :, :3]
    SLIX.io.imwrite_rgb(f"{output_path_name}vector_distribution_{args['colormap']}.tiff", vector_image)
    if not args['disable_colorbubble']:
        SLIX.io.imwrite_rgb(f"{args['output']}/color_bubble_{args['colormap']}.tiff",
                            SLIX.visualization.color_bubble(available_colormaps[args['colormap']]))


def write_fom(args, direction_image, output_path_name) -> None:
    logger = get_logger(__name__)
    if args['inclination'] is not None:
        inclination_image = read_inclination(args)
    else:
        inclination_image = numpy.zeros_like(direction_image)

    output_data_type = '.' + args['output_type']
    if output_data_type not in ['.h5', '.tiff', '.tif']:
        logger.error('Output data type is not supported. Please choose a valid '
                     'datatype!')
        exit(1)
    saturation = None
    value = None
    if args['saturation']:
        saturation = SLIX.io.imread(args['saturation'])
    if args['value']:
        value = SLIX.io.imread(args['value'])
    rgb_fom = SLIX.visualization.direction(direction_image, inclination_image, saturation, value,
                                           available_colormaps[args['colormap']])
    SLIX.io.imwrite_rgb(f"{output_path_name}fom_{args['colormap']}{output_data_type}", rgb_fom)
    if not args['disable_colorbubble']:
        SLIX.io.imwrite_rgb(f"{args['output']}/color_bubble_{args['colormap']}.tiff",
                            SLIX.visualization.color_bubble(available_colormaps[args['colormap']]))


def read_inclination(args) -> numpy.ndarray:
    inclination_image = None
    for inclination_file in args['inclination']:
        single_inclination_image = SLIX.io.imread(inclination_file)
        if inclination_image is None:
            inclination_image = single_inclination_image
        else:
            if len(inclination_image.shape) == 2:
                inclination_image = numpy.stack((inclination_image,
                                                 single_inclination_image),
                                                axis=-1)
            else:
                inclination_image = numpy.concatenate((inclination_image,
                                                       single_inclination_image
                                                       [:, :, numpy.newaxis]),
                                                      axis=-1)
    return inclination_image


if __name__ == "__main__":
    main()
