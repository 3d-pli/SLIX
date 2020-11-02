#!/usr/bin/env python3

from SLIX import toolbox
import numpy
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, SUPPRESS
import os
import tqdm


def create_argument_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Creation of feature set from '
                                        'scattering image.',
                            add_help=False
                            )
    # Required parameters
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i',
                          '--input',
                          nargs='*',
                          help='Input files (.nii or .tiff/.tif).',
                          required=True)
    required.add_argument('-o',
                          '--output',
                          help='Output folder where images will be saved to',
                          required=True)
    # Optional parameters
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--prominence_threshold',
                          type=float,
                          default=0.08,
                          help='Change the threshold for prominent peaks. '
                               'Peaks with lower prominences will not be used'
                               ' for further evaluation. (Default: 8%% of'
                               ' total signal amplitude.) Only recommended for'
                               ' experienced users!')
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
    image.add_argument('--direction',
                       action='store_true',
                       help='Add crossing directions (dir_1, dir_2, dir_3)'
                       )
    image.add_argument('--peaks',
                       action='store_true',
                       help='Add number of peaks below prominence and above '
                            'prominence')
    image.add_argument('--peakprominence',
                       action='store_true',
                       help='Add average peak prominence for each pixel')
    image.add_argument('--peakwidth',
                       action='store_true',
                       help='Add average width of all peaks detected')
    image.add_argument('--peakdistance',
                       action='store_true',
                       help='Add distance between two peaks if two peaks are '
                            'detected')
    image.add_argument('--optional',
                       action='store_true',
                       help='Adds Max/Min/Non Crossing Direction to the output'
                            ' images.')
    image.add_argument('--no_centroids',
                       action='store_false',
                       help='Disable centroid calculation. Not recommended.')
    # Return generated parser
    return parser


def main():
    parser = create_argument_parser()
    arguments = parser.parse_args()
    args = vars(arguments)

    DIRECTION = True
    PEAKS = True
    PEAKWIDTH = True
    PEAKPROMINENCE = True
    PEAKDISTANCE = True

    if args['direction'] or args['peaks'] or args['peakprominence'] or \
            args['peakwidth'] or args['peakdistance']:
        DIRECTION = args['direction']
        PEAKS = args['peaks']
        PEAKPROMINENCE = args['peakprominence']
        PEAKWIDTH = args['peakwidth']
        PEAKDISTANCE = args['peakdistance']
    OPTIONAL = args['optional']

    print(
        'SLI Feature Generator:\n' +
        'Chosen feature maps:\n' +
        'Direction maps: ' + str(DIRECTION) + '\n' +
        'Peak maps: ' + str(PEAKS) + '\n' +
        'Peak prominence map: ' + str(PEAKPROMINENCE) + '\n' +
        'Peak width map: ' + str(PEAKWIDTH) + '\n' +
        'Peak distance map: ' + str(PEAKDISTANCE) + '\n' +
        'Optional maps: ' + str(OPTIONAL) + '\n'
    )

    paths = args['input']
    if not isinstance(paths, list):
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    tqdm_paths = tqdm.tqdm(paths)
    for path in tqdm_paths:
        filename_without_extension = \
            os.path.splitext(os.path.basename(path))[0]
        output_path_name = args['output'] + '/' + filename_without_extension
        tqdm_paths.set_description(filename_without_extension)

        output_string = ""

        image = numpy.fromfile(path, sep='\n')
        image = image[numpy.newaxis, numpy.newaxis, ...]
        significant_peaks = toolbox. \
            significant_peaks(image,
                              low_prominence=args['prominence_threshold'],
                              use_gpu=False)

        if PEAKS:
            peaks = toolbox.peaks(image, use_gpu=False)
            output_string += 'High Prominence Peaks,' + \
                             str(int(numpy.sum(significant_peaks, axis=-1)))\
                             + '\n'
            output_string += 'Low Prominence Peaks,' + \
                             str(int(numpy.sum(peaks, axis=-1) -
                                 numpy.sum(significant_peaks, axis=-1)))\
                             + '\n'

        if PEAKPROMINENCE:
            mean_prominence = toolbox.mean_peak_prominence(image,
                                                           significant_peaks,
                                                           use_gpu=False)

            output_string += 'Mean Prominence,' \
                             + str(float(mean_prominence.flatten())) + '\n'

        if PEAKWIDTH:
            mean_peak_width = toolbox.mean_peak_width(image,
                                                      significant_peaks,
                                                      use_gpu=False)
            output_string += 'Mean peak width,' \
                             + str(float(mean_peak_width)) \
                             + '\n'

        if args['no_centroids']:
            centroids = toolbox. \
                centroid_correction(image, significant_peaks,
                                    use_gpu=False)
        else:
            centroids = numpy.zeros(image.shape)

        if PEAKDISTANCE:
            mean_peak_distance = toolbox.mean_peak_distance(significant_peaks,
                                                            centroids,
                                                            use_gpu=False)
            output_string += 'Mean peak distance,' \
                             + str(float(mean_peak_distance)) + '\n'

        if DIRECTION:
            direction = toolbox.direction(significant_peaks, centroids,
                                          use_gpu=False)
            output_string += 'Direction,' + str(direction.flatten()) + '\n'

        if OPTIONAL:
            min_img = numpy.min(image, axis=-1)
            output_string += 'Min,' + str(float(min_img.flatten())) + '\n'
            del min_img

            max_img = numpy.max(image, axis=-1)
            output_string += 'Max,' + str(float(max_img.flatten())) + '\n'
            del max_img

            avg_img = numpy.average(image, axis=-1)
            output_string += 'Avg,' + str(float(avg_img.flatten())) + '\n'
            del avg_img

        with open(output_path_name+'.csv', mode='w') as f:
            f.write(output_string)
            f.flush()
