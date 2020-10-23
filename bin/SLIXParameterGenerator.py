#/usr/bin/env python

from SLIX import toolbox, io
import tifffile
import numpy
import argparse
import os
import time


DIRECTION = True
PEAKS = True
PEAKWIDTH = True
PEAKPROMINENCE = True
PEAKDISTANCE = True
OPTIONAL = False


def create_argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Creation of feature set from scattering image.',
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
    optional.add_argument('-v',
                          '--verbose',
                          action='store_true')
    optional.add_argument('--with_mask',
                          action='store_true',
                          help='Use mask to try to remove some of the background')
    optional.add_argument('--mask_threshold',
                          type=int,
                          default=10,
                          help='Value for filtering background noise when calculating masks.'
                               'Higher values might result in the removal of some of the gray matter in the mask'
                               'but will remove the background more effectively.')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    # Parameters to select which images will be generated
    image = parser.add_argument_group('output choice (none = all except optional)')
    image.add_argument('--direction',
                       action='store_true',
                       help='Add crossing directions (dir_1, dir_2, dir_3)'
                       )
    image.add_argument('--peaks',
                       action='store_true',
                       help='Add number of peaks below prominence and above prominence')
    image.add_argument('--peakprominence',
                       action='store_true',
                       help='Add average peak prominence for each pixel')
    image.add_argument('--peakwidth',
                       action='store_true',
                       help='Add average width of all peaks detected')
    image.add_argument('--peakdistance',
                       action='store_true',
                       help='Add distance between two peaks if two peaks are detected')
    image.add_argument('--optional',
                       action='store_true',
                       help='Adds Max/Min/Non Crossing Direction to the output images.')
    image.add_argument('--no_centroids',
                       action='store_false',
                       help='Disable centroid calculation. Not recommended.')
    # Return generated parser
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    arguments = parser.parse_args()
    args = vars(arguments)

    if args['direction'] or args['peaks'] or args['peakprominence'] or args['peakwidth'] or args['peakdistance']:
        DIRECTION = args['direction']
        PEAKS = args['peaks']
        PEAKPROMINENCE = args['peakprominence']
        PEAKWIDTH = args['peakwidth']
        PEAKDISTANCE = args['peakdistance']
    OPTIONAL = args['optional']

    if args['verbose']:
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

    for path in paths:
        folder = os.path.dirname(path)
        filename_without_extension = os.path.splitext(os.path.basename(path))[0]
        output_path_name = args['output'] + '/' + filename_without_extension
        image = io.imread(path)
        if args['verbose']:
            print(path)
            start_time = time.time()

        peaks = None
        print('peaks')
        peaks = toolbox.peaks(image)
        prominence_full = toolbox.peak_prominence(image, peak_image=peaks).astype('float32')
        significant_peaks = peaks.copy()
        significant_peaks[prominence_full < 0.08] = 0
        if PEAKS:
            io.imwrite(output_path_name+'_all_peak_positions.tiff', peaks)
            io.imwrite(output_path_name+'_all_peak_positions.tiff', significant_peaks)

        if PEAKPROMINENCE:
            print('peakprominence')
            peak_prominence_full = toolbox.peak_prominence(image, peak_image=peaks,
                                                           kind_of_normalization=1).astype('float32')
            io.imwrite(output_path_name+'_prominence.tiff', peak_prominence_full)
            del peak_prominence_full

            peak_prominence_full = toolbox.peak_prominence(image, peak_image=peaks).astype('float32')
            peak_prominence_full[peak_prominence_full < 0.08] = 0
            io.imwrite(output_path_name+'_prominence_filtered.tiff', peak_prominence_full)

        if PEAKWIDTH:
            print('peakwidth')
            peak_width_full = toolbox.peak_width(image, significant_peaks)
            io.imwrite(output_path_name+'_peak_width.tiff', peak_width_full)
            del peak_width_full

        if args['no_centroids']:
            print('centroids')
            centroids = toolbox.centroid_correction(image, significant_peaks)
            io.imwrite(output_path_name+'_centroid_peaks.tiff', centroids)
        else:
            print('no centroids')
            centroids = numpy.zeros(image.shape)

        if PEAKDISTANCE:
            peak_distance_full = toolbox.peak_distance(significant_peaks, centroids)
            io.imwrite(output_path_name+'_peak_distance.tiff', peak_distance_full)
            del peak_distance_full

        if DIRECTION:
            direction = toolbox.direction(significant_peaks, centroids)
            for dim in range(direction.shape[-1]):
                io.imwrite(output_path_name+'_direction_'+str(dim)+'.tiff', direction[:, :, dim])

        if args['verbose']:
            print("--- %s seconds ---" % (time.time() - start_time))
