#!/usr/bin/env python3

import argparse
import multiprocessing
import os

import numpy
import pymp
from PIL import Image

# Imports
import Library.ScatterPlotToolbox as toolbox

# Default parameters. Will be changed when using the argument parser when calling the program.
DIRECTION = True
PEAKS = True
PEAKWIDTH = True
PEAKPROMINENCE = True
PEAKDISTANCE = True
OPTIONAL = False


def full_pipeline(PATH, OUTPUT, ROISIZE, APPLY_MASK, APPLY_SMOOTHING, MASK_THRESHOLD):
    """
    Generates feature maps based on given parameters and write them into an output directory based on the OUTPUT argument.
    Depending on the global set parameters by the argument parser only a subset of the possible feature maps will be generated.

    Args:
        PATH: Path to SLI-measurement
        OUTPUT: Output file path without any extension. This path will be extended with the tags of the respective feature maps.
        ROISIZE: Downsampling argument. Will reduce the image dimensions to reduce memory usage and time.
        APPLY_MASK: Generate a mask before evaluating feature maps to remove the background from the remaining tissue. Threshold is based on MASK_THRESHOLD.
        APPLY_SMOOTHING: Reduce image noise by applying a Savitzky-Golay filter with a window length of 9 and polynomial order of 2
        MASK_THRESHOLD: Set threshold for the APPLY_MASK parameter.

    Returns: None
    """
    image = toolbox.read_image(PATH)
    print(PATH)
    path_name = OUTPUT
    roiset = toolbox.zaxis_roiset(image, ROISIZE)
    if APPLY_SMOOTHING:
        print('Smoothing will be applied.')
        roiset = toolbox.smooth_roiset(roiset, 9, 2)
    if APPLY_MASK:
        mask = toolbox.create_background_mask(roiset, MASK_THRESHOLD)
        roiset[mask, :] = 0
    print("Roi finished")

    """
    Corresponding boolean values for selected_parameters
    0 : Min
    1 : Max
    2 : Average
    3 : Low Prominence Peaks
    4 : High Prominence Peaks
    5 : Peakwidth
    6 : Peakprominence
    7 : Peakdistance
    8 : Non Crossing Direction
    9-11 : Crossing Direction
    """
    selected_methods = [OPTIONAL, OPTIONAL, OPTIONAL, PEAKS, PEAKS, PEAKWIDTH, PEAKPROMINENCE, PEAKDISTANCE, OPTIONAL,
                        DIRECTION]
    parameter_maps = generate_feature_maps(roiset, selected_methods)
    current_index = 0
    if OPTIONAL:
        # Maximum
        max_array = parameter_maps[:, current_index]
        max_image = toolbox.reshape_array_to_image(max_array, image.shape[0], ROISIZE)
        Image.fromarray(max_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(path_name + '_max.tiff')
        print("Max image written")
        current_index += 1

        # Minimum
        min_array = parameter_maps[:, current_index]
        min_image = toolbox.reshape_array_to_image(min_array, image.shape[0], ROISIZE)
        Image.fromarray(min_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(path_name + '_min.tiff')
        print("Min image written")
        current_index += 1

        # Average
        avg_array = parameter_maps[:, current_index]
        avg_image = toolbox.reshape_array_to_image(avg_array, image.shape[0], ROISIZE)
        Image.fromarray(avg_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(path_name + '_avg.tiff')
        print("Avg image written")
        current_index += 1

    if PEAKS:
        # Low Prominence
        low_prominence_array = parameter_maps[:, current_index]
        low_peak_image = toolbox.reshape_array_to_image(low_prominence_array.astype('int8'), image.shape[0], ROISIZE)
        Image.fromarray(low_peak_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(
            path_name + '_low_prominence_peaks.tiff')
        print('Low Peaks Written')
        current_index += 1

        # High Prominence
        high_prominence_array = parameter_maps[:, current_index]
        high_peak_image = toolbox.reshape_array_to_image(high_prominence_array.astype('int8'), image.shape[0], ROISIZE)
        Image.fromarray(high_peak_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(
            path_name + '_high_prominence_peaks.tiff')
        print('High Peaks Written')
        current_index += 1

    if PEAKWIDTH:
        # Peakwidth
        peakwidth_array = parameter_maps[:, current_index]
        peakwidth_image = toolbox.reshape_array_to_image(peakwidth_array, image.shape[0], ROISIZE)
        Image.fromarray(peakwidth_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(
            path_name + '_peakwidth.tiff')
        print("Peakwidth written")
        current_index += 1

    if PEAKPROMINENCE:
        # Peakprominence
        peakprominence_array = parameter_maps[:, current_index]
        peakprominence_image = toolbox.reshape_array_to_image(peakprominence_array, image.shape[0], ROISIZE)
        Image.fromarray(peakprominence_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(
            path_name + '_peakprominence.tiff')
        print("Peakprominence written")
        current_index += 1

    if PEAKDISTANCE:
        distance_array = parameter_maps[:, current_index]
        distance_image = toolbox.reshape_array_to_image(distance_array, image.shape[0], ROISIZE)
        Image.fromarray(distance_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(
            path_name + '_peakdistance.tiff')
        print("Peakdistance written")
        current_index += 1

    if OPTIONAL:
        # Direction Non Crossing
        direction_array = parameter_maps[:, current_index]
        direction_image = toolbox.reshape_array_to_image(direction_array, image.shape[0], ROISIZE)
        Image.fromarray(direction_image).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(
            path_name + '_non_crossing_dir.tiff')
        print("Non Crossing Direction written")
        current_index += 1

    if DIRECTION:
        # Direction Crossing
        direction_array = parameter_maps[:, current_index:]
        dir_1 = toolbox.reshape_array_to_image(direction_array[:, 0], image.shape[0], ROISIZE)
        dir_2 = toolbox.reshape_array_to_image(direction_array[:, 1], image.shape[0], ROISIZE)
        dir_3 = toolbox.reshape_array_to_image(direction_array[:, 2], image.shape[0], ROISIZE)
        Image.fromarray(dir_1).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(path_name + '_dir_1.tiff')
        Image.fromarray(dir_2).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(path_name + '_dir_2.tiff')
        Image.fromarray(dir_3).resize(image.shape[:2][::-1], resample=Image.NEAREST).save(path_name + '_dir_3.tiff')
        print("Crossing Directions written")


def generate_feature_maps(roiset, selected_parameters):
    """
    Corresponding boolean values for selected_parameters
    0 : Min
    1 : Max
    2 : Average
    3 : Low Prominence Peaks
    4 : High Prominence Peaks
    5 : Peakwidth
    6 : Peakprominence
    7 : Peakdistance
    8 : Non Crossing Direction
    9-11 : Crossing Direction
    """

    number_of_parameter_maps = numpy.count_nonzero(selected_parameters)
    if selected_parameters[-1]:
        number_of_parameter_maps += 2
    resulting_parameter_maps = pymp.shared.array((roiset.shape[0], number_of_parameter_maps), dtype=numpy.float)

    with pymp.Parallel(toolbox.CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            current_index = 0

            if numpy.any(selected_parameters[3:]):
                peaks = toolbox.all_peaks(roi)
            if numpy.any(selected_parameters[4:6]):
                peak_positions_high_non_centroid = toolbox.peak_positions(peaks, roi, centroid_calculation=False)
            if numpy.any(selected_parameters[7:]):
                peak_positions_high = toolbox.peak_positions(peaks, roi)

            if selected_parameters[0]:
                resulting_parameter_maps[i, current_index] = roi.min()
                current_index += 1
            if selected_parameters[1]:
                resulting_parameter_maps[i, current_index] = roi.max()
                current_index += 1
            if selected_parameters[2]:
                resulting_parameter_maps[i, current_index] = roi.mean()
                current_index += 1
            if selected_parameters[3]:
                peak_positions_low_non_centroid = toolbox.peak_positions(peaks, roi, 0, toolbox.TARGET_PROMINENCE,
                                                                         centroid_calculation=False)
                resulting_parameter_maps[i, current_index] = len(peak_positions_low_non_centroid)
                current_index += 1
            if selected_parameters[4]:
                resulting_parameter_maps[i, current_index] = len(peak_positions_high_non_centroid)
                current_index += 1
            if selected_parameters[5]:
                resulting_parameter_maps[i, current_index] = toolbox.peakwidth(peak_positions_high_non_centroid, roi,
                                                                               len(peak_positions_high_non_centroid),
                                                                               len(roi) // 2)
                current_index += 1
            if selected_parameters[6]:
                resulting_parameter_maps[i, current_index] = toolbox.prominence(peak_positions_high_non_centroid, roi,
                                                                                len(peak_positions_high_non_centroid))
                current_index += 1
            if selected_parameters[7]:
                resulting_parameter_maps[i, current_index] = toolbox.peakdistance(peak_positions_high,
                                                                                  len(peak_positions_high),
                                                                                  len(roi) // 2)
                current_index += 1
            if selected_parameters[8]:
                resulting_parameter_maps[i, current_index] = toolbox.non_crossing_direction(peak_positions_high,
                                                                                            len(peak_positions_high),
                                                                                            len(roi) // 2)
                current_index += 1
            if selected_parameters[9]:
                resulting_parameter_maps[i, current_index:current_index + 3] = toolbox.crossing_direction(
                    peak_positions_high, len(peak_positions_high), len(roi) // 2)
                current_index += 3

    return resulting_parameter_maps


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
    optional.add_argument('--with_mask',
                          action='store_true',
                          help='Use mask to try to remove some of the background')
    optional.add_argument('--mask_threshold',
                          type=int,
                          default=10,
                          help='Value for filtering background noise when calculating masks.'
                               'Higher values might result in the removal of some of the gray matter in the mask'
                               'but will remove the background more effectively.')
    optional.add_argument('--with_smoothing',
                          action='store_true',
                          help='Apply smoothing for individual roi curves for noisy images.'
                               'Recommended for measurements with less than 5 degree between each image.')
    # optional.add_argument('--no_centroid_calculation',
    #                      action='store_true',
    #                      help='Disable centroid calculation. Not recommended!')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    # Computational parameters
    compute = parser.add_argument_group('computational arguments')
    compute.add_argument('-r', '--roisize',
                         type=int,
                         help='Roisize which will be used to calculate images.'
                              'This effectively equals downsampling and will speed up the calculation.'
                              'Images will be upscaled later to retain the same size as the input images',
                         default=1)
    compute.add_argument('--num_procs',
                         type=int,
                         help='Number of processes used',
                         default=min(16, multiprocessing.cpu_count()))
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
    # Return generated parser
    return parser


if __name__ == '__main__':
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
    toolbox.CPU_COUNT = args['num_procs']

    print(
        'SLI Feature Generator:\n'
        'Number of threads: ' + str(toolbox.CPU_COUNT) + '\n\n'
                                                         'Chosen feature maps:\n' +
        'Direction maps: ' + str(DIRECTION) + '\n' +
        'Peak maps: ' + str(PEAKS) + '\n' +
        'Peakprominence map: ' + str(PEAKPROMINENCE) + '\n' +
        'Peakwidth map: ' + str(PEAKWIDTH) + '\n' +
        'Peakdistance map: ' + str(PEAKDISTANCE) + '\n' +
        'Optional maps: ' + str(OPTIONAL) + '\n'
    )

    paths = args['input']
    if not type(paths) is list:
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    for path in paths:
        folder = os.path.dirname(path)
        filename_without_extension = os.path.splitext(os.path.basename(path))[0]
        full_pipeline(path, args['output'] + '/' + filename_without_extension, args['roisize'], args['with_mask'],
                      args['with_smoothing'], args['mask_threshold'])
