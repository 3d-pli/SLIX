#!/usr/bin/env python3

import argparse
import os

import numpy
import tqdm
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

import SLIX.toolbox as toolbox


def full_pipeline(filepath, output_filename, with_smoothing=True, with_plots=False):
    """
    Example pipeline how a single line profile can be processed using SLIX. Here, significant values like the number of
    peaks and their peak positions are determined using the SLIX library. Resulting features are written into a
    non-binary text file allowing further studies.

    Args:
        filepath: Input path for line profile.
        output_filename: Output file pattern for generated features. A file with 'output_filename'.txt will be
         generated. If 'with_plots' is True 'output_filename'.png will also be generated
        with_smoothing: Apply the Savitzky-Golay filter with a polynomial order of 2 and window length of 45 to
        the given line profile.
        with_plots: Create two plots where the line profile as well as all detected peak positions are shown.
        If the parameter 'with_smoothing' is used both the smoothed and original line profile is shown.

    Returns: None
    """
    line_profile = numpy.fromfile(filepath, dtype=numpy.float, sep='\n')
    # When line profiles are smoothed
    if with_smoothing:
        line_profile_expanded = numpy.concatenate((line_profile, line_profile, line_profile))
        line_profile_smoothed = savgol_filter(line_profile_expanded, 45, 2)

        first_measurement = len(line_profile_smoothed) // 3 // 2
        last_measurement = len(line_profile_smoothed) - first_measurement
        line_profile_smoothed = line_profile_smoothed[first_measurement:last_measurement]

        line_profile_smoothed = numpy.expand_dims(line_profile_smoothed, axis=0)
        num_measurements = first_measurement
    else:
        line_profile_expanded = numpy.concatenate(
            (line_profile[-len(line_profile) // 2:], line_profile, line_profile[:len(line_profile) // 2]))
        line_profile_smoothed = numpy.expand_dims(line_profile_expanded, axis=0)
        num_measurements = len(line_profile) // 4

    peaks = toolbox.all_peaks(line_profile_smoothed.flatten())
    max_array = line_profile_smoothed.max()
    min_array = line_profile_smoothed.min()

    if with_plots:
        if with_smoothing:
            plt.plot(line_profile[2 * num_measurements:-2 * num_measurements])
            plt.plot(line_profile_smoothed.flatten()[num_measurements:-2*num_measurements])
        else:
            plt.plot(line_profile_smoothed.flatten()[num_measurements:-2*num_measurements])
        acc_peaks = toolbox.accurate_peak_positions(peaks, line_profile_smoothed.flatten(), centroid_calculation=False)
        acc_peaks = acc_peaks - num_measurements
        plt.plot(acc_peaks, line_profile_smoothed.flatten()[num_measurements:-2*num_measurements][acc_peaks], 'o',
                 label='Peak position')

    acc_peaks = toolbox.accurate_peak_positions(peaks, line_profile_smoothed.flatten(), centroid_calculation=True)
    directions = toolbox.crossing_direction(acc_peaks, len(line_profile))
    prominence = toolbox.prominence(peaks, line_profile_expanded)

    if with_plots:
        plt.plot((acc_peaks-num_measurements), [
            line_profile_smoothed.flatten()[num_measurements:-2*num_measurements][int(numpy.floor(peak))] + (
                        peak - int(peak)) * (
                    line_profile_smoothed.flatten()[num_measurements:-num_measurements][int(numpy.ceil(peak))] -
                    line_profile_smoothed.flatten()[num_measurements:-num_measurements][
                        int(numpy.floor(peak))]) for peak in (acc_peaks-num_measurements)], 'x', label='Corrected peak position')
        plt.plot([24, 24], [line_profile.min(), line_profile.max()])
        plt.legend()
        plt.savefig(output_filename + '.png', dpi=600)
        plt.close()

    # Convert peak calculation to angle for simplified comparison with 3D-PLI measurements
    # acc_peaks = (acc_peaks * 180.0 / num_measurements) % 360
    acc_peaks = acc_peaks - 2 * num_measurements

    # Generate output parameters for file
    output = 'Max: ' + str(max_array) + '\nMin: ' + str(min_array) + '\nNum_Peaks: ' + str(len(acc_peaks)) + \
             '\nPeak_Pos: [' + " ".join(str(x) for x in acc_peaks) + ']\nDirections: [' + \
             " ".join(str(x) for x in directions) + ']\nProminence: ' + str(prominence)
    with open(output_filename + '.txt', 'w') as f:
        f.write(output)
        f.flush()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Creation of feature set from scattering image.')
    parser.add_argument('-i', '--input',
                        nargs='*', help='Input path / files.',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Output folder',
                        required=True)
    parser.add_argument('--smoothing',
                        required=False,
                        action='store_true',
                        default=False)
    parser.add_argument('--with_plots',
                        action='store_true')
    parser.add_argument('--target_peak_height',
                        type=float,
                        required=False,
                        help='EXPERIENCED USERS ONLY: Replaces target peak height for peak evaluation.',
                        default=toolbox.TARGET_PEAK_HEIGHT)
    arguments = parser.parse_args()
    args = vars(arguments)

    paths = args['input']
    if not isinstance(paths, list):
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    toolbox.TARGET_PEAK_HEIGHT = args['target_peak_height']

    for i in tqdm.tqdm(range(len(paths))):
        folder = os.path.dirname(paths[i])
        filename_without_extension = os.path.splitext(os.path.basename(paths[i]))[0]
        full_pipeline(paths[i], args['output'] + '/' + filename_without_extension, args['smoothing'],
                      args['with_plots'])
