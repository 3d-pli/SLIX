#!/usr/bin/env python3

import argparse
import os

import numpy
import tqdm
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

import Library.ScatterPlotToolbox as toolbox

def full_pipeline(PATH, NAME, with_smoothing=True, with_plots=False):
    roiset = numpy.fromfile(PATH, dtype=numpy.float, sep='\n')
    if with_smoothing:
        roiset = numpy.concatenate((roiset, roiset, roiset))
        roiset_rolled = savgol_filter(roiset, 45, 2)
        z_begin = len(roiset_rolled) // 3 // 2
        z_end = len(roiset_rolled) - z_begin
        roiset_rolled = roiset_rolled[z_begin:z_end]
        roiset_rolled = numpy.expand_dims(roiset_rolled, axis=0)
        z = z_begin
    else:
        roiset = numpy.concatenate((roiset[-len(roiset) // 2:], roiset, roiset[:len(roiset) // 2]))
        roiset_rolled = numpy.expand_dims(roiset, axis=0)
        z = len(roiset) // 4

    max_array = toolbox.max_array_from_roiset(roiset_rolled)
    min_array = toolbox.min_array_from_roiset(roiset_rolled)
    peak_array = toolbox.peak_array_from_roiset(roiset_rolled)
    if with_plots:
        if with_smoothing:
            plt.plot(roiset[2 * z:-2 * z + 3])
            plt.plot(roiset_rolled.flatten()[z:-z + 3])
        else:
            plt.plot(roiset[z:-z + 3])
        peaks = toolbox.get_peaks_from_roi(roiset_rolled.flatten(), centroid_calculation=False)
        peaks = peaks - z
        plt.plot(peaks, roiset_rolled.flatten()[z:-z + 3][peaks], 'o')
    peaks = toolbox.get_peaks_from_roi(roiset_rolled.flatten(), centroid_calculation=True)
    peaks = peaks - z
    if with_plots:
        plt.plot(peaks, [roiset_rolled.flatten()[z:-z + 3][int(numpy.floor(peak))] + (peak - int(peak)) * (
                roiset_rolled.flatten()[z:-z + 3][int(numpy.ceil(peak))] - roiset_rolled.flatten()[z:-z + 3][
            int(numpy.floor(peak))]) for peak in peaks], 'x')
        plt.savefig(NAME + '.png', dpi=600)
        plt.close()
    # Convert peak calculation to angle
    peaks = (peaks * 180.0 / z) % 360

    # Generate output parameters for file
    output = 'Max: ' + str(max_array) + '\nMin: ' + str(min_array) + '\nNum_Peaks: [' + " ".join(
        str(x) for x in peak_array) + ']\nPeak_Pos: [' + " ".join(str(x) for x in peaks) + ']'
    with open(NAME + '.txt', 'w') as f:
        f.write(output)
        f.flush()


if __name__ == '__main__':
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
    if not type(paths) is list:
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    toolbox.TARGET_PEAK_HEIGHT = args['target_peak_height']

    for i in tqdm.tqdm(range(len(paths))):
        folder = os.path.dirname(paths[i])
        filename_without_extension = os.path.splitext(os.path.basename(paths[i]))[0]
        full_pipeline(paths[i], args['output'] + '/' + filename_without_extension, args['smoothing'],
                      args['with_plots'])
