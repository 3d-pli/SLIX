#!/usr/bin/env python3

import Library.ScatterPlotToolbox as toolbox
import numpy
import argparse
import os
from matplotlib import pyplot as plt
from PIL import Image

import pandas
from scipy.signal import peak_widths, savgol_filter

def full_pipeline(PATH, NAME, with_smoothing=True):
    print(PATH)
    roiset = numpy.fromfile(PATH, dtype=numpy.float, sep='\n')
    if with_smoothing:
        roiset = numpy.concatenate((roiset, roiset, roiset))
        roiset_rolled = savgol_filter(roiset, 45, 2)
        z_begin = len(roiset_rolled)//3//2
        z_end = len(roiset_rolled) - z_begin
        roiset_rolled = roiset_rolled[z_begin:z_end]
        roiset_rolled = numpy.expand_dims(roiset_rolled, axis=0)
        z = z_begin
    else:
        roiset = numpy.concatenate((roiset[-len(roiset)//2:], roiset, roiset[:len(roiset)//2]))
        roiset_rolled = numpy.expand_dims(roiset, axis=0)
        z = len(roiset)//4

    #print("Roi finished")
    max_array = toolbox.max_array_from_roiset(roiset_rolled)
    #print("Max image finished")
    min_array = toolbox.min_array_from_roiset(roiset_rolled)
    #print("Min image finished")
    peak_array = toolbox.peak_array_from_roiset(roiset_rolled)
    if with_smoothing:
        plt.plot(roiset[2*z:-2*z+3])
        plt.plot(roiset_rolled.flatten()[z:-z+3])
    else:
        plt.plot(roiset[z:-z+3])
    peaks = toolbox.get_peaks_from_roi(roiset_rolled.flatten(), centroid_calculation=False)
    peaks = peaks - z
    plt.plot(peaks, roiset_rolled.flatten()[z:-z+3][peaks], 'o')
    peaks = toolbox.get_peaks_from_roi(roiset_rolled.flatten(), centroid_calculation=True)
    peaks = peaks - z
    plt.plot(peaks, [roiset_rolled.flatten()[z:-z+3][int(numpy.floor(peak))] + (peak - int(peak)) * (roiset_rolled.flatten()[z:-z+3][int(numpy.ceil(peak))] - roiset_rolled.flatten()[z:-z+3][int(numpy.floor(peak))]) for peak in peaks], 'x')
    plt.savefig(NAME+'.png', dpi=600)
    plt.close()
    # Convert peak calculation to angle for comparison with delft data
    peaks = 180 / z * peaks 
    
    # Generate output parameters for file
    output = 'Max: ' + str(max_array) + '\nMin: ' + str(min_array) + '\nNum_Peaks: ' + str(peak_array) + '\nPeak_Pos: ' + str(peaks)
    with open(NAME+'.txt', 'w') as f:
        f.write(output)
        f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Creation of feature set from scattering image.')
    parser.add_argument('-i', '--input', nargs='*', help=('Input path / files.'), required=True)
    parser.add_argument('-o', '--output', help=('Output folder'), required=True)
    parser.add_argument('--smoothing', required=False, action='store_true', default=False)
    arguments = parser.parse_args()
    args = vars(arguments)
    
    paths = args['input']
    if not type(paths) is list:
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    for path in paths:
        folder = os.path.dirname(path)
        filename_without_extension = os.path.splitext(os.path.basename(path))[0]
        full_pipeline(path, args['output'] + '/' + filename_without_extension, args['smoothing'])
