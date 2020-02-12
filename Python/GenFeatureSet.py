#!/usr/bin/env python3

import Library.ScatterPlotToolbox as toolbox
import numpy
import argparse
import os

from matplotlib import pyplot as plt
from PIL import Image

def full_pipeline(PATH, NAME, ROISIZE, APPLY_MASK = True, MASK_THRESHOLD = 10):
    image = toolbox.read_image(PATH)
    print(image.shape)
    path_name = NAME
    roiset = toolbox.zaxis_roiset(image, ROISIZE)
    if APPLY_MASK:
        mask = toolbox.create_background_mask(roiset, MASK_THRESHOLD)
        roiset[mask, :] = 0
    print("Roi finished")
    max_array = toolbox.max_array_from_roiset(roiset)
    max_image = toolbox.reshape_array_to_image(max_array, image.shape[0], ROISIZE)
    Image.fromarray(max_image).resize(image.shape[:2][::-1]).save(path_name+'_max.tiff')
    print("Max image written")
    min_array = toolbox.min_array_from_roiset(roiset)
    min_image = toolbox.reshape_array_to_image(min_array, image.shape[0], ROISIZE)
    Image.fromarray(min_image).resize(image.shape[:2][::-1]).save(path_name+'_min.tiff')
    print("Min image written")
    peak_array = toolbox.peak_array_from_roiset(roiset)
    peak_image = toolbox.reshape_array_to_image(peak_array, image.shape[0], ROISIZE)
    Image.fromarray(peak_image).resize(image.shape[:2][::-1]).save(path_name+'_peaks.tiff')
    print("Peak image written")
    direction_array = toolbox.non_crossing_direction_array_from_roiset(roiset)
    direction_image = toolbox.reshape_array_to_image(direction_array, image.shape[0], ROISIZE)
    Image.fromarray(direction_image).resize(image.shape[:2][::-1]).save(path_name+'_non_crossing_dir.tiff')
    print("Non Crossing Direction written")
    peakwidth_array = toolbox.peakwidth_array_from_roiset(roiset)
    peakwidth_image = toolbox.reshape_array_to_image(peakwidth_array, image.shape[0], ROISIZE)
    Image.fromarray(peakwidth_image).resize(image.shape[:2][::-1]).save(path_name+'_peakwidth.tiff')
    print("Peakwidth written")
    crossing_array = numpy.where(peak_array > 2, 255, 0).astype('uint8')
    crossing_image = toolbox.reshape_array_to_image(crossing_array, image.shape[0], ROISIZE)
    Image.fromarray(crossing_image).resize(image.shape[:2][::-1]).save(path_name+'_crossing.tiff')
    print("4 Peak written")
    direction_array = toolbox.crossing_direction_array_from_roiset(roiset)
    dir_1 = toolbox.reshape_array_to_image(direction_array[:, 0], image.shape[0], ROISIZE)
    dir_2 = toolbox.reshape_array_to_image(direction_array[:, 1], image.shape[0], ROISIZE)
    Image.fromarray(dir_1).resize(image.shape[:2][::-1]).save(path_name+'_dir_1.tiff')
    Image.fromarray(dir_2).resize(image.shape[:2][::-1]).save(path_name+'_dir_2.tiff')
    print("Crossing Directions written")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Creation of feature set from scattering image.')
    parser.add_argument('-i', '--input', help=('Input path / files.'))
    parser.add_argument('-o', '--output', help=('Output folder'))
    parser.add_argument('-r', '--roisize', type=int, help=('Roisize which will be used to calculate images. Default = 1'), default=1)
    parser.add_argument('--with_mask', action='store_true')
    parser.add_argument('--mask_threshold', type=int, default=10, help=('Value for filtering background noise when calculating masks. Lower values might retain more background noise but will also affect the brain structure less.'))
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
        full_pipeline(path, args['output'] + '/' + filename_without_extension, args['roisize'], args['with_mask'], args['mask_threshold'])
