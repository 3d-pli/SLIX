#!/usr/bin/env python
# coding: utf-8

import sys
import numpy
from PIL import Image
from sklearn import cluster
import h5py
import nibabel
from matplotlib import pyplot as plt
import matplotlib
import tifffile
from scipy.signal import find_peaks
import peakutils

def get_roi_set(PATH, ROISIZE):
    # Load NIfTI dataset
    if PATH.endswith('.nii'):
        data: numpy.memmap = nibabel.load(PATH).get_fdata()
    else:
        data: numpy.memmap = tifffile.imread(PATH)
        data: numpy.memmap = numpy.moveaxis(data, 0, -1)

    # Create region of interest element
    x: int = data.shape[0]
    y: int = data.shape[1]
    z: int = data.shape[2]

    print("Shape: %d x %d" %(x, y))

    roi_set: list = []
    if ROISIZE > 1:
        for i in range(0, x, ROISIZE):
            for j in range(0, y, ROISIZE):
                roi: numpy.memmap = data[i:i+ROISIZE, j:j+ROISIZE, :]
                average_per_dimension: numpy.memmap = numpy.average(numpy.average(roi, axis=1), axis=0).flatten()
                average_per_dimension = numpy.concatenate((average_per_dimension[-z//2:], average_per_dimension, average_per_dimension[:z//2]))
                roi_set.append(average_per_dimension)
    else:
        roi_set: list = data.reshape((x * y, 24))
            
    return numpy.array(roi_set, dtype="float32"), x, y, z


def gen_peak_image(PATH, ROISIZE, filename):
    roi_img, x, y, z = get_roi_set(PATH, ROISIZE)
    
    peak_image = numpy.empty(roi_img.shape[0])
    for i, roi in enumerate(roi_img):
        # Generate peaks
        peaks = peakutils.indexes(roi, thres=0.3, min_dist=3)
        # Only consider peaks which are in bounds
        peaks = peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)]
        # Filter double peak
        if z//2 in peaks and len(roi)-z//2 in peaks:
            peaks = peaks[1:]
        peak_image[i] = len(peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)])
    # Rescale image
    peak_image_reshaped = peak_image.reshape((numpy.ceil(x/ROISIZE).astype('int'),roi_img.shape[0]//numpy.ceil(x/ROISIZE).astype('int')))
    Image.fromarray(peak_image_reshaped).resize((y, x)).save(filename)


def gen_direction_fom(PATH, ROISIZE, filename):
    roi_img, x, y, z = get_roi_set(PATH, ROISIZE)
    
    peak_image = numpy.empty(roi_img.shape[0])
    for i, roi in enumerate(roi_img):
        peaks = peakutils.indexes(roi, thres=0.3, min_dist=3)
        # Filter peaks outside of boundries
        peaks = peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)]
        if z//2 in peaks and len(roi)-z//2 in peaks:
            peaks = peaks[1:]
        amount_of_peaks = len(peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)])
        # Scale peaks correctly for direction
        peaks = (peaks - z//2) * 360 / z
        # Change behaviour based on amount of peaks (steep, crossing, ...)
        if amount_of_peaks == 1:
            peak_image[i] = (270 - peaks[0])%180
        elif amount_of_peaks == 2:
            pos = (270 - ((peaks[1]+peaks[0])//2))%180
            peak_image[i] = pos
        else:
            peak_image[i] = 0
    
    peak_image_reshaped = peak_image.reshape((numpy.ceil(x/ROISIZE).astype('int'),roi_img.shape[0]//numpy.ceil(x/ROISIZE).astype('int')))
    Image.fromarray(peak_image_reshaped).resize((y, x)).save(filename)

if __name__ == '__main__':
    INPUT_NAME = ''
    OUTPUT_NAME_NUM_PEAKS = ''
    OUTPUT_NAME_DIR_MAP = ''
    ROISIZE = 3
    
    gen_peak_image(INPUT_NAME, ROISIZE, OUTPUT_NAME_NUM_PEAKS)
    gen_direction_fom(INPUT_NAME, ROISIZE, OUTPUT_NAME_DIR_MAP)
