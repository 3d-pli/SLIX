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

    roi_set: list = []
    if ROISIZE > 1:
        for i in range(0, x, ROISIZE):
            for j in range(0, y, ROISIZE):
                roi: numpy.memmap = data[i:i+ROISIZE, j:j+ROISIZE, :]
                average_per_dimension: numpy.memmap = numpy.average(numpy.average(roi, axis=1), axis=0).flatten()
                average_per_dimension = numpy.concatenate((average_per_dimension[-z//2:], average_per_dimension, average_per_dimension[:z//2]))
                roi_set.append(average_per_dimension)
    else:
        roi_set: list = data.reshape((x * y, z))
            
    return numpy.array(roi_set, dtype="float32"), x, y, z

def gen_minmax_image(PATH, ROISIZE, filename_max, filename_min):
    roi_img, x, y, z = get_roi_set(PATH, ROISIZE)
    
    max_image = numpy.empty(roi_img.shape[0])
    min_image = numpy.empty(roi_img.shape[0])
    for i, roi in enumerate(roi_img):
        max_image[i] = numpy.max(roi)
        min_image[i] = numpy.min(roi)
    max_image_reshaped = max_image.reshape(numpy.ceil(x/ROISIZE).astype('int'),roi_img.shape[0]//numpy.ceil(x/ROISIZE).astype('int'))
    min_image_reshaped = min_image.reshape(numpy.ceil(x/ROISIZE).astype('int'),roi_img.shape[0]//numpy.ceil(x/ROISIZE).astype('int'))
    Image.fromarray(max_image_reshaped).resize((y, x)).save(filename_max)
    Image.fromarray(min_image_reshaped).resize((y, x)).save(filename_min)

def gen_avg_image(PATH, ROISIZE, filename_avg):
    roi_img, x, y, z = get_roi_set(PATH, ROISIZE)
    
    avg_image = numpy.empty(roi_img.shape[0])
    for i, roi in enumerate(roi_img):
        avg_image[i] = numpy.mean(roi)
    avg_image_reshaped = avg_image.reshape(numpy.ceil(x/ROISIZE).astype('int'),roi_img.shape[0]//numpy.ceil(x/ROISIZE).astype('int'))
    Image.fromarray(avg_image_reshaped).resize((y, x)).save(filename_avg)

if __name__ == '__main__':
    INPUT_NAME = ''
    OUTPUT_NAME_MAX = ''
    OUTPUT_NAME_MIN = ''
    OUTPUT_NAME_AVG = ''
    ROISIZE = 3

    gen_avg_image(INPUT_NAME, ROISIZE, OUTPUT_NAME_AVG)
    gen_minmax_image(INPUT_NAME, ROISIZE, OUTPUT_NAME_MAX, OUTPUT_NAME_MIN)
