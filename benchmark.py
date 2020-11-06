#!/usr/bin/env python
import time
from SLIX import io, toolbox
import cupy

image = io.imread('Vervet1818_s0512_60um_SLI_090_Stack_1day.nii')
use_gpu = True

times = []

for i in range(10):
    print(i)
    start = time.time()
    if use_gpu:
        image = cupy.array(image)
    sig_peaks = toolbox.significant_peaks(image, use_gpu=use_gpu, return_numpy=False)
    centroid = toolbox.centroid_correction(image, sig_peaks, use_gpu=use_gpu, return_numpy=False)
    toolbox.direction(sig_peaks, centroid, use_gpu=use_gpu)
    toolbox.num_peaks(image, use_gpu=use_gpu)
    toolbox.peak_distance(sig_peaks, centroid, use_gpu=use_gpu)
    toolbox.mean_peak_distance(sig_peaks, centroid, use_gpu=use_gpu)
    toolbox.peak_prominence(image, sig_peaks, use_gpu=use_gpu)
    toolbox.mean_peak_prominence(image, sig_peaks, use_gpu=use_gpu)
    toolbox.peak_width(image, sig_peaks, use_gpu=use_gpu)
    toolbox.mean_peak_width(image, sig_peaks, use_gpu=use_gpu)
    total_time = time.time() - start
    times.append(total_time)

import numpy

print(numpy.array(times).mean(), numpy.array(times).std())

