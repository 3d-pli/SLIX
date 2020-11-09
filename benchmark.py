#!/usr/bin/env python
from SLIX import io, toolbox
import time
import numpy
if toolbox.gpu_available:
    import cupy
    use_gpu = True
else:
    use_gpu = False
import urllib.request
import tqdm.auto

# Download example image for visualization
input_file_name = './Vervet1818_s0512_60um_SLI_090_Stack_1day.nii'
url = 'https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/' \
      'hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/' \
      'Vervet1818_s0512_60um_SLI_090_Stack_1day.nii'
urllib.request.urlretrieve(url, input_file_name)
image = io.imread(input_file_name)

# Main benchmark loop
times = []
for i in tqdm.tqdm(range(10)):
    start = time.time()
    if use_gpu:
        image = cupy.array(image)
    sig_peaks = toolbox.significant_peaks(image, use_gpu=use_gpu,
                                          return_numpy=False)
    centroid = toolbox.centroid_correction(image, sig_peaks, use_gpu=use_gpu,
                                           return_numpy=False)
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

# Show mean time and standard deviation
average_time = numpy.array(times).mean()
std_time = numpy.array(times).std()
print('Used GPU:', toolbox.gpu_available)
print('Average time is:', average_time, '+-', std_time)

