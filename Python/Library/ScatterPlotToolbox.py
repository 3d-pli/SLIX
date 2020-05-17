import nibabel
import numpy
import multiprocessing
import tifffile
import sys
from read_roi import read_roi_zip
from scipy.signal import peak_widths, savgol_filter, find_peaks, peak_prominences

import pymp
pymp.config.nested = True
from pymp import shared

from matplotlib import pyplot as plt

BACKGROUND_COLOR = -1
CPU_COUNT = min(16, multiprocessing.cpu_count())
MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 2

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.94
TARGET_PROMINENCE = 0.08

def read_image(FILEPATH):
    """
    Reads iamge file and returns it.
    
    Arguments:
        FILEPATH {str} -- Path to image

    Returns:
        numpy.array -- Image with shape [x, y, z] where [x, y] is the size of a single image and z specifies the number of images
    """
    # Load NIfTI dataset
    if FILEPATH.endswith('.nii'):
        data = nibabel.load(FILEPATH).get_fdata()
        data = numpy.squeeze(numpy.swapaxes(data, 0, 1))
    else:
        data = tifffile.imread(FILEPATH)
        data = numpy.squeeze(numpy.moveaxis(data, 0, -1))

    return data

def create_background_mask(IMAGE, threshold=10):
    """Creates a background mask based on given threshhold. As all background pixels are near zero when looking through the z-axis plot
    this method should remove most of the background allowing better approximations using the available features. It is advised to use this function.
    
    Arguments:
        IMAGE {numpy.array} -- 2D/3D-image containing the z-axis in the last dimension
    
    Keyword Arguments:
        threshold {int} -- Threshhold for mask creation (default: {10})
    
    Returns:
        numpy.array -- 1D/2D-image which masks the background as True and foreground as False
    """
    mask = numpy.max(IMAGE < threshold, axis=-1)
    return mask

def zaxis_from_imagej_roiset(IMAGE, PATH_TO_ROISET, extend=True):
    rois = read_roi_zip(PATH_TO_ROISET)
    x, y, z = IMAGE.shape[0], IMAGE.shape[1], IMAGE.shape[2]
    
    if extend:
        roi_set = pymp.shared.array((len(rois.items()), 2*z), dtype='float32')
    else:
        roi_set = pymp.shared.array((len(rois.items()), z), dtype='float32')

    roi_values = list(dict(rois.items()).values())
    name_set = list(dict(rois.items()).keys())

    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(1, len(rois)):
                value = roi_values[i]
                name = value['name']
                width, height = value['width'], value['height']
                roi_type = value['type']
                left, top = value['left'], value['top']
                center = (left + width/2, top + height/2)

                if width == height and roi_type == 'oval':
                    x_indices = numpy.arange(top, top+height+1)
                    y_indices = numpy.arange(left, left+width+1)
                    rectangle_indices = numpy.array(numpy.meshgrid(x_indices, y_indices)).T.reshape(-1, 2)
                    rectangle_indices = rectangle_indices[(rectangle_indices[:, 0] - center[1])**2 + (rectangle_indices[:, 1] - center[0])**2 < width*height]
                    
                    roi = IMAGE[rectangle_indices[:, 0], rectangle_indices[:, 1], :]
                    average_per_dimension = numpy.average(roi, axis=0).flatten()
                    if extend:
                        average_per_dimension = numpy.concatenate((average_per_dimension[-z//2:], average_per_dimension, average_per_dimension[:z//2]))
                    name_set[i] = name
                    roi_set[i] = average_per_dimension
                else:
                    continue

    return roi_set[1:], name_set[1:]

def zaxis_roiset(IMAGE, ROISIZE, extend=True):
    """
    Create z-axis profile of given image by creating a roiset image containing the average value of pixels within the specified ROISIZE. 
    The returned image will have twice the size in the z-axis as the both halfs will be doubled for the peak detection.

    
    Arguments:
        IMAGE {numpy.memmap} -- Image containing multiple images in a z-stack 
        ROISIZE {int} -- Size in pixels which are used to create the region of interest image
    
    Returns:
        numpy.array -- Image with shape [x/ROISIZE, y/ROISIZE, 2*z] containing the average value of the given roiset for each image in z-axis.
    """
    # Get image dimensions
    x = IMAGE.shape[0]
    y = IMAGE.shape[1]
    z = IMAGE.shape[2]
    nx = numpy.ceil(x/ROISIZE).astype('int')
    ny = numpy.ceil(y/ROISIZE).astype('int')
    
    if extend:
        roi_set = pymp.shared.array((nx * ny, 2*z), dtype='float32')
    else:
        roi_set = pymp.shared.array((nx * ny, z), dtype='float32')

    # Roisize == 1 is exactly the same as the original image
    if ROISIZE > 1:
        with pymp.Parallel(CPU_COUNT) as p:
            for i in p.range(0, nx):
                for j in range(0, ny):
                    # Create average of selected ROI and append two halfs to the front and back
                    roi = IMAGE[ROISIZE*i:ROISIZE*i+ROISIZE, ROISIZE*j:ROISIZE*j+ROISIZE, :]
                    average_per_dimension = numpy.average(numpy.average(roi, axis=1), axis=0).flatten()
                    if extend:
                        average_per_dimension = numpy.concatenate((average_per_dimension[-z//2:], average_per_dimension, average_per_dimension[:z//2]))
                    roi_set[i*ny + j] = average_per_dimension
    else:
        with pymp.Parallel(CPU_COUNT) as p:
            for i in p.range(0, nx):
                for j in range(0, ny):
                    # Create average of selected ROI and append two halfs to the front and back
                    roi = IMAGE[i, j, :]
                    if extend:
                        roi = numpy.concatenate((roi[-z//2:], roi, roi[:z//2]))
                    roi_set[i*ny + j] = roi
            
    return roi_set

def smooth_roiset(roiset, range=45, polynom_order=2):
    roiset_c = numpy.concatenate((roiset, roiset, roiset))
    roiset_rolled = savgol_filter(roiset_c, range, polynom_order)
    roiset_rolled = roiset_rolled[len(roiset):-len(roiset)]
    return roiset_rolled

def normalize_roi(roi, kind_of_normalizaion=0):
    """
    Normalize given line profile by using different normalization techniques based on the kind_of_normalization parameter

    0 : Scale line profile to be between 0 and 1
    1 : Divide line profile through it's mean value
    
    Arguments:
        roi {numpy.memmap} -- Line profile of a singular pixel / region of interest
        kind_of_normalization {int} -- Normalization technique which will be used for the calculation 
    
    Returns:
        numpy.array -- Normalized line profile of the given roi parameter
    """
    roi = roi.copy().astype('float32')
    if not numpy.all(roi == 0):
        if roi.max() == roi.min():
            nroi = numpy.ones(roi.shape)
        else:
            if kind_of_normalizaion == 0:
                nroi = (roi - roi.min()) / (roi.max() - roi.min())
            elif kind_of_normalizaion == 1:
                nroi = roi / numpy.mean(roi)
        return nroi
    else:
        return roi

def get_peaks_from_roi(roi, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True, centroid_calculation=True):
    z = roi.shape[0] // 2

    roi = normalize_roi(roi)
    # Generate peaks
    maxima, _ = find_peaks(roi, prominence=(low_prominence, high_prominence))

    # Only consider peaks which are in bounds
    if cut_edges:
        maxima = maxima[(maxima >= z//2) & (maxima <= len(roi)-z//2)]
        # Filter double peak
        if numpy.all(numpy.isin([z//2, len(roi)-z//2], maxima)):
            maxima = maxima[1:]

    if centroid_calculation:
        reverse_roi = -1 * roi
        minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
        centroid_maxima = maxima.copy().astype('float32')

        for i in range(maxima.shape[0]):
            peak = maxima[i]
            target_peak_height = roi[maxima[i]] - roi[maxima].max() * (1 - TARGET_PEAK_HEIGHT)
            minima_distances = peak - minima

            lpos = rpos = peak

            # Check for minima in left and set left position accordingly
            target_distances = (minima_distances <= MAX_DISTANCE_FOR_CENTROID_ESTIMATION) & (minima_distances > 0)
            if target_distances.any():
                lpos = peak - minima_distances[target_distances].min()
            # Look for 80% of the peak height    
            below_target_peak_height = numpy.argwhere(roi[peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION : peak] < target_peak_height)
            if len(below_target_peak_height) > 0:
                below_target_peak_height = below_target_peak_height.max()
                tlpos = peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION + below_target_peak_height
                if tlpos < lpos:
                    lpos = tlpos
            else:
                tlpos = peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION
                if tlpos < lpos:
                    lpos = tlpos

            # Repeat for right bound
            target_distances = (minima_distances >= -MAX_DISTANCE_FOR_CENTROID_ESTIMATION) & (minima_distances < 0)
            if target_distances.any():
                rpos = peak - minima_distances[target_distances].min()
            # Look for 80% of the peak height
            below_target_peak_height = numpy.argwhere(roi[peak : peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION] < target_peak_height)
            if len(below_target_peak_height) > 0:
                below_target_peak_height = below_target_peak_height.min()
                trpos = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION - below_target_peak_height
                if trpos > rpos:
                    rpos = trpos
            else:
                trpos = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION
                if trpos > rpos:
                    rpos = trpos

            # Create sampling to get exact 80% of peak height
            def create_sampling(roi, peak, left, right, target_peak_height, number_of_samples):
                sampling = numpy.interp(numpy.linspace(left-1, right+1, (right-left + 2)*100), numpy.arange(left-1, right+1), roi[left-1:right+1])
                if roi[left] > target_peak_height:
                    left_bound = number_of_samples
                else:
                    try:
                        choices = numpy.argwhere(sampling[:(peak-left+1) * number_of_samples] < target_peak_height)
                        if len(choices) > 0:
                            left_bound = choices.max()
                        else:
                            left_bound = number_of_samples
                    except ValueError:
                        print(roi[left], target_peak_height)
                if roi[right] > target_peak_height:
                    right_bound = len(sampling)-number_of_samples
                else:
                    choices = numpy.argwhere(sampling[(peak-left+1) * number_of_samples:] < target_peak_height)
                    if len(choices) > 0:
                        right_bound = (peak-left+1) * number_of_samples + choices.min()
                    else:
                        right_bound = len(sampling)-number_of_samples
        
                return sampling, left_bound, right_bound

            sampling, lbound, rbound = create_sampling(roi, peak, lpos, rpos, target_peak_height, NUMBER_OF_SAMPLES)
            int_lpos = (lpos-1) + 1/NUMBER_OF_SAMPLES * lbound
            int_rpos = (lpos-1) + 1/NUMBER_OF_SAMPLES * rbound

            # Move at max one entry on the x-coordinate axis to the left or right to prevent too much movement
            centroid = numpy.sum(numpy.linspace(int_lpos, int_rpos+1e-10, len(sampling[lbound:rbound])) * sampling[lbound:rbound]) / numpy.sum(sampling[lbound:rbound])
            if numpy.abs(centroid - peak) > 1:
                centroid = peak + numpy.sign(centroid - peak)  
            centroid_maxima[i] = centroid

        maxima = centroid_maxima

    return maxima


def reshape_array_to_image(image, x, ROISIZE):
    """
    Convert array back to image keeping the lower resolution based on the ROISIZE.
    
    Arguments:
        image {numpy.array} -- Array created by other methods with lower resolution based on ROISIZE
        x {int} -- Size of original image in x-dimension
        ROISIZE {int} -- Size of the ROI used for evaluating the roiset
    
    Returns:
        numpy.array -- Reshaped image based on the input array
    """
    image_reshaped = image.reshape((numpy.ceil(x/ROISIZE).astype('int'), image.shape[0]//numpy.ceil(x/ROISIZE).astype('int')))
    return image_reshaped

def peak_array_from_roiset(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True, centroid_calculation=True):    
    """
    Generate array visualizing the number of peaks present in each pixel of a given roiset.
    
    Arguments:
        roiset {numpy.array} -- Roiset generated with the zaxis_roiset method
    
    Returns:
        numpy.array -- 2D-Array with the amount of peaks in each pixel
    """
    peak_arr = pymp.shared.array((roiset.shape[0]), dtype='uint8')

    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = get_peaks_from_roi(roi, low_prominence, high_prominence, cut_edges, centroid_calculation)
            peak_arr[i] = len(peaks)
    return peak_arr

def peakwidth_array_from_roiset(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    peakwidth_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    z = roiset.shape[1]//2
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = numpy.array(get_peaks_from_roi(roi, low_prominence, high_prominence, cut_edges, centroid_calculation=False), dtype='int64')
            if len(peaks) > 0:
                widths = peak_widths(roi, peaks, rel_height=0.5)
                peakwidth_array[i] = numpy.mean(widths[0])
            else:
                peakwidth_array[i] = 0
            peakwidth_array[i] = (peakwidth_array[i]) * (360.0 / z)
    return peakwidth_array


def peakprominence_array_from_roiset(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    prominence_arr = pymp.shared.array((roiset.shape[0]), dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            peak_roi = normalize_roi(roiset[i])
            prominence_roi = normalize_roi(roiset[i], kind_of_normalizaion=1)
            peaks = get_peaks_from_roi(peak_roi, low_prominence, high_prominence, cut_edges, False)
            prominence_arr[i] = 0 if len(peaks) == 0 else numpy.mean(peak_prominences(prominence_roi, peaks)[0])
    return prominence_arr

def distance_array_from_roiset(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True, centroid_calculation=True):
    dist_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    z = roiset.shape[1]//2
    
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = get_peaks_from_roi(roi, low_prominence, high_prominence, cut_edges, centroid_calculation)
            amount_of_peaks = len(peaks)

            # Scale peaks correctly for direction
            peaks = (peaks - z//2) * (360.0 / z)  

            # Compute peak distance for curves with 1-2 detected peaks
            if amount_of_peaks == 1:    # distance for one peak = 0
                dist_array[i] = 0
            elif amount_of_peaks >= 2 and amount_of_peaks % 2 == 0:
                distances = peaks[1::2] - peaks[::2]
                dist_array[i] = distances.mean()
                if dist_array[i] > 180:
                    dist_array[i] = 360 - dist_array[i]
            else:
                dist_array[i] = BACKGROUND_COLOR
    return dist_array

def non_crossing_direction_array_from_roiset(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True, centroid_calculation=True):
    dir_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    z = roiset.shape[1]//2
    
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = get_peaks_from_roi(roi, low_prominence, high_prominence, cut_edges, centroid_calculation)
            amount_of_peaks = len(peaks)

            # Scale peaks correctly for direction
            peaks = (peaks - z//2) * (360.0 / z)
            # Change behaviour based on amount of peaks (steep, crossing, ...)
            if amount_of_peaks == 1:
                dir_array[i] = (270 - peaks[0])%180
            elif amount_of_peaks == 2:
                pos = (270 - ((peaks[1]+peaks[0])/2.0))%180
                dir_array[i] = pos
            else:
                dir_array[i] = BACKGROUND_COLOR
    return dir_array

def crossing_direction_array_from_roiset(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True, centroid_calculation=True):
    dir_array = pymp.shared.array((roiset.shape[0], 3), dtype='float32')
    z = roiset.shape[1]//2
    
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = get_peaks_from_roi(roi, low_prominence, high_prominence, cut_edges, centroid_calculation)
            amount_of_peaks = len(peaks)

            # Scale peaks correctly for direction
            peaks = (peaks - z//2) * (360.0 / z)
            # Change behaviour based on amount of peaks (steep, crossing, ...)
            if amount_of_peaks == 1:
                dir_array[i, 0] = (270 - peaks[0])%180
                dir_array[i, 1] = BACKGROUND_COLOR
                dir_array[i, 2] = BACKGROUND_COLOR
            elif amount_of_peaks == 2:
                dir_array[i, 0] = (270 - ((peaks[1]+peaks[0])/2.0))%180
                dir_array[i, 1] = BACKGROUND_COLOR
                dir_array[i, 2] = BACKGROUND_COLOR
            elif amount_of_peaks == 4:
                if(numpy.abs((peaks[2] - peaks[0]) - 180) < 35):
                    dir_array[i, 0] = (270 - ((peaks[2]+peaks[0])/2.0))%180   
                else:
                    dir_array[i, 0] = BACKGROUND_COLOR
                if(numpy.abs((peaks[3] - peaks[1]) - 180) < 35):
                    dir_array[i, 1] = (270 - ((peaks[3]+peaks[1])/2.0))%180
                else:
                    dir_array[i, 1] = BACKGROUND_COLOR
                dir_array[i, 2] = BACKGROUND_COLOR    
            elif amount_of_peaks == 6:
                if(numpy.abs((peaks[3] - peaks[0]) - 180) < 35):
                    dir_array[i, 0] = (270 - ((peaks[3]+peaks[0])/2.0))%180   
                else:
                    dir_array[i, 0] = BACKGROUND_COLOR 
                if(numpy.abs((peaks[4] - peaks[1]) - 180) < 35):
                    dir_array[i, 1] = (270 - ((peaks[4]+peaks[1])/2.0))%180
                else:
                    dir_array[i, 1] = BACKGROUND_COLOR
                if(numpy.abs((peaks[5] - peaks[2]) - 180) < 35):
                    dir_array[i, 2] = (270 - ((peaks[5]+peaks[2])/2.0))%180
                else:
                    dir_array[i, 2] = BACKGROUND_COLOR
            else:
                dir_array[i] = BACKGROUND_COLOR

    return dir_array

def max_array_from_roiset(roiset):
    max_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            max_array[i] = numpy.max(roi)
    return max_array

def min_array_from_roiset(roiset):
    min_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            min_array[i] = numpy.min(roi)
    return min_array

def avg_array_from_roiset(roiset):
    avg_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            avg_array[i] = numpy.mean(roi)
    return avg_array
