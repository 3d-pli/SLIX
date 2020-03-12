import nibabel
import numpy
import multiprocessing
import peakutils
import tifffile
import sys
from read_roi import read_roi_zip
from scipy.signal import peak_widths, savgol_filter, find_peaks, peak_prominences

import pymp
from pymp import shared

BACKGROUND_COLOR = -1
CPU_COUNT = min(12, multiprocessing.cpu_count())

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
    mask = numpy.all(IMAGE < threshold, axis=-1)
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

def normalize_roi(roi):
    if not numpy.all(roi == 0):
        if roi.max() == roi.min():
            nroi = numpy.ones(roi.shape)
        else:
            nroi = (roi - roi.min()) / (roi.max() - roi.min())
            #nroi = roi / numpy.mean(roi)
        return nroi
    else:
        return roi

def get_peaks_from_roi(roi, low_prominence=0.1, high_prominence=None, cut_edges=True, centroid_calculation=True):
    z = roi.shape[0] // 2
    #print(z)
    roi = normalize_roi(roi)
    # Generate peaks
    maxima, _ = find_peaks(roi, prominence=(low_prominence, high_prominence))
    # Only consider peaks which are in bounds
    if cut_edges:
        maxima = maxima[(maxima >= z//2) & (maxima <= len(roi)-z//2)]
        # Filter double peak
        if numpy.all(numpy.isin([z//2, len(roi)-z//2], maxima)):
            maxima = maxima[1:]

    # Correct position of maxima
    # Reverse curve
    if centroid_calculation:
        reverse_roi = -1 * roi
        minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
        centroid_maxima = maxima.copy().astype('float32')

        for i in range(maxima.shape[0]):
            peak = maxima[i]
            #distance = numpy.min(numpy.abs(minima - peak))
            distance = 2
            lpos = peak - distance
            rpos = peak + distance
            centroid = numpy.sum(numpy.arange(lpos, rpos+1, 1) * roi[lpos:rpos+1]) / numpy.sum(roi[lpos:rpos+1])
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

def peak_array_from_roiset(roiset, low_prominence=0.1, high_prominence=None, cut_edges=True, centroid_calculation=True):    
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

"""def peakwidth_array_from_roiset(roiset, low_prominence=0.1, high_prominence=None, cut_edges=True):
    peak_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    z = roiset.shape[1]//2

    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = numpy.array(get_peaks_from_roi(roi, low_prominence, high_prominence, cut_edges), dtype='int64')

            if len(peaks) > 0:
                widths = peak_widths(roi, peaks, rel_height=0.5)
                peak_array[i] = numpy.mean(widths[0])
            else:
                peak_array[i] = 0

    return peak_array"""

def peakprominence_array_from_roiset(roiset, low_prominence=0.1, high_prominence=None, cut_edges=True, centroid_calculation=True):
    peak_arr = pymp.shared.array((roiset.shape[0]), dtype='float32')
    z = roiset.shape[1]//2

    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = normalize_roi(roiset[i])
            peaks = get_peaks_from_roi(roi, low_prominence, high_prominence, cut_edges, centroid_calculation)
            peak_arr[i] = 0 if len(peaks) == 0 else numpy.mean(peak_prominences(roi, peaks)[0])
    return peak_arr

def non_crossing_direction_array_from_roiset(roiset, low_prominence=0.1, high_prominence=None, cut_edges=True, centroid_calculation=True):
    peak_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
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
                peak_array[i] = (270 - peaks[0])%180
            elif amount_of_peaks == 2:
                pos = (270 - ((peaks[1]+peaks[0])/2.0))%180
                peak_array[i] = pos
            else:
                peak_array[i] = BACKGROUND_COLOR
    return peak_array

def crossing_direction_array_from_roiset(roiset, low_prominence=0.1, high_prominence=None, cut_edges=True, centroid_calculation=True):
    peak_array = pymp.shared.array((roiset.shape[0], 2), dtype='float32')
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
                peak_array[i] = (270 - peaks[0])%180
            elif amount_of_peaks == 2:
                pos = (270 - ((peaks[1]+peaks[0])/2.0))%180
                peak_array[i] = pos
            elif amount_of_peaks == 3:
                if(numpy.abs((peaks[0] - peaks[2]) - 180) < 35):
                    peak_array[i, 0] = (270 - ((peaks[2]+peaks[0])/2.0))%180
                    peak_array[i, 1] = (270 - peaks[1])%180
                elif(numpy.abs((peaks[1] - peaks[0]) - 180) < 35):
                    peak_array[i, 0] = (270 - ((peaks[1]+peaks[0])/2.0))%180
                    peak_array[i, 1] = (270 - peaks[2])%180 
                elif(numpy.abs((peaks[1] - peaks[2]) - 180) < 35):
                    peak_array[i, 0] = (270 - ((peaks[1]+peaks[2])/2.0))%180
                    peak_array[i, 1] = (270 - peaks[0])%180
                else:
                    peak_array[i] = BACKGROUND_COLOR 
            elif amount_of_peaks == 4:
                if(numpy.abs((peaks[3] - peaks[1]) - 180) < 35):
                    peak_array[i, 1] = (270 - ((peaks[3]+peaks[1])/2.0))%180
                else:
                    peak_array[i, 1] = BACKGROUND_COLOR
                if(numpy.abs((peaks[2] - peaks[0]) - 180) < 35):
                    peak_array[i, 0] = (270 - ((peaks[2]+peaks[0])/2.0))%180   
                else:
                    peak_array[i] = BACKGROUND_COLOR    
            else:
                peak_array[i] = BACKGROUND_COLOR
    return peak_array

def max_array_from_roiset(roiset):
    #TODO: I'm totally sure that that code can be optimized
    max_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            max_array[i] = numpy.max(roi)
    return max_array

def min_array_from_roiset(roiset):
    #TODO: I'm totally sure that that code can be optimized
    min_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            min_array[i] = numpy.min(roi)
    return min_array

def avg_array_from_roiset(roiset):
    #TODO: I'm totally sure that that code can be optimized
    avg_array = pymp.shared.array((roiset.shape[0]), dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            avg_array[i] = numpy.mean(roi)
    return avg_array