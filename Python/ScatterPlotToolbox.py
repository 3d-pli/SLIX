import nibabel
import numpy
import peakutils
import tifffile

BACKGROUND_COLOR = 0

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
        data = numpy.moveaxis(data, 0, -1)

    return data

def zaxis_roiset(IMAGE, ROISIZE):
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

    roi_set = []
    # Roisize == 1 is exactly the same as the original image
    if ROISIZE > 1:
        # TODO: Find optimization for this loop
        for i in range(0, x, ROISIZE):
            for j in range(0, y, ROISIZE):
                # Create average of selected ROI and append two halfs to the front and back
                roi = IMAGE[i:i+ROISIZE, j:j+ROISIZE, :]
                average_per_dimension = numpy.average(numpy.average(roi, axis=1), axis=0).flatten()
                average_per_dimension = numpy.concatenate((average_per_dimension[-z//2:], average_per_dimension, average_per_dimension[:z//2]))
                roi_set.append(average_per_dimension)
    else:
        # Flatten two axis together as some algorithms expect a 2D input array
        roi_set = IMAGE.reshape((x * y, z))
            
    return numpy.array(roi_set, dtype="float32")

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
    image_reshaped = image.reshape((numpy.ceil(x/ROISIZE).astype('int'),image.shape[0]//numpy.ceil(x/ROISIZE).astype('int')))
    return image_reshaped

def peak_array_from_roiset(roiset):    
    """
    Generate array visualizing the number of peaks present in each pixel of a given roiset.
    
    Arguments:
        roiset {numpy.array} -- Roiset generated with the zaxis_roiset method
    
    Returns:
        numpy.array -- 2D-Array with the amount of peaks in each pixel
    """
    peak_array = numpy.empty(roiset.shape[0])
    z = roiset.shape[1]//2
    
    for i, roi in enumerate(roiset):
        # Generate peaks
        peaks = peakutils.indexes(roi, thres=0.3, min_dist=3)
        # Only consider peaks which are in bounds
        peaks = peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)]
        # Filter double peak
        if numpy.all(numpy.isin(peaks, [z//2, len(roi)-z//2])):
            peaks = peaks[1:]
        peak_array[i] = len(peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)])
    return peak_array


def non_crossing_direction_array_from_roiset(roiset):
    peak_array = numpy.empty(roiset.shape[0])
    z = roiset.shape[1]//2
    
    for i, roi in enumerate(roiset):
        # Generate peaks
        peaks = peakutils.indexes(roi, thres=0.3, min_dist=3)
        # Only consider peaks which are in bounds
        peaks = peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)]
        # Filter double peak
        if numpy.all(numpy.isin(peaks, [z//2, len(roi)-z//2])):
            peaks = peaks[1:]

        amount_of_peaks = len(peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)])
        # Scale peaks correctly for direction
        peaks = (peaks - z//2) * 360 / z
        # Change behaviour based on amount of peaks (steep, crossing, ...)
        if amount_of_peaks == 1:
            peak_array[i] = (270 - peaks[0])%180
        elif amount_of_peaks == 2:
            pos = (270 - ((peaks[1]+peaks[0])//2))%180
            peak_array[i] = pos
        else:
            peak_array[i] = BACKGROUND_COLOR
    return peak_array

def crossing_direction_array_from_roiset(roiset):
    peak_array = numpy.empty((roiset.shape[0], 2))
    z = roiset.shape[1]//2
    
    for i, roi in enumerate(roiset):
        # Generate peaks
        peaks = peakutils.indexes(roi, thres=0.3, min_dist=3)
        # Only consider peaks which are in bounds
        peaks = peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)]
        # Filter double peak
        if numpy.all(numpy.isin(peaks, [z//2, len(roi)-z//2])):
            peaks = peaks[1:]

        amount_of_peaks = len(peaks[(peaks >= z//2) & (peaks <= len(roi)-z//2)])
        # Scale peaks correctly for direction
        peaks = (peaks - z//2) * 360 / z
        # Change behaviour based on amount of peaks (steep, crossing, ...)
        if amount_of_peaks == 1:
            peak_array[i, :] = (270 - peaks[0])%180
        elif amount_of_peaks == 2:
            pos = (270 - ((peaks[1]+peaks[0])//2))%180
            peak_array[i, :] = pos
        elif amount_of_peaks == 4:
            # TODO: Check crossing fibers
            if(numpy.abs((peaks[3] - peaks[1]) - 180) < 35):
                peak_array[i, 1] = (270 - ((peaks[3]+peaks[1])//2))%180
            else:
                peak_array[i, 1] = BACKGROUND_COLOR
            if(numpy.abs((peaks[2] - peaks[0]) - 180) < 35):
                peak_array[i, 0] = (270 - ((peaks[2]+peaks[0])//2))%180   
            else:
                peak_array[i, 0] = BACKGROUND_COLOR     
        else:
            peak_array[i] = BACKGROUND_COLOR
    return peak_array

def max_array_from_roiset(roiset):
    #TODO: I'm totally sure that that code can be optimized
    max_array = numpy.empty(roiset.shape[0])
    for i, roi in enumerate(roiset):
        max_array[i] = numpy.max(roi)
    return max_array

def min_array_from_roiset(roiset):
    #TODO: I'm totally sure that that code can be optimized
    min_array = numpy.empty(roiset.shape[0])
    for i, roi in enumerate(roiset):
        min_array[i] = numpy.min(roi)
    return min_array

def avg_array_from_roiset(roiset):
    #TODO: I'm totally sure that that code can be optimized
    avg_array = numpy.empty(roiset.shape[0])
    for i, roi in enumerate(roiset):
        avg_array[i] = numpy.mean(roi)
    return avg_array