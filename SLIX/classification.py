import numpy


def full_mask(high_prominence_peaks, low_prominence_peaks, peakdistance,
              max_image):
    """
    This method classifies the resulting parameter maps of an SLI measurement to
    identify the areas of the image. Based on the other methods in this module,
    a mask combining the results of the other methods is created. The resulting
    mask will therefore contain values to separate flat, crossing and inclined
    fibers.

    ----------

    The resulting mask is a binary mask with the following values:
    0: The area neither flat, crossing or inclined.
    1: The area is a flat fiber.
    2: The area is contains two crossing fibers.
    3: The area is contains three crossing fibers.
    4: The area is lightly inclined.
    5: The area is inclined.
    6: The area is strongly inclined.

    Args:
        high_prominence_peaks: numpy.ndarray containing the number of peaks
                               with a high prominence.

        low_prominence_peaks: numpy.ndarray containing the number of peaks
                              with a low prominence.

        peakdistance: numpy.ndarray containing the distance between the peaks.

        max_image: numpy.ndarray containing the maximum signal of the image.

    Returns:
        numpy.ndarray containing the binary mask.

    """
    crossing = crossing_mask(high_prominence_peaks, max_image)
    flat = flat_mask(high_prominence_peaks, low_prominence_peaks, peakdistance)
    inclined = inclinated_mask(high_prominence_peaks, peakdistance, max_image, flat)

    return_mask = flat.astype(numpy.uint8)
    return_mask[crossing == 1] = 2
    return_mask[crossing == 2] = 3
    return_mask[inclined == 2] = 4
    return_mask[inclined == 3] = 5
    return_mask[inclined == 4] = 6

    return return_mask


def crossing_mask(high_prominence_peaks, max_image):
    """
    This method classifies the resulting parameter maps of an SLI measurement to
    identify the areas of the image where the underlying fiber structure
    is probably a crossing one. To do so, the method uses the following
    criteria:
    1. The maximum signal during the measurement is above the mean signal
       of the maximum image.
    2. The number of peaks is either four (two crossing fibers) or six (three
       crossing fibers).

    --------------

    The resulting mask is a binary mask with the following values:
    0: The area is not a crossing one.
    1: The area is a crossing one with two crossing fibers.
    2: The area is a crossing one with three crossing fibers.

    Args:
        high_prominence_peaks: numpy.ndarray containing the number of peaks
                               with a high prominence.

        max_image: numpy.ndarray containing the maximum signal of the image.

    Returns:
        numpy.ndarray containing the binary mask.
    """
    crossing = (high_prominence_peaks == 4) | (high_prominence_peaks == 6)
    crossing = crossing.astype(numpy.uint8)
    mean_flat_signal = numpy.average(max_image)
    crossing[(max_image < mean_flat_signal)] = 0
    crossing[high_prominence_peaks == 4] = 1
    crossing[high_prominence_peaks == 6] = 2
    return crossing


def inclinated_mask(high_prominence_peaks, peakdistance,
                    max_image, flat_mask):
    """
    This method classifies the resulting parameter maps of an SLI measurement to
    identify the areas of the image where the underlying fiber structure
    is probably an inclined one. To do so, the method uses the following
    criteria:

    Flat fibers:
    1. The maximum signal during the measurement is above the mean signal
       of the maximum image.
    2. The number of peaks is two (one flat fiber)

    Inclined fibers:
    Three different scenarios are possible:
    1. Two peaks are present and the peak distance is between 120° and
       150° (lightly inclined fiber)
    2. Two peaks are present and the peak distance is below 120°
       (inclined fiber)
    3. One single peak is present (steep fiber)

    ----------

    The resulting mask is a binary mask with the following values:
    0: The area is neither a flat nor an inclined one.
    1: The area is a flat one.
    2: The area is a lightly inclined one.
    3: The area is an inclined one.
    4: The area is a steep one.

    Args:
        high_prominence_peaks: numpy.ndarray containing the number of peaks
                               with a high prominence.

        peakdistance: Mean distance between the prominent peaks in
                      high_prominence_peaks.

        max_image: numpy.ndarray containing the maximum signal of the image.

        flat_mask: numpy.ndarray containing the binary mask of the flat areas.

    Returns:
        numpy.ndarray containing the binary mask.
    """
    inclinated_areas = numpy.zeros(high_prominence_peaks.shape,
                                   dtype=numpy.uint8)
    mean_flat_signal = numpy.average(max_image[flat_mask > 0])
    two_peak_mask = (high_prominence_peaks == 2) & \
                    (max_image > mean_flat_signal)

    inclinated_areas[flat_mask > 0] = 1
    inclinated_areas[two_peak_mask &
                     (peakdistance > 120) &
                     (peakdistance < 150)] = 2
    inclinated_areas[two_peak_mask & (peakdistance < 120)] = 3
    inclinated_areas[high_prominence_peaks == 1] = 4

    return inclinated_areas


def flat_mask(high_prominence_peaks, low_prominence_peaks, peakdistance):
    """
    This method classifies the resulting parameter maps of an SLI measurement to
    identify the areas of the image where the underlying fiber structure
    is probably a flat one. To do so, the method uses the following
    criteria:
    1. Two prominent peaks are present.
    2. The peak distance is between 145° and 215°. A peak distance of 180° is
       expected for a completely flat fiber, but small deviations for example
       through the sampling steps of the measurement are possible.
    3. No more than two low prominent peaks are present. Completely flat
       fibers generally have a very stable signal and therefore a low
       amount of low prominent peaks.

    ----------

    The resulting mask is a binary mask with the following values:
    0: The area is not a flat fiber.
    1: The area is a flat fiber.

    Args:
        high_prominence_peaks: numpy.ndarray containing the number of peaks

        low_prominence_peaks: numpy.ndarray containing the number of peaks
                              which are below the threshold of prominence
                              needed to be considered as a prominent peak.

        peakdistance: Mean distance between the prominent peaks in
                      high_prominence_peaks.

    Returns:
        numpy.ndarray containing the binary mask.

    """
    return (high_prominence_peaks == 2) & (low_prominence_peaks < 2) & \
           (peakdistance > 145) & (peakdistance < 215)
