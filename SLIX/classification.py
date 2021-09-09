import numpy


def crossing_mask(high_prominence_peaks, max_image):
    crossing = (high_prominence_peaks == 4) | (high_prominence_peaks == 6)
    crossing = crossing.astype(numpy.uint8)
    mean_flat_signal = numpy.average(max_image)
    crossing[(max_image < mean_flat_signal)] = 0
    crossing[high_prominence_peaks == 4] = 1
    crossing[high_prominence_peaks == 6] = 2
    return crossing


def inclinated_mask(high_prominence_peaks, peakdistance,
                    max_image, flat_mask):
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
    inclinated_areas[high_prominence_peaks == 1] = 3

    return inclinated_areas


def flat_mask(high_prominence_peaks, low_prominence_peaks, peakdistance):
    return (high_prominence_peaks == 2) & (low_prominence_peaks < 2) & \
           (peakdistance > 145) & (peakdistance < 215)
