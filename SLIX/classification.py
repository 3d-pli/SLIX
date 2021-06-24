import numpy


def crossing_mask(high_prominence_peaks, max_image, flat_mask):
    crossing = (high_prominence_peaks == 4) | (high_prominence_peaks == 6)
    mean_flat_signal = numpy.average(max_image[flat_mask])
    std_flat_signal = numpy.std(max_image[flat_mask])
    crossing[(max_image < mean_flat_signal - std_flat_signal) |
             (max_image > mean_flat_signal + std_flat_signal)] = False
    return crossing


def inclinated_mask(high_prominence_peaks, peakdistance, max_image, flat_mask):
    resulting_labeling = numpy.zeros(high_prominence_peaks.shape, dtype=numpy.uint8)
    resulting_labeling[high_prominence_peaks == 1] = 1
    mean_flat_signal = numpy.average(max_image[flat_mask])
    resulting_labeling[(high_prominence_peaks == 2) & (peakdistance < 145) & (max_image > mean_flat_signal)] = 2
    return resulting_labeling


def flat_mask(high_prominence_peaks, low_prominence_peaks, peakdistance):
    return (high_prominence_peaks == 2) & (low_prominence_peaks < 2) & (peakdistance > 145) & (peakdistance < 215)