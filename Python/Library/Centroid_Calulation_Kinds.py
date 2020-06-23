import numpy
from scipy.signal import find_peaks, peak_prominences, peak_widths

MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 2
low_prominence = 0.08
high_prominence = None
centroid_calculation = True
roi = numpy.empty(24, dtype=float)

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.80

"""
Berechnung der Peakposition mit Centroid-Berechnung
Höhe wird mit Berücksichtigt.
Es wird über das Target Intervall integriert
"""
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
                sampling = numpy.empty(number_of_samples * (right - left + 2))
                for i in range(left-1, right+1):
                    for j in range(number_of_samples):
                        sampling[(i - (left-1)) * number_of_samples + j] = roi[i] + (roi[i+1] - roi[i]) * j/number_of_samples   

                if roi[left] > target_peak_height:
                    left_bound = 0
                else:
                    left_bound = numpy.argwhere(sampling[:(peak-left+1) * number_of_samples] < target_peak_height).max()
                if roi[right] > target_peak_height:
                    right_bound = len(sampling)
                else:
                    right_bound = (peak-left+1) * number_of_samples + numpy.argwhere(sampling[(peak-left+1) * number_of_samples:] < target_peak_height).min()

                #sampling[left_bound:right_bound] = sampling[left_bound]

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

"""
Berechnung der Peakpositionen mit Centroid-Berechnung
Höhe der Peaks wird mit berücksichtigt (80% Höhe)
Es wird eine Gerade durch gezogen
"""
if centroid_calculation:
        reverse_roi = -1 * roi
        minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
        centroid_maxima = maxima.copy().astype('float32')

        for i in range(maxima.shape[0]):
            peak = maxima[i]
            target_peak_height = TARGET_PEAK_HEIGHT * roi[maxima[i]]
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
                sampling = numpy.empty(number_of_samples * (right - left + 2))
                for i in range(left-1, right+1):
                    for j in range(number_of_samples):
                        sampling[(i - (left-1)) * number_of_samples + j] = roi[i] + (roi[i+1] - roi[i]) * j/number_of_samples   

                if roi[left] > target_peak_height:
                    left_bound = 0
                else:
                    left_bound = numpy.argwhere(sampling[:(peak-left+1) * number_of_samples] < target_peak_height).max()
                if roi[right] > target_peak_height:
                    right_bound = len(sampling)
                else:
                    right_bound = (peak-left+1) * number_of_samples + numpy.argwhere(sampling[(peak-left+1) * number_of_samples:] < target_peak_height).min()

                sampling[left_bound:right_bound] = sampling[left_bound]

                return sampling, left_bound, right_bound

            sampling, lbound, rbound = create_sampling(roi, peak, lpos, rpos, target_peak_height, number_of_samples)
            int_lpos = (lpos-1) + 1/NUMBER_OF_SAMPLES * lbound
            int_rpos = (lpos-1) + 1/NUMBER_OF_SAMPLES * rbound

            # Move at max one entry on the x-coordinate axis to the left or right to prevent too much movement
            centroid = (int_rpos + int_lpos) / 2
            if numpy.abs(centroid - peak) > 1:
                centroid = peak + numpy.sign(centroid - peak)  
            centroid_maxima[i] = centroid

        maxima = centroid_maxima

"""
Berechnung der Peakpositionen mit Centroid-Berechnung
Höhe der Peaks wird mit berücksichtigt (80% Höhe)
Es wird noch keine Gerade durch gezogen
"""
if centroid_calculation:
        reverse_roi = -1 * roi
        minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
        centroid_maxima = maxima.copy().astype('float32')

        for i in range(maxima.shape[0]):
            peak = maxima[i]
            target_peak_height = 0.8 * roi[maxima[i]]
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
                # TODO: Create linear function to get exact target point for centroid calculation
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
                # TODO: Create linear function to get exact target point for centroid calculation
                trpos = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION - below_target_peak_height
                if trpos > rpos:
                    rpos = trpos
            else:
                trpos = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION
                if trpos > rpos:
                    rpos = trpos

            # Move at max one entry on the x-coordinate axis to the left or right to prevent too much movement
            centroid = numpy.sum(numpy.arange(lpos, rpos+1, 1) * roi[lpos:rpos+1]) / numpy.sum(roi[lpos:rpos+1])
            if numpy.abs(centroid - peak) > 1:
                centroid = peak + numpy.sign(centroid - peak)  
            centroid_maxima[i] = centroid

        maxima = centroid_maxima

"""
Berechnung der Peakpositionen mit Centroid-Berechnung
Schrittweite ist +/- 2 von der Peakposition aus.
"""
# Correct position of maxima
# Reverse curve
if centroid_calculation:
    reverse_roi = -1 * roi
    minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
    centroid_maxima = maxima.copy().astype('float32')

    for i in range(maxima.shape[0]):
        peak = maxima[i]
        distance = MAX_DISTANCE_FOR_CENTROID_ESTIMATION
        lpos = peak - distance
        rpos = peak + distance
        centroid = numpy.sum(numpy.arange(lpos, rpos+1, 1) * roi[lpos:rpos+1]) / numpy.sum(roi[lpos:rpos+1])
        if numpy.abs(centroid - peak) > 1:
            centroid = peak + numpy.sign(centroid - peak)
        centroid_maxima[i] = centroid

    maxima = centroid_maxima

"""
Berechnung der Peakpositionen mit Centroid-Berechnung
Schrittweite ist von Minimum zu Minimum 
"""

# Correct position of maxima
# Reverse curve
if centroid_calculation:
    reverse_roi = -1 * roi
    minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
    centroid_maxima = maxima.copy().astype('float32')

    for i in range(maxima.shape[0]):
        peak = maxima[i]
        distance = numpy.min(numpy.abs(minima - peak))
        lpos = peak - distance
        rpos = peak + distance
        centroid = numpy.sum(numpy.arange(lpos, rpos+1, 1) * roi[lpos:rpos+1]) / numpy.sum(roi[lpos:rpos+1])
        if numpy.abs(centroid - peak) > 1:
            centroid = peak + numpy.sign(centroid - peak)
        centroid_maxima[i] = centroid

    maxima = centroid_maxima


"""
Berechnung mit Centroid
Suche nach Minima auf beiden Seiten. Wenn kein Minima vorhanden, dann gehe maximale Schrittweite
"""

if centroid_calculation:
    reverse_roi = -1 * roi
    minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
    centroid_maxima = maxima.copy().astype('float32')

    for i in range(maxima.shape[0]):
        peak = maxima[i]
        left_side_peaks = minima[minima < peak]
        if len(left_side_peaks) > 0:
            distance = numpy.min(numpy.abs(peak - left_side_peaks))
            lpos = peak - distance
        else:
            lpos = peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION

        right_side_peaks = minima[minima > peak]
        if len(right_side_peaks) > 0:
            distance = numpy.min(numpy.abs(peak - right_side_peaks))
            lpos = peak + distance
        else:
            lpos = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION
        centroid = numpy.sum(numpy.arange(lpos, rpos+1, 1) * roi[lpos:rpos+1]) / numpy.sum(roi[lpos:rpos+1])
        if numpy.abs(centroid - peak) > 1:
            centroid = peak + numpy.sign(centroid - peak)
        centroid_maxima[i] = centroid

    maxima = centroid_maxima