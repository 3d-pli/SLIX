from Library.ScatterPlotToolbox import *


class TestScatterPlotToolBox:
    def test_all_peaks(self):
        # Create an absolute simple peak array
        arr = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        real_peaks = numpy.argwhere(arr == 1).flatten()
        toolbox_peaks = all_peaks(arr, cut_edges=False)
        assert numpy.all(toolbox_peaks == real_peaks)

        # cut_edges should remove the peak position 1
        toolbox_peaks = all_peaks(arr)
        assert numpy.all(toolbox_peaks == real_peaks[1:])

    def test_num_peaks_image(self):
        pass

    def test_peak_positions(self):
        # Create an absolute simple peak array
        arr = numpy.array([0, 1, 0, 0.07, 0, 1, 0, 0.07, 0, 1, 0])
        # Test if high and low prominence separation is working as intended
        high_peaks = numpy.argwhere(arr == 1).flatten()
        low_peaks = numpy.argwhere(arr == 0.07).flatten()

        toolbox_peaks = all_peaks(arr, cut_edges=False)
        toolbox_high_peaks = peak_positions(toolbox_peaks, arr, centroid_calculation=False)
        toolbox_low_peaks = peak_positions(toolbox_peaks, arr, low_prominence=0, high_prominence=TARGET_PROMINENCE,
                                           centroid_calculation=False)
        assert numpy.all(high_peaks == toolbox_high_peaks)
        assert numpy.all(low_peaks == toolbox_low_peaks)

    def test_peakdistance(self):
        pass

    def test_peakdistance_image(self):
        pass

    def test_prominence(self):
        # Create an absolute simple peak array
        arr = numpy.array([0, 1, 0, 0.1, 0, 1, 0, 0.1, 0, 1, 0])
        comparison = normalize(arr, kind_of_normalizaion=1)

        toolbox_peaks = all_peaks(arr, cut_edges=False)
        toolbox_prominence = prominence(toolbox_peaks, arr, len(toolbox_peaks))
        assert numpy.isclose(toolbox_prominence, numpy.mean(comparison[comparison > 0]))

    def test_prominence_image(self):
        pass

    def test_peakwidth(self):
        pass

    def test_peakwidth_image(self):
        pass

    def test_crossing_direction(self):
        # Test for one direction with 180°+-35° distance
        two_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        expected_direction = numpy.array([135, BACKGROUND_COLOR, BACKGROUND_COLOR])
        peaks = all_peaks(two_peak_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, two_peak_arr, centroid_calculation=False)
        toolbox_direction = crossing_direction(high_peaks, len(high_peaks), len(two_peak_arr))
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for two directions with 180°+-35° distance
        four_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        expected_direction = numpy.array([135, 60, BACKGROUND_COLOR])
        peaks = all_peaks(four_peak_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, four_peak_arr, centroid_calculation=False)
        toolbox_direction = crossing_direction(high_peaks, len(high_peaks), len(two_peak_arr))
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for three directions with 180°+-35° distance
        six_peak_arr = numpy.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
        expected_direction = numpy.array([135, 105, 60])
        peaks = all_peaks(six_peak_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, six_peak_arr, centroid_calculation=False)
        toolbox_direction = crossing_direction(high_peaks, len(high_peaks), len(two_peak_arr))
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for angle outside of 180°+-35° distance
        error_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        expected_direction = numpy.array([BACKGROUND_COLOR, BACKGROUND_COLOR, BACKGROUND_COLOR])
        peaks = all_peaks(error_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, error_arr, centroid_calculation=False)
        toolbox_direction = crossing_direction(high_peaks, len(high_peaks), len(error_arr))
        assert numpy.all(expected_direction == toolbox_direction)

        error_arr = numpy.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        expected_direction = numpy.array([BACKGROUND_COLOR, BACKGROUND_COLOR, 60])
        peaks = all_peaks(error_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, error_arr, centroid_calculation=False)
        toolbox_direction = crossing_direction(high_peaks, len(high_peaks), len(error_arr))
        assert numpy.all(expected_direction == toolbox_direction)

    def test_crossing_direction_image(self):
        pass

    def test_non_crossing_direction(self):
        # Test for one peak
        one_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        expected_direction = 45
        peaks = all_peaks(one_peak_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, one_peak_arr, centroid_calculation=False)
        toolbox_direction = non_crossing_direction(high_peaks, len(high_peaks), len(one_peak_arr))
        assert expected_direction == toolbox_direction

        # Test for two peaks
        two_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        expected_direction = 135
        peaks = all_peaks(two_peak_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, two_peak_arr, centroid_calculation=False)
        toolbox_direction = non_crossing_direction(high_peaks, len(high_peaks), len(two_peak_arr))
        assert expected_direction == toolbox_direction

        # Test for four peaks
        four_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        expected_direction = BACKGROUND_COLOR
        peaks = all_peaks(four_peak_arr, cut_edges=False)
        high_peaks = peak_positions(peaks, four_peak_arr, centroid_calculation=False)
        toolbox_direction = non_crossing_direction(high_peaks, len(high_peaks), len(two_peak_arr))
        assert expected_direction == toolbox_direction

    def test_non_crossing_direction_image(self):
        pass

    def test_centroid_correction(self):
        pass

    def test_read_image(self):
        pass

    def test_create_background_mask(self):
        test_array = (numpy.random.random(10000) * 256).astype('int')
        expected_results = test_array < 10
        toolbox_mask = create_background_mask(test_array[..., numpy.newaxis])
        assert numpy.all(expected_results == toolbox_mask)

    def test_normalize(self):
        test_array = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=numpy.float)
        # Normalization kind == 0 -> Scale to 0..1
        expected_array = test_array / test_array.max()
        normalized_array = normalize(test_array)
        assert numpy.all(numpy.isclose(expected_array, normalized_array))
        # Normalization kind == 1 -> Divide by mean value of array
        expected_array = test_array / test_array.mean()
        normalized_array = normalize(test_array, kind_of_normalizaion=1)
        assert numpy.all(numpy.isclose(expected_array, normalized_array))

    def test_reshape_array_to_image(self):
        pass
