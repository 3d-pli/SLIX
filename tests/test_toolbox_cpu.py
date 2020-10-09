from SLIX.SLIX_CPU import toolbox
from SLIX.SLIX_CPU import _toolbox
import numpy


class TestToolbox:
    def test_all_peaks(self):
        # Create an absolute simple peak array
        arr = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 1)
        arr = arr.reshape((1, 1, 11))
        real_peaks = arr == 1
        toolbox_peaks = toolbox.peaks(arr)
        assert numpy.all(toolbox_peaks == real_peaks)

        # Test one single peak
        arr = numpy.array(([0, 1, 0, 0, 0, 0]), dtype=bool)
        arr = arr.reshape((1, 1, 6))
        real_peaks = arr == 1
        toolbox_peaks = toolbox.peaks(arr)
        assert numpy.all(toolbox_peaks == real_peaks)

        # Test one single peak
        arr = numpy.array(([0, 1, 1, 0, 0, 0]), dtype=bool)
        arr = arr.reshape((1, 1, 6))
        real_peaks = arr == 1
        real_peaks[0, 0, 2] = False
        toolbox_peaks = toolbox.peaks(arr)
        assert numpy.all(toolbox_peaks == real_peaks)

        # Test one single peak
        arr = numpy.array(([0, 1, 1, 1, 0, 0]), dtype=bool)
        arr = arr.reshape((1, 1, 6))
        real_peaks = arr == 1
        real_peaks[0, 0, 1] = False
        real_peaks[0, 0, 3] = False
        toolbox_peaks = toolbox.peaks(arr)
        assert numpy.all(toolbox_peaks == real_peaks)

        arr = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] * 1)
        arr = arr.reshape((1, 1, 11))
        real_peaks = arr == 1
        toolbox_peaks = toolbox.peaks(arr)
        assert numpy.all(toolbox_peaks == real_peaks)

    def test_num_peaks(self):
        # Create an absolute simple peak array
        test_arr = numpy.array(([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=bool)
        test_arr = test_arr.reshape((1, 1, 20))

        real_peaks = test_arr == True
        toolbox_peaks = toolbox.num_peaks(real_peaks)
        expected_value = numpy.count_nonzero(real_peaks[0, 0, :])
        assert numpy.all(toolbox_peaks == expected_value)

    def test_peak_prominence(self):
        # Create an absolute simple peak array
        arr = numpy.array([0, 1, 0, 0.07, 0, 1, 0, 0.07, 0, 1, 0] * 1)
        arr = arr.reshape((1, 1, 11))
        # Test if high and low prominence separation is working as intended
        high_peaks = arr == 1
        low_peaks = arr == 0.07

        toolbox_peaks = toolbox.peaks(arr)
        toolbox_prominence = toolbox.peak_prominence(arr, toolbox_peaks)
        toolbox_high_peaks = toolbox_peaks.copy()
        toolbox_high_peaks[toolbox_prominence < toolbox.TARGET_PROMINENCE] = False
        toolbox_low_peaks = toolbox_peaks.copy()
        toolbox_low_peaks[toolbox_prominence >= toolbox.TARGET_PROMINENCE] = False
        assert numpy.all(high_peaks == toolbox_high_peaks)
        assert numpy.all(low_peaks == toolbox_low_peaks)

    def test_peakdistance(self):
        # Test one peak
        test_arr = numpy.array(([True, False, False] + [False] * 21))
        test_arr = test_arr.reshape((1, 1, 24))
        expected_distance = 360

        toolbox_peaks = toolbox.peaks(test_arr)
        toolbox_distance = toolbox.peak_distance(toolbox_peaks, numpy.zeros(toolbox_peaks.shape, dtype=float))
        toolbox_distance_reduced = numpy.sum(toolbox_distance, axis=-1)
        assert numpy.all(toolbox_distance_reduced == expected_distance)

        # Test two peaks
        test_arr = numpy.array(([False, False, True, False, False, False, False, True, False] + [False] * 15))
        test_arr = test_arr.reshape((1, 1, 24))
        expected_distance = 75

        toolbox_peaks = toolbox.peaks(test_arr)
        toolbox_distance = toolbox.peak_distance(toolbox_peaks, numpy.zeros(toolbox_peaks.shape, dtype=float))
        assert toolbox_distance[0, 0, 2] == expected_distance
        assert toolbox_distance[0, 0, 7] == 360 - expected_distance

        # Test four peaks
        test_arr = numpy.array(([False, False, True, False, False, False, False, True, False,
                                 False, False, True, False, False, False, True, False, False] + [False] * 6))
        test_arr = test_arr.reshape((1, 1, 24))
        expected_distance_1 = 135
        expected_distance_2 = 120

        toolbox_peaks = toolbox.peaks(test_arr)
        toolbox_distance = toolbox.peak_distance(toolbox_peaks, numpy.zeros(toolbox_peaks.shape, dtype=float))
        assert toolbox_distance[0, 0, 2] == expected_distance_1
        assert toolbox_distance[0, 0, 11] == 360 - expected_distance_1

        assert toolbox_distance[0, 0, 7] == expected_distance_2
        assert toolbox_distance[0, 0, 15] == 360 - expected_distance_2

    def test_peakwidth(self):
        test_arr = numpy.array([0, 0.5, 1, 0.5, 0] + [0] * 19)
        test_arr = test_arr.reshape((1, 1, 24))
        expected_width = 30

        toolbox_peaks = toolbox.peaks(test_arr)
        toolbox_width = toolbox.peak_width(test_arr, toolbox_peaks)
        assert toolbox_width[0, 0, 2] == expected_width
        assert numpy.sum(toolbox_width) == expected_width

    def test_direction(self):
        # Test for one peak
        one_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        one_peak_arr = one_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([45, _toolbox.BACKGROUND_COLOR, _toolbox.BACKGROUND_COLOR])
        peaks = toolbox.peaks(one_peak_arr)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(one_peak_arr.shape))
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for one direction with 180°+-35° distance
        two_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        two_peak_arr = two_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([135, _toolbox.BACKGROUND_COLOR, _toolbox.BACKGROUND_COLOR])
        peaks = toolbox.peaks(two_peak_arr)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape))
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for two directions with 180°+-35° distance
        four_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        four_peak_arr = four_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([135, 60, _toolbox.BACKGROUND_COLOR])
        peaks = toolbox.peaks(four_peak_arr)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape))
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for three directions with 180°+-35° distance
        six_peak_arr = numpy.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
        six_peak_arr = six_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([135, 105, 60])
        peaks = toolbox.peaks(six_peak_arr)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape))
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for angle outside of 180°+-35° distance
        error_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        error_arr = error_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([82.5, _toolbox.BACKGROUND_COLOR, _toolbox.BACKGROUND_COLOR])
        peaks = toolbox.peaks(error_arr)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape))
        assert numpy.all(expected_direction == toolbox_direction)

        error_arr = numpy.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        error_arr = error_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([_toolbox.BACKGROUND_COLOR, _toolbox.BACKGROUND_COLOR, 60])
        peaks = toolbox.peaks(error_arr)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape))
        assert numpy.all(expected_direction == toolbox_direction)

    def test_centroid_correction(self):
        # simple test case: one distinct peak
        test_array = numpy.array([0] * 9 + [1] + [0] * 14)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks)
        assert numpy.isclose(toolbox_centroid[0, 0, 9], 0)

        # simple test case: one distinct peak
        test_array = numpy.array([0] * 8 + [0.5, 1, 0.5] + [0] * 13)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks)
        assert numpy.isclose(toolbox_centroid[0, 0, 9], 0)

        # simple test case: centroid is between two measurements
        test_array = numpy.array([0] * 8 + [1, 1] + [0] * 14)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array)
        print(test_high_peaks)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks)
        assert numpy.isclose(toolbox_centroid[0, 0, 8], 0.5)

        # more complicated test case: wide peak plateau
        test_array = numpy.array([0] * 8 + [1, 1, 1] + [0] * 13)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks)
        assert numpy.isclose(toolbox_centroid[0, 0, 9], 0)

    def test_normalize(self):
        test_array = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=numpy.float)
        # Normalization kind == 0 -> Scale to 0..1
        expected_array = test_array / test_array.max()
        normalized_array = toolbox.normalize(test_array)
        assert numpy.all(numpy.isclose(expected_array, normalized_array))
        # Normalization kind == 1 -> Divide by mean value of array
        expected_array = test_array / test_array.mean()
        normalized_array = toolbox.normalize(test_array, kind_of_normalization=1)
        assert numpy.all(numpy.isclose(expected_array, normalized_array))
