import SLIX
from SLIX.CPU import _toolbox as ntoolbox
import numpy


class TestNumbaToolboxCPU:
    def test_peak_cleanup(self):
        test_one_peak = numpy.array([0, 1, 0, 0]).reshape((1, 4))
        result = ntoolbox._peaks(test_one_peak)
        assert numpy.all(test_one_peak == result)

        test_two_peak = numpy.array([0, 1, 1, 0]).reshape((1, 4))
        result = ntoolbox._peaks(test_two_peak)
        assert numpy.all(numpy.array([0, 1, 0, 0]) == result)

        test_three_peak = numpy.array([0, 1, 1, 1, 0]).reshape((1, 5))
        result = ntoolbox._peaks(test_three_peak)
        assert numpy.all(numpy.array([0, 0, 1, 0, 0]) == result)

        test_double_three_peak = numpy.array([0, 1, 1, 1, 0, 1, 1, 1, 0]).reshape((1, 9))
        result = ntoolbox._peaks(test_double_three_peak)
        assert numpy.all(numpy.array([0, 0, 1, 0, 0, 0, 1, 0, 0]) == result)

    def test_prominence(self):
        test_array = numpy.array([0, 0.1, 0.2, 0.4, 0.8, 1, 0.5, 0.7, 0.9, 0.5, 0.3, 0.95, 0]).reshape((1, 13))
        peaks = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]).reshape((1, 13))

        expected_prominence = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0.4, 0, 0, 0.65, 0]).reshape((1, 13))
        toolbox_prominence = ntoolbox._prominence(test_array, peaks)

        assert numpy.all(numpy.isclose(expected_prominence, toolbox_prominence))

    def test_peakwidth(self):
        test_array = numpy.array([0, 0.1, 0.2, 0.5, 0.8, 1, 0.77, 0.7, 0.66, 0.5, 0.74, 0.98, 0.74]).reshape((1, 13))
        peaks = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]).reshape((1, 13))
        prominence = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.48, 0]).reshape((1, 13))
        expected_width = numpy.array([0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 2, 0]).reshape((1, 13))

        toolbox_width = ntoolbox._peakwidth(test_array, peaks, prominence, 0.5)
        assert numpy.all(toolbox_width == expected_width)

    def test_peakdistance(self):
        test_arr = numpy.array(([False, False, True, False, False, False, False, True, False] + [False] * 15))\
                        .reshape((1, 24))
        expected_distance = 75

        toolbox_distance = ntoolbox._peakdistance(test_arr, numpy.zeros(test_arr.shape, dtype=float), numpy.array([2]))
        assert toolbox_distance[0, 2] == expected_distance
        assert toolbox_distance[0, 7] == 360 - expected_distance

    def test_direction(self):
        # Test for one peak
        one_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
                            .reshape((1, 24))
        expected_direction = numpy.array([45, ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR])
        toolbox_direction = ntoolbox._direction(one_peak_arr, numpy.zeros(one_peak_arr.shape), numpy.array([1]), 3, 0)
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for one direction with 180°+-35° distance
        two_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])\
                            .reshape((1, 24))
        expected_direction = numpy.array([135, ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR])
        toolbox_direction = ntoolbox._direction(two_peak_arr, numpy.zeros(two_peak_arr.shape), numpy.array([2]), 3, 0)
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for (invalid) two directions with 180°+-35° distance
        four_peak_arr = numpy.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])\
                            .reshape((1, 24))
        expected_direction = numpy.array([ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR])
        toolbox_direction = ntoolbox._direction(four_peak_arr, numpy.zeros(four_peak_arr.shape), numpy.array([4]), 3, 0)
        assert numpy.all(expected_direction == toolbox_direction)

    def test_centroid_correction_bases(self):
        # simple test case: one distinct peak
        test_array = numpy.array([0] * 9 + [1] + [0] * 14).reshape((1, 24))
        test_high_peaks = SLIX.toolbox.peaks(test_array, use_gpu=False)
        test_reverse_peaks = SLIX.toolbox.peaks(-test_array, use_gpu=False)

        left_bases, right_bases = ntoolbox._centroid_correction_bases(test_array, test_high_peaks, test_reverse_peaks)
        assert numpy.sum(left_bases) == 1
        assert numpy.sum(right_bases) == 1

        # simple test case: one distinct peak
        test_array = numpy.array([0] * 8 + [0.95, 1, 0.5] + [0] * 13).reshape((1, 24))
        test_high_peaks = SLIX.toolbox.peaks(test_array, use_gpu=False)
        test_reverse_peaks = SLIX.toolbox.peaks(-test_array, use_gpu=False)

        left_bases, right_bases = ntoolbox._centroid_correction_bases(test_array, test_high_peaks, test_reverse_peaks)
        assert numpy.sum(left_bases) == 2
        assert numpy.sum(right_bases) == 1

        # simple test case: centroid is between two measurements
        test_array = numpy.array([0] * 8 + [1, 1] + [0] * 14).reshape((1, 24))
        test_high_peaks = SLIX.toolbox.peaks(test_array, use_gpu=False)
        test_reverse_peaks = SLIX.toolbox.peaks(-test_array, use_gpu=False)

        left_bases, right_bases = ntoolbox._centroid_correction_bases(test_array, test_high_peaks, test_reverse_peaks)
        assert numpy.sum(left_bases) == 1
        assert numpy.sum(right_bases) == 2

        # more complicated test case: wide peak plateau
        test_array = numpy.array([0] * 8 + [1, 1, 1] + [0] * 13).reshape((1, 24))
        test_high_peaks = SLIX.toolbox.peaks(test_array, use_gpu=False)
        test_reverse_peaks = SLIX.toolbox.peaks(-test_array, use_gpu=False)

        left_bases, right_bases = ntoolbox._centroid_correction_bases(test_array, test_high_peaks, test_reverse_peaks)
        assert numpy.sum(left_bases) == 2
        assert numpy.sum(right_bases) == 2

    def test_centroid(self):
        image = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((1, 24))
        left = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((1, 24))
        right = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((1, 24))
        peak = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((1, 24))

        result_centroid = ntoolbox._centroid(image, peak, left, right)
        assert numpy.sum(result_centroid) == 0
