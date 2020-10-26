import SLIX
from SLIX.SLIX_CPU import _toolbox as ntoolbox
import numpy


class TestNumbaToolboxCPU:
    def test_peak_cleanup(self):
        test_one_peak = numpy.array([0, 1, 0, 0]).reshape((1, 4))
        result = ntoolbox._peak_cleanup(test_one_peak)
        assert numpy.all(test_one_peak == result)

        test_two_peak = numpy.array([0, 1, 1, 0]).reshape((1, 4))
        result = ntoolbox._peak_cleanup(test_two_peak)
        assert numpy.all(numpy.array([0, 1, 0, 0]) == result)

        test_three_peak = numpy.array([0, 1, 1, 1, 0]).reshape((1, 5))
        result = ntoolbox._peak_cleanup(test_three_peak)
        assert numpy.all(numpy.array([0, 0, 1, 0, 0]) == result)

        test_double_three_peak = numpy.array([0, 1, 1, 1, 0, 1, 1, 1, 0]).reshape((1, 9))
        result = ntoolbox._peak_cleanup(test_double_three_peak)
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
        test_arr = numpy.array(([False, False, True, False, False, False, False, True, False] + [False] * 15))
        test_arr = test_arr.reshape((1, 24))
        expected_distance = 75

        toolbox_distance = ntoolbox._peakdistance(test_arr, numpy.zeros(test_arr.shape, dtype=float), numpy.array([2]))
        assert toolbox_distance[0, 2] == expected_distance
        assert toolbox_distance[0, 7] == 360 - expected_distance


    def test_direction(self):
        pass

    def test_centroid_correction_bases(self):
        pass

    def test_centroid(self):
        pass
