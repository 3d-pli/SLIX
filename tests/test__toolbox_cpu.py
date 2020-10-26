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
