import SLIX
from SLIX.SLIX_GPU import _toolbox as ntoolbox
import cupy


class TestNumbaToolboxGPU:
    pass
    """def test_peak_cleanup(self):
        test_one_peak = cupy.array([0, 1, 0, 0]).reshape((4, 1))
        result = ntoolbox._peak_cleanup(test_one_peak)
        assert cupy.all(test_one_peak == result)

        test_two_peak = cupy.array([0, 1, 1, 0]).reshape((4, 1))
        result = ntoolbox._peak_cleanup(test_two_peak)
        assert cupy.all(cupy.array([0, 1, 0, 0]) == result)

        test_three_peak = cupy.array([0, 1, 1, 1, 0]).reshape((5, 1))
        result = ntoolbox._peak_cleanup(test_three_peak)
        assert cupy.all(cupy.array([0, 0, 1, 0, 0]) == result)"""
