from SLIX import toolbox
import numpy
import pytest

if toolbox.gpu_available:
    use_gpu_arr = [True, False]
else:
    use_gpu_arr = [False]


class TestToolbox:
    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_background_mask(self, use_gpu):
        average_image = numpy.zeros((256, 256, 10))
        for i in range(0, 128):
            average_image[i, :] = 0
            average_image[i + 128, :] = 128

        toolbox_mask = toolbox.background_mask(average_image, use_gpu=use_gpu)
        print(toolbox_mask)
        assert numpy.all(toolbox_mask[:128, :] == True)
        assert numpy.all(toolbox_mask[128:, :] == False)

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_all_peaks(self, use_gpu):
        # Create an absolute simple peak array
        arr = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        arr = arr.reshape((1, 1, 11))
        real_peaks = arr == 1
        toolbox_peaks = toolbox.peaks(arr, use_gpu=use_gpu)
        assert numpy.all(toolbox_peaks == real_peaks)

        # Test one single peak
        arr = numpy.array(([0, 1, 0, 0, 0, 0]), dtype=bool)
        arr = arr.reshape((1, 1, 6))
        real_peaks = arr == 1
        toolbox_peaks = toolbox.peaks(arr, use_gpu=use_gpu)
        assert numpy.all(toolbox_peaks == real_peaks)

        # Test one single peak
        arr = numpy.array(([0, 1, 1, 0, 0, 0]), dtype=bool)
        arr = arr.reshape((1, 1, 6))
        real_peaks = arr == 1
        real_peaks[0, 0, 2] = False
        toolbox_peaks = toolbox.peaks(arr, use_gpu=use_gpu)
        assert numpy.all(toolbox_peaks == real_peaks)

        # Test one single peak
        arr = numpy.array(([0, 1, 1, 1, 0, 0]), dtype=bool)
        arr = arr.reshape((1, 1, 6))
        real_peaks = arr == 1
        real_peaks[0, 0, 1] = False
        real_peaks[0, 0, 3] = False
        toolbox_peaks = toolbox.peaks(arr, use_gpu=use_gpu)
        assert numpy.all(toolbox_peaks == real_peaks)

        arr = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] * 1)
        arr = arr.reshape((1, 1, 11))
        real_peaks = arr == 1
        toolbox_peaks = toolbox.peaks(arr, use_gpu=use_gpu)
        assert numpy.all(toolbox_peaks == real_peaks)

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_num_peaks(self, use_gpu):
        # Create an absolute simple peak array
        test_arr = numpy.array(([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=bool)
        test_arr = test_arr.reshape((1, 1, 20))

        real_peaks = test_arr == True
        toolbox_peaks = toolbox.num_peaks(real_peaks, use_gpu=use_gpu)
        expected_value = numpy.count_nonzero(real_peaks[0, 0, :])
        assert numpy.all(toolbox_peaks == expected_value)

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_peak_prominence(self, use_gpu):
        # Create an absolute simple peak array
        arr = numpy.array([0, 1, 0, 0.07, 0, 1, 0, 0.07, 0, 1, 0] * 1)
        arr = arr.reshape((1, 1, 11))
        # Test if high and low prominence separation is working as intended
        high_peaks = arr == 1
        low_peaks = arr == 0.07

        toolbox_peaks = toolbox.peaks(arr, use_gpu=use_gpu)
        toolbox_prominence = toolbox.peak_prominence(arr, toolbox_peaks, kind_of_normalization=0, use_gpu=use_gpu)
        toolbox_high_peaks = toolbox_peaks.copy()
        toolbox_high_peaks[toolbox_prominence < toolbox.cpu_toolbox.TARGET_PROMINENCE] = False
        toolbox_low_peaks = toolbox_peaks.copy()
        toolbox_low_peaks[toolbox_prominence >= toolbox.cpu_toolbox.TARGET_PROMINENCE] = False
        assert numpy.all(high_peaks == toolbox_high_peaks)
        assert numpy.all(low_peaks == toolbox_low_peaks)

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_mean_peakprominence(self, use_gpu):
        test_arr = numpy.array([0, 1, 0, 0.1, 0, 1, 0, 0.1, 0, 1, 0]).reshape((1, 1, 11))
        comparison = toolbox.cpu_toolbox.normalize(test_arr, kind_of_normalization=1)

        toolbox_peaks = toolbox.peaks(test_arr)
        toolbox_prominence = toolbox.mean_peak_prominence(test_arr, toolbox_peaks,
                                                          kind_of_normalization=1, use_gpu=use_gpu)
        assert numpy.isclose(toolbox_prominence, numpy.mean(comparison[comparison > 0]))

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_peakdistance(self, use_gpu):
        # Test one peak
        test_arr = numpy.array(([True, False, False] + [False] * 21))
        test_arr = test_arr.reshape((1, 1, 24))
        expected_distance = 360

        toolbox_peaks = toolbox.peaks(test_arr, use_gpu=use_gpu)
        toolbox_distance = toolbox.peak_distance(toolbox_peaks,
                                                 numpy.zeros(toolbox_peaks.shape, dtype=float), use_gpu=use_gpu)
        toolbox_distance_reduced = numpy.sum(toolbox_distance, axis=-1)
        assert numpy.all(toolbox_distance_reduced == expected_distance)

        # Test two peaks
        test_arr = numpy.array(([False, False, True, False, False, False, False, True, False] + [False] * 15))
        test_arr = test_arr.reshape((1, 1, 24))
        expected_distance = 75

        toolbox_peaks = toolbox.peaks(test_arr, use_gpu=use_gpu)
        toolbox_distance = toolbox.peak_distance(toolbox_peaks,
                                                 numpy.zeros(toolbox_peaks.shape, dtype=float), use_gpu=use_gpu)
        assert toolbox_distance[0, 0, 2] == expected_distance
        assert toolbox_distance[0, 0, 7] == 360 - expected_distance

        # Test four peaks
        test_arr = numpy.array(([False, False, True, False, False, False, False, True, False,
                                 False, False, True, False, False, False, True, False, False] + [False] * 6))
        test_arr = test_arr.reshape((1, 1, 24))
        expected_distance_1 = 135
        expected_distance_2 = 120

        toolbox_peaks = toolbox.peaks(test_arr, use_gpu=use_gpu)
        toolbox_distance = toolbox.peak_distance(toolbox_peaks,
                                                 numpy.zeros(toolbox_peaks.shape, dtype=float), use_gpu=use_gpu)
        assert toolbox_distance[0, 0, 2] == expected_distance_1
        assert toolbox_distance[0, 0, 11] == 360 - expected_distance_1

        assert toolbox_distance[0, 0, 7] == expected_distance_2
        assert toolbox_distance[0, 0, 15] == 360 - expected_distance_2

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_mean_peakdistance(self, use_gpu):
        # Test four peaks
        test_arr = numpy.array(([False, False, True, False, False, False, False, True, False,
                                 False, False, True, False, False, False, True, False, False] + [False] * 6))
        test_arr = test_arr.reshape((1, 1, 24))
        expected_distance = (135 + 120) / 2

        toolbox_peaks = toolbox.peaks(test_arr, use_gpu=use_gpu)
        toolbox_distance = toolbox.mean_peak_distance(toolbox_peaks, numpy.zeros(toolbox_peaks.shape,
                                                                                 dtype=float), use_gpu=use_gpu)
        assert numpy.isclose(toolbox_distance[0, 0], expected_distance)

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_peakwidth(self, use_gpu):
        test_arr = numpy.array([0, 0.5, 1, 0.5, 0] + [0] * 19)
        test_arr = test_arr.reshape((1, 1, 24))
        expected_width = 30

        toolbox_peaks = toolbox.peaks(test_arr, use_gpu=use_gpu)
        toolbox_width = toolbox.peak_width(test_arr, toolbox_peaks, use_gpu=use_gpu)
        assert toolbox_width[0, 0, 2] == expected_width
        assert numpy.sum(toolbox_width) == expected_width

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_mean_peakwidth(self, use_gpu):
        test_arr = numpy.array([0, 0.5, 1, 0.5, 0, 0, 0.5, 0.6, 0.7, 1, 0.7, 0.6, 0.5] + [0] * 11)
        test_arr = test_arr.reshape((1, 1, 24))
        expected_width = 60

        toolbox_peaks = toolbox.peaks(test_arr, use_gpu=use_gpu)
        toolbox_width = toolbox.mean_peak_width(test_arr, toolbox_peaks, use_gpu=use_gpu)
        assert toolbox_width[0, 0] == expected_width

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_direction(self, use_gpu):
        # Test for one peak
        one_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        one_peak_arr = one_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([45, -1, -1])
        peaks = toolbox.peaks(one_peak_arr, use_gpu=use_gpu)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(one_peak_arr.shape), use_gpu=use_gpu)
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for one direction with 180°+-35° distance
        two_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        two_peak_arr = two_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([135, -1, -1])
        peaks = toolbox.peaks(two_peak_arr, use_gpu=use_gpu)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape), use_gpu=use_gpu)
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for two directions with 180°+-35° distance
        four_peak_arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        four_peak_arr = four_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([135, 60, -1])
        peaks = toolbox.peaks(four_peak_arr, use_gpu=use_gpu)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape), use_gpu=use_gpu)
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for three directions with 180°+-35° distance
        six_peak_arr = numpy.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
        six_peak_arr = six_peak_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([135, 105, 60])
        peaks = toolbox.peaks(six_peak_arr, use_gpu=use_gpu)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape), use_gpu=use_gpu)
        assert numpy.all(expected_direction == toolbox_direction)

        # Test for one angle outside of 180°+-35° distance
        arr = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        arr = arr.reshape((1, 1, 24))
        expected_direction = numpy.array([82.5, -1, -1])
        peaks = toolbox.peaks(arr, use_gpu=use_gpu)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(arr.shape), use_gpu=use_gpu)
        assert numpy.all(expected_direction == toolbox_direction)

        error_arr = numpy.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        error_arr = error_arr.reshape((1, 1, 24))
        expected_direction = numpy.array([-1, -1, -1])
        peaks = toolbox.peaks(error_arr, use_gpu=use_gpu)
        toolbox_direction = toolbox.direction(peaks, numpy.zeros(two_peak_arr.shape), use_gpu=use_gpu)
        assert numpy.all(expected_direction == toolbox_direction)

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_centroid_correction(self, use_gpu):
        # simple test case: one distinct peak
        test_array = numpy.array([0] * 9 + [1] + [0] * 14)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array, use_gpu=use_gpu)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks, use_gpu=use_gpu)
        assert numpy.isclose(toolbox_centroid[0, 0, 9], 0)

        # simple test case: one distinct peak
        test_array = numpy.array([0] * 8 + [0.5, 1, 0.5] + [0] * 13)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array, use_gpu=use_gpu)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks, use_gpu=use_gpu)
        assert numpy.isclose(toolbox_centroid[0, 0, 9], 0)

        # simple test case: centroid is between two measurements
        test_array = numpy.array([0] * 8 + [1, 1] + [0] * 14)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array, use_gpu=use_gpu)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks, use_gpu=use_gpu)
        assert numpy.isclose(toolbox_centroid[0, 0, 8], 0.5)

        # more complicated test case: wide peak plateau
        test_array = numpy.array([0] * 8 + [1, 1, 1] + [0] * 13)
        test_array = test_array.reshape((1, 1, 24))
        test_high_peaks = toolbox.peaks(test_array, use_gpu=use_gpu)

        toolbox_centroid = toolbox.centroid_correction(test_array, test_high_peaks, use_gpu=use_gpu)
        assert numpy.isclose(toolbox_centroid[0, 0, 9], 0)

    @pytest.mark.parametrize("use_gpu", use_gpu_arr)
    def test_unit_vectors(self, use_gpu):
        direction = numpy.arange(0, 180).reshape((1, 1, 180))
        unit_x, unit_y = toolbox.unit_vectors(direction, use_gpu=use_gpu)
        assert numpy.all(numpy.isclose(numpy.rad2deg(numpy.arctan2(unit_y, unit_x)), 180 - direction))