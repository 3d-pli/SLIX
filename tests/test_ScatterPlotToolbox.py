import pytest
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
        pass

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
        pass

    def test_crossing_direction_image(self):
        pass

    def test_non_crossing_direction(self):
        pass

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


