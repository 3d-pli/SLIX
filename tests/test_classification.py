import numpy
import SLIX.io
import SLIX.classification


class TestClassification:
    def test_crossing_mask(self):
        max_image = SLIX.io.imread('./tests/files/cluster/cluster_max.tiff')
        high_prominence_peaks = SLIX.io.imread('./tests/files/cluster/cluster_high_prominence_peaks.tiff')
        cluster = SLIX.classification.crossing_mask(high_prominence_peaks, max_image)

        expected_result = SLIX.io.imread('./tests/files/cluster/results/cluster_crossing_mask.tiff')
        assert numpy.all(cluster == expected_result)

    def test_flat_mask(self):
        high_prominence_peaks = SLIX.io.imread('./tests/files/cluster/cluster_high_prominence_peaks.tiff')
        low_prominence_peaks = SLIX.io.imread('./tests/files/cluster/cluster_low_prominence_peaks.tiff')
        peakdistance = SLIX.io.imread('./tests/files/cluster/cluster_peakdistance.tiff')
        cluster = SLIX.classification.flat_mask(high_prominence_peaks, low_prominence_peaks, peakdistance)

        expected_result = SLIX.io.imread('./tests/files/cluster/results/cluster_flat_mask.tiff')
        assert numpy.all(cluster == expected_result)

    def test_inclinated_mask(self):
        max_image = SLIX.io.imread('./tests/files/cluster/cluster_max.tiff')
        high_prominence_peaks = SLIX.io.imread('./tests/files/cluster/cluster_high_prominence_peaks.tiff')
        peakdistance = SLIX.io.imread('./tests/files/cluster/cluster_peakdistance.tiff')
        flat_mask = SLIX.io.imread('./tests/files/cluster/results/cluster_flat_mask.tiff')
        cluster = SLIX.classification.inclinated_mask(high_prominence_peaks, peakdistance, max_image, flat_mask)

        expected_result = SLIX.io.imread('./tests/files/cluster/results/cluster_inclination_mask.tiff')
        assert numpy.all(cluster == expected_result)

    def test_full_mask(self):
        max_image = SLIX.io.imread('./tests/files/cluster/cluster_max.tiff')
        high_prominence_peaks = SLIX.io.imread('./tests/files/cluster/cluster_high_prominence_peaks.tiff')
        low_prominence_peaks = SLIX.io.imread('./tests/files/cluster/cluster_low_prominence_peaks.tiff')
        peakdistance = SLIX.io.imread('./tests/files/cluster/cluster_peakdistance.tiff')
        cluster = SLIX.classification.full_mask(high_prominence_peaks, low_prominence_peaks,
                                                peakdistance, max_image)

        expected_result = SLIX.io.imread('./tests/files/cluster/results/cluster_classification_mask.tiff')
        assert numpy.all(cluster == expected_result)

