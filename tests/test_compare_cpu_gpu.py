import SLIX.SLIX_CPU
import SLIX.SLIX_GPU
import SLIX
import numpy

class TestToolbox:
    def setup_method(self, method):
        self.example = SLIX.toolbox.read_image('./files/demo.nii')

    def test_compare_peaks(self):
        gpu_peaks = SLIX.SLIX_GPU.toolbox.peaks(self.example)
        cpu_peaks = SLIX.SLIX_CPU.toolbox.peaks(self.example)

        assert numpy.all(cpu_peaks == gpu_peaks)

    def test_compare_num_peaks(self):
        pass

    def test_compare_prominence(self):
        pass

    def test_compare_peakwidth(self):
        pass

    def test_compare_peakdistance(self):
        pass

    def test_compare_centroids(self):
        pass

    def test_compare_direction(self):
        pass
