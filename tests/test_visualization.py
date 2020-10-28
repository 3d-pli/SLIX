from matplotlib.testing.decorators import image_comparison
from SLIX import toolbox, io, visualization
import matplotlib
import pytest
import shutil
import os

matplotlib.use('agg')


class TestVisualization:
    @image_comparison(baseline_images=['parameter_map'], remove_text=True, extensions=['png'])
    def test_visualize_parameter_map(self):
        example = io.imread('tests/files/demo.nii')
        prominence = toolbox.mean_peak_prominence(example, kind_of_normalization=1)
        visualization.visualize_parameter_map(prominence)

    @image_comparison(baseline_images=['unit_vectors'], remove_text=True, extensions=['png'])
    def test_visualize_unit_vectors(self):
        example = io.imread('tests/files/demo.nii')
        peaks = toolbox.significant_peaks(example)
        centroid = toolbox.centroid_correction(example, peaks)
        direction = toolbox.direction(peaks, centroid)
        unit_x, unit_y = toolbox.unit_vectors(direction)

        visualization.visualize_unit_vectors(unit_x, unit_y, thinout=10)


@pytest.fixture(scope="session", autouse=True)
def run_around_tests(request):
    # A test function will be run at this point
    yield

    # Code that will run after your test, for example:
    def remove_test_dir():
        if os.path.isdir('result_images'):
            shutil.rmtree('result_images')
    request.addfinalizer(remove_test_dir)
