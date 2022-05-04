import numpy
from SLIX import toolbox, io, visualization
import matplotlib
from matplotlib import pyplot as plt
import pytest
import shutil
import os

matplotlib.use('agg')

available_colormaps = [
    {'rgb': visualization.Colormap.rgb},
    {'hsvBlack': visualization.Colormap.hsv_black},
    {'hsvWhite': visualization.Colormap.hsv_white},
    {'rgb_r': visualization.Colormap.rgb_reverse},
    {'hsvBlack_r': visualization.Colormap.hsv_black_reverse},
    {'hsvWhite_r': visualization.Colormap.hsv_white_reverse}
]


class TestVisualization:
    @pytest.mark.parametrize('colormap', available_colormaps)
    def test_visualize_unit_vectors(self, colormap):
        colormap_name = list(colormap.keys())[0]
        colormap_function = list(colormap.values())[0]

        example = io.imread('tests/files/demo.nii')
        peaks = toolbox.significant_peaks(example, use_gpu=False)
        centroid = toolbox.centroid_correction(example, peaks, use_gpu=False)
        direction = toolbox.direction(peaks, centroid, use_gpu=False)
        unit_x, unit_y = toolbox.unit_vectors(direction, use_gpu=False)
        visualization.unit_vectors(unit_x, unit_y, thinout=10, colormap=colormap_function)
        plt.savefig(f'tests/output/vis/unit_vectors_{colormap_name}.tiff', dpi=100,
                    bbox_inches='tight')

        orig = io.imread(f'tests/files/vis/unit_vectors_{colormap_name}.tiff')
        to_compare = io.imread(f'tests/output/vis/unit_vectors_{colormap_name}.tiff')

        if numpy.all(numpy.isclose(orig - to_compare, 0)):
            assert True
        else:
            io.imwrite(f'tests/output/vis/unit_vectors_{colormap_name}-diff.tiff', orig - to_compare)
            assert False

    def test_unit_vectors_single_direction(self):
        example = io.imread('tests/files/demo.nii')
        peaks = toolbox.significant_peaks(example, use_gpu=False)
        centroid = toolbox.centroid_correction(example, peaks, use_gpu=False)
        direction = toolbox.direction(peaks, centroid, number_of_directions=1, use_gpu=False)
        unit_x, unit_y = toolbox.unit_vectors(direction, use_gpu=False)
        visualization.unit_vectors(unit_x, unit_y, thinout=10)
        plt.savefig('tests/output/vis/unit_vectors_single_dir.tiff', dpi=100,
                    bbox_inches='tight')

        orig = io.imread('tests/files/vis/unit_vectors_single_dir.tiff')
        to_compare = io.imread('tests/output/vis/unit_vectors_single_dir.tiff')

        if numpy.all(numpy.isclose(orig - to_compare, 0)):
            assert True
        else:
            io.imwrite('tests/output/vis/unit_vectors_single_dir-diff.tiff', orig - to_compare)
            assert False

    @pytest.mark.parametrize('colormap', available_colormaps)
    def test_visualize_unit_vector_distribution(self, colormap):
        colormap_name = list(colormap.keys())[0]
        colormap_function = list(colormap.values())[0]

        example = io.imread('tests/files/demo.nii')
        peaks = toolbox.significant_peaks(example, use_gpu=False)
        centroid = toolbox.centroid_correction(example, peaks, use_gpu=False)
        direction = toolbox.direction(peaks, centroid, use_gpu=False)
        unit_x, unit_y = toolbox.unit_vectors(direction, use_gpu=False)
        visualization.unit_vector_distribution(unit_x, unit_y, thinout=15, vector_width=5,
                                               alpha=0.01, colormap=colormap_function)

        plt.savefig(f'tests/output/vis/unit_vector_distribution_{colormap_name}.tiff', dpi=100,
                    bbox_inches='tight')

        orig = io.imread(f'tests/files/vis/unit_vector_distribution_{colormap_name}.tiff')
        to_compare = io.imread(f'tests/output/vis/unit_vector_distribution_{colormap_name}.tiff')

        if numpy.all(numpy.isclose(orig - to_compare, 0)):
            assert True
        else:
            io.imwrite(f'tests/output/vis/unit_vector_distribution_{colormap_name}-diff.tiff', orig - to_compare)
            assert False

    def test_weight_unit_vector(self):
        example = io.imread('tests/files/demo.nii')
        peaks = toolbox.significant_peaks(example, use_gpu=False)
        centroid = toolbox.centroid_correction(example, peaks, use_gpu=False)
        direction = toolbox.direction(peaks, centroid, use_gpu=False)
        avg = numpy.average(example, axis=-1)
        avg = (avg - avg.min()) / (numpy.percentile(avg, 95) - avg.min())
        unit_x, unit_y = toolbox.unit_vectors(direction, use_gpu=False)
        visualization.unit_vectors(unit_x, unit_y, weighting=avg, thinout=10, scale=10)
        plt.savefig('tests/output/vis/unit_vectors_weighted.tiff', dpi=100,
                    bbox_inches='tight')

        orig = io.imread('tests/files/vis/unit_vectors_weighted.tiff')
        to_compare = io.imread('tests/output/vis/unit_vectors_weighted.tiff')

        if numpy.all(numpy.isclose(orig - to_compare, 0)):
            assert True
        else:
            io.imwrite('tests/output/vis/unit_vectors_weighted-diff.tiff', orig - to_compare)
            assert False

    def test_weight_unit_vector_distribution(self):
        example = io.imread('tests/files/demo.nii')
        peaks = toolbox.significant_peaks(example, use_gpu=False)
        centroid = toolbox.centroid_correction(example, peaks, use_gpu=False)
        direction = toolbox.direction(peaks, centroid, use_gpu=False)
        avg = numpy.average(example, axis=-1)
        avg = (avg - avg.min()) / (numpy.percentile(avg, 95) - avg.min())
        unit_x, unit_y = toolbox.unit_vectors(direction, use_gpu=False)
        visualization.unit_vector_distribution(unit_x, unit_y, weighting=avg, thinout=15,
                                               scale=15, vector_width=5, alpha=0.01)

        plt.savefig('tests/output/vis/unit_vectors_distribution_weighted.tiff', dpi=100,
                    bbox_inches='tight')

        orig = io.imread('tests/files/vis/unit_vectors_distribution_weighted.tiff')
        to_compare = io.imread('tests/output/vis/unit_vectors_distribution_weighted.tiff')

        if numpy.all(numpy.isclose(orig - to_compare, 0)):
            assert True
        else:
            io.imwrite('tests/output/vis/unit_vectors_distribution_weighted-diff.tiff', orig - to_compare)
            assert False

    def test_visualize_parameter_map(self):
        example = io.imread('tests/files/demo.nii')
        prominence = toolbox.mean_peak_prominence(example, kind_of_normalization=1, use_gpu=False)
        visualization.parameter_map(prominence, colorbar=False)
        plt.savefig('tests/output/vis/parameter_map.tiff', dpi=100,
                    bbox_inches='tight')

        orig = io.imread('tests/files/vis/parameter_map.tiff')
        to_compare = io.imread('tests/output/vis/parameter_map.tiff')

        assert numpy.all(numpy.isclose(orig - to_compare, 0))

    def test_visualize_direction_one_dir(self):
        image = numpy.arange(0, 180)
        hsv_image = visualization.direction(image)
        assert numpy.all(hsv_image[0, :] == [255, 0, 0])
        assert numpy.all(hsv_image[30, :] == [255, 255, 0])
        assert numpy.all(hsv_image[60, :] == [0, 255, 0])
        assert numpy.all(hsv_image[90, :] == [0, 255, 255])
        assert numpy.all(hsv_image[120, :] == [0, 0, 255])
        assert numpy.all(hsv_image[150, :] == [255, 0, 255])

    def test_visualize_direction_multiple_dir(self):
        first_dir = numpy.arange(0, 180)[..., numpy.newaxis, numpy.newaxis]
        second_dir = (first_dir + 30) % 180
        second_dir[0:45] = -1
        third_dir = (first_dir + 60) % 180
        third_dir[0:90] = -1
        fourth_dir = (first_dir + 90) % 180
        fourth_dir[0:135] = -1
        stack_direction = numpy.concatenate((first_dir,
                                             second_dir,
                                             third_dir,
                                             fourth_dir),
                                            axis=-1)
        hsv_image = visualization.direction(stack_direction)

        # Check first direction
        assert numpy.all(hsv_image[0, 0, :] == [255, 0, 0])
        assert numpy.all(hsv_image[1, 1, :] == [255, 0, 0])
        assert numpy.all(hsv_image[0, 1, :] == [255, 0, 0])
        assert numpy.all(hsv_image[1, 0, :] == [255, 0, 0])

        assert numpy.all(hsv_image[60, 0, :] == [255, 255, 0])
        assert numpy.all(hsv_image[61, 1, :] == [255, 255, 0])
        assert numpy.all(hsv_image[60, 1, :] == [255, 255, 0])
        assert numpy.all(hsv_image[61, 0, :] == [255, 255, 0])

        # Probe check second direction
        assert numpy.all(hsv_image[120, 0, :] == [0, 255, 0])
        assert numpy.all(hsv_image[121, 1, :] == [0, 255, 0])
        assert numpy.all(hsv_image[120, 1, :] == [0, 255, 255])
        assert numpy.all(hsv_image[121, 0, :] == [0, 255, 255])

        # Probe check third direction
        assert numpy.all(hsv_image[240, 0, :] == [0, 0, 255])
        assert numpy.all(hsv_image[240, 1, :] == [255, 0, 0])
        assert numpy.all(hsv_image[241, 0, :] == [255, 0, 255])
        assert numpy.all(hsv_image[241, 1, :] == [0, 0, 0])

        # Probe check fourth direction
        assert numpy.all(hsv_image[300, 0, :] == [255, 0, 255])
        assert numpy.all(hsv_image[300, 1, :] == [255, 255, 0])
        assert numpy.all(hsv_image[301, 0, :] == [255, 0, 0])
        assert numpy.all(hsv_image[301, 1, :] == [0, 255, 0])


@pytest.fixture(scope="session", autouse=True)
def run_around_tests(request):
    if not os.path.isdir('tests/output/vis'):
        os.makedirs('tests/output/vis')

    # A test function will be run at this point
    yield

    def remove_test_dir():
        if os.path.isdir('tests/output/vis'):
            # shutil.rmtree('tests/output/vis')
            pass

    request.addfinalizer(remove_test_dir)


@pytest.fixture(scope="function", autouse=True)
def run_around_single_test(request):
    plt.clf()
    plt.cla()
    plt.close()
    plt.axis('off')

    # A test function will be run at this point
    yield
