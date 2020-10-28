from SLIX import io
import pytest
import os
import shutil
import numpy

from matplotlib import pyplot as plt

class TestIO:
    def test_read_hdf5(self):
        image = io.hdf5_read('tests/files/demo.h5', '/Image')
        assert image.shape == (170, 163, 24)

    def test_write_hdf5(self):
        test_arr = numpy.random.rand(100, 110, 24)
        io.hdf5_write('tests/output/test_write_hdf5.h5', '/Image', test_arr)
        assert os.path.isfile('tests/output/test_write_hdf5.h5')

        image = io.hdf5_read('tests/output/test_write_hdf5.h5', '/Image')
        assert image.shape == test_arr.shape
        assert numpy.all(numpy.isclose(test_arr, image))

    def test_read_tiff(self):
        image = io.imread('tests/files/demo.tiff')
        assert image.shape == (170, 163, 24)

    def test_write_tiff(self):
        test_arr = numpy.random.rand(100, 110, 24)
        io.imwrite('tests/output/test_write_tiff.tiff', test_arr)
        assert os.path.isfile('tests/output/test_write_tiff.tiff')

        image = io.imread('tests/output/test_write_tiff.tiff')
        assert image.shape == test_arr.shape
        assert numpy.all(numpy.isclose(test_arr, image))

    def test_read_nifti(self):
        image = io.imread('tests/files/demo.nii')
        assert image.shape == (566, 542, 24)

    def test_write_nifti(self):
        test_arr = numpy.random.rand(100, 110, 24)
        io.imwrite('tests/output/test_write_nifti.nii', test_arr)
        assert os.path.isfile('tests/output/test_write_nifti.nii')

        image = io.imread('tests/output/test_write_nifti.nii')
        assert image.shape == test_arr.shape
        assert numpy.all(numpy.isclose(test_arr, image))


@pytest.fixture(scope="session", autouse=True)
def run_around_tests(request):
    # Code that will run before your test, for example:
    assert os.path.isfile('tests/files/demo.nii')
    assert os.path.isfile('tests/files/demo.tiff')
    assert os.path.isfile('tests/files/demo.h5')
    if not os.path.isdir('tests/output/'):
        os.mkdir('tests/output/')

    # A test function will be run at this point
    yield

    # Code that will run after your test, for example:
    def remove_test_dir():
        if os.path.isdir('tests/output/'):
            shutil.rmtree('tests/output/')
    request.addfinalizer(remove_test_dir)
