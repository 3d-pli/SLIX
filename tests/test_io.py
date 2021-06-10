from SLIX import io
import pytest
import os
import shutil
import numpy


class TestIO:
    def test_read_hdf5(self):
        image = io.imread('tests/files/demo.h5')
        assert image.shape == (170, 163, 24)

    def test_write_hdf5(self):
        test_arr = numpy.random.rand(100, 110, 24)

        writer = io.H5FileWriter()
        writer.open('tests/output/test_write_hdf5.h5')
        writer.write_dataset('/Image', test_arr)
        writer.close()

        assert os.path.isfile('tests/output/test_write_hdf5.h5')

        ############

        reader = io.H5FileReader()
        reader.open('tests/output/test_write_hdf5.h5')
        image = reader.read('/Image')
        reader.close()

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
        io.imwrite('tests/output/test_write_nifti.nii.gz', test_arr)
        assert os.path.isfile('tests/output/test_write_nifti.nii.gz')

        image = io.imread('tests/output/test_write_nifti.nii.gz')
        assert image.shape == test_arr.shape
        assert numpy.all(numpy.isclose(test_arr, image))

    def test_read_interoperable(self):
        test_arr = numpy.random.rand(10, 11, 24)
        io.imwrite('tests/output/test_write_interoperable.tiff', test_arr)
        assert os.path.isfile('tests/output/test_write_interoperable.tiff')

        image = io.imread('tests/output/test_write_interoperable.tiff')
        assert image.shape == test_arr.shape
        assert numpy.all(numpy.isclose(test_arr, image))

        io.imwrite('tests/output/test_write_interoperable.h5', test_arr)
        assert os.path.isfile('tests/output/test_write_interoperable.h5')

        image = io.imread('tests/output/test_write_interoperable.h5')
        assert image.shape == test_arr.shape
        assert numpy.all(numpy.isclose(test_arr, image))

        io.imwrite('tests/output/test_write_interoperable.nii', test_arr)
        assert os.path.isfile('tests/output/test_write_interoperable.nii')

        image = io.imread('tests/output/test_write_interoperable.nii')
        assert image.shape == test_arr.shape
        assert numpy.all(numpy.isclose(test_arr, image))

        io.imwrite('tests/output/test_write_interoperable.nii.gz', test_arr)
        assert os.path.isfile('tests/output/test_write_interoperable.nii.gz')

        image = io.imread('tests/output/test_write_interoperable.nii.gz')
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
