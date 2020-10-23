import numpy
import tifffile
import nibabel
import h5py
from PIL import Image


def hdf5_read(filepath, dataset):
    pass


def hdf5_write(filepath, dataset):
    pass


def imread(filepath):
    """
    Reads image file and returns it.
    Supported file formats: NIfTI, Tiff.

    Arguments:
        filepath: Path to image

    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size of a single image and z specifies the number
                     of measurements
    """
    # Load NIfTI dataset
    if filepath.endswith('.nii'):
        data = nibabel.load(filepath).get_fdata()
        data = numpy.squeeze(numpy.swapaxes(data, 0, 1))
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        data = tifffile.imread(filepath)
        data = numpy.squeeze(numpy.moveaxis(data, 0, -1))
    else:
        data = numpy.array(Image.open(filepath))
    return data


def imwrite(data, filepath):
    if filepath.endswith('.nii'):
        nibabel.save(nibabel.Nifti1Image(numpy.swapaxes(data, 0, 1), numpy.eye(4)), filepath)
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        tifffile.imwrite(filepath, numpy.moveaxis(data, -1, 0))
    else:
        Image.fromarray(data).save(filepath)