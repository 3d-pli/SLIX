import numpy
import tifffile
import nibabel
import h5py
from PIL import Image
import sys
import getpass


def hdf5_read(filepath, dataset):
    with h5py.File(filepath, mode='r') as file:
        return file[dataset][:]


def hdf5_write(filepath, dataset, data):
    with h5py.File(filepath, mode='w') as file:
        file_dataset = file.create_dataset(dataset, data.shape, numpy.float32, data=data)
        file_dataset.attrs['created_by'] = getpass.getuser()
        file_dataset.attrs['software'] = sys.argv[0]
        file_dataset.attrs['software_parameters'] = ' '.join(sys.argv[1:])
        file_dataset.attrs['filename'] = filepath


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
    if len(data.shape) == 3:
        swap_axes = True
    if filepath.endswith('.nii'):
        if swap_axes:
            save_data = numpy.swapaxes(data, 0, 1)
        else:
            save_data = data
        nibabel.save(nibabel.Nifti1Image(save_data, numpy.eye(4)), filepath)
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        if swap_axes:
            save_data = numpy.swapaxes(data, -1, 0)
        else:
            save_data = data
        tifffile.imwrite(filepath, save_data)
    else:
        Image.fromarray(data).save(filepath)