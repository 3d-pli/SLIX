from PIL import Image
import numpy
import tifffile
import nibabel
import h5py
import sys
import getpass
import re
import os
import glob

MEASUREMENT_REGEX_LEFT = r'.*_+p0*'
MEASUREMENT_REGEX_RIGHT = r'_+.*\.(tif{1,2}|jpe*g|nii|h5|png)'


def read_folder(filepath):
    """
    Reads multiple image files from a folder and returns the resulting stack.
    The images are checked with the MEASUREMENT_REGEX_LEFT and
    MEASUREMENT_REGEX_RIGHT.
    Supported file formats: NIfTI, Tiff.

    Arguments:
        filepath: Path to folder

    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size
        of a single image and z specifies the number of measurements
    """

    files_in_folder = glob.glob(filepath+'/*')
    image = None

    # Measurement index
    index = 1
    file_found = True

    while file_found:
        file_found = False

        # Check if files contain the needed regex for our measurements
        for file in files_in_folder:
            match = re.fullmatch(MEASUREMENT_REGEX_LEFT + str(index) + MEASUREMENT_REGEX_RIGHT, file)
            if match is not None:
                measurement_image = imread(file)
                file_found = True
                if image is None:
                    image = measurement_image
                elif len(image.shape) == 2:
                    image = numpy.stack((image, measurement_image), axis=-1)
                else:
                    image = numpy.concatenate((image, measurement_image[:, :, numpy.newaxis]), axis=-1)
                break

        index = index + 1

    return image


def hdf5_read(filepath, dataset):
    """
    Reads image file and returns it.

    Arguments:
        filepath: Path to image
        dataset: Path to dataset in HDF5 file

    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size of a
        single image and z specifies the number of measurements
    """
    with h5py.File(filepath, mode='r') as file:
        data = file[dataset][:]
        data = numpy.moveaxis(data, 0, -1)
        return data


def hdf5_write(filepath, dataset, data, mode='w'):
    """
    Write generated image to given filepath.

    Arguments:
        filepath: Path to image
        dataset: Path to dataset in HDF5 file
        data: Data which will be written to the disk
        mode: Mode with which the HDF5 file will be created.
        Please change 'w' to 'a' if appending to a already exisiting HDF5 file
    Returns:
        None
    """
    with h5py.File(filepath, mode=mode) as file:
        data = numpy.moveaxis(data, -1, 0)
        file_dataset = file.create_dataset(dataset, data.shape, numpy.float32,
                                           data=data)
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
        numpy.array: Image with shape [x, y, z] where [x, y] is the size
        of a single image and z specifies the number of measurements
    """
    if os.path.isdir(filepath):
        data = read_folder(filepath)
    # Load NIfTI dataset
    elif filepath.endswith('.nii'):
        data = nibabel.load(filepath).get_fdata()
        if len(data.shape) > 2:
            data = numpy.squeeze(numpy.swapaxes(data, 0, 1))
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        data = tifffile.imread(filepath)
        if len(data.shape) > 2:
            data = numpy.squeeze(numpy.moveaxis(data, 0, -1))
    else:
        data = numpy.array(Image.open(filepath))
    return data


def imwrite(filepath, data):
    """
    Write generated image to given filepath.
    Supported file formats: NIfTI, Tiff.
    Other file formats are only indirectly supported and might result in
    errors.

    Arguments:
        filepath: Path to image
        data: Data which will be written to the disk

    Returns:
        None
    """
    if len(data.shape) == 3:
        swap_axes = True
    else:
        swap_axes = False

    save_data = data.copy()
    if isinstance(save_data.dtype, (int, numpy.int32, numpy.int64)):
        save_data = save_data.astype('int32')
    else:
        save_data = save_data.astype('float32')

    if filepath.endswith('.nii'):
        if swap_axes:
            save_data = numpy.swapaxes(save_data, 0, 1)
        else:
            save_data = save_data
        nibabel.save(nibabel.Nifti1Image(save_data, numpy.eye(4)), filepath)
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        if swap_axes:
            save_data = numpy.moveaxis(save_data, -1, 0)
        else:
            save_data = save_data
        tifffile.imwrite(filepath, save_data)
    else:
        Image.fromarray(save_data).save(filepath)
