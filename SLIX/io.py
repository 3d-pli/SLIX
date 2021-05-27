from typing import Union

from PIL import Image

import SLIX
from .attributemanager import AttributeHandler
import numpy
import tifffile
import nibabel
import h5py
import sys
import re
import os
import glob
import datetime


class H5FileReader:
    def __init__(self):
        self.path = None
        self.file = None
        self.content = None

    def open(self, path):
        if not path == self.path:
            self.close()
            self.path = path
            self.file = h5py.File(path, 'r')

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.path = None
            self.content = None

    def read(self, dataset):
        if self.content is None:
            self.content = {}
        if dataset not in self.content.keys():
            self.content[dataset] = self.file[dataset][:]
            if len(self.content[dataset].shape) == 3:
                shape = numpy.array(self.content[dataset].shape)
                self.content[dataset] = numpy.swapaxes(self.content[dataset],
                                                       shape.argmin(), -1)
        return self.content[dataset]


class H5FileWriter:
    def __init__(self):
        self.path = None
        self.file = None

    def add_symlink(self, dataset, symlink_path):
        if self.file is None:
            return
        self.file[symlink_path] = self.file[dataset]
        self.file.flush()

    def add_plim_attributes(self, stack_path, dataset='/Image'):
        if self.path is None or self.file is None:
            return

        if dataset not in self.file:
            self.file.create_dataset(dataset)
        output_handler = AttributeHandler(self.file[dataset])

        if stack_path[:-3] == ".h5":
            original_file = h5py.File(stack_path, 'r')
            original_dataset = original_file[dataset]
            original_handler = AttributeHandler(original_dataset)
            original_handler.copy_all_attributes_to(output_handler)
            original_file.close()

        output_handler.add_creator()
        output_handler.set_attribute('software', sys.argv[0])
        output_handler.set_attribute('software_revision',
                                     SLIX.__version__)
        output_handler.set_attribute('creation_time',
                                     datetime.datetime.now()
                                     .strftime('%Y-%m-%d %H:%M:%S'))
        output_handler.set_attribute('software_parameters',
                                     ' '.join(sys.argv[1:]))
        output_handler.set_attribute('image_modality', "Placeholder")
        output_handler.add_id()

        self.file.flush()

    def write_attribute(self, dataset, attr_name, value):
        if self.file is None:
            return

        if dataset not in self.file:
            self.file.create_dataset(dataset)
        output_handler = AttributeHandler(self.file[dataset])
        output_handler.set_attribute(attr_name, value)
        self.file.flush()

    def write_dataset(self, dataset, content):
        if self.file is None:
            return

        if dataset not in self.file:
            if len(content.shape) == 3:
                shape = numpy.array(content.shape)
                content = numpy.swapaxes(content, shape.argmin(), -1)
            self.file.create_dataset(dataset, content.shape,
                                     dtype=content.dtype, data=content)
        self.file.flush()

    def close(self):
        if self.file is None:
            return

        self.file.flush()
        self.file.close()

        self.path = None
        self.file = None

    def open(self, path):
        if self.path != path:
            self.close()
            self.path = path
            self.file = h5py.File(path, mode='w')


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
    file_regex = r'.*_+p[0-9]+_?.*\.(tif{1,2}|jpe*g|nii|h5|png)'

    files_in_folder = glob.glob(filepath+'/*')
    matching_files = []
    for file in files_in_folder:
        if re.match(file_regex, file) is not None:
            matching_files.append(file)
    matching_files.sort(key=__natural_sort_filenames_key)
    image = None

    # Check if files contain the needed regex for our measurements
    for file in matching_files:
        print(file)
        measurement_image = imread(file)
        if image is None:
            image = measurement_image
        elif len(image.shape) == 2:
            image = numpy.stack((image, measurement_image), axis=-1)
        else:
            image = numpy.concatenate((image,
                    measurement_image[:, :, numpy.newaxis]), axis=-1)

    return image


def __natural_sort_filenames_key(string, regex=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in regex.split(string)]


def imread(filepath, dataset="/Image"):
    """
    Reads image file and returns it.
    Supported file formats: HDF5, NIfTI, Tiff.

    Arguments:
        filepath: Path to image
        dataset: When reading a HDF5 file, a dataset is required.
                 Default: '/Image'

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
            data = numpy.squeeze(numpy.swapaxes(data, 0, -1))
    elif filepath.endswith('.h5'):
        reader = H5FileReader()
        reader.open(filepath)
        data = reader.read(dataset)
        reader.close()
        return data
    else:
        data = numpy.array(Image.open(filepath))
    return data


def imwrite(filepath, data, dataset='/Image', original_stack_path=""):
    """
    Write generated image to given filepath.
    Supported file formats: HDF5, NIfTI, Tiff.
    Other file formats are only indirectly supported and might result in
    errors.

    Arguments:
        filepath: Path to image
        data: Data which will be written to the disk
        dataset: When writing a HDF5 file, a dataset is required.
                 Default: '/Image'
        original_stack_path: Path to the original image stack used to create
                             this content. Only required when a HDF5 file
                             is written.
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
    elif isinstance(save_data.dtype, numpy.uint8):
        pass
    else:
        save_data = save_data.astype('float32')

    if filepath.endswith('.nii'):
        save_data = numpy.swapaxes(save_data, 0, 1)
        nibabel.save(nibabel.Nifti1Image(save_data, numpy.eye(4)), filepath)
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        save_data = numpy.swapaxes(save_data, -1, 0)
        tifffile.imwrite(filepath, save_data)
    elif filepath.endswith('.h5'):
        writer = H5FileWriter()
        writer.open(filepath)
        writer.write_dataset(dataset, save_data)
        writer.add_plim_attributes(original_stack_path, dataset)
        writer.add_symlink(dataset, '/pyramid/00')
        writer.close()
    else:
        Image.fromarray(save_data).save(filepath)


def imwrite_rgb(filepath, data, dataset='/Image', original_stack_path=""):
    """
        Write generated RGB image to given filepath.
        Supported file formats: HDF5, Tiff.
        Other file formats are only indirectly supported and might result in
        errors.

        Arguments:
            filepath: Path to image
            data: Data which will be written to the disk
            dataset: When reading a HDF5 file, a dataset is required.
                     Default: '/Image'
            original_stack_path: Path to the original image stack used to create
                                 this content. Only required when a HDF5 file
                                 is written.
        Returns:
            None
        """
    save_data = data.copy()
    axis = numpy.argwhere(numpy.array(save_data.shape) == 3).flatten()
    if len(axis) == 0:
        print('Cannot create RGB image as no dimension has a depth of 3.')
        return

    if filepath.endswith('.tiff') or filepath.endswith('.tif'):
        save_data = numpy.moveaxis(save_data, axis[0], 0)
        tifffile.imwrite(filepath, save_data, photometric='rgb')
    elif filepath.endswith('.h5'):
        writer = H5FileWriter()
        writer.open(filepath)
        save_data = numpy.moveaxis(save_data, axis[0], 0)
        writer.write_dataset(dataset, save_data)
        writer.add_plim_attributes(original_stack_path, dataset)
        writer.add_symlink(dataset, '/pyramid/00')
        writer.close()
    else:
        print("File type is not supported. "
              "Supported file types are .h5, .tif(f)")
