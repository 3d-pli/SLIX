from PIL import Image
import numpy
import tifffile
import nibabel
import h5py
import sys
import re
import os
import glob
import datetime
import SLIX
from .attributemanager import AttributeHandler

__all__ = ['H5FileReader', 'H5FileWriter', 'imread', 'imwrite', 'imwrite_rgb']

nibabel.openers.Opener.default_compresslevel = 9


class H5FileReader:
    """
    This class allows to read HDF5 files from your file system.
    It supports reading datasets but not reading attributes.
    """

    def __init__(self):
        self.path = None
        self.file = None
        self.content = None

    def open(self, path):
        """

        Args:

            path: Path on the filesystem to the HDF5 file which will be read

        Returns:

            None
        """
        if not path == self.path:
            self.close()
            self.path = path
            self.file = h5py.File(path, 'r')

    def close(self):
        """
        Close the currently opened file, if any is open.

        Returns:

            None
        """
        if self.file is not None:
            self.file.close()
            self.file = None
            self.path = None
            self.content = None

    def read(self, dataset):
        """
        Read a dataset from the currently opened file, if any is open.
        The content of the dataset will be stored for future use.

        Args:

            dataset: Path to the dataset within the HDF5

        Returns:

            The opened dataset.

        """
        if self.content is None:
            self.content = {}
        if dataset not in self.content.keys():
            self.content[dataset] = numpy.squeeze(self.file[dataset][:])
        return self.content[dataset]


class H5FileWriter:
    """
    This class allows to write HDF5 files to your file system.
    It supports writing datasets and writing attributes.
    """

    def __init__(self):
        self.path = None
        self.file = None

    def add_symlink(self, dataset, symlink_path):
        """
        Adds a symbolic link from one dataset to another path.

        Args:

            dataset: Dataset path within the HDF5

            symlink_path: Path to the symlink.

        Returns:

            None

        """
        if self.file is None:
            return
        self.file[symlink_path] = self.file[dataset]
        self.file.flush()

    def add_plim_attributes(self, stack_path, dataset='/Image'):
        """
        PLIM is a package used in the 3D-PLI group to read and write multiple
        attributes from/to a HDF5 file. The basic functionality is added in
        attributemanager.py. Calling this method will write many attributes to
        the HDF5 file at the given dataset.

        This includes: A unique ID, the creator, software parameters,
                       creation time, software_revision, image_modality and
                       all attributes from the original stack, if it was a
                       HDF5

        Args:
            stack_path: Path of the stack that was used to calculate the
                        content which will be written to the HDF5 file.
            dataset: Dataset where the attributes shall be written to.

        Returns:

            None
        """
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
        """
        Write a single attribute to a dataset.

        Args:

            dataset: Path to the dataset within the HDF5

            attr_name: Name of the attribute which shall be written.

            value: Value of the attribute that shall be written.

        Returns:

            None
        """
        if self.file is None:
            return

        if dataset not in self.file:
            self.file.create_dataset(dataset)
        output_handler = AttributeHandler(self.file[dataset])
        output_handler.set_attribute(attr_name, value)
        self.file.flush()

    def write_dataset(self, dataset, content):
        """
        Write a dataset to the currently opened HDF5 file, if any is open.
        If no HDF5 file is open, this method will return without writing.

        Args:

            dataset: Path to the dataset within the HDF5 file.

            content: Content which shall be written.

        Returns:

            None
        """
        if self.file is None:
            return

        if dataset not in self.file:
            # Change compression algorithm for large files as it can take
            # very long for the compression to finish
            if len(content.shape) == 3:
                self.file.create_dataset(dataset, content.shape,
                                         dtype=content.dtype, data=content,
                                         compression='lzf', shuffle=True)
            else:
                self.file.create_dataset(dataset, content.shape,
                                         dtype=content.dtype, data=content,
                                         compression='gzip',
                                         compression_opts=9,
                                         shuffle=True)
        else:
            self.file[dataset] = content

        self.file.flush()

    def close(self):
        """
        Close the currently opened file.

        Returns:

            None
        """
        if self.file is None:
            return

        self.file.flush()
        self.file.close()

        self.path = None
        self.file = None

    def open(self, path):
        """
        Open a new HDF5 file with the given path. If another file was opened,
        it will be closed first.

        Args:

            path: Path to the HDF5 file.

        Returns:

            None
        """
        if self.path != path:
            self.close()
            self.path = path
            self.file = h5py.File(path, mode='w')


def read_folder(filepath):
    """
    Reads multiple image files from a folder and returns the resulting stack.
    To find the images in the right order, a regex is used which will search
    for files with the following pattern:
    [prefix]_p[Nr][suffix]. The start number doesn't need to be 0.
    The files are sorted with a natural sort, meaning that files like
    0002, 1, 004, 3 will be sorted as 1, 0002, 3, 004.

    The follwing regex is used to find the measurements:
    ".*_+p[0-9]+_?.*\.(tif{1,2}|jpe*g|nii|h5|png)"

    Supported file formats for the image file equal the supported formats of
    SLIX.imread.

    Args:

        filepath: Path to folder

    Returns:

        numpy.array: Image with shape [x, y, z] where [x, y] is the size
        of a single image and z specifies the number of measurements
    """
    fileregex = r'.*_+p[0-9]+_?.*\.(tif{1,2}|jpe*g|nii|h5|png)'

    files_in_folder = glob.glob(filepath + '/*')
    matching_files = []
    for file in files_in_folder:
        if re.match(fileregex, file) is not None:
            matching_files.append(file)
    matching_files.sort(key=__natural_sort_filenames_key)
    image = None

    # Check if files contain the needed regex for our measurements
    for file in matching_files:
        measurement_image = imread(file)
        if image is None:
            image = measurement_image
        elif len(image.shape) == 2:
            image = numpy.stack((image, measurement_image), axis=-1)
        else:
            image = numpy.concatenate((image,
                                       measurement_image
                                       [:, :, numpy.newaxis]), axis=-1)

    return image


def __natural_sort_filenames_key(string, regex=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in regex.split(string)]


def imread(filepath, dataset="/Image"):
    """
    Reads image file and returns it.
    Supported file formats: HDF5, NIfTI, Tiff.

    Args:

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
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        data = nibabel.load(filepath).get_fdata()
        data = numpy.squeeze(numpy.swapaxes(data, 0, 1))
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        data = tifffile.imread(filepath)
        if len(data.shape) == 3:
            data = numpy.squeeze(numpy.moveaxis(data, 0, -1))
    elif filepath.endswith('.h5'):
        reader = H5FileReader()
        reader.open(filepath)
        data = reader.read(dataset)
        if len(data.shape) == 3:
            data = numpy.squeeze(numpy.moveaxis(data, 0, -1))
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

    Args:

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
    save_data = data.copy()
    if save_data.dtype == bool:
        save_data = save_data.astype(numpy.uint8)
    elif save_data.dtype == numpy.float64:
        save_data = save_data.astype(numpy.float32)
    elif save_data.dtype == numpy.int64:
        save_data = save_data.astype(numpy.int32)
    elif save_data.dtype == numpy.uint64:
        save_data = save_data.astype(numpy.uint32)

    if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        save_data = numpy.swapaxes(save_data, 0, 1)
        nibabel.save(nibabel.Nifti1Image(save_data, numpy.eye(4)),
                     filepath)

    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        if len(save_data.shape) == 3:
            save_data = numpy.moveaxis(save_data, -1, 0)
        tifffile_version_date = datetime.datetime.strptime(
            tifffile.__version__, '%Y.%m.%d')
        tifffile_comparison_date = datetime.datetime.strptime(
            '2020.10.02', '%Y.%m.%d')
        if tifffile_version_date > tifffile_comparison_date:
            tifffile.imwrite(filepath, save_data, compression=8)
        else:
            tifffile.imwrite(filepath, save_data, compress=9)

    elif filepath.endswith('.h5'):
        if len(save_data.shape) == 3:
            save_data = numpy.moveaxis(save_data, -1, 0)
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

        Args:

            filepath: Path to image

            data: Data which will be written to the disk

            dataset: When reading a HDF5 file, a dataset is required.
                     Default: '/Image'

            original_stack_path: Path to the original image stack used to
                                 create this content. Only required when a
                                 HDF5 file is written.
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
        tifffile.imwrite(filepath, save_data, photometric='rgb',
                         compression=8)
    elif filepath.endswith('.h5'):
        writer = H5FileWriter()
        writer.open(filepath)
        writer.write_dataset(dataset, save_data)
        writer.add_plim_attributes(original_stack_path, dataset)
        writer.add_symlink(dataset, '/pyramid/00')
        writer.close()
    else:
        print("File type is not supported. "
              "Supported file types are .h5, .tif(f)")
