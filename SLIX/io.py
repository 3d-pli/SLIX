import numpy
import tifffile
import nibabel


def read_image(filepath):
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
    else:
        data = tifffile.imread(filepath)
        data = numpy.squeeze(numpy.moveaxis(data, 0, -1))

    return data
