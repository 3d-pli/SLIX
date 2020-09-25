import numpy
import cupy
import nibabel
import tifffile
from numba import cuda

# DEFAULT PARAMETERS
BACKGROUND_COLOR = -1
MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 2

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.94
TARGET_PROMINENCE = 0.08


def read_image(FILEPATH):
    """
    Reads image file and returns it.
    Supported file formats: NIfTI, Tiff.

    Arguments:
        FILEPATH: Path to image

    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size of a single image and z specifies the number
                     of measurements
    """
    # Load NIfTI dataset
    if FILEPATH.endswith('.nii'):
        data = nibabel.load(FILEPATH).get_fdata()
        data = numpy.squeeze(numpy.swapaxes(data, 0, 1))
    else:
        data = tifffile.imread(FILEPATH)
        data = numpy.squeeze(numpy.moveaxis(data, 0, -1))

    return data


def peaks(image, return_numpy=True):
    gpu_image = cupy.array(image, dtype='float32')
    right = cupy.roll(gpu_image, 1, axis=-1) - gpu_image
    left = cupy.roll(gpu_image, -1, axis=-1) - gpu_image
    del gpu_image

    peaks = (left < 0) & (right <= 0)
    del right
    del left

    if return_numpy:
        peaks_cpu = cupy.asnumpy(peaks)
        del peaks

        return peaks_cpu
    else:
        return peaks


def num_peaks(image, return_numpy=True):
    gpu_image = cupy.array(image, dtype='float64')
    right = cupy.roll(gpu_image, 1, axis=-1) - gpu_image
    left = cupy.roll(gpu_image, -1, axis=-1) - gpu_image
    del gpu_image

    peaks = (left < 0) & (right <= 0)
    del right
    del left

    resulting_image = cupy.empty((peaks.shape[:2]))
    resulting_image[:, :] = cupy.count_nonzero(peaks, axis=-1)
    if return_numpy:
        resulting_image_cpu = cupy.asnumpy(resulting_image)
        del resulting_image
        return resulting_image_cpu
    else:
        return resulting_image


def normalize(image, kind_of_normalization=0, return_numpy=True):
    gpu_image = cupy.array(image, dtype='float32')
    if kind_of_normalization == 0:
        gpu_image = (gpu_image - cupy.min(gpu_image, axis=-1)[..., None]) / \
                    (cupy.max(gpu_image, axis=-1)[..., None] - cupy.min(gpu_image, axis=-1)[..., None])
    else:
        gpu_image = gpu_image / cupy.mean(gpu_image, axis=-1)[..., None]

    if return_numpy:
        gpu_image_cpu = cupy.asnumpy(gpu_image)
        del gpu_image
        return gpu_image_cpu
    else:
        return gpu_image


@cuda.jit('void(float32[:, :], int8[:, :], float32[:, :])')
def _prominence(image, peak_array, result_image):
    idx = cuda.grid(1)
    sub_image = image[idx]
    sub_peak_array = peak_array[idx]

    for pos in range(len(sub_peak_array)):
        if sub_peak_array[pos] == 1:
            i_min = -len(sub_peak_array) / 2
            i_max = int(len(sub_peak_array) * 1.5)

            i = pos
            left_min = image[idx, pos]
            while i_min <= i and image[idx, i] <= image[idx, pos]:
                if image[idx, i] < left_min:
                    left_min = image[idx, i]
                i -= 1

            i = pos
            right_min = sub_image[pos]
            while i <= i_max and sub_image[i % len(sub_peak_array)] <= sub_image[pos]:
                if sub_image[i % len(sub_peak_array)] < right_min:
                    right_min = sub_image[i % len(sub_peak_array)]
                i += 1

            result_image[idx, pos] = sub_image[pos] - max(left_min, right_min)
        else:
            result_image[idx, pos] = 0


def peak_prominence(image, peak_image=None, kind_of_normalization=0, return_numpy=True):
    gpu_image = cupy.array(image, dtype='float32')
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image)
    else:
        gpu_peak_image = peaks(gpu_image, return_numpy=False)
    gpu_image = normalize(gpu_image, kind_of_normalization, return_numpy=False)

    [image_x, image_y, image_z] = gpu_image.shape

    gpu_image = gpu_image.reshape(image_x * image_y, image_z)
    gpu_peak_image = gpu_peak_image.reshape(image_x * image_y, image_z).astype('int8')
    result_img = numpy.zeros((image_x * image_y, image_z), dtype='float32')
    result_img_gpu = cuda.to_device(result_img)
    del result_img

    # https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding_utils.pyx
    threadsperblock = 256
    blockspergrid = (image_x * image_y + (threadsperblock - 1)) // threadsperblock
    _prominence[blockspergrid, threadsperblock](gpu_image, gpu_peak_image, result_img_gpu)
    #cuda.synchronize()

    result_img_gpu = cupy.asarray(result_img_gpu.reshape((image_x, image_y, image_z)))

    if peak_image is None:
        del gpu_peak_image

    if return_numpy:
        if isinstance(image, type(numpy.zeros(0))):
            del gpu_image
        result_img_cpu = cupy.asnumpy(result_img_gpu)
        del result_img_gpu
        return result_img_cpu
    else:
        return result_img_gpu


def peak_width():
    pass


def direction():
    pass



