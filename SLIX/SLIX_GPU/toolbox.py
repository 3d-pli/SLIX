import numpy
import cupy
from numba import cuda
from SLIX.SLIX_GPU._toolbox import _direction, _prominence, _peakwidth


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
    result_img_gpu = cupy.empty((image_x * image_y, image_z), dtype='float32')

    # https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding_utils.pyx
    threads_per_block = 256
    blocks_per_grid = (image_x * image_y + (threads_per_block - 1)) // threads_per_block
    _prominence[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image, result_img_gpu)
    cuda.synchronize()

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


def direction(peak_image, number_of_directions=3, return_numpy=True):
    gpu_peak_image = cupy.array(peak_image)
    [image_x, image_y, image_z] = gpu_peak_image.shape

    gpu_peak_image = gpu_peak_image.reshape(image_x * image_y, image_z).astype('int8')
    result_img_gpu = cupy.empty((image_x * image_y, number_of_directions), dtype='float32')
    number_of_peaks = cupy.count_nonzero(gpu_peak_image, axis=-1).astype('int8')

    threads_per_block = 256
    blocks_per_grid = (image_x * image_y + (threads_per_block - 1)) // threads_per_block
    _direction[blocks_per_grid, threads_per_block](gpu_peak_image, number_of_peaks, result_img_gpu)
    cuda.synchronize()

    result_img_gpu = cupy.asarray(result_img_gpu.reshape((image_x, image_y, number_of_directions)))

    if peak_image is None:
        del gpu_peak_image

    if return_numpy:
        result_img_cpu = cupy.asnumpy(result_img_gpu)
        del result_img_gpu
        return result_img_cpu
    else:
        return result_img_gpu



