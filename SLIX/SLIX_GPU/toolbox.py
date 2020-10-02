import cupy
import numpy
from numba import cuda

from SLIX.SLIX_GPU._toolbox import _direction, _prominence, _peakwidth, _peakdistance, TARGET_PROMINENCE, \
    _centroid_correction_bases, _centroid, _peak_cleanup


def peaks(image, return_numpy=True):
    gpu_image = cupy.array(image, dtype='int32')
    right = cupy.roll(gpu_image, 1, axis=-1) - gpu_image
    left = cupy.roll(gpu_image, -1, axis=-1) - gpu_image
    del gpu_image

    peaks = (left <= 0) & (right <= 0)
    reshape = False
    if len(peaks.shape) == 3:
        reshape = True
        [image_x, image_y, image_z] = peaks.shape
        peaks = peaks.reshape(image_x * image_y, image_z)
    else:
        [image_x, image_z] = peaks.shape
        image_y = 1
    del right
    del left

    resulting_peaks = cupy.empty(peaks.shape, dtype='int8')
    threads_per_block = 256
    blocks_per_grid = (image_x * image_y + (threads_per_block - 1)) // threads_per_block
    print(resulting_peaks.shape, peaks.shape)
    _peak_cleanup[blocks_per_grid, threads_per_block](peaks, resulting_peaks)

    if reshape:
        resulting_peaks = resulting_peaks.reshape(image_x, image_y, image_z)
    del peaks

    if return_numpy:
        peaks_cpu = cupy.asnumpy(resulting_peaks)
        del resulting_peaks

        return peaks_cpu.astype('bool')
    else:
        return resulting_peaks.astype('bool')


def num_peaks(image, return_numpy=True):
    peak_image = peaks(image, return_numpy=False)

    resulting_image = cupy.empty((peak_image.shape[:2]))
    resulting_image[:, :] = cupy.count_nonzero(peak_image, axis=-1)
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


def mean_peak_prominence(image, peak_image=None, kind_of_normalization=0, return_numpy=True):
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image)
    else:
        gpu_peak_image = peaks(peak_image, return_numpy=False)
    peak_prominence_gpu = peak_prominence(image, peak_image, kind_of_normalization, return_numpy=False)
    peak_prominence_gpu = cupy.sum(peak_prominence_gpu, axis=-1) / cupy.maximum(1,
                                                                                cupy.count_nonzero(gpu_peak_image,
                                                                                                   axis=-1))

    del gpu_peak_image
    if return_numpy:
        peak_width_cpu = cupy.asnumpy(peak_prominence_gpu)
        del peak_prominence_gpu
        return peak_width_cpu
    else:
        return peak_prominence_gpu


def peak_width(image, peak_image=None, target_height=0.5, return_numpy=True):
    gpu_image = cupy.array(image, dtype='float32')
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image)
    else:
        gpu_peak_image = peaks(gpu_image, return_numpy=False)
    [image_x, image_y, image_z] = gpu_image.shape

    gpu_image = gpu_image.reshape(image_x * image_y, image_z)
    gpu_peak_image = gpu_peak_image.reshape(image_x * image_y, image_z).astype('int8')
    gpu_prominence = cupy.empty((image_x * image_y, image_z), dtype='float32')
    result_image_gpu = cupy.empty((image_x * image_y, image_z), dtype='float32')

    # https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding_utils.pyx
    threads_per_block = 256
    blocks_per_grid = (image_x * image_y + (threads_per_block - 1)) // threads_per_block
    _prominence[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image, gpu_prominence)
    cuda.synchronize()

    _peakwidth[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image, gpu_prominence, result_image_gpu,
                                                   target_height)
    del gpu_prominence
    if peak_image is None:
        del gpu_peak_image

    result_image_gpu = cupy.asarray(result_image_gpu.reshape((image_x, image_y, image_z)))
    result_image_gpu = result_image_gpu * (360.0 / image_z)

    if return_numpy:
        if isinstance(image, type(numpy.zeros(0))):
            del gpu_image
        result_image_cpu = cupy.asnumpy(result_image_gpu)
        del result_image_gpu
        return result_image_cpu
    else:
        return result_image_gpu


def mean_peak_width(image, peak_image=None, target_height=0.5, return_numpy=True):
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image)
    else:
        gpu_peak_image = peaks(peak_image, return_numpy=False)
    peak_width_gpu = peak_width(image, gpu_peak_image, target_height, return_numpy=False)
    peak_width_gpu = cupy.sum(peak_width_gpu, axis=-1) / cupy.maximum(1, cupy.count_nonzero(gpu_peak_image, axis=-1))

    del gpu_peak_image
    if return_numpy:
        peak_width_cpu = cupy.asnumpy(peak_width_gpu)
        del peak_width_gpu
        return peak_width_cpu
    else:
        return peak_width_gpu


def peak_distance(peak_image, centroids, return_numpy=True):
    gpu_peak_image = cupy.array(peak_image)
    gpu_centroids = cupy.array(centroids)
    [image_x, image_y, image_z] = gpu_peak_image.shape

    gpu_peak_image = gpu_peak_image.reshape(image_x * image_y, image_z).astype('int8')
    gpu_centroids = gpu_centroids.reshape(image_x * image_y, image_z).astype('float32')
    number_of_peaks = cupy.count_nonzero(gpu_peak_image, axis=-1).astype('int8')
    result_img_gpu = cupy.zeros((image_x * image_y, image_z), dtype='float32')

    threads_per_block = 256
    blocks_per_grid = (image_x * image_y + (threads_per_block - 1)) // threads_per_block
    _peakdistance[blocks_per_grid, threads_per_block](gpu_peak_image, gpu_centroids, number_of_peaks, result_img_gpu)
    cuda.synchronize()

    result_img_gpu = cupy.asarray(result_img_gpu.reshape((image_x, image_y, image_z)))

    if peak_image is None:
        del gpu_peak_image

    if return_numpy:
        result_img_cpu = cupy.asnumpy(result_img_gpu)
        del result_img_gpu
        return result_img_cpu
    else:
        return result_img_gpu


def mean_peak_distance(peak_image, centroids, return_numpy=True):
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image)
    else:
        gpu_peak_image = peaks(peak_image, return_numpy=False)
    peak_distance_gpu = peak_distance(peak_image, centroids, return_numpy=False)
    peak_distance_gpu = cupy.sum(peak_distance_gpu, axis=-1) / cupy.maximum(1,
                                                                            cupy.count_nonzero(gpu_peak_image,
                                                                                               axis=-1))

    del gpu_peak_image
    if return_numpy:
        peak_width_cpu = cupy.asnumpy(peak_distance_gpu)
        del peak_distance_gpu
        return peak_width_cpu
    else:
        return peak_distance_gpu


def direction(peak_image, centroids, number_of_directions=3, return_numpy=True):
    gpu_peak_image = cupy.array(peak_image)
    gpu_centroids = cupy.array(centroids)
    [image_x, image_y, image_z] = gpu_peak_image.shape

    gpu_peak_image = gpu_peak_image.reshape(image_x * image_y, image_z).astype('int8')
    gpu_centroids = gpu_centroids.reshape(image_x * image_y, image_z).astype('float32')
    result_img_gpu = cupy.empty((image_x * image_y, number_of_directions), dtype='float32')
    number_of_peaks = cupy.count_nonzero(gpu_peak_image, axis=-1).astype('int8')

    threads_per_block = 256
    blocks_per_grid = (image_x * image_y + (threads_per_block - 1)) // threads_per_block
    _direction[blocks_per_grid, threads_per_block](gpu_peak_image, gpu_centroids, number_of_peaks, result_img_gpu)
    cuda.synchronize()
    del number_of_peaks

    result_img_gpu = cupy.asarray(result_img_gpu.reshape((image_x, image_y, number_of_directions)))

    if peak_image is None:
        del gpu_peak_image

    if return_numpy:
        result_img_cpu = cupy.asnumpy(result_img_gpu)
        del result_img_gpu
        return result_img_cpu
    else:
        return result_img_gpu


def centroid_correction(image, peak_image, low_prominence=TARGET_PROMINENCE, high_prominence=None, return_numpy=True):
    gpu_image = cupy.array(image, dtype='float32')
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image, dtype='uint8')
    else:
        gpu_peak_image = peaks(gpu_image, return_numpy=False).astype('uint8')
    if low_prominence is None:
        low_prominence = -cupy.inf
    if high_prominence is None:
        high_prominence = -cupy.inf

    [image_x, image_y, image_z] = gpu_image.shape
    gpu_image = gpu_image.reshape(image_x * image_y, image_z)
    gpu_peak_image = gpu_peak_image.reshape(image_x * image_y, image_z)

    gpu_reverse_image = -1 * gpu_image
    gpu_reverse_peaks = peaks(gpu_reverse_image, return_numpy=False).astype('uint8')
    gpu_reverse_prominence = cupy.empty(gpu_image.shape, dtype='float32')
    threads_per_block = 256
    blocks_per_grid = (image_x * image_y + (threads_per_block - 1)) // threads_per_block
    _prominence[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image, gpu_reverse_prominence)
    cuda.synchronize()
    del gpu_reverse_image

    gpu_reverse_peaks[gpu_reverse_prominence < low_prominence] = False
    gpu_reverse_peaks[gpu_reverse_prominence > high_prominence] = False
    del gpu_reverse_prominence

    gpu_left_bases = cupy.empty(gpu_image.shape, dtype='int8')
    gpu_right_bases = cupy.empty(gpu_image.shape, dtype='int8')
    _centroid_correction_bases[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image,
                                                                   gpu_reverse_peaks, gpu_left_bases, gpu_right_bases)
    cuda.synchronize()
    del gpu_reverse_peaks

    # Centroid calculation based on left_bases and right_bases
    gpu_centroid_peaks = cupy.empty(gpu_image.shape, dtype='float32')
    _centroid[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image, gpu_left_bases,
                                                  gpu_right_bases, gpu_centroid_peaks)
    cuda.synchronize()
    if peak_image is None:
        del gpu_peak_image
    del gpu_right_bases
    del gpu_left_bases

    gpu_centroid_peaks = gpu_centroid_peaks.reshape((image_x, image_y, image_z))

    if return_numpy:
        result_img_cpu = cupy.asnumpy(gpu_centroid_peaks)
        del gpu_centroid_peaks
        return result_img_cpu
    else:
        return gpu_centroid_peaks
