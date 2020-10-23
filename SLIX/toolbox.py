try:
    try:
        import cupy
        cupy.empty(0)
        from SLIX.SLIX_GPU import toolbox as gpu_toolbox
    except cupy.cuda.runtime.CUDARuntimeError:
        pass
except ModuleNotFoundError:
    pass
from SLIX.SLIX_CPU import toolbox as cpu_toolbox


def peaks(image, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.peaks(image, return_numpy)
    else:
        return cpu_toolbox.peaks(image)


def direction(peak_image, centroids, number_of_directions=3, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.direction(peak_image, centroids, number_of_directions, return_numpy)
    else:
        return cpu_toolbox.direction(peak_image, number_of_directions)


def peak_distance(peak_image, centroids, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.peak_distance(peak_image, centroids, return_numpy)
    else:
        return cpu_toolbox.peak_distance(peak_image)


def mean_peak_distance(peak_image, centroids, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.mean_peak_distance(peak_image, centroids, return_numpy)
    else:
        return cpu_toolbox.mean_peak_distance(peak_image)


def peak_prominence(image, peak_image=None, kind_of_normalization=0, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.peak_prominence(image, peak_image, kind_of_normalization, return_numpy)
    else:
        return cpu_toolbox.peak_prominence(image, peak_image, kind_of_normalization)


def mean_peak_prominence(image, peak_image=None, kind_of_normalization=0, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.mean_peak_prominence(image, peak_image, kind_of_normalization, return_numpy)
    else:
        return cpu_toolbox.mean_peak_prominence(image, peak_image, kind_of_normalization)


def peak_width(image, peak_image=None, target_height=0.5, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.peak_width(image, peak_image, target_height, return_numpy=return_numpy)
    else:
        return cpu_toolbox.peak_width(image, peak_image, target_height)


def mean_peak_width(image, peak_image=None, target_height=0.5, use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.mean_peak_width(image, peak_image, target_height, return_numpy=return_numpy)
    else:
        return cpu_toolbox.mean_peak_width(image, peak_image, target_height)


def centroid_correction(image, peak_image, low_prominence=cpu_toolbox.TARGET_PROMINENCE, high_prominence=None,
                        use_gpu=True, return_numpy=True):
    if use_gpu:
        return gpu_toolbox.centroid_correction(image, peak_image, low_prominence, high_prominence, return_numpy)
    else:
        return cpu_toolbox.centroid_correction(image, peak_image, low_prominence, high_prominence)


def unit_vectors(direction, use_gpu=True, return_numpy=True):
    pass