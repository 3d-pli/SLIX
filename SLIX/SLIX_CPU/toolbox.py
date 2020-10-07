import numpy
from SLIX.SLIX_CPU._toolbox import _direction, _prominence, _peakwidth, _peakdistance, \
    _centroid, _centroid_correction_bases, TARGET_PROMINENCE


def peaks(image):
    image = numpy.array(image, dtype=numpy.float32)
    right = numpy.roll(image, 1, axis=-1) - image
    left = numpy.roll(image, -1, axis=-1) - image

    resulting_image = (left <= 0) & (right <= 0) & (image != 0)
    del right
    del left

    return resulting_image


def num_peaks(image):
    image = numpy.array(image, dtype=numpy.float32)
    right = numpy.roll(image, 1, axis=-1) - image
    left = numpy.roll(image, -1, axis=-1) - image
    del image

    resulting_image = (left < 0) & (right <= 0)
    del right
    del left

    resulting_image = numpy.empty((resulting_image.shape[:2]))
    resulting_image[:, :] = numpy.count_nonzero(resulting_image, axis=-1)
    return resulting_image


def normalize(image, kind_of_normalization=0):
    image = numpy.array(image, dtype=numpy.float32)
    if kind_of_normalization == 0:
        image = (image - numpy.min(image, axis=-1)[..., None]) / \
                numpy.maximum(1e-15, numpy.max(image, axis=-1)[..., None] - numpy.min(image, axis=-1)[..., None])
    else:
        image = image / numpy.maximum(1e-15, numpy.mean(image, axis=-1)[..., None])
    return image


def peak_prominence(image, peak_image=None, kind_of_normalization=0):
    image = numpy.array(image, dtype=numpy.float32)
    if peak_image is None:
        peak_image = peaks(image)
    else:
        peak_image = numpy.array(peak_image)
    image = normalize(image, kind_of_normalization)

    [image_x, image_y, image_z] = image.shape

    image = image.reshape(image_x * image_y, image_z)
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('int8')

    result_img = _prominence(image, peak_image)

    result_img = result_img.reshape((image_x, image_y, image_z))
    return result_img


def mean_peak_prominence(image, peak_image=None, kind_of_normalization=0):
    if peak_image is not None:
        peak_image = numpy.array(peak_image)
    else:
        peak_image = peaks(peak_image)
    result_img = peak_prominence(image, peak_image, kind_of_normalization)
    result_img = numpy.sum(result_img, axis=-1) / numpy.maximum(1,
                                                                numpy.count_nonzero(peak_image,
                                                                                    axis=-1))
    return result_img


def peak_width(image, peak_image=None, target_height=0.5):
    image = numpy.array(image, dtype=numpy.float32)
    if peak_image is not None:
        peak_image = numpy.array(peak_image)
    else:
        peak_image = peaks(image)
    [image_x, image_y, image_z] = image.shape

    image = image.reshape(image_x * image_y, image_z)
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('int8')

    prominence = _prominence(image, peak_image)
    result_image = _peakwidth(image, peak_image, prominence, target_height)

    result_image = result_image.reshape((image_x, image_y, image_z))
    result_image = result_image * (360.0 / image_z)

    return result_image


def mean_peak_width(image, peak_image=None, target_height=0.5):
    if peak_image is not None:
        peak_image = numpy.array(peak_image)
    else:
        peak_image = peaks(peak_image)
    result_img = peak_width(image, peak_image, target_height)
    result_img = numpy.sum(result_img, axis=-1) / numpy.maximum(1, numpy.count_nonzero(peak_image, axis=-1))

    return result_img


def peak_distance(peak_image, centroids):
    peak_image = numpy.array(peak_image)
    [image_x, image_y, image_z] = peak_image.shape

    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('int8')
    number_of_peaks = numpy.count_nonzero(peak_image, axis=-1).astype('int8')

    result_img = _peakdistance(peak_image, centroids, number_of_peaks)
    result_img = result_img.reshape((image_x, image_y, image_z))

    return result_img


def mean_peak_distance(peak_image, centroids):
    if peak_image is not None:
        peak_image = numpy.array(peak_image)
    else:
        peak_image = peaks(peak_image)
    result_image = peak_distance(peak_image, centroids)
    result_image = numpy.sum(result_image, axis=-1) / numpy.maximum(1,
                                                                    numpy.count_nonzero(peak_image,
                                                                                        axis=-1))
    return result_image


def direction(peak_image, centroids, number_of_directions=3):
    peak_image = numpy.array(peak_image)
    [image_x, image_y, image_z] = peak_image.shape

    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('int8')
    number_of_peaks = numpy.count_nonzero(peak_image, axis=-1).astype('int8')

    result_img = _direction(peak_image, centroids, number_of_peaks, number_of_directions)
    result_img = result_img.reshape((image_x, image_y, number_of_directions))

    return result_img


def centroid_correction(image, peak_image, low_prominence=TARGET_PROMINENCE, high_prominence=None):
    print('centroid_correction')
    if peak_image is None:
        peak_image = peaks(image).astype('uint8')
    if low_prominence is None:
        low_prominence = -numpy.inf
    if high_prominence is None:
        high_prominence = -numpy.inf

    [image_x, image_y, image_z] = image.shape
    image = image.reshape(image_x * image_y, image_z)
    peak_image = peak_image.reshape(image_x * image_y, image_z)

    reverse_image = -1 * image
    reverse_peaks = peaks(reverse_image).astype('uint8')
    reverse_prominence = _prominence(image, peak_image)

    reverse_peaks[reverse_prominence < low_prominence] = False
    reverse_peaks[reverse_prominence > high_prominence] = False

    left_bases, right_bases = _centroid_correction_bases(image, peak_image, reverse_peaks)

    # Centroid calculation based on left_bases and right_bases
    centroid_peaks = _centroid(image, peak_image, left_bases, right_bases)
    centroid_peaks = centroid_peaks.reshape((image_x, image_y, image_z))

    return centroid_peaks
