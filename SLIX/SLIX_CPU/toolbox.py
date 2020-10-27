import numpy
from SLIX.SLIX_CPU._toolbox import _direction, _prominence, _peakwidth, _peakdistance, \
    _centroid, _centroid_correction_bases, _peak_cleanup, TARGET_PROMINENCE


def peaks(image):
    image = numpy.array(image, dtype=numpy.float32)
    image = normalize(image)
    right = numpy.roll(image, 1, axis=-1) - image
    left = numpy.roll(image, -1, axis=-1) - image

    resulting_image = (left <= 1e-10) & (right <= 1e-10) & numpy.invert(numpy.isclose(image, 0))
    del right
    del left

    if len(image.shape) == 3:
        [image_x, image_y, image_z] = resulting_image.shape
        resulting_image = resulting_image.reshape(image_x * image_y, image_z)
    resulting_image = _peak_cleanup(resulting_image)
    if len(image.shape) == 3:
        resulting_image = resulting_image.reshape(image_x, image_y, image_z)
    return resulting_image.astype('bool')


def num_peaks(image=None, peak_image=None):
    if peak_image is None and image is not None:
        peak_image = peaks(image)
    elif peak_image is not None:
        peak_image = numpy.array(peak_image)
    else:
        raise ValueError('Either image or peak_image has to be defined.')

    return numpy.count_nonzero(peak_image, axis=-1)


def normalize(image, kind_of_normalization=0):
    image = numpy.array(image, dtype=numpy.float32)
    if kind_of_normalization == 0:
        image = (image - image.min(axis=-1)[..., None]) \
                / numpy.maximum(1e-15, image.max(axis=-1)[..., None] - image.min(axis=-1)[..., None])
    else:
        image = image / numpy.maximum(1e-15, numpy.mean(image, axis=-1)[..., None])
    return image


def peak_prominence(image, peak_image=None, kind_of_normalization=0):
    image = numpy.array(image, dtype=numpy.float32)
    if peak_image is None:
        peak_image = peaks(image).astype('uint8')
    else:
        peak_image = numpy.array(peak_image).astype('uint8')
    image = normalize(image, kind_of_normalization)

    [image_x, image_y, image_z] = image.shape

    image = image.reshape(image_x * image_y, image_z)
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')

    result_img = _prominence(image, peak_image)

    result_img = result_img.reshape((image_x, image_y, image_z))
    return result_img


def mean_peak_prominence(image, peak_image=None, kind_of_normalization=0):
    if peak_image is not None:
        peak_image = numpy.array(peak_image).astype('uint8')
    else:
        peak_image = peaks(image).astype('uint8')
    result_img = peak_prominence(image, peak_image, kind_of_normalization)
    result_img = numpy.sum(result_img, axis=-1) / numpy.maximum(1,
                                                                numpy.count_nonzero(peak_image,
                                                                                    axis=-1))
    return result_img


def peak_width(image, peak_image=None, target_height=0.5):
    image = numpy.array(image, dtype='float32')
    if peak_image is not None:
        peak_image = numpy.array(peak_image).astype('uint8')
    else:
        peak_image = peaks(image).astype('uint8')

    [image_x, image_y, image_z] = image.shape

    image = image.reshape(image_x * image_y, image_z)
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')

    prominence = _prominence(image, peak_image)
    result_image = _peakwidth(image, peak_image, prominence, target_height)

    result_image = result_image.reshape((image_x, image_y, image_z))
    result_image = result_image * 360.0 / image_z

    return result_image


def mean_peak_width(image, peak_image=None, target_height=0.5):
    if peak_image is not None:
        peak_image = numpy.array(peak_image).astype('uint8')
    else:
        peak_image = peaks(image).astype('uint8')
    result_img = peak_width(image, peak_image, target_height)
    result_img = numpy.sum(result_img, axis=-1) / numpy.maximum(1, numpy.count_nonzero(peak_image, axis=-1))

    return result_img


def peak_distance(peak_image, centroids):
    peak_image = numpy.array(peak_image).astype('uint8')
    [image_x, image_y, image_z] = peak_image.shape

    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')
    number_of_peaks = numpy.count_nonzero(peak_image, axis=-1).astype('uint8')
    centroids = centroids.reshape(image_x * image_y, image_z).astype('float32')

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
    peak_image = numpy.array(peak_image).astype('uint8')
    [image_x, image_y, image_z] = peak_image.shape

    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')
    centroids = centroids.reshape(image_x * image_y, image_z).astype('float32')
    number_of_peaks = numpy.count_nonzero(peak_image, axis=-1).astype('uint8')

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
    image = image.reshape(image_x * image_y, image_z).astype('float32')
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')

    reverse_image = -1 * image
    # TODO: This is not pretty coding
    reverse_peaks = peaks(reverse_image.reshape((image_x, image_y, image_z)))\
        .astype('uint8')\
        .reshape(image_x * image_y, image_z)
    reverse_prominence = _prominence(reverse_image, reverse_peaks)

    reverse_peaks[reverse_prominence < low_prominence] = False
    reverse_peaks[reverse_prominence > high_prominence] = False

    left_bases, right_bases = _centroid_correction_bases(image, peak_image, reverse_peaks)
    # Centroid calculation based on left_bases and right_bases
    centroid_peaks = _centroid(image, peak_image, left_bases, right_bases)
    centroid_peaks = centroid_peaks.reshape((image_x, image_y, image_z))

    return centroid_peaks


def unit_vectors(direction):
    directions_rad = numpy.deg2rad(direction)
    UnitX = -numpy.sin(0.5 * numpy.pi) * numpy.cos(directions_rad)
    UnitY = numpy.sin(0.5 * numpy.pi) * numpy.sin(directions_rad)

    UnitX[numpy.isclose(direction, -1)] = 0
    UnitY[numpy.isclose(direction, -1)] = 0

    return UnitX, UnitY
