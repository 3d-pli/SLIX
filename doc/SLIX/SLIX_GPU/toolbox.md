Module SLIX.SLIX_GPU.toolbox
============================

Functions
---------

    
`background_mask(image, threshold=10, return_numpy=True)`
:   Creates a background mask by setting all image pixels with low scattering
    signals to zero. As all background pixels are near zero for all images in
    the SLI image stack, this method should remove most of the background
    allowing for better approximations using the available features.
    It is advised to use this function.
    
    Parameters
    ----------
    image: Complete SLI measurement image stack as a 2D/3D Numpy array
    threshold: Threshhold for mask creation (default: 10)
    return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
    array will be returned.
    
    Returns
    -------
    numpy.array: 1D/2D-image which masks the background as True and foreground
    as False

    
`centroid_correction(image, peak_image, low_prominence=0.08, high_prominence=None, return_numpy=True)`
:   Correct peak positions from a line profile by looking at only the peak with
    a given threshold using a centroid calculation. If a minimum is found in
    the considered interval, this minimum will be used as the limit instead.
    The range for the peak correction is limited by
    MAX_DISTANCE_FOR_CENTROID_ESTIMATION.
    
    Parameters
    ----------
    image: Original line profile used to detect all peaks. This array will be
    further analyzed to better determine the peak positions.
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    NumPy array with the positions of all detected peak positions corrected
    with the centroid calculation.

    
`direction(peak_image, centroids, number_of_directions=3, return_numpy=True)`
:   Calculate up to `number_of_directions` direction angles based on the given
    peak positions.
    If more than `number_of_directions*2` peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a
    direction angle of BACKGROUND_COLOR.
    The peak positions are determined by the position of the corresponding peak
    pairs (i.e. 6 peaks: 1+4, 2+5, 3+6).
    If two peaks are too far away or too near (outside of 180°±35°), the
    direction angle will be considered as invalid, resulting in a direction
    angle of BACKGROUND_COLOR.
    
    Parameters
    ----------
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack.
    centroids: Centroids resulting from `centroid_correction` for more accurate
    results.
    number_of_directions: Number of directions which shall be generated.
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    NumPy array with the shape (x, y, `number_of_directions`) containing up to
    `number_of_directions` direction angles. x equals the number of pixels of
    the SLI image series. If a direction angle is invalid or missing, the array
    entry will be BACKGROUND_COLOR instead.

    
`mean_peak_distance(peak_image, centroids, return_numpy=True)`
:   Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image
    series.
    
    Parameters
    ----------
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    centroids: Use centroid calculation to better determine the peak position
    regardless of the number of
    measurements / illumination angles used.
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    NumPy array of floating point values containing the mean peak distance of
    the line profiles in degrees.

    
`mean_peak_prominence(image, peak_image=None, kind_of_normalization=0, return_numpy=True)`
:   Calculate the mean peak prominence of all given peak positions within a
    line profile. The line profile will be normalized by dividing the line
    profile through its mean value. Therefore, values above 1 are possible.
    
    Parameters
    ----------
    image: Original line profile used to detect all peaks. This array will be
    further
           analyzed to better determine the peak positions.
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    kind_of_normalization: Normalize given line profile by using a
    normalization technique based on the kind_of_normalization parameter.
       0 : Scale line profile to be between 0 and 1
       1 : Divide line profile through its mean value
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    Floating point value containing the mean peak prominence of the line
    profile in degrees.

    
`mean_peak_width(image, peak_image=None, target_height=0.5, return_numpy=True)`
:   Calculate the mean peak width of all given peak positions within a line
    profile.
    
    Parameters
    ----------
    image: Original line profile used to detect all peaks. This array will be
    further analyzed to better determine the peak positions.
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    target_height: Relative peak height in relation to the prominence of the
    given peak.
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    NumPy array where each entry corresponds to the mean peak width of the
    line profile. The values are in degree.

    
`normalize(image, kind_of_normalization=0, return_numpy=True)`
:   Normalize given line profile by using a normalization technique based on
    the kind_of_normalization parameter.
    
    0 : Scale line profile to be between 0 and 1
    1 : Divide line profile through its mean value
    
    Arguments:
        image: Full SLI measurement (series of images) which is
               prepared for the pipeline using the SLIX toolbox methods.
        kind_of_normalization: Normalization technique which will be used for
        the calculation
        return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns:
        numpy.array -- Image where each pixel is normalized by the last axis
        of the image

    
`num_peaks(image=None, peak_image=None, return_numpy=True)`
:   Calculate the number of peaks from each line profile in an SLI image series
    by detecting all peaks and applying thresholds to remove unwanted peaks.
    
    Parameters
    ----------
    image: Full SLI measurement (series of images) which is prepared for the
    pipeline using the SLIX toolbox methods.
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    Array where each entry corresponds to the number of detected peaks within
    the first dimension of the SLI image series.

    
`peak_distance(peak_image, centroids, return_numpy=True)`
:   Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image series.
    
    Parameters
    ----------
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    centroids: Use centroid calculation to better determine the peak position
    regardless of the number of measurements / illumination angles used.
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    NumPy array of floating point values containing the peak distance of the
    line profiles in degrees in their respective peak position. The first peak
    of each peak pair will show the distance between peak_1 and peak_2 while
    the second peak will show 360 - (peak_2 - peak_1).

    
`peak_prominence(image, peak_image=None, kind_of_normalization=0, return_numpy=True)`
:   Calculate the peak prominence of all given peak positions within a line
    profile. The line profile will be normalized by dividing the line profile
    through its mean value. Therefore, values above 1 are possible.
    
    Parameters
    ----------
    image: Original line profile used to detect all peaks. This array will be
    further analyzed to better determine the peak positions.
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    kind_of_normalization: Normalize given line profile by using a
    normalization technique based on the kind_of_normalization parameter.
       0 : Scale line profile to be between 0 and 1
       1 : Divide line profile through its mean value
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    Floating point value containing the mean peak prominence of the line
    profile in degrees.

    
`peak_width(image, peak_image=None, target_height=0.5, return_numpy=True)`
:   Calculate the peak width of all given peak positions within a line profile.
    
    Parameters
    ----------
    image: Original line profile used to detect all peaks. This array will be
    further analyzed to better determine the peak positions.
    peak_image: Boolean NumPy array specifying the peak positions in the full
    SLI stack
    target_height: Relative peak height in relation to the prominence of the
    peak
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    NumPy array where each entry corresponds to the peak width of the line
    profile. The values are in degree.

    
`peaks(image, return_numpy=True)`
:   Detect all peaks from a full SLI measurement. Peaks will not be filtered
    in any way. To detect only significant peaks, filter the peaks by using the
    prominence as a threshold.
    
    Parameters
    ----------
    image: Complete SLI measurement image stack as a 2D/3D Numpy array
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    2D/3D boolean image containing masking the peaks with `True`

    
`unit_vectors(direction, return_numpy=True)`
:   Calculate the unit vectors (UnitX, UnitY) from a given direction angle.
    
    Parameters
    ----------
    direction: 3D NumPy array - direction angles in degrees
    return_numpy:  Specifies if a CuPy or Numpy array will be returned.
    
    Returns
    -------
    UnitX, UnitY: 3D NumPy array, 3D NumPy array
        x- and y-vector component in arrays