Module SLIX.toolbox
===================

Functions
---------

    
`accurate_peak_positions(peak_positions, line_profile, low_prominence=0.08, high_prominence=inf, centroid_calculation=True)`
:   Post-processing method after peaks have been calculated using the 'all_peaks' method. The peak are filtered based
    on their peak prominence. Additionally, peak positions can be corrected by applying centroid corrections based on the
    line profile.
    
    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    centroid_calculation: Use centroid calculation to better determine the peak position regardless of the number of
    measurements / illumination angles used.
    
    Returns
    -------
    NumPy array with the positions of all detected peaks.

    
`all_peaks(line_profile, cut_edges=True)`
:   Detect all peaks from a given line profile in an SLI measurement. Peaks will not be filtered in any way.
    To detect only significant peaks, use the 'peak_positions' method and apply thresholds.
    
    Parameters
    ----------
    line_profile: 1D-NumPy array with all intensity values of a single image pixel in the stack.
    cut_edges: If True, only consider peaks within the second third of all detected peaks.
    
    Returns
    -------
    List with the positions of all detected peaks.

    
`centroid_correction(line_profile, peak_positions, low_prominence=0.08, high_prominence=inf)`
:   Correct peak positions from a line profile by looking at only the peak with a given threshold using a centroid
    calculation. If a minimum is found in the considered interval, this minimum will be used as the limit instead.
    The range for the peak correction is limited by MAX_DISTANCE_FOR_CENTROID_ESTIMATION.
    
    Parameters
    ----------
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    peak_positions: Detected peak positions of the 'all_peaks' method.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    
    Returns
    -------
    NumPy array with the positions of all detected peak positions corrected with the centroid calculation.

    
`create_background_mask(IMAGE, threshold=10)`
:   Creates a background mask by setting all image pixels with low scattering signals to zero. As all background pixels are near zero for all images in the SLI image stack, this method should remove most of the background allowing for better approximations using the
    available features. It is advised to use this function.
    
    Arguments:
        IMAGE: 2D/3D-image containing the z-axis in the last dimension
    
    Keyword Arguments:
        threshold: Threshhold for mask creation (default: {10})
    
    Returns:
        numpy.array: 1D/2D-image which masks the background as True and foreground as False

    
`create_roiset(IMAGE, ROISIZE, extend=True)`
:   Create roi set of the given image by creating an image containing the average value of pixels within the
    specified ROISIZE. The returned image will have twice the size in the third axis as the both halfs will be doubled
    for the peak detection.
    
    Arguments:
        IMAGE: Image containing multiple images in a 3D-stack
        ROISIZE: Size in pixels which are used to create the region of interest image
    
    Returns:
        numpy.array: Image with shape [x/ROISIZE, y/ROISIZE, 2*'number of measurements'] containing the average value
        of the given roi for each image in z-axis.

    
`create_sampling(line_profile, peak_positions, left_bound, right_bound, target_peak_height, number_of_samples=100)`
:   Parameters
    ----------
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    peak_positions: Detected peak positions of the 'all_peaks' method.
    left_bound: Left bound for linear interpolation.
    right_bound: Right bound for linear interpolation.
    target_peak_height: Targeted peak height for centroid calculation.
    number_of_samples: Number of samples used for linear interpolation.
    
    Returns
    -------
    Linear interpolated array, new left bound, new right bound for centroid calculation.

    
`crossing_direction(peak_positions, number_of_measurements)`
:   Calculate up to three direction angles based on the given peak positions. If more than six peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The peak positions are determined by the position of the corresponding peak pairs (i.e. 6 peaks: 1+4, 2+5, 3+6).
    If two peaks are too far away or too near (outside of 180°±35°), the direction angle will be considered as invalid,
    resulting in a direction angle of BACKGROUND_COLOR.
    
    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    number_of_measurements: Number of measurements during a full SLI measurement, i.e. the number of points in the line
    profile.
    
    Returns
    -------
    NumPy array with the shape (3,) containing up to three direction angles. If a direction angle is invalid or missing,
    the array entry will be BACKGROUND_COLOR instead.

    
`crossing_direction_image(roiset, low_prominence=0.08, high_prominence=inf, cut_edges=True)`
:   Calculate up to three direction angles based on the given peak positions. If more than six peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The peak positions are determined by the position of the corresponding peak pairs (i.e. 6 peaks: 1+4, 2+5, 3+6).
    If two peaks are too far away or too near (outside of 180°±35°), the direction angle will be considered as invalid,
    resulting in a direction angle of BACKGROUND_COLOR.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.
    
    Parameters
    ----------
    roiset: Full SLI measurement (image series) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: If True, only consider peaks within the second third of all detected peaks.
    
    Returns
    -------
    NumPy array with the shape (x, 3) containing up to three direction angles. 
    x equals the number of pixels of the SLI image series. If a direction angle is invalid or missing, the array entry
    will be BACKGROUND_COLOR instead.

    
`non_crossing_direction(peak_positions, number_of_measurements)`
:   Calculate one direction angle based on the given peak positions. If more than two peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The direction angle is determined by the mid position between two peaks.
    
    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    number_of_measurements: Number of images in an SLI image stack, i.e. the number of points in the line
    profile.
    
    Returns
    -------
    Floating point value containing the direction angle in degrees.
    If a direction angle is invalid or missing, the returned value will be BACKGROUND_COLOR instead.

    
`non_crossing_direction_image(roiset, low_prominence=0.08, high_prominence=inf, cut_edges=True)`
:   Calculate one direction angle based on the given peak positions. If more than two peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The direction angle is determined by the mid position between two peaks.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in SLIXParameterGenerator.
    
    Parameters
    ----------
    roiset: Full SLI measurement (image series) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: If True, only consider peaks within the second third of all detected peaks.
    
    Returns
    -------
    NumPy array of floating point values containing the direction angle in degree.
    If a direction angle is invalid or missing, the returned value will be BACKGROUND_COLOR instead.

    
`normalize(roi, kind_of_normalization=0)`
:   Normalize given line profile by using a normalization technique based on the kind_of_normalization parameter.
    
    0 : Scale line profile to be between 0 and 1
    1 : Divide line profile through its mean value
    
    Arguments:
        roi: Line profile of a single pixel / region of interest
        kind_of_normalization: Normalization technique which will be used for the calculation
    
    Returns:
        numpy.array -- Normalized line profile of the given roi parameter

    
`num_peaks_image(roiset, low_prominence=0.08, high_prominence=inf, cut_edges=True)`
:   Calculate the number of peaks from each line profile in an SLI image series by detecting all peaks and applying thresholds to
    remove unwanted peaks.
    
    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: If True, only consider peaks within the second third of all detected peaks.
    
    Returns
    -------
    NumPy array where each entry corresponds to the number of detected peaks within the first dimension of the SLI image series.

    
`peakdistance(peak_positions, number_of_measurements)`
:   Calculate the mean peak distance in degrees between two corresponding peaks within a line profile.
    
    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    number_of_measurements: Number of images in the SLI image stack, i.e. the number of points in one
    line profile.
    
    Returns
    -------
    Floating point value containing the mean peak distance of the line profile in degrees.

    
`peakdistance_image(roiset, low_prominence=0.08, high_prominence=inf, cut_edges=True, centroid_calculation=True)`
:   Calculate the mean peak distance in degrees between two corresponding peaks for each line profile in an SLI image
    series.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.
    
    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: If True, only consider peaks within the second third of all detected peaks.
    centroid_calculation: Use centroid calculation to better determine the peak position regardless of the number of
    measurements / illumination angles used.
    
    Returns
    -------
    NumPy array of floating point values containing the mean peak distance of the line profiles in degrees.

    
`peakwidth(peak_positions, line_profile, number_of_measurements)`
:   Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    number_of_measurements: Number of measurements during a full SLI measurement, i.e. the number of points in one line
    profile.
    
    Returns
    -------
    Floating point value containing the mean peak width of the line profile in degrees.

    
`peakwidth_image(roiset, low_prominence=0.08, high_prominence=inf, cut_edges=True)`
:   Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.
    
    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: If True, only consider peaks within the second third of all detected peaks.
    
    Returns
    -------
    NumPy array where each entry corresponds to the mean peak width of the line profile.

    
`prominence(peak_positions, line_profile)`
:   Calculate the mean peak prominence of all given peak positions within a line profile. The line profile will be
    normalized by dividing the line profile through its mean value. Therefore, values above 1 are possible.
    
    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    
    Returns
    -------
    Floating point value containing the mean peak prominence of the line profile in degrees.

    
`prominence_image(roiset, low_prominence=0.08, high_prominence=inf, cut_edges=True)`
:   Calculate the mean peak prominence of all given peak positions for each line profile in an SLI image series. Each
    line profile will be normalized by dividing the line profile through its mean value. Therefore, values above 1 are
    possible.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.
    
    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: If True, only consider peaks within the second third of all detected peaks.
    
    Returns
    -------
    NumPy array where each entry corresponds to the mean peak prominence of the line profile.

    
`read_image(FILEPATH)`
:   Reads image file and returns it.
    Supported file formats: NIfTI, Tiff.
    
    Arguments:
        FILEPATH: Path to image
    
    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size of a single image and z specifies the number
                     of measurements

    
`reshape_array_to_image(image, x, ROISIZE)`
:   Convert array back to image keeping the lower resolution based on the ROISIZE.
    
    Arguments:
        image: Array created by other methods with lower resolution based on ROISIZE
        x: Size of original image in x-dimension
        ROISIZE: Size of the ROI used for evaluating the roiset
    
    Returns:
        numpy.array -- Reshaped image based on the input array

    
`smooth_roiset(roiset, range=45, polynom_order=2)`
:   Applies Savitzky-Golay filter to given roiset and returns the smoothened measurement.
    
    Args:
        roiset: Flattened image with the dimensions [x*y, z] where z equals the number of measurements
        range: Used window length for the Savitzky-Golay filter
        polynom_order: Used polynomial order for the Savitzky-Golay filter
    
    Returns: Line profiles with applied Savitzky-Golay filter and the same shape as the original roi set.