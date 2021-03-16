Module SLIX.visualization
=========================

Functions
---------

    
`downsample(image, kernel_size, background_value=-1, background_threshold=0.5)`
:   Reduce image dimensions of a parameter map by replacing (N x N) pixels by
    their median value for each image. Image pixels with undefined values
    (background) will not be considered for computing the median,
    except when the fraction of background pixels lies above the defined
    threshold.
    
    Parameters
    ----------
    image: 2D or 3D parameter map (single image or image stack) calculated with
     SLIX.toolbox.
    kernel_size: Downsampling parameter N (defines how many image pixels
    (N x N) are replaced by their median value).
    background_value: Background value of the parameter map. This is generally
    -1 but can differ for unit vector maps.
    background_threshold: Fraction of background pixels in the considered
    (N x N) area for which the image pixels are set to background_value.
    If the fraction of background pixels lies below this defined threshold,
    background pixels will not be considered for computing the median.
    
    Returns
    -------
    2D or 3D NumPy array with reduced image dimensions.

    
`visualize_parameter_map(parameter_map, fig=None, ax=None, alpha=1, cmap='viridis', vmin=0, vmax=None, colorbar=True)`
:   This method will create a Matplotlib plot based on imshow to display the
    given parameter map in different colors. The parameter map is plotted to
    the current axis and figure. If neither is given, the method will
    create a new subfigure. To show the results, please use pyplot.show().
    
    Parameters
    ----------
    parameter_map: 2D parameter map calculated with SLIX.toolbox.
    fig: Matplotlib figure. If None, a new subfigure will be created for fig
    and ax.
    ax: Matplotlib axis. If None, a new subfigure will be created for fig
    and ax.
    alpha: Apply alpha to Matplotlib plots to overlay them with some other
    image like the averaged transmitted light intensity.
    cmap: Matplotlib color map which is used for displaying the image.
    vmin: Minimum value in the resulting plot. If any value is below vmin,
    it will be displayed in black.
    vmax: Maximum value in the resulting plot. If any value is above vmax,
    it will be displayed in white.
    colorbar: Boolean value controlling if a color bar will be displayed in
    the current subplot.
    
    Returns
    -------
    The current Matplotlib figure and axis. The image can be shown with
    pyplot.show().

    
`visualize_unit_vectors(UnitX, UnitY, thinout=1, ax=None, alpha=1, background_threshold=0.5)`
:   This method will create a Matplotlib plot based on quiver to represent the
    given unit vectors as colored lines (vector map).
    Parameters like thinout can be used to reduce the computing load. If
    thinout = 1, the resulting vectors might not be visible
    without zooming in significantly. Here, the vectors will only be plotted
    to the current axis. To show the results, please use pyplot.show().
    
    Parameters
    ----------
    UnitX: Unit vector components along the x-axis (3D NumPy array).
    UnitY: Unit vector components along the y-axis (3D NumPy array).
    thinout: Downsampling parameter N (defines how many vectors N x N are
    replaced by one vector using the downsample function).
    Unit vectors will be thinned out using downsampling and thinning in
    combination. This will increase the
    vector size in the resulting image but will also reduce the information
    density. Please use with caution.
    ax: Matplotlib axis. If None, the current context axis will be used.
    alpha: Apply alpha to Matplotlib plots to overlay them with some other
    other image like the averaged transmitted light intensity.
    background_threshold: If the fraction of background pixels (number of
    pixels without vector within N x N pixels) exceeds this threshold,
    the downsampled pixel will not show a vector.
    
    Returns
    -------
    The current Matplotlib axis. The image can be shown with pyplot.show().