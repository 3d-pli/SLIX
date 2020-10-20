Module SLIX.visualization
=========================

Functions
---------

    
`downsample(image, sample_size, background_value=-1, background_threshold=0.5)`
:   Reduce image dimensions of a parameter map by applying a median filter in each image in the z-axis.
    The background will not be considered for the median filter except when the magnitude of it is above the given
    threshold.
    
    Parameters
    ----------
    image: 2D or 3D parameter map calculated with SLIX.toolbox.
    sample_size: Down sampling parameter.
    background_value: Background value in this parameter map. This is generally -1 but can differ for unit vectors.
    background_threshold: If magnitude of the background values exceeds this value, the downsampled image will have
                          background_value as it's resulting pixel value.
    
    Returns
    -------
    2D or 3D Numpy array with reduced image dimensions

    
`unit_vectors(directions)`
:   Calculates the unit vector from direction and inclination
    
    Parameters
    ----------
    directions : 3d-array
        direction in radian
    
    Returns
    -------
    res : 3d-array, 3d-array
        x- and y-vector component in arrays

    
`visualize_parameter_map(parameter_map, fig=None, ax=None, alpha=1, cmap='viridis', vmin=0, vmax=None, colorbar=True)`
:   This method will create a Matplotlib plot based on imshow to represent the given parameter map.
    Here, the parameter map be plotted to the current axis and figure. If none is applied, the method will create a new
    subfigure. To show the results, please use pyplot.show().
    
    Parameters
    ----------
    parameter_map: 2D parameter map calculated with SLIX.toolbox.
    fig: Matplotlib figure. If None a new subfigure will be created for fig and ax.
    ax: Matplotlib axis. If None a new subfigure will be created for fig and ax.
    alpha: Apply alpha to Matplotlib plots to overlay them with some other plots like the original measurement.
    cmap: Matplotlib color map which is used for the shown image.
    vmin: Minimum value in the resulting plot. If any value is below vmin it will be displayed in black.
    vmax: Maximum value in the resulting plot. If any value is above vmax it will be displayed in white.
    colorbar: Boolean value controlling if a color bar will be displayed in the current subplot.
    
    Returns
    -------
    The current Matplotlib figure and axis. The image can be shown with pyplot.show().

    
`visualize_unit_vectors(UnitX, UnitY, thinout=1, ax=None, alpha=1, background_threshold=0.5)`
:   This method will create a Matplotlib plot based on quiver to represent the given unit vectors in a more readable
    way. Parameters like thinout can be used to reduce the computing load. If thinout = 1 the resulting vectors might
    not be visible without zooming in significantly.
    Here, the vectors will only be plotted to the current axis. To show the results, please use pyplot.show().
    
    Parameters
    ----------
    UnitX: Unit vectors in x-axis
    UnitY: Unit vectors in y-axis
    thinout: Unit vectors will be thinned out using downsampling and thinning in combination. This will increase the
             vector size in the resulting image but will also reduce the information density. Please use with caution.
    ax: Matplotlib axis. If none, the current context axis will be used.
    alpha: Apply alpha to Matplotlib plots to overlay them with some other plots like the original measurement.
    background_threshold: If magnitude of the background values exceeds this value, the downsampled image will have
                          background_value as it's resulting pixel value.
    
    Returns
    -------
    The current Matplotlib axis. The image can be shown with pyplot.show().