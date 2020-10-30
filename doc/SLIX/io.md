Module SLIX.io
==============

Functions
---------

    
`hdf5_read(filepath, dataset)`
:   Reads image file and returns it.
    
    Arguments:
        filepath: Path to image
        dataset: Path to dataset in HDF5 file
    
    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size of a single image and z specifies the number
                     of measurements

    
`hdf5_write(filepath, dataset, data, mode='w')`
:   Write generated image to given filepath.
    
    Arguments:
        filepath: Path to image
        dataset: Path to dataset in HDF5 file
        data: Data which will be written to the disk
        mode: Mode with which the HDF5 file will be created. Please change 'w' to 'a' if appending to a already
        exisiting HDF5 file
    Returns:
        None

    
`imread(filepath)`
:   Reads image file and returns it.
    Supported file formats: NIfTI, Tiff.
    
    Arguments:
        filepath: Path to image
    
    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size of a single image and z specifies the number
                     of measurements

    
`imwrite(filepath, data)`
:   Write generated image to given filepath.
    Supported file formats: NIfTI, Tiff.
    Other file formats are only indirectly supported and might result in errors.
    
    Arguments:
        filepath: Path to image
        data: Data which will be written to the disk
    
    Returns:
        None