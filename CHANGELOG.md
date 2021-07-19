# Changelog

## v2.3.0
### Added
- Added `SLIXCluster`, a tool which can be used to separate regions based on classification data through SLI measurements. The parameters aren't final yet and might change in the future.
- Added `SLIX.classification` which includes the classification methods of generated parameter maps.
- Added a new parameter to `SLIXVisualizeParameter [...] vector` named `--distribution` which allows to plot all vectors in a selected threshold region. This allows to see which regions have a high confidence in their reported orientation and which regions might not give any information. Please note that the performance for full measurements isn't that high in the current state.
- Added `SLIX.visualization.unit_vector_distribution` which is used for the creating of the image described in the last bullet point.

### Changed
- Restructured some of the hidden methods to other Python files to make the basic package infrastructure easier to read.
- Renamed methods in `SLIX.visualization` to prevent the repetition of `visualize`
- The README.md will now use GitHub asset links instead of the old repository used during the first implementation of SLIX.
- Changed the structure for the command line programs from `_cmd.py` to a package named `_cmd` containing all programs in separate files. Both solutions work but the new solution allows SLIX to scale more easily in the future.

### Fixed
- Fixed a bug in the tests of the visualization images where the Matplotlib figure wasn't cleared as expected. The tests did check the right things though. It only resulted in a problem when adding another test for the new parameter.

## v2.2.0
### Added
- Added SLIXVisualizeParameter to the toolbox which enables visualizing directions either as a fiber orientation map or as vectors seen before in the example Jupyter notebook
- Added a new paragraph to the README.md to explain the new SLIXVisualizeParameter
- Added a new method in visualization.py to generate a fiber orientation map from a direction. 
- Added a `--correctdir` option to SLIXParameterGenerator to accommodate for a camera shift during the measurement which impacts the resulting direction angle.
- Added a `--smoothing` option to SLIXParameterGenerator as there are now two methods to smooth the line profiles of a measurement (Savitzky-Golay and Fourier)
- Added a `--output_type` option to SLIXParameterGenerator. SLIX now generally supports TIFF, HDF5 and NIfTI files. HDF5 files require that the image is locaed at `/Image`
- Added a `--unit_vector` option to write unit vectors (as `.nii`). The functionality was implemented already but there wasn't a parameter for users.
- Added attributemanager.py which handles the newly added attributes in HDF5 files. 
- Added the option to read an entire folder of 2D image files instead of an image stack when following the following regex syntax: `.*_+p[0-9]+_?.*\.(tif{1,2}|jpe*g|nii|h5|png)`
- Added `imwrite_rgb` to io.py for writing fiber orientation maps as TIFF or HDF5.
- .nii.gz files can now be read. While this was technically possible before this change, a if clause prevented the usage of the right library

### Changed
- Added compression to all available data types (.tiff, .nii.gz, .h5). nii files will only be written as a compressed file if you use --output_type .nii.gz. Other data types will be compressed automatically.
- Overhaul of the documentation. The path changed from /doc to /docs. The documentation is now hosted on https://3d-pli.github.io/SLIX/
and isn't solely in the GitHub wiki anymore.
- Changed the datatype for the detected number of peaks from `int` to `uint16` because there shouldn't be more than 65535 peaks in a measurement.
- There is now only a single `_cmd.py` instead of two `cmd_[...].py` which cleans up the package a bit.
- Added documentation to missing methods.
- Changed how the `background_threshold` parameter works. Now, you define a fraction. If the fraction of background pixels lies above this defined threshold,
background pixels will not be considered for computing the median for visualizing a downscaled vector image.
- The CPU and GPU separation of the SLIX toolbox are now protected as users should only use the main methods.
- Changed the error message when the GPU cannot be used due to errors with Numba. Previously the same message shown when CuPy couldn't be initialized was shown.
- Changed the background mask algorithm when using `SLIXParameterGenerator`. Instead of a fix value it is not based on the average image histogram. The parameter remains disabled by default.
- Removed the `--mask_threshold` parameter from `SLIXParameterGenerator` in process of the previously bullet point.

### Fixed
- Fixed a bug where the direction of a line profile with two peaks wasn't generated when the distance between the peaks was outside of 180° +- 35°.
- Fixed a few bugs regarding image orientations when reading / writing different data types
- When importing a module from SLIX, the underlying modules were visible. This is now resolved.
- Fixed a bug which could result in white pixels inside of the visualized direciton.
- Fixed the missing parallelization of the peak generation when using only the GPU. Previously an issue with Numba prevented this. A change in the code structure now allowed to implement parallelization.

## v2.0

### Added
- The parameter `--detailed` got added to SLIXParameterGenerator. When using this parameter, 3 dimensional parameter maps get created which give a more detailed look at the analysis of the underlying data.
- Added a method to thin out the measurement by using either a plain thin out, median thin out or average thin out.
- Added support for reading HDF5 files in the API.

### Changed
- The entire package was overhauled to include support for GPUs and support for Windows. The toolbox will now use Numba and CuPY instead of PyMP.
- General performance improvements were made to ensure that large measurements can be evaluated.
- Restructuring of the whole toolbox. Now instead of toolbox.py and visualization.py there are also preparation.py and io.py.
- The unit vector method is now in toolbox.py instead of visualization.py

### Fixed

## v1.2.2

### Added

### Changed
- This issue fixes some typos in the README.md and tutorial.
- Introduced a troubleshooting section for line profiles with 3/5 peaks.
- Further explanations in README.md, paper.md and tutorial.

### Fixed

## v1.2.1

### Added
Included GitHub Actions workflow to publish the repository on PyPI when creating a tag.

### Changed
The example Jupyter Notebook was expanded to include how you could transfer the generated parameter maps to unit vector maps which could be used for tractography applications.

### Fixed

## v1.2.0

### Added
- Added GitHub workflow for automated testing
- Added CONTRIBUTING.md
- Greatly expanded README.md to include more examples, tutorials and generally more information on how to use the toolbox.
- Added progress bars to most methods so that the user can see how long each task will take approximately
- Added visualization to SLIX. Now parameter maps and unit vectors can be visualized using Matplotlib. A tutorial was added under examples.
- Added API documentation to the repository and in form of markdown documents in /doc

### Changed
- Added two new parameters in SLIXParameterGenerator

### Fixed
- Fixed a bug where the program could crash if the user did use the toolbox and did not set an upper prominence bound
- Fixed a bug where some methods could cause a crash because of a left over parameter that wasn't used anymore
- Fixed a bug where the NUMBER_OF_SAMPLES parameter would not be correctly applied to the sampling algorithm if the number of samples got changed.
- Wrong datatypes when calling the program will now result in an error message instead of a crash because of tifffile
- Fixed a bug where the crossing direction could not be correctly reshaped by the toolbox
- Fixed some smaller bugs in both programs in /bin

## v1.1.1

### Added

### Changed
- Additional updates to the documentation of the package and README.md

### Fixed
- Fixed a bug where the generation of the parameter map for the mean prominence of an image would result in a type error in SLIXParameterGenerator.

## v1.1

### Added

### Changed
- Updates to the documentation and source code
- Both the documentation and source code should now be easier to understand

### Fixed

## v1.0.0
Initial release of SLIX on GitHub

