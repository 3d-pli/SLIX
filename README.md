[![DOI](https://joss.theoj.org/papers/10.21105/joss.02675/status.svg)](https://doi.org/10.21105/joss.02675)

Documentation wiki: https://3d-pli.github.io/SLIX/

# SLIX &ndash; Scattered Light Imaging ToolboX

![https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/SLIX_Logo.png](https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/SLIX_Logo.png)


<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=3 orderedList=false} -->

<!-- code_chunk_output -->

- [Introduction](#introduction)
  - [SLI Measurement](#sli-measurement)
  - [SLI Profiles](#sli-profiles)
  - [Parameter Maps](#parameter-maps)
- [System recommendations](#system-recommendations)
- [Installation of SLIX](#installation-of-slix)
- [Evaluation of SLI Profiles](#evaluation-of-sli-profiles)
  - [Required Arguments](#required-arguments)
  - [Optional Arguments](#optional-arguments)
  - [Example](#example)
- [Generation of Parameter Maps](#generation-of-parameter-maps)
  - [Required Arguments](#required-arguments-1)
  - [Optional Arguments](#optional-arguments-1)
  - [Example](#example-1)
  - [Resulting Parameter Maps](#resulting-parameter-maps)
- [Visualization of Parameter Maps](#visualization-of-parameter-maps)
- [Tutorial](#tutorial)
- [Performance Metrics](#performance-metrics)
- [Authors](#authors)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

<!-- /code_chunk_output -->


## Introduction 

*Scattered Light Imaging (SLI)* is a novel neuroimaging technique that allows to explore the substructure of nerve fibers, especially in regions with crossing nerve fibers, in whole brain sections with micrometer resolution ([Menzel et al. (2020)](https://arxiv.org/abs/2008.01037)). By illuminating histological brain sections from different angles and measuring the transmitted light under normal incidence, characteristic light intensity profiles (SLI profiles) can be obtained which provide crucial information such as the directions of crossing nerve fibers in each measured image pixel. 

This repository contains the *Scattered Light Imaging ToolboX (SLIX)* &ndash; an open-source Python package that allows a fully automated evaluation of SLI measurements and the generation of different parameter maps. The purpose of SLIX is twofold: First, it allows to transform the raw data of SLI measurements (SLI image stack) to human-readable parameter maps that can be used for further analysis and interpreted by researchers. To this end, SLIX also contains functions to visualize the resulting parameter maps, e.g. as colored vector maps. Second, the results of SLIX can be processed further for use in tractography algorithms. For example, the resulting fiber direction maps can be stored as angles or as unit vectors, which can be used as input for streamline tractography algorithms ([Nolden et al. (2019)](https://doi.org/10.1007/978-3-658-25326-4_17)).

The figure belows shows the different steps, from the SLI measurement to the generation of parameter maps: 

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/figure_Doku.jpg" height="327">

### SLI Measurement
The sample is illuminated from different angles, with constant polar angle <img src="https://render.githubusercontent.com/render/math?math=\theta"> and different equidistant azimuthal angles <img src="https://render.githubusercontent.com/render/math?math=\phi"> (starting on top and rotating clock-wise), see figure (a). A camera behind the sample records an image of the transmitted light under normal incidence for each direction of illumination, yielding a series of images (b). 

The **SLI image stack** is used as input for *SLIX* (.nii or .tiff are accepted). The software assumes that the measurement has been performed with equidistant angles over a full range of 360°. The number of images defines the illumination angles (e.g. when using 24 images as input, the software assumes that the images were recorded with <img src="https://render.githubusercontent.com/render/math?math=\phi"> = 0°,15°,...,345°).

### SLI Profiles
Each pixel in the SLI image stack contains a light intensity profile (**SLI profile** <img src="https://render.githubusercontent.com/render/math?math=I(\phi)">), which is characteristic for the brain tissue structure at this point (see [Menzel et al. (2020)](https://arxiv.org/abs/2008.01037) for more details). With *SLIX*, it is possible to automatically extract and evaluate the SLI profiles for all image pixels. 

The **peak positions** are computed with [scipy.signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html), taking the 360° periodicity of the signal into account. To account for inaccuracies introduced by the discretization of the SLI profile, the determined peak positions are corrected by calculating the geometric center of the peak tips with a height corresponding to 6% of the total signal amplitude. The value of 6% turned out to be the best choice to obtain reliable fiber orientations (see [Menzel et al. (2020)](https://arxiv.org/abs/2008.01037), Appx. B), but can be changed by the user. Figure (c) shows an SLI profile with 15° discretization and the corrected peak positions as vertical lines.

To avoid that peaks caused by noise or details in the fiber structure impair the computed fiber direction angles, only *prominent* peaks are used for further evaluation. The **peak prominence** ([scipy.signal.peak_prominences](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences)) indicates how strongly a peak stands out from the background signal and is defined by the vertical distance between the top of the peak and the higher of the two neighboring minima (see figure (c), in red). If not defined otherwise by the user, peaks with a prominence above 8% of the total signal amplitude (max-min) are considered as prominent peaks. The value of 8% turned out to be the optimal choice for the generation of reliable fiber orientations (best compromise between correctly and wrongly detected peaks for regions with known fiber orientations, see [Menzel et al. (2020)](https://arxiv.org/abs/2008.01037), Appx. A). 

The **peak width** (see figure (c), in dark blue) is determined as the full width of the peak at a height corresponding to the peak height minus half of the peak prominence. 

The in-plane **fiber direction angles** <img src="https://render.githubusercontent.com/render/math?math=\varphi"> are computed from the (corrected) mid positions of prominent peak pairs with a pair-wise distance of (180 +/- 35)°, see figure (c) in green/magenta. The range of possible distances has been limited to ensure that the out-of-plane angles of the nerve fibers are not too large (the direction angles of inclined crossing fibers cannot be reliably determined).

### Parameter Maps
By evaluating the SLI profiles of each image pixel, *SLIX* generates different parameter maps, which provide various information about the investigated brain tissue: 
- The [average map](#average) shows the overall scattering of the tissue; [maximum](#maximum) and [minimum](#minimum) can be used to get an idea of the signal amplitude and the signal-to-noise ratio.
- The [number of non-prominent peaks](#low-prominence-peaks) and the [number of prominent peaks](#high-prominence-peaks) indicate the clarity of the signal (regions with indefinite scattering signals, such as background or regions with a small number of nerve fibers, show a higher number of non-prominent peaks); the [average peak prominence](#average-peak-prominence) indicates the reliability of the peak positions.
- The [average peak width](#average-peak-width) and the [peak distance](#peak-distance) correlate with the out-of-plane angle of the nerve fibers (in-plane fibers show two narrow peaks with a large distance of about 180°).
- The [direction maps](#direction-angles) show the in-plane direction angles of the nerve fibers for up to three different crossing fibers. The fiber directions can be represented by a colored vector map, as described in the [tutorial](#tutorial) below.

With [`SLIXLineplotParameterGenerator`](#evaluation-of-sli-profiles), it is possible to evaluate individual SLI profiles and compute characteristics such as the number of peaks, their positions, and in-plane fiber direction angles. For a given SLI image stack, [`SLIXParameterGenerator`](#generation-of-parameter-maps) computes the desired parameter maps for all image pixels.

## System recommendations
- Operating System: Windows, Linux, MacOS
- Python version: Python 3.6+
- Processor: At least four threads if executed on CPU only
- RAM: 8 GiB or more
- (optional GPU: NVIDIA GPU supported by CUDA 9.0+)

## Installation of SLIX

##### How to clone SLIX (for further work)
```bash
git clone git@github.com:3d-pli/SLIX.git
cd SLIX

# A virtual environment is recommended:
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```

##### How to install SLIX as Python package
```bash
# Install via PyPi
pip install SLIX

# Install after cloning locally
git clone git@github.com:3d-pli/SLIX.git
cd SLIX
pip install .
```

##### Run SLIX locally

```bash
git clone git@github.com:3d-pli/SLIX.git
cd SLIX
python3 SLIXParameterGenerator.py [options]
python3 SLIXLineplotParameterGenerator.py [options]

# alternatively, after installation of SLIX
pip3 install SLIX
SLIXParameterGenerator [options]
SLIXLineplotParameterGenerator [options]
```

## Evaluation of SLI Profiles

`SLIXLineplotParameterGenerator` allows the evaluation of individual SLI profiles (txt-files with a list of intensity values): For each SLI profile, the maximum, minimum, number of prominent peaks, corrected peak positions, fiber direction angles, and average peak prominence are computed and stored in a text file. The user has the option to generate a plot of the SLI profile that shows the profile together with the peak positions before/after correction. The corrected peak positions are determined by calculating the geometric center of the peak tip, improving the accuracy of the determined peak positions in discretized SLI profiles. If not defined otherwise, the peak tip height corresponds to 6% of the total signal amplitude (for derivation, see [Menzel et al. (2020)](https://arxiv.org/abs/2008.01037), Appx. B).

```
SLIXLineplotParameterGenerator -i [INPUT-TXT-FILES] -o [OUTPUT-FOLDER] [[parameters]]
```
### Required Arguments
| Argument      | Function                                                                    |
| ------------------- | --------------------------------------------------------------------------- |
| `-i, --input`  | Input text files, describing the SLI profiles (list of intensity values). |
| `-o, --output` | Output folder used to store the characteristics of the SLI profiles in a txt-file (Max, Min, Num_Peaks, Peak_Pos, Directions, Prominence). Will be created if not existing. |

### Optional Arguments
| Argument      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--smoothing`  | Smoothing of SLI profiles before evaluation. The smoothing is performed using a Savitzky-Golay filter with 45 sampling points and a second order polynomial. (Designed for measurements with <img src="https://render.githubusercontent.com/render/math?math=\Delta\phi"> < 5° steps to reduce the impact of irrelevant details in the fiber structure, cf. orange vs. black curve in Figure 1c in the [paper](https://github.com/3d-pli/SLIX/blob/master/paper/paper.pdf).) |
| `--with_plots` | Generates plots (png-files) showing the SLI profiles and the determined peak positions (orange dots: before correction; green crosses: after correction). |
| `--prominence_threshold` | Change the threshold for prominent peaks. Peaks with lower prominences will not be used for further evaluation. (Default: 8% of total signal amplitude.) Only recommended for experienced users! (default: 0.08) |


### Example
The following example demonstrates the evaluation of two SLI profiles, which can be found in the "examples" folder of the SLIX repository:
```
SLIXLineplotParameterGenerator -i examples/*.txt -o output --with_plots
```
The resulting plot and txt-file are shown below, exemplary for one of the SLI profiles (90-Stack-1647-1234.txt):

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/90-Stack-1647-1234.png" height="327">


```
Max: 119.0
Min: 68.0
Num_Peaks: 4
Peak_Pos: [2.5935516 7.723009 14.294096 20.10292]
Directions: [143.3426513671875 61.305511474609375 -1.0]
Prominence: 0.27323946356773376
```
The plot shows the SLI profile derived from a stack of 24 images. The x-axis displays the number of images, the y-axis the measured light intensity [a.u.]. To detect peaks at the outer boundaries, the profile was extended taking the 360° periodicity of the signal into account; the red line indicates the boundary. The orange dots are the original peak positions, the green crosses indicate the corrected peak positions, taking the discretization of the profile into account.

The txt-file lists the maximum and minimum intensity values in the SLI profile, the number of prominent peaks (i.e. peaks with a prominence above 8% of the total signal amplitude), the corrected peak positions (on the x-axis), the direction angles of the nerve fibers (in degrees, derived from the mid-positions between peak pairs), and the average prominence of the peaks normalized by the average of the signal.

## Generation of Parameter Maps

`SLIXParameterGenerator` allows the generation of different parameter maps from an SLI image stack.

```
SLIXParameterGenerator -i [INPUT-STACK] -o [OUTPUT-FOLDER] [[parameters]]
```
### Required Arguments

| Argument        | Function                                                |
| ---------------- | ------------------------------------------------------- |
| `-i, --input`    | Input file: SLI image stack (as .tif(f) or .nii).      |
| `-o, --output`   | Output folder where resulting parameter maps (.tiff) will be stored. Will be created if not existing. |


### Optional Arguments

| Argument          | Function                                                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--thinout`    | Average every NxN pixels in the SLI image stack and run the evaluation on the resulting (downsampled) images. (Default: N=1) |
| `--with_mask`      | Consider all image pixels with low scattering as background: Pixels for which the maximum intensity value of the SLI profile is below a defined threshold (`--mask_threshold`) are set to zero and will not be further evaluated.                                                                |
| `--correctdir` | Correct the resulting direction angle by a floating point value (in degree). This is useful when the stack or camera was rotated. |
| `--smoothing [args]` | Apply smoothing to the SLI profiles for each image pixel before evaluation. Available options are low pass fourier filtering `fourier` and Savitzky-Golay filtering `savgol`. With both options, you can input up to two numbers to specify the parameters for the smoothing algorithm. With fourier, you are able to choose soft threshold for the fourier filter in percent `0.2 = 20%` and a smoothing multiplier (range `0--1`, higher values mean more smoothing) (e.g. `fourier 0.1 0.02`). With Savitzky-Golay you can choose the window length and the polynomial order (e.g. `savgol 45 2`)|
| `--prominence_threshold` | Change the threshold for prominent peaks. Peaks with lower prominences will not be used for further evaluation. (Default: 8% of total signal amplitude.) Only recommended for experienced users! |
| `--detailed` | Save 3D images in addition to 2D mean images which include more detailed information but will need a lot more disk space. |
| `--disable_gpu` | Use the CPU in combination with Numba instead of the GPU variant. This is only recommended if your GPU is significantly slower than your CPU. |
| `--no_centroids` | Disable centroid calculation for the parameter maps. This is absolutely not recommended and will result in worse parameter maps but can lower the computing time significantly. |
| `--output_type` | Define the output data type of the parameter images. Default = tiff. Supported types: nii, h5, tiff. (default: tiff) |

The arguments listed below determine which parameter maps will be generated from the SLI image stack.  If any such argument (except `–-optional`) is used, no parameter map besides the ones specified will be generated. If none of these arguments is used, all parameter maps except the optional ones will be generated: peakprominence, number of (prominent) peaks, peakwidth, peakdistance, direction angles in crossing regions.

| Argument       | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--peakprominence`| Generate a parameter map (`_peakprominence.tiff`) containing the average prominence ([scipy.signal.peak_prominence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences)) of an SLI profile (image pixel), normalized by the average of the SLI profile. |
| `--peaks`         | Generate two parameter maps (`_low_prominence_peaks.tiff` and `_high_prominence_peaks.tiff`) containing the number of peaks in an SLI profile (image pixel) with a prominence below and above the defined prominence_threshold (Default: 8% of the total signal amplitude). |
| `--peakwidth`     | Generate a parameter map (`_peakwidth.tiff`) containing the average peak width (in degrees) of all prominent peaks in an SLI profile. |
| `--peakdistance`  | Generate a parameter map (`_peakdistance.tiff`) containing the distance between two prominent peaks (in degrees) in an SLI profile. Pixels for which the SLI profile shows more/less than two prominent peaks are set to `-1`. |
| `--direction`     | Generate three parameter maps (`_dir_1.tiff`, `_dir_2.tiff`, `_dir_3.tiff`) indicating up to three in-plane direction angles of (crossing) fibers (in degrees). If any or all direction angles cannot be determined for an image pixel, this pixel is set to `-1` in the respective map.|
| `--unit_vectors`  | Generate unit vectors maps (`.nii`) from direction images. |
| `--optional`      | Generate four additional parameter maps: average value of each SLI profile (`_avg.tiff`), maximum value of each SLI profile (`_max.tiff`), minimum value of each SLI profile (`_min.tiff`), and in-plane direction angles (in degrees) in regions without crossings (`_dir.tiff`). Image pixels for which the SLI profile shows more than two prominent peaks are set to `-1` in the direction map. |

### Example
The following example demonstrates the generation of the parameter maps, for two artificially crossing sections of human optic tracts (left) and the upper left corner of a coronal vervet brain section (right): 

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/Screenshot_Demo1.png" height="327">&nbsp;&nbsp;<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/Screenshot_Demo2.png" height="327">

#### How to run the demo yourself:

##### 1. Download the needed files:

Command line:
```bash
wget https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Human_Brain/optic_tracts_crossing_sections/SLI-human-Sub-01_2xOpticTracts_s0037_30um_SLI_105_Stack_3days_registered.nii
wget https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/Vervet1818_s0512_60um_SLI_090_Stack_1day.nii
```
Links:

[SLI-human-Sub-01_2xOpticTracts_s0037_30um_SLI_105_Stack_3days_registered.nii](https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Human_Brain/optic_tracts_crossing_sections/SLI-human-Sub-01_2xOpticTracts_s0037_30um_SLI_105_Stack_3days_registered.nii)

[Vervet1818_s0512_60um_SLI_090_Stack_1day.nii](https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/Vervet1818_s0512_60um_SLI_090_Stack_1day.nii)

##### 2. Run SLIX:
```bash
SLIXParameterGenerator -i ./SLI-human-Sub-01_2xOpticTracts_s0037_30um_SLI_105_Stack_3days_registered.nii -o . --thinout 5

SLIXParameterGenerator -i ./Vervet1818_s0512_60um_SLI_090_Stack_1day.nii -o . --thinout 10 --direction
```

The execution of both commands should take around one minute max. The resulting parameter maps will be downsampled. To obtain full resolution parameter maps, do not use the `roisize` option. In this case, the computing time will be higher (around 25 times higher for the first example and 100 times higher for the second example). 

To display the resulting parameter maps, you can use e.g. [ImageJ](https://imagej.net/Download) or [Fiji](https://imagej.net/Fiji/Downloads). More examples (SLI image stacks and resulting parameter maps) can be found on the [EBRAINS data repository](https://kg.ebrains.eu/search/instances/Dataset/de08dcb8-28d4-4f8f-b019-ef0f8924d5d4).

### Resulting Parameter Maps

All 12 parameter maps that can be generated with *SLIX* are shown below, exemplary for the coronal vervet brain section used in the above demo (available [here](https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/Vervet1818_s0512_60um_SLI_090_Stack_1day.nii)). In contrast to the above demo, the parameter maps were generated with full resolution. For testing purposes, we suggest to run the evaluation on downsampled images, e.g. with `--thinout 3`, which greatly speeds up the generation of the parameter maps.
```bash
SLIXParameterGenerator -i ./Vervet1818_s0512_60um_SLI_090_Stack_1day.nii -o .
```

##### Average
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/avg.jpg" width="327">

`_average.tiff` shows the average intensity for each SLI profile (image pixel). Regions with high scattering show higher values.

##### Low Prominence Peaks
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/low_prominence_peaks.jpg" width="327">

`_low_prominence_peaks.tiff` shows the number of non-prominent peaks for each image pixel, i.e. peaks that have a prominence below 8% of the total signal amplitude (max-min) of the SLI profile and are not used for further evaluation. For a reliable reconstruction of the direction angles, this number should be small, ideally zero.

##### High Prominence Peaks
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/high_prominence_peaks.jpg" width="327">

`_high_prominence_peaks.tiff` shows the number of prominent peaks for each image pixel, i.e. peaks with a prominence above 8% of the total signal amplitude (max-min) of the SLI profile. The position of these peaks is used to compute the fiber direction angles.

##### Average Peak Prominence
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/peakprominence.jpg" width="327">

`_peakprominence.tiff` shows the average prominence of the peaks for each image pixel, normalized by the average of each profile. The higher the value, the clearer the signal.

##### Average Peak Width
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/peakwidth.jpg" width="327">

`_peakwidth.tiff` shows the average width of all prominent peaks for each image pixel. A small peak width implies that the fiber directions can be precisely determined. Larger peak widths occur for out-of-plane fibers and/or fibers with small crossing angles.

##### Peak Distance
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/peakdistance.jpg" width="327">

`_peakdistance.tiff` shows the distance between two prominent peaks for each image pixel. If an SLI profile contains only one peak, the distance is zero. In regions with crossing nerve fibers, the distance is not defined and the image pixels are set to `-1`. The peak distance is a measure for the out-of-plane angle of the fibers: A peak distance of about 180° implies that the region contains in-plane fibers; the more the fibers point out of the section plane, the smaller the peak distance becomes. For fibers with an inclination angle of about 70° and above, a single broad peak is expected.

##### Direction Angles
The in-plane direction angles are only computed if the SLI profile has one, two, four, or six prominent peaks with a pair-wise distance of (180 +/- 35)°. Otherwise, the image pixel is set to `-1`. The direction angle is computed from the mid position of one peak pair, or (in case of only one peak) from the position of the peak itself. All direction angles are in degrees (with 0° being along the positive x axis, and 90° along the positive y-axis). 

In case of three or five prominent peaks, the peak pairs cannot be safely assigned and are therefore not evaluated. SLI profiles with three peaks could also be caused e.g. by strongly inclined crossing fibers, where a reliable determination of the in-plane fiber directions is not possible. If an SLI profile contains three prominent peaks, it might also be the case that the forth peak lies below the prominence threshold because the signal is not very clear. To account for missing peaks, the user has the possibility to decrease the prominence threshold by setting another value with `--prominence_threshold`. However, this should be done with caution, as this also causes non-significant peaks that are generated by noise or other artifacts to be considered for evaluation and might yield wrong fiber direction angles. Therefore, we strongly recommend to read the derivation in [Menzel et al. (2020)](https://arxiv.org/abs/2008.01037), Appx. A, before adjusting the threshold.

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/dir_1.jpg" width="327">

`_dir_1.tiff` shows the first detected fiber direction angle. 

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/dir_2.jpg" width="327">

`_dir_2.tiff` shows the second detected fiber direction angle (only defined in regions with two or three crossing fibers).

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/dir_3.jpg" width="327">

`_dir_3.tiff` shows the third detected fiber direction angle (only defined in regions with three crossing fibers).

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/dir.jpg" width="327">

`_dir.tiff` shows the fiber direction angle only in regions with one or two prominent peaks, i.e. excluding regions with crossing fibers.

##### Maximum 
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/max.jpg" width="327">

`_max.tiff` shows the maximum of the SLI profile for each image pixel. 

##### Minimum
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/min.jpg" width="327">

`_min.tiff` shows the minimum of the SLI profile for each image pixel. 
To obtain a measure for the signal-to-noise, the difference between maximum and minimum can be divided by the average.

## Visualization of parameter maps
With `SLIXVisualizeParameter` the direction can be visualized easily. Just put the direction files as an input argument and generate either fiber orientation maps
or a visualization of the unit vectors easily.

```bash
SLIXVisualizeParameter -i [INPUT-DIRECTIONS] -o [OUTPUT-FOLDER] {fom, vector}
```

### Required Arguments

| Argument          | Function                                                |
| ----------------- | ------------------------------------------------------- |
| `-i, --input`     | Input file: SLI direction stack (as .tif(f) or .nii).      |
| `-o, --output`    | Output folder where resulting parameter maps (.tiff) will be stored. |

### Subarguments
`SLIXVisualizeParameter` supports both the creation of FOMs of direction images, and the visualization
of unit vectors. Depending on what you want to do, there is an argument which follows the input and output options.
The arguments are either `fom` for the creation of a FOM, or `vector` for the creation of a vector visualization

#### Argument `fom`
| Argument               | Function                                                      |
| ---------------------- | ------------------------------------------------------------- |
| `--output_type`     | Define the output data type of the parameter images. Default = tiff. Supported types: h5, tiff.      |


#### Argument `vector`
| Argument               | Function                                                      |
| ---------------------- | ------------------------------------------------------------- |
| `--slimeasurement`     | SLI measurement used for the generation of the direction. Required.      |
| `--alpha`              | Factor for the vectors which will be used during visualization. A higher value means that the vectors will be more visible. (Value range: 0 -- 1) |
| `--thinout`            | Thin out vectors by an integer value. A thinout of 20 means that both the x-axis and y-axis are thinned by a value of 20. Default = 20 |
| `--scale`              | Increases the scale of the vectors. A higher scale means that the vectors in the resulting image are longer. This can be helpful if many pixels of the input image are empty but you don't want to use the thinout option to see results. If the scale option isn't used, the vectors are scaled by the thinout option.               |
| `--vector_width`       | Change the default vector width shown in the resulting image. This can be useful if only a small number of vectors will be shown (for example when using a large thinout) |
| `--threshold`          | When using the thinout option, you might not want to get a vector for a lonely vector in the base image. This parameter defines a threshold for the allowed percentage of background pixels to be present. If more pixels than the threshold are background pixels, no vector will be shown. (Value range: 0 -- 1) |


### Example
```bash
# Download the image
wget https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/Vervet1818_s0512_60um_SLI_090_Stack_1day.nii

# Generate direction
SLIXParameterGenerator -i Vervet1818_s0512_60um_SLI_090_Stack_1day.nii -o Output/ --direction

# Use direction to create a FOM
SLIXVisualizeParameter -i Output/Vervet1818_s0512_60um_SLI_090_Stack_1day_dir_1.tiff Output/Vervet1818_s0512_60um_SLI_090_Stack_1day_dir_2.tiff Output/Vervet1818_s0512_60um_SLI_090_Stack_1day_dir_3.tiff -o Output fom

# Visualize the unit vectors from the direction
SLIXVisualizeParameter -i Output/Vervet1818_s0512_60um_SLI_090_Stack_1day_dir_1.tiff Output/Vervet1818_s0512_60um_SLI_090_Stack_1day_dir_2.tiff Output/Vervet1818_s0512_60um_SLI_090_Stack_1day_dir_3.tiff -o Output vector --slimeasurement Vervet1818_s0512_60um_SLI_090_Stack_1day.nii
```

### Resulting images
#### FOM
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/fom.jpg" width="654">

The following color bubble is used for the visualization of the orientation map. The color will match the angle of the direction.

<img src="./assets/FOM-Colorbubble-HSV.jpg" width="128">

The fiber orientation map, which is generated by `SLIXVisualizeParameter` presents each pixel of the raw measurement as a four pixel region.
Each pixel in that region will get a color based on the number of directions in the measurement as well as the orientation of each direction.

When only one direction is present, all four pixels will be mapped to the HSV color of that direction.

Two directions will be shown in a cross pattern. An example is shown below.

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/dir_fom_two_dirs.png">

When three directions are present, the first three pixel will have the HSV color of the direction. The fourth pixel will be black (no direction). 

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/dir_fom_three_dirs.png">

#### Unit vector map
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/vector.jpg" width="327">

## Tutorial
The [Jupyter notebook](https://github.com/3d-pli/SLIX/blob/master/examples/Visualization_Example.ipynb) demonstrates how SLIX can be used to analyze SLI measurements and to visualize the results. 
For example, it allows to display the generated parameter maps in different colors, and to show the orientations of (crossing) nerve fibers as colored lines (vector maps) by computing unit vector maps from the direction maps. 
The following vector map has been generated with the function `visualize_unit_vectors`, using `alpha = 0.8` (defining the transparency of the background image), 
`thinout = 20` (i.e. 20 x 20 pixels were evaluated together), 
and `background_threshold = 0.25` (i.e. if more than 25% of the evaluated pixels are `-1`, no vector will be computed). 

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/master/assets/output_unit_vectors.png" height="327">

## Performance Metrics
The actual runtime depends on the complexity of the SLI image stack. Especially the number of images in the stack and the number of image pixels can have a big influence. To test the performance, one SLI image stack from the coronal vervet brain section (containing 24 images with 2469x3272 pixels each) was analyzed by running `benchmark.py`. This script will create all parameter maps (non detailed ones in addition to all detailed parameter maps) without any downsampling. All performance measurements were taken without times for reading and writing files. When utilizing the GPU and parameter maps are necessary for further operations, they are kept on the GPU to reduce processing time. The SLI measurement, high prominence peaks and centroids are therefore calculated only once each iteration and are used throughout the whole benchmark. Each benchmark used an Anaconda environment with Python 3.8.5 and all neccessary packages installed.

| CPU | Operating system | With GPU | Time in seconds for [this](https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/Vervet1818_s0512_60um_SLI_090_Stack_1day.nii) example (8.078.658 pixels) |
| ------------ | --------------------- | ----------------- | --------------------- |
| AMD Ryzen 3700X | Manjaro (Nov 9th 2020) | Disabled | 23.000 ± 1.374 |
| AMD Ryzen 3700X | Manjaro (Nov 9th 2020) | NVIDIA GTX 1070 | 32.013 ± 0.371 |
| AMD Ryzen 3700X | Manjaro (Dez 8th 2020) | NVIDIA RTX 3070 | 22.467 ± 0.370 |
| Intel Core i3-2120 | Ubuntu 18.04 LTS | -N/A- | 86.138 ± 3.57 |
| Intel Core i5-3470 | Ubuntu 20.04 LTS | Disabled | 77.768 ± 6.101 |
| Intel Core i5-3470 | Ubuntu 20.04 LTS | NVIDIA GTX 1070 | 34.169 ± 0.975 |
| Intel Core i5-8350U | Ubuntu 20.10 | -N/A- | 58.945 ± 2.799 |
| Intel Core i7-7820HQ | MacOS Big Sur | -N/A- | 55.709 ± 3.446 |
| 2x Intel Xeon CPU E5-2690 | Ubuntu 18.04 LTS | Disabled | 42.363 ± 3.475 |
| 2x Intel Xeon CPU E5-2690 | Ubuntu 18.04 LTS | NVIDIA GTX 1080 | 27.712 ± 4.052 |

## Authors
- Jan André Reuter
- Miriam Menzel

## References
|                                                                                                                                                                                                                |                                                                                                                                                              |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [![](https://www.fz-juelich.de/SharedDocs/Bilder/INM/INM-1/DE/PLI/PLI-GruppenLogo.png?__blob=thumbnail)](https://www.fz-juelich.de/inm/inm-1/EN/Forschung/Fibre%20Architecture/Fibre%20Architecture_node.html) | [Fiber Architecture - INM1 - Forschungszentrum Jülich](https://www.fz-juelich.de/inm/inm-1/EN/Forschung/Fibre%20Architecture/Fibre%20Architecture_node.html) |
|                                                 [![](https://sos-ch-dk-2.exo.io/public-website-production/img/HBP.png)](https://www.humanbrainproject.eu/en/)                                                  | [Human Brain Project](https://www.humanbrainproject.eu/en/)           

## Acknowledgements
This open source software code was developed in part or in whole in the Human Brain Project, funded from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 785907 and 945539 ("Human Brain Project" SGA2 and SGA3). The project also received funding from the Helmholtz Association port-folio theme "Supercomputing and Modeling for the Human Brain".

## License
This software is released under MIT License
```
Copyright (c) 2020 Forschungszentrum Jülich / Jan André Reuter.
Copyright (c) 2020 Forschungszentrum Jülich / Miriam Menzel.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.python input parameters -i -o

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
