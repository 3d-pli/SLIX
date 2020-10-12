# SLIX - Scattered Light Imaging ToolboX

![https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/SLIX_Logo.png](https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/SLIX_Logo.png)

# Introduction 

*Scattered Light Imaging (SLI)* is a novel neuroimaging technique that resolves the substructure of nerve fibers, especially in regions with crossing nerve fibers, in whole brain sections with micrometer resolution ([Menzel et al. (2020b)](https://arxiv.org/abs/2008.01037)). The measurement principle was first introduced by [Menzel et al. (2020a)](http://dx.doi.org/10.1103/PhysRevX.10.021002): A histological brain section is illuminated under oblique incidence of light from different angles. The measurement is performed with a constant polar angle of illumination and different azimuthal angles (*directions of illumination* <img src="https://render.githubusercontent.com/render/math?math=\phi">). For each direction of illumination, the intensity of light that is transmitted under normal incidence is recorded. The resulting images form a series (*SLI image stack*) in which each image pixel contains a light intensity profile (*SLI profile* <img src="https://render.githubusercontent.com/render/math?math=I(\phi)">).

This repository contains the *Scattered Light Imaging ToolboX (SLIX)* - an open-source Python package that allows a fully automated evaluation of SLI measurements and the generation of different parameter maps. For a given SLI image stack, `SLIXParameterGenerator` is able to compute up to 12 (8 + 4 optional) parameter maps providing different information about the measured brain tissue sample, e.g. the individual in-plane direction angles of the nerve fibers for regions with up to three crossing nerve fiber bundles. Individual parameter maps can be selected through command line parameters. With `SLIXLineplotParameterGenerator`, it is possible to use existing SLI profiles (txt-files with a list of intensity values) as input and compute the corresponding parameter set (txt-file) for each SLI profile, which contains the number of peaks, the position (<img src="https://render.githubusercontent.com/render/math?math=\phi">) of the maximum and minimum, and the peak positions.

## How to clone SLIX
```
git clone git@github.com:3d-pli/SLIX.git
cd SLIX

# A virtual environment is recommended:
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```

## How to install SLIX as Python package
```
git clone git@github.com:3d-pli/SLIX.git
cd SLIX

python3 setup.py install
```

## `SLIXParameterGenerator`

Main tool to create desired parameter maps from an SLI image stack.

```
SLIXParameterGenerator -i [INPUT-STACK] -o [OUTPUT-FOLDER] [[parameters]]
```

### Required parameters

| Parameter        | Function                                                |
| ---------------- | ------------------------------------------------------- |
| `-i, --input`    | Input file: SLI image stack (as .tif(f) or .nii).      |
| `-o, --output`   | Output folder where resulting parameter maps will be stored. Will be created if not existing. |

### Optional parameters

| Parameter          | Function                                                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-r, --roisize`    | Average every NxN pixels of the SLI images and run the evaluation on the resulting (downsampled) images. Later on, the images will be upscaled to match the input file dimensions. (Default: -r 1) |
| `--with_mask`      | Remove the background based on the maximum value of each image pixel. May include gray matter.                                                                |
| `--mask_threshold` | Set the threshold for the background mask. Pixels for which the maximum intensity value of the SLI profile is below the threshold, will be considered as background. Higher values might remove the background better but will also include more of the gray matter. (Default = 10) |
| `--num_procs`      | Run the program with the selected number of processes. (Default = either 16 threads or the maximum number of threads available.)                                  |
| `--with_smoothing` | Apply smoothing to the SLI profiles for each image pixel before evaluation. The smoothing is performed using a Savitzky-Golay filter with 45 sampling points and a second order polynomial. (Designed for measurements with <img src="https://render.githubusercontent.com/render/math?math=\Delta\phi"> = 5° steps.)                                                                                     |

### Output
Additional parameters that determine which parameter maps will be generated from the SLI image stack. If no parameter is used, the following parameter maps will be generated: peaks, direction, peakwidth, peakprominence, peakdistance. If any parameter (except `–-optional`) is used, no parameter map besides the ones specified will be generated.

| Parameter      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--peaks`         | Generate two parameter maps (`_low_prominence_peaks.tiff` and `_high_prominence_peaks.tiff`) containing the number of peaks of the SLI profiles for a prominence below and above 8% of the maximum signal amplitude. |
| `--direction`     | Generate three parameter maps (`_dir_1.tiff`, `_dir_2.tiff`, `_dir_3.tiff`) indicating up to three in-plane direction angles of (crossing) fibers. If any or all direction angles cannot be determined for an image pixel, this pixel is set to `-1` in the respective map. |
| `--peakwidth`     | Generate a parameter map (`_peakwidth.tiff`) containing the average peak width of all peaks in an SLI profile (image pixel) with a prominence above 8% of the maximum signal amplitude. |
| `--peakprominence`| Generate a parameter map (`_peakprominence.tiff`) containing the average prominence ([scipy.signal.peak_prominence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences)) of an SLI profile (image pixel), normalized by the average of the SLI profile. |
| `--peakdistance`  | Generate a parameter map (`_peakdistance.tiff`) containing the distance between two peaks of an SLI profile (image pixel) with a prominence above 8%. All other pixels are set to `-1`. |
| `--optional`      | Generate additional parameter maps: maximum value of each SLI profile (`_max.tiff`), minimum value of each SLI profile (`_min.tiff`), in-plane direction angles in regions without crossings (`_dir.tiff`). |
### Example
![](https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/demo.gif)

### Resulting parameter maps

The parameter maps shown below have been generated from the SLI image stack of a coronal vervet monkey brain section (upper left corner), which is available [here](https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/Vervet1818_s0512_60um_SLI_090_Stack_1day.nii). More data (SLI image stacks and parameter maps) are available on the [EBRAINS data repository](https://doi.org/10.25493/XA4S-XXZ). The parameter maps were generated with full resolution, i.e. without downsampling (`--roisize 1`). For testing on a less powerful computer, it is recommended to run the evaluation on downsampled images, e.g. with `--roisize 10`, which greatly speeds up the generation of the parameter maps.
 
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/dir_1.jpg" width="327"><img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/high_prominence_peaks.jpg" width="327">

Direction(`_dir_1.tiff`)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;High Prominence Peaks
 
<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/low_prominence_peaks.jpg" width="327"><img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/peakprominence.jpg" width="327">

Low Prominence Peaks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Peakprominence

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/peakwidth.jpg" width="327"><img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/peakdistance.jpg" width="327">

Peakwidth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Peakdistance

<img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/max.jpg" width="327"><img src="https://jugit.fz-juelich.de/j.reuter/slix/-/raw/assets/min.jpg" width="327">

Maximum&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Minimum

## Additional tools

### `SLIXLineplotParameterGenerator`
Individual evaluation of SLI profiles (txt-files with list of intensity values): max/min, number of peaks, peak positions

```
SLIXLineplotParameterGenerator -i [INPUT-TXT-FILES] -o [OUTPUT-FOLDER] [[parameters]]
```

| Parameter      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `-i, --input`  | Input text files, describing the SLI profiles (list of intensity values). |
| `-o, --output` | Output folder used to store the FeatureSet (txt-file containing the characteristics of the SLI profiles): max, min, num_peaks, peak_positions. Will be created if not existing. |
| `--smoothing`  | Smoothing of SLI profiles before evaluation. |
| `--with_plots` | Generates png-files showing the SLI profiles and the determined peak positions (with/without correction). |
| `--target_peak_height` | Change peak height used for correcting the peak positions (Default: 6% of peak height). Only recommended for experienced users! |

## Performance metrics
All performance measurements were taken without times for reading and writing files. Real measurements were used for the analysis. The actual runtime depends on the complexity of the measurements. Especially the number of rotation angles per pixel can have a big influence. 

For this test four different measurements were analyzed by running the program with different core counts and averaging the number of pixels evaluated per second. In total, 32.314.632 line profiles were evaluated for this performance evaluation.

Our testing system consists of an AMD Ryzen 3700X at 3.60 - 4.425 GHz paired with 16 GiB DDR4-3000 memory. Other system configurations might take longer or shorter to compute the parameter maps.

| Thread count | Average pixels per second | Time in minutes for [this](https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000048_ScatteredLightImaging_pub/Vervet_Brain/coronal_sections/Vervet1818_s0512_60um_SLI_090_Stack_1day.nii) example (8.078.658 pixels) |
| ------------ | --------------------- | --------------------- |
| 4 | 4416 | 29:41 |
| 6 | 6232 | 21:36 |
| 8 | 7654 | 17:34 |
| 10 | 8326 | 16:10 |
| 12 | 9384 | 13:54 |
| 16 | 11300 | 11:54 |

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

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
