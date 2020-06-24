# SLIX -- Scattered Light Imaging Toolbox

# Introduction 

Simulation studies by [Menzel. et. al  (2020)](http://dx.doi.org/10.1103/PhysRevX.10.021002) have shown that individual nerve fiber orientations in regions with crossing fibers are revealed through scattering patterns. In more recent studies [Menzel and Pereira (2020)](http://dx.doi.org/10.17632/dp496jpd7h.3) used coherent Fourier scatterometry to measure the scattering patterns for real brain tissue samples. The results validate simulation studies. However, only small brain sections with a limited resolution can be measured with this technique.

Scattered Light Imaging presents a neuroimaging technique that is able to resolve the substructure of nerve fiber crossings in whole brain sections with micrometer resolution. It was first introduced in [Menzel. et. al  (2020)](http://dx.doi.org/10.1103/PhysRevX.10.021002). A brain section is illuminated under oblique incidence of light from different angles. While this technique can measure only a limited number of scattering angles (along a circle in the fullscattering pattern), it can be used to study the scattering behavior of a wholebrain section at once and with micrometer resolution.

This repository represents the toolbox used for the feature generation of Scattered Light Imaging measurements. Given a scattering measurement `GenFeatureSet.py` is able to calculate up to 11 (8 + 3 optional) feature maps describing the scattering behaviour of the given measurement. Individual feature maps can be selected through command line parameters. Both `GenFeatureSet.py` and `GenLinePlotFeatureSet.py` show how the underlying toolbox can be used.

## How to install the toolbox
```
git clone git@github.com:Thyre/SLIX.git
cd SLIX

# If a virtual environment is needed:
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```

## `GenFeatureSet.py`

Main tool to create desired feature maps from SLI image stacks.


### Required Parameters

| Parameter        | Function                                                |
| ---------------- | ------------------------------------------------------- |
| `-i, --input`    | Input file(s) (SLI-Image stack as .tif(f) or .nii)      |
| `-o, --output`   | Output folder where images will be stored. Will be created if not existing. |

### Optional parameters

| Parameter          | Function                                                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-r, --roisize`    | Average every NxN pixels of the image and run calculation on resulting image. Image will be upscaled later to match the input file dimensions.      |
| `--with_mask`      | Remove background based on the maximum value of each pixel. May include gray matter.                                                                |
| `--mask_threshold` | Set threshold for the background mask. Higher values might remove the background better but will also include more of the gray matter. Default = 10 |
| `--num_procs`      | Run tool with selected number of processes. Default = either 16 threads or the maximum number of threads available                                  |
| `--with_smoothing` | Apply smoothing to the image stack. Designed for 5° measurement                                                                                     |
| `--without_centroid_calculation`| Disable centroid calculation (Integral over plateau of peak). Not recommended! |

### Output
Additional parameters to change which images will be generated. If no parameter is used
direction/number of peaks/peakprominence/peakwidth/peakdistance are generated. If a
parameter besides `–-optional` is used no map besides the ones specified will be generated.

| Parameter      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--peaks`         | Generate two feature maps with the number of peaks for a prominence below / above 0.08 named `_low_prominence_peaks.tiff` and `_high_prominence_peaks.tiff` |
| `--direction`     | Generate three direction maps (`_dir_1.tiff`, `_dir_2.tiff`, `_dir_3.tiff`) where up to three different directions are shown in fiber crossing regions. Non-evaluable pixels are shown with a value of `-1`|
| `--peakwidth`     | Generate a feature map with the average peak width of all peaks with a prominence above 0.08 |
| `--peakprominence`| Generate a feature map average prominence of line profile, normalized by mean |
| `--peakdistance`  | Generate a feature map with the distance between two peaks for each pixel where two high prominence peaks are detected. Each other pixel is `-1`|
| `--optional`      | Generate additional feature maps (max values, min values, direction map without crossing regions)|

### Example
![](https://raw.githubusercontent.com/Thyre/SLIX/assets/demo.gif?token=ADRHMEJ6GV4BDOEOSZE43QS67RLRU)

### Resulting feature maps
 
<img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/dir_1.jpg?token=ADRHMEJLLCNNCFKDNVMVUF267RLVY" width="327"><img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/high_prominence_peaks.jpg?token=ADRHMEJHT6FA7OH52LJFASS67RL3W" width="327">

Direction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; High Prominence Peaks
 
<img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/low_prominence_peaks.jpg?token=ADRHMENJC4X7TW75LZKQNPC67RL42" width="327"><img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/peakprominence.jpg?token=ADRHMEJRY33RLYVFIX2NNKK67RL6E" width="327">

Low Prominence Peaks &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Peakprominence

<img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/peakwidth.jpg?token=ADRHMEIXBQIDLWA5QYZZPIC67RL7I" width="327"><img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/peakdistance.jpg?token=ADRHMEJ2VA23CX3L3ZSEOSK67RMAM" width="327">

Peakwidth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Peakdistance

<img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/max.jpg?token=ADRHMEOKKMB6N4I7CU6UDFK67RMEY" width="327"><img src="https://raw.githubusercontent.com/Thyre/SLIX/assets/min.jpg?token=ADRHMEPT5PG5F4JNTBR6UNC67RMF4" width="327">

Maximum &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Minimum


## Additional tools

### `GenLinePlotFeatureSet.py`
Evaluation of line profiles : max/min, number of peaks, peak positions

```
./GenLinePlotFeatureSet.py [parameters]
```

| Parameter      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `-i, --input`  | Input text files (generated with LineProfiles.py), describing a line profile (Delft/SLI)                          |
| `-o, --output` | Output folder (used to store FeatureSet: max, min, num_peaks, peak_positions) Will be created if not existing. |
| `--smoothing`  | For scatterometry line profiles |
| `--with_plots` | Generates png-files with line profiles and peak positions (with/without centroid) |
| `--target_peak_height` | Change peak height used for centroid calculation |

## Authors
- Jan André Reuter
- Miriam Menzel

## License
This project is licensed under the GPLv3 License - see the [LICENSE](https://github.com/Thyre/SLIX/blob/master/LICENSE) file for details
