from SLIX import toolbox
import tifffile
import numpy
import argparse

USE_GPU=True


def create_argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Creation of feature set from scattering image.',
                                     add_help=False
                                     )
    # Required parameters
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i',
                          '--input',
                          nargs='*',
                          help='Input files (.nii or .tiff/.tif).',
                          required=True)
    required.add_argument('-o',
                          '--output',
                          help='Output folder where images will be saved to',
                          required=True)
    # Optional parameters
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--with_mask',
                          action='store_true',
                          help='Use mask to try to remove some of the background')
    optional.add_argument('--mask_threshold',
                          type=int,
                          default=10,
                          help='Value for filtering background noise when calculating masks.'
                               'Higher values might result in the removal of some of the gray matter in the mask'
                               'but will remove the background more effectively.')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    # Computational parameters
    compute = parser.add_argument_group('computational arguments')
    compute.add_argument('--no_gpu',
                         type=int,
                         help='Disable the usage of the GPU and use the CPU implementation instead. Not recommended.',
                         action='store_false')
    # Parameters to select which images will be generated
    image = parser.add_argument_group('output choice (none = all except optional)')
    image.add_argument('--direction',
                       action='store_true',
                       help='Add crossing directions (dir_1, dir_2, dir_3)'
                       )
    image.add_argument('--peaks',
                       action='store_true',
                       help='Add number of peaks below prominence and above prominence')
    image.add_argument('--peakprominence',
                       action='store_true',
                       help='Add average peak prominence for each pixel')
    image.add_argument('--peakwidth',
                       action='store_true',
                       help='Add average width of all peaks detected')
    image.add_argument('--peakdistance',
                       action='store_true',
                       help='Add distance between two peaks if two peaks are detected')
    image.add_argument('--optional',
                       action='store_true',
                       help='Adds Max/Min/Non Crossing Direction to the output images.')
    # Return generated parser
    return parser


if __name__ == "__main__":
    #image = toolbox.read_image('/data/PLI-LAP2/TESTREIHEN/Streumessungen/1_Streumessung/90_Vervet_512/90_Stack.tif')
    image = toolbox.read_image('/home/jreuter/AktuelleArbeit/90_Stack.tif')
    print(image.shape)

    import time

    start_time = time.time()
    peaks = toolbox.peaks(image, use_gpu=USE_GPU)
    #tifffile.imwrite('/tmp/peak_positions.tiff', numpy.swapaxes(peaks, -1, 0))

    peak_prominence_full = toolbox.peak_prominence(image, peak_image=peaks, kind_of_normalization=1, use_gpu=USE_GPU).astype('float32')
    #tifffile.imwrite('/tmp/prominence.tiff', numpy.swapaxes(peak_prominence_full, -1, 0))
    del peak_prominence_full

    peak_prominence_full = toolbox.peak_prominence(image, peak_image=peaks, use_gpu=USE_GPU).astype('float32')
    peaks[peak_prominence_full < 0.08] = False
    peak_prominence_full[peak_prominence_full < 0.08] = 0
    #tifffile.imwrite('/tmp/prominence_filtered.tiff', numpy.swapaxes(peak_prominence_full, -1, 0))
    #tifffile.imwrite('/tmp/peak_positions_filtered.tiff', numpy.swapaxes(peaks, -1, 0))

    peak_width_full = toolbox.peak_width(image, peaks, use_gpu=USE_GPU)
    #tifffile.imwrite('/tmp/peak_width.tiff', numpy.swapaxes(peak_width_full, -1, 0))
    del peak_width_full

    peak_distance_full = toolbox.peak_distance(peaks, use_gpu=USE_GPU)
    #tifffile.imwrite('/tmp/peak_distance.tiff', numpy.swapaxes(peak_distance_full, -1, 0))

    direction = toolbox.direction(peaks, use_gpu=USE_GPU)
    #for dim in range(direction.shape[-1]):
        #tifffile.imwrite('/tmp/direction_'+str(dim)+'.tiff', direction[:, :, dim])

    from SLIX.SLIX_GPU.toolbox import centroid_correction
    centroid_peaks = centroid_correction(image, peaks)
    tifffile.imwrite('/tmp/centroid_peaks.tiff', numpy.swapaxes(centroid_peaks, -1, 0))

    print("--- %s seconds ---" % (time.time() - start_time))
