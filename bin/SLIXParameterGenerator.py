#!/usr/bin/env python3

from SLIX import toolbox, io
if toolbox.gpu_available:
    import cupy
import numpy
import argparse
import os
import tqdm

DIRECTION = True
PEAKS = True
PEAKWIDTH = True
PEAKPROMINENCE = True
PEAKDISTANCE = True
OPTIONAL = False


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
    optional.add_argument('--detailed',
                          action='store_true',
                          help='Save 3D images in addition to 2D mean images which include more detailed information'
                          ' but will need a lot more disk space.')
    optional.add_argument('--with_mask',
                          action='store_true',
                          help='Use mask to try to remove some of the background')
    optional.add_argument('--mask_threshold',
                          type=float,
                          default=10,
                          help='Value for filtering background noise when calculating masks.'
                               'Higher values might result in the removal of some of the gray matter in the mask'
                               'but will remove the background more effectively.')
    optional.add_argument('--disable_gpu',
                          action='store_false',
                          help='Use the CPU in combination with Numba instead of the GPU variant. This is only'
                               ' recommended if your GPU is significantly slower than your CPU.')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
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
    image.add_argument('--no_centroids',
                       action='store_false',
                       help='Disable centroid calculation. Not recommended.')
    # Return generated parser
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    arguments = parser.parse_args()
    args = vars(arguments)

    if args['direction'] or args['peaks'] or args['peakprominence'] or args['peakwidth'] or args['peakdistance']:
        DIRECTION = args['direction']
        PEAKS = args['peaks']
        PEAKPROMINENCE = args['peakprominence']
        PEAKWIDTH = args['peakwidth']
        PEAKDISTANCE = args['peakdistance']
    OPTIONAL = args['optional']
    if toolbox.gpu_available:
        toolbox.gpu_available = args['disable_gpu']

    print(
        'SLI Feature Generator:\n' +
        'Chosen feature maps:\n' +
        'Direction maps: ' + str(DIRECTION) + '\n' +
        'Peak maps: ' + str(PEAKS) + '\n' +
        'Peak prominence map: ' + str(PEAKPROMINENCE) + '\n' +
        'Peak width map: ' + str(PEAKWIDTH) + '\n' +
        'Peak distance map: ' + str(PEAKDISTANCE) + '\n' +
        'Optional maps: ' + str(OPTIONAL) + '\n'
    )

    paths = args['input']
    if not isinstance(paths, list):
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    tqdm_paths = tqdm.tqdm(paths)
    for path in tqdm_paths:
        folder = os.path.dirname(path)
        filename_without_extension = os.path.splitext(os.path.basename(path))[0]
        output_path_name = args['output'] + '/' + filename_without_extension
        tqdm_paths.set_description(filename_without_extension)

        image = io.imread(path)
        if toolbox.gpu_available:
            image = cupy.array(image)
        if args['with_mask']:
            mask = toolbox.background_mask(image, args['mask_threshold'], use_gpu=toolbox.gpu_available)
            image[mask, :] = 0
            io.imwrite(output_path_name + '_background_mask.tiff', mask)

        significant_peaks = toolbox.significant_peaks(image, use_gpu=toolbox.gpu_available,
                                                      return_numpy=not toolbox.gpu_available)
        if toolbox.gpu_available:
            significant_peaks_cpu = significant_peaks.get()
        else:
            significant_peaks_cpu = significant_peaks

        if PEAKS:
            peaks = toolbox.peaks(image, use_gpu=toolbox.gpu_available)
            if args['detailed']:
                io.imwrite(output_path_name+'_all_peaks_detailed.tiff', peaks)
                io.imwrite(output_path_name+'_high_prominence_peaks_detailed.tiff', significant_peaks_cpu)
            io.imwrite(output_path_name+'_high_prominence_peaks.tiff', numpy.sum(significant_peaks_cpu, axis=-1))
            io.imwrite(output_path_name+'_low_prominence_peaks.tiff',
                       numpy.sum(peaks, axis=-1) - numpy.sum(significant_peaks_cpu, axis=-1))

        if PEAKPROMINENCE:
            if args['detailed']:
                peak_prominence_full = toolbox.peak_prominence(image, peak_image=significant_peaks,
                                                               kind_of_normalization=1, use_gpu=toolbox.gpu_available)
                io.imwrite(output_path_name+'_prominence_detailed.tiff', peak_prominence_full)
                del peak_prominence_full
            io.imwrite(output_path_name+'_prominence.tiff',
                       toolbox.mean_peak_prominence(image, significant_peaks, use_gpu=toolbox.gpu_available))

        if PEAKWIDTH:
            if args['detailed']:
                peak_width_full = toolbox.peak_width(image, significant_peaks, use_gpu=toolbox.gpu_available)
                io.imwrite(output_path_name+'_peakwidth_detailed.tiff', peak_width_full)
                del peak_width_full
            io.imwrite(output_path_name+'_peakwidth.tiff',
                       toolbox.mean_peak_width(image, significant_peaks, use_gpu=toolbox.gpu_available))

        if args['no_centroids']:
            centroids = toolbox.centroid_correction(image, significant_peaks, use_gpu=toolbox.gpu_available,
                                                    return_numpy=not toolbox.gpu_available)
            if args['detailed']:
                if toolbox.gpu_available:
                    centroids_cpu = centroids.get()
                else:
                    centroids_cpu = centroids
                io.imwrite(output_path_name+'_centroid_correction.tiff', centroids_cpu)
        else:
            if toolbox.gpu_available:
                centroids = cupy.zeros(image.shape)
            else:
                centroids = numpy.zeros(image.shape)

        if PEAKDISTANCE:
            if args['detailed']:
                peak_distance_full = toolbox.peak_distance(significant_peaks, centroids, use_gpu=toolbox.gpu_available)
                io.imwrite(output_path_name + '_distance_detailed.tiff', peak_distance_full)
                del peak_distance_full
            io.imwrite(output_path_name + '_distance.tiff',
                       toolbox.mean_peak_distance(significant_peaks, centroids, use_gpu=toolbox.gpu_available))

        if DIRECTION:
            direction = toolbox.direction(significant_peaks, centroids, use_gpu=toolbox.gpu_available)
            for dim in range(direction.shape[-1]):
                io.imwrite(output_path_name+'_direction_'+str(dim+1)+'.tiff', direction[:, :, dim])

        if OPTIONAL:
            min_img = numpy.min(image, axis=-1)
            io.imwrite(output_path_name + '_min.tiff', min_img)
            del min_img

            max_img = numpy.max(image, axis=-1)
            io.imwrite(output_path_name + '_max.tiff', max_img)
            del max_img

            avg_img = numpy.average(image, axis=-1)
            io.imwrite(output_path_name + '_avg.tiff', avg_img)
            del avg_img

            non_crossing_direction = toolbox.direction(significant_peaks, centroids,
                                                       number_of_directions=1, use_gpu=toolbox.gpu_available)
            io.imwrite(output_path_name + '_dir.tiff', non_crossing_direction)
