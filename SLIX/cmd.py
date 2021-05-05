#!/usr/bin/env python3
import SLIX.io
from SLIX import toolbox, io, preparation
if toolbox.gpu_available:
    import cupy

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, SUPPRESS
import numpy
from matplotlib import pyplot as plt
import os
import tifffile
import tqdm


def create_argument_parser_full_image():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Creation of feature set from '
                                        'scattering image.',
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
                          help='Save 3D images in addition to 2D mean images '
                               'which include more detailed information'
                          ' but will need a lot more disk space.')
    optional.add_argument('--with_mask',
                          action='store_true',
                          help='Use mask to try to remove some of the '
                               'background')
    optional.add_argument('--mask_threshold',
                          type=float,
                          default=10,
                          help='Value for filtering background noise when '
                               'calculating masks. Higher values might result '
                               'in the removal of some of the gray matter in '
                               'the mask but will remove the background '
                               'more effectively.')
    optional.add_argument('--thinout',
                          type=int,
                          default=1,
                          help='Average every NxN pixels in the SLI image '
                               'stack and run the evaluation on the resulting '
                               '(downsampled) images.')
    optional.add_argument('--smoothing',
                          type=str,
                          nargs="*",
                          help='Apply smoothing for each line profile for '
                               'noisy images. Recommended for measurements'
                               ' with less than 5 degree between each image.'
                               'Available options: "fourier" or "savgol"'
                               '. The parameters of those algorithms can be '
                               'set with additional parameters. For example'
                               ' --smoothing fourier 10 20 or '
                               ' --smoothing savgol 45 3')
    optional.add_argument('--disable_gpu',
                          action='store_false',
                          help='Use the CPU in combination with Numba instead '
                               'of the GPU variant. This is only recommended '
                               'if your GPU is significantly slower than your '
                               'CPU.')
    optional.add_argument('--prominence_threshold',
                          type=float,
                          default=0.08,
                          help='Change the threshold for prominent peaks. Peaks'
                               ' with lower prominences will not be used'
                               ' for further evaluation. Only recommended for'
                               ' experienced users!')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )
    # Parameters to select which images will be generated
    image = parser.add_argument_group('output choice (none = all except '
                                      'optional)')
    image.add_argument('--direction',
                       action='store_true',
                       help='Add crossing directions (dir_1, dir_2, dir_3)'
                       )
    image.add_argument('--peaks',
                       action='store_true',
                       help='Add number of peaks below prominence and above '
                            'prominence')
    image.add_argument('--peakprominence',
                       action='store_true',
                       help='Add average peak prominence for each pixel')
    image.add_argument('--peakwidth',
                       action='store_true',
                       help='Add average width of all peaks detected')
    image.add_argument('--peakdistance',
                       action='store_true',
                       help='Add distance between two peaks if two peaks are '
                            'detected')
    image.add_argument('--unit_vectors',
                       action='store_true',
                       help='Write unit vector images from direction')
    image.add_argument('--optional',
                       action='store_true',
                       help='Adds Max/Min/Non Crossing Direction to the output'
                            ' images.')
    image.add_argument('--no_centroids',
                       action='store_false',
                       help='Disable centroid calculation. Not recommended.')
    # Return generated parser
    return parser


def create_argument_parser_line_profile():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Creation of feature set from '
                                        'scattering image.',
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
    optional.add_argument('--prominence_threshold',
                          type=float,
                          default=0.08,
                          help='Change the threshold for prominent peaks. '
                               'Peaks with lower prominences will not be used'
                               ' for further evaluation. (Default: 8%% of'
                               ' total signal amplitude.) Only recommended for'
                               ' experienced users!')
    optional.add_argument('--with_plots',
                          action='store_true',
                          help='Generates plots (png-files) showing the SLI '
                               'profiles and the determined peak positions '
                               '(orange dots: before correction; '
                               'green crosses: after correction).')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )
    # Parameters to select which images will be generated
    image = parser.add_argument_group('output choice (none = all except '
                                      'optional)')
    image.add_argument('--direction',
                       action='store_true',
                       help='Add crossing directions (dir_1, dir_2, dir_3)'
                       )
    image.add_argument('--peaks',
                       action='store_true',
                       help='Add number of peaks below prominence and above '
                            'prominence')
    image.add_argument('--peakprominence',
                       action='store_true',
                       help='Add average peak prominence for each pixel')
    image.add_argument('--peakwidth',
                       action='store_true',
                       help='Add average width of all peaks detected')
    image.add_argument('--peakdistance',
                       action='store_true',
                       help='Add distance between two peaks if two peaks are '
                            'detected')
    image.add_argument('--optional',
                       action='store_true',
                       help='Adds Max/Min/Non Crossing Direction to the output'
                            ' images.')
    image.add_argument('--no_centroids',
                       action='store_false',
                       help='Disable centroid calculation. Not recommended.')
    # Return generated parser
    return parser


def create_argument_parser_visualization():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Creation of feature set from '
                                        'scattering image.',
                            add_help=False
                            )
    # Required parameters
    parser.add_argument('-i',
                          '--input',
                          nargs='*',
                          help='Input direction (.tiff) images. '
                               'Please put all directions in the right order.',
                          required=True)
    parser.add_argument('-o',
                          '--output',
                          help='Output folder where images will be saved to.',
                          required=True)
    parser.add_argument('--fom', action="store_true",
                        help="Write approximate fiber orientation map from"
                             " direction images.")
    parser.add_argument('--vector', type=str,
                        help="Write vector orientation map from direction"
                             " images. Please add the corresponding measure"
                             "ment image for the background.")
    parser.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )

    # Return generated parser
    return parser

def main_full_image():
    parser = create_argument_parser_full_image()
    arguments = parser.parse_args()
    args = vars(arguments)

    DIRECTION = True
    PEAKS = True
    PEAKWIDTH = True
    PEAKPROMINENCE = True
    PEAKDISTANCE = True

    if args['direction'] or args['peaks'] or args['peakprominence'] or\
            args['peakwidth'] or args['peakdistance']:
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

    number_of_param_maps = numpy.count_nonzero([DIRECTION,
                                                PEAKS,
                                                PEAKPROMINENCE,
                                                PEAKWIDTH,
                                                PEAKDISTANCE,
                                                OPTIONAL,
                                                args['smoothing'] is not None,
                                                args['with_mask'],
                                                not args['no_centroids']]) + 1
    tqdm_paths = tqdm.tqdm(paths)
    tqdm_step = tqdm.tqdm(total=number_of_param_maps)
    for path in tqdm_paths:
        if os.path.isdir(path):
            while path[-1] == "/":
                path = path[:-1]
            filename_without_extension = path.split("/")[-1]
        else:
            filename_without_extension = \
                os.path.splitext(os.path.basename(path))[0]
        output_path_name = args['output'] + '/' + filename_without_extension
        tqdm_paths.set_description(filename_without_extension)

        tqdm_step.set_description('Reading image')
        image = io.imread(path)
        if args['thinout'] > 1:
            image = preparation.thin_out(image, args['thinout'],
                                         strategy='average')
            io.imwrite(output_path_name + '_image.tiff', image)
        tqdm_step.update(1)

        if args['smoothing']:
            tqdm_step.set_description('Applying smoothing')

            algorithm = args['smoothing'][0]
            if algorithm == "fourier":
                low_percentage = 10
                if len(args['smoothing']) > 1:
                    low_percentage = int(args['smoothing'][1])

                high_percentage = 25
                if len(args['smoothing']) > 2:
                    high_percentage = int(args['smoothing'][2])

                image = preparation.low_pass_fourier_smoothing(image,
                                                               low_percentage,
                                                               high_percentage)
            elif algorithm == "savgol":
                window_length = 45
                if len(args['smoothing']) > 1:
                    window_length = int(args['smoothing'][1])

                poly_order = 2
                if len(args['smoothing']) > 2:
                    poly_order = int(args['smoothing'][2])

                image = preparation.savitzky_golay_smoothing(image,
                                                             window_length,
                                                             poly_order)

            else:
                print("Unknown option. Please use either "
                      "'fourier' or 'savgol'!")

            tqdm_step.update(1)
            io.imwrite(output_path_name+'_'+algorithm+'_smoothed.tiff', image)

        if toolbox.gpu_available:
            image = cupy.array(image)

        if args['with_mask']:
            tqdm_step.set_description('Creating mask')
            mask = toolbox.background_mask(image, args['mask_threshold'],
                                           use_gpu=toolbox.gpu_available)
            image[mask, :] = 0
            io.imwrite(output_path_name + '_background_mask.tiff', mask)
            tqdm_step.update(1)

        tqdm_step.set_description('Generating peaks')
        significant_peaks = toolbox.\
            significant_peaks(image,
                              low_prominence=args['prominence_threshold'],
                              use_gpu=toolbox.gpu_available,
                              return_numpy=not toolbox.gpu_available)
        if toolbox.gpu_available:
            significant_peaks_cpu = significant_peaks.get()
        else:
            significant_peaks_cpu = significant_peaks

        if PEAKS:
            peaks = toolbox.peaks(image, use_gpu=toolbox.gpu_available)

            io.imwrite(output_path_name+'_high_prominence_peaks.tiff',
                       numpy.sum(significant_peaks_cpu, axis=-1))
            io.imwrite(output_path_name+'_low_prominence_peaks.tiff',
                       numpy.sum(peaks, axis=-1) -
                       numpy.sum(significant_peaks_cpu, axis=-1))

            if args['detailed']:
                io.imwrite(output_path_name+'_all_peaks_detailed.tiff', peaks)
                io.imwrite(
                    output_path_name+'_high_prominence_peaks_detailed.tiff',
                    significant_peaks_cpu)

        tqdm_step.update(1)

        if PEAKPROMINENCE:
            tqdm_step.set_description('Generating peak prominence')

            io.imwrite(output_path_name+'_peakprominence.tiff',
                       toolbox.
                       mean_peak_prominence(image, significant_peaks,
                                            use_gpu=toolbox.gpu_available))

            if args['detailed']:
                peak_prominence_full = \
                    toolbox.peak_prominence(image,
                                            peak_image=significant_peaks,
                                            kind_of_normalization=1,
                                            use_gpu=toolbox.gpu_available)
                io.imwrite(output_path_name+'_peakprominence_detailed.tiff',
                           peak_prominence_full)
                del peak_prominence_full

            tqdm_step.update(1)

        if PEAKWIDTH:
            tqdm_step.set_description('Generating peak width')

            io.imwrite(output_path_name+'_peakwidth.tiff',
                       toolbox.mean_peak_width(image, significant_peaks,
                                               use_gpu=toolbox.gpu_available))

            if args['detailed']:
                peak_width_full = \
                    toolbox.peak_width(image, significant_peaks,
                                       use_gpu=toolbox.gpu_available)
                io.imwrite(output_path_name+'_peakwidth_detailed.tiff',
                           peak_width_full)
                del peak_width_full

            tqdm_step.update(1)

        if args['no_centroids']:
            tqdm_step.set_description('Generating centroids')

            centroids = toolbox.\
                centroid_correction(image, significant_peaks,
                                    use_gpu=toolbox.gpu_available,
                                    return_numpy=not toolbox.gpu_available)

            if args['detailed']:
                if toolbox.gpu_available:
                    centroids_cpu = centroids.get()
                else:
                    centroids_cpu = centroids
                io.imwrite(output_path_name+'_centroid_correction.tiff',
                           centroids_cpu)
            tqdm_step.update(1)
        else:
            # If no centroids are used, use zeros for all values instead.
            if toolbox.gpu_available:
                centroids = cupy.zeros(image.shape)
            else:
                centroids = numpy.zeros(image.shape)

        if PEAKDISTANCE:
            tqdm_step.set_description('Generating peak distance')

            io.imwrite(output_path_name + '_peakdistance.tiff',
                       toolbox.
                       mean_peak_distance(significant_peaks, centroids,
                                          use_gpu=toolbox.gpu_available))

            if args['detailed']:
                peak_distance_full = toolbox.\
                    peak_distance(significant_peaks, centroids,
                                  use_gpu=toolbox.gpu_available)
                io.imwrite(output_path_name + '_peakdistance_detailed.tiff',
                           peak_distance_full)
                del peak_distance_full

            tqdm_step.update(1)

        if DIRECTION:
            tqdm_step.set_description('Generating direction')
            direction = toolbox.direction(significant_peaks, centroids,
                                          use_gpu=toolbox.gpu_available)
            for dim in range(direction.shape[-1]):
                io.imwrite(output_path_name+'_dir_'+str(dim+1)+'.tiff',
                           direction[:, :, dim])
            tqdm_step.update(1)

        if OPTIONAL:
            tqdm_step.set_description('Generating optional maps')
            if toolbox.gpu_available:
                image = image.get()
            min_img = numpy.min(image, axis=-1)
            io.imwrite(output_path_name + '_min.tiff', min_img)
            del min_img

            max_img = numpy.max(image, axis=-1)
            io.imwrite(output_path_name + '_max.tiff', max_img)
            del max_img

            avg_img = numpy.average(image, axis=-1)
            io.imwrite(output_path_name + '_avg.tiff', avg_img)
            del avg_img

            non_crossing_direction = toolbox.\
                direction(significant_peaks, centroids,
                          number_of_directions=1,
                          use_gpu=toolbox.gpu_available)
            io.imwrite(output_path_name + '_dir.tiff', non_crossing_direction)
            tqdm_step.update(1)

        tqdm_step.reset()
    tqdm_step.close()


def main_line_profile():
    parser = create_argument_parser_line_profile()
    arguments = parser.parse_args()
    args = vars(arguments)

    DIRECTION = True
    PEAKS = True
    PEAKWIDTH = True
    PEAKPROMINENCE = True
    PEAKDISTANCE = True

    if args['direction'] or args['peaks'] or args['peakprominence'] or \
            args['peakwidth'] or args['peakdistance']:
        DIRECTION = args['direction']
        PEAKS = args['peaks']
        PEAKPROMINENCE = args['peakprominence']
        PEAKWIDTH = args['peakwidth']
        PEAKDISTANCE = args['peakdistance']
    OPTIONAL = args['optional']

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
        filename_without_extension = \
            os.path.splitext(os.path.basename(path))[0]
        output_path_name = args['output'] + '/' + filename_without_extension
        tqdm_paths.set_description(filename_without_extension)

        output_string = ""

        image = numpy.fromfile(path, sep='\n')
        image = image[numpy.newaxis, numpy.newaxis, ...]
        significant_peaks = toolbox. \
            significant_peaks(image,
                              low_prominence=args['prominence_threshold'],
                              use_gpu=False)

        if PEAKS:
            peaks = toolbox.peaks(image, use_gpu=False)
            output_string += 'High Prominence Peaks,' + \
                             str(int(numpy.sum(significant_peaks, axis=-1))) \
                             + '\n'
            output_string += 'Low Prominence Peaks,' + \
                             str(int(numpy.sum(peaks, axis=-1) -
                                     numpy.sum(significant_peaks, axis=-1))) \
                             + '\n'

        if PEAKPROMINENCE:
            mean_prominence = toolbox.mean_peak_prominence(image,
                                                           significant_peaks,
                                                           use_gpu=False)

            output_string += 'Mean Prominence,' \
                             + str(float(mean_prominence.flatten())) + '\n'

        if PEAKWIDTH:
            mean_peak_width = toolbox.mean_peak_width(image,
                                                      significant_peaks,
                                                      use_gpu=False)
            output_string += 'Mean peak width,' \
                             + str(float(mean_peak_width)) \
                             + '\n'

        if args['no_centroids']:
            centroids = toolbox. \
                centroid_correction(image, significant_peaks,
                                    use_gpu=False)
        else:
            centroids = numpy.zeros(image.shape)

        if PEAKDISTANCE:
            mean_peak_distance = toolbox.mean_peak_distance(significant_peaks,
                                                            centroids,
                                                            use_gpu=False)
            output_string += 'Mean peak distance,' \
                             + str(float(mean_peak_distance)) + '\n'

        if DIRECTION:
            direction = toolbox.direction(significant_peaks, centroids,
                                          use_gpu=False)
            output_string += 'Direction,' + str(direction.flatten()) + '\n'

        if OPTIONAL:
            min_img = numpy.min(image, axis=-1)
            output_string += 'Min,' + str(float(min_img.flatten())) + '\n'
            del min_img

            max_img = numpy.max(image, axis=-1)
            output_string += 'Max,' + str(float(max_img.flatten())) + '\n'
            del max_img

            avg_img = numpy.average(image, axis=-1)
            output_string += 'Avg,' + str(float(avg_img.flatten())) + '\n'
            del avg_img

        if args['with_plots']:
            image = image.flatten()
            plt.plot(image)
            significant_peaks = numpy.argwhere(significant_peaks.flatten())\
                .flatten()
            centroids = centroids.flatten()[significant_peaks]
            plt.plot(significant_peaks,
                     image[significant_peaks], 'x',
                     label='Peak position')

            centroid_positions = numpy.where(centroids < 0,
                                             # True
                                             image[significant_peaks] -
                                             (image[significant_peaks] -
                                              image[significant_peaks - 1]) *
                                             numpy.abs(centroids),
                                             # False
                                             image[significant_peaks] +
                                             (image[(significant_peaks + 1) %
                                                    len(image)] -
                                              image[significant_peaks]) *
                                             centroids)

            plt.plot(significant_peaks + centroids,
                     centroid_positions, 'o', label='Corrected peak position')
            plt.legend()
            plt.savefig(output_path_name + '.png', dpi=100)
            plt.close()

        with open(output_path_name + '.csv', mode='w') as f:
            f.write(output_string)
            f.flush()


def main_visualize():
    parser = create_argument_parser_visualization()
    arguments = parser.parse_args()
    args = vars(arguments)

    filename_without_extension = \
        os.path.splitext(os.path.basename(args['input'][0]))[0]
    output_path_name = args['output'] + '/' + filename_without_extension

    direction_image = None
    for direction_file in args['input']:
        single_direction_image = SLIX.io.imread(direction_file)
        if direction_image is None:
            direction_image = single_direction_image
        else:
            if len(direction_image.shape) == 2:
                direction_image = numpy.stack((direction_image,
                                               single_direction_image),
                                              axis=-1)
            else:
                direction_image = numpy.concatenate((direction_image,
                                                     single_direction_image
                                                     [:, :, numpy.newaxis]),
                                                    axis=-1)

    if args['fom']:
        rgb_fom = SLIX.visualization.visualize_direction(direction_image)
        rgb_fom = (255 * numpy.moveaxis(rgb_fom, -1, 0)).astype(numpy.uint8)
        print(rgb_fom.dtype)
        tifffile.imwrite(output_path_name+'_fom.tiff', rgb_fom,
                         photometric='rgb')

    if args['vector'] is not None:
        image = SLIX.io.imread(args['vector'])
        UnitX, UnitY = SLIX.toolbox.unit_vectors(direction_image)

        thinout = 20
        alpha = 0.8
        background_threshold = 0.65

        plt.imshow(numpy.max(image, axis=-1), cmap='gray')
        SLIX.visualization.visualize_unit_vectors(UnitX, UnitY,
                                                  thinout=thinout, alpha=alpha,
                                                  background_threshold=
                                                  background_threshold)
        plt.axis('off')
        plt.savefig(output_path_name+'_vector.tiff', dpi=1000)
        plt.clf()
