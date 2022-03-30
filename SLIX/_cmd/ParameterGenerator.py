from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from typing import Optional, Tuple

import SLIX.io
from SLIX import io, toolbox, preparation
from SLIX._logging import get_logger
import os
import glob
import numpy
import tqdm
import re

if toolbox.gpu_available:
    import cupy


def create_argument_parser():
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
    optional.add_argument('--thinout',
                          type=int,
                          default=1,
                          help='Average every NxN pixels in the SLI image '
                               'stack and run the evaluation on the resulting '
                               '(downsampled) images.')
    optional.add_argument('--correctdir',
                          default=0,
                          type=float,
                          help='Correct the resulting direction angle by a'
                               ' floating point value. This is useful when the'
                               ' stack or camera was rotated.')
    optional.add_argument('--smoothing',
                          type=str,
                          nargs=3,
                          help='Apply smoothing for each line profile for '
                               'noisy images. Recommended for measurements'
                               ' with less than 5 degree between each image. '
                               'Available options: "fourier" or "savgol"'
                               '. The parameters of those algorithms can be '
                               'set with additional parameters. For example'
                               ' --smoothing fourier 0.25 0.025 or '
                               ' --smoothing savgol 45 3')
    optional.add_argument('--prominence_threshold',
                          type=float,
                          default=0.08,
                          help='Change the threshold for prominent peaks. Peaks'
                               ' with lower prominences will not be used'
                               ' for further evaluation. Only recommended for'
                               ' experienced users!')
    optional.add_argument('--output_type',
                          required=False,
                          default='tiff',
                          help='Define the output data type of the parameter'
                               ' images. Default = tiff. Supported types:'
                               ' nii, h5, tiff.')
    optional.add_argument('--disable_gpu',
                          action='store_false',
                          help='Use the CPU in combination with Numba instead '
                               'of the GPU variant. This is only recommended '
                               'if your GPU is significantly slower than your '
                               'CPU.')
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
    image.add_argument('--inclination_sign',
                       action='store_true',
                       help='Add the sign of the inclination angle')
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


def get_file_pattern(path):
    regex = io._fileregex
    files_in_folder = glob.glob(path + '/*')
    pattern = None
    for file in files_in_folder:
        if re.match(regex, file) is not None:
            pattern = file
            break
    # Remove folder
    pattern = os.path.splitext(os.path.basename(pattern))[0]
    pattern = re.sub(r'_+p[0-9]+_?', '_', pattern)
    return pattern


def smooth_image(args, image, output_path_name) -> Optional[Tuple[numpy.ndarray, str]]:
    algorithm = args['smoothing'][0]
    if algorithm == "fourier":
        low_percentage = 0.25
        if len(args['smoothing']) > 1:
            low_percentage = float(args['smoothing'][1])

        smoothing_factor = 0.025
        if len(args['smoothing']) > 2:
            smoothing_factor = float(args['smoothing'][2])
        output_path_name = f'{output_path_name}' \
                           f'_{algorithm}_{low_percentage}_' \
                           f'{smoothing_factor}'

        image = preparation.low_pass_fourier_smoothing(image,
                                                       low_percentage,
                                                       smoothing_factor)
    elif algorithm == "savgol":
        window_length = 45
        if len(args['smoothing']) > 1:
            window_length = int(args['smoothing'][1])

        poly_order = 2
        if len(args['smoothing']) > 2:
            poly_order = int(args['smoothing'][2])
        output_path_name = output_path_name + f'_{algorithm}_' \
                                              f'{window_length}_' \
                                              f'{poly_order}'

        image = preparation.savitzky_golay_smoothing(image,
                                                     window_length,
                                                     poly_order)
    else:
        return None
    return image, output_path_name


def main():
    logger = get_logger("SLIXParameterGenerator")
    parser = create_argument_parser()
    arguments = parser.parse_args()
    args = vars(arguments)

    DIRECTION = True
    PEAKS = True
    PEAKWIDTH = True
    PEAKPROMINENCE = True
    PEAKDISTANCE = True
    INCLINATION_SIGN = True
    UNIT_VECTORS = False
    output_data_type = '.' + args['output_type']

    if output_data_type not in ['.nii', '.nii.gz', '.h5', '.tiff', '.tif']:
        logger.error('Output data type is not supported. Please choose a valid '
                     'datatype!')
        exit(1)

    if args['direction'] or args['peaks'] or args['peakprominence'] or \
            args['peakwidth'] or args['peakdistance'] or \
            args['unit_vectors'] or args['inclination_sign']:
        DIRECTION = args['direction']
        PEAKS = args['peaks']
        PEAKPROMINENCE = args['peakprominence']
        PEAKWIDTH = args['peakwidth']
        PEAKDISTANCE = args['peakdistance']
        INCLINATION_SIGN = args['inclination_sign']
        UNIT_VECTORS = args['unit_vectors']
    OPTIONAL = args['optional']
    if toolbox.gpu_available:
        toolbox.gpu_available = args['disable_gpu']

    logger.info(
        f'\nSLI Feature Generator:\n' +
        f'Chosen feature maps:\n' +
        f'Direction maps: {DIRECTION} \n' +
        f'Peak maps: {PEAKS} \n' +
        f'Peak prominence map: {PEAKPROMINENCE} \n' +
        f'Peak width map: {PEAKWIDTH} \n' +
        f'Peak distance map: {PEAKDISTANCE} \n' +
        f'Inclination sign map: {INCLINATION_SIGN} \n' +
        f'Unit vector maps: {UNIT_VECTORS} \n' +
        f'Optional maps: {OPTIONAL} \n'
    )

    paths = args['input']
    if not isinstance(paths, list):
        paths = [paths]

    if not SLIX.io.check_output_dir(args['output']):
        exit(1)

    number_of_param_maps = numpy.count_nonzero([DIRECTION,
                                                PEAKS,
                                                PEAKPROMINENCE,
                                                PEAKWIDTH,
                                                PEAKDISTANCE,
                                                INCLINATION_SIGN,
                                                OPTIONAL,
                                                UNIT_VECTORS,
                                                args['smoothing'] is not None,
                                                args['with_mask'],
                                                not args['no_centroids']]) + 1
    tqdm_paths = tqdm.tqdm(paths)
    tqdm_step = tqdm.tqdm(total=number_of_param_maps)
    for path in tqdm_paths:
        if os.path.isdir(path):
            filename_without_extension = get_file_pattern(path)
        else:
            filename_without_extension = \
                os.path.splitext(os.path.basename(path))[0]
        output_path_name = f'{args["output"]}/{filename_without_extension}'
        tqdm_paths.set_description(filename_without_extension)

        tqdm_step.set_description('Reading image')
        image = io.imread(path)
        while len(image.shape) < 3:
            image = image[numpy.newaxis, ...]

        if os.path.isdir(path):
            io.imwrite(f'{output_path_name}_Stack' + output_data_type, image)

        if args['thinout'] > 1:
            image = preparation.thin_out(image, args['thinout'],
                                         strategy='average')
            output_path_name = f'{output_path_name}_thinout_{args["thinout"]}'
            io.imwrite(output_path_name + output_data_type, image)
        tqdm_step.update(1)

        if args['smoothing']:
            tqdm_step.set_description('Applying smoothing')
            result = smooth_image(args, image, output_path_name)
            if result is None:
                logger.error(f"Unknown smoothing option. "
                             f"Please use either 'fourier' or 'savgol'!")
                exit(1)
            else:
                image = result[0]
                output_path_name = result[1]
                del result

            tqdm_step.update(1)
            io.imwrite(output_path_name + output_data_type, image)

        if toolbox.gpu_available:
            image = cupy.array(image)

        if args['with_mask']:
            tqdm_step.set_description('Creating mask')
            mask = toolbox.background_mask(image, use_gpu=toolbox.gpu_available,
                                           return_numpy=not toolbox.gpu_available)
            image[mask, :] = 0
            if toolbox.gpu_available:
                mask = mask.get()
            io.imwrite(f'{output_path_name}_background_mask'
                       f'{output_data_type}', mask)
            tqdm_step.update(1)

        tqdm_step.set_description('Generating peaks')
        significant_peaks = toolbox. \
            significant_peaks(image,
                              low_prominence=args['prominence_threshold'],
                              use_gpu=toolbox.gpu_available,
                              return_numpy=not toolbox.gpu_available)
        if toolbox.gpu_available:
            significant_peaks_cpu = significant_peaks.get()
        else:
            significant_peaks_cpu = significant_peaks

        if PEAKS:
            peaks = toolbox.peaks(image, use_gpu=toolbox.gpu_available,
                                  return_numpy=True)

            io.imwrite(f'{output_path_name}_high_prominence_peaks'
                       f'{output_data_type}',
                       numpy.sum(significant_peaks_cpu, axis=-1,
                                 dtype=numpy.uint16))
            io.imwrite(f'{output_path_name}_low_prominence_peaks'
                       f'{output_data_type}',
                       numpy.sum(peaks, axis=-1, dtype=numpy.uint16) -
                       numpy.sum(significant_peaks_cpu, axis=-1,
                                 dtype=numpy.uint16))

            if args['detailed']:
                io.imwrite(f'{output_path_name}_all_peaks_detailed'
                           f'{output_data_type}', peaks)
                io.imwrite(
                    f'{output_path_name}_high_prominence_peaks_detailed'
                    f'{output_data_type}',
                    significant_peaks_cpu
                )

            tqdm_step.update(1)

        if PEAKPROMINENCE:
            tqdm_step.set_description('Generating peak prominence')

            io.imwrite(f'{output_path_name}_peakprominence'
                       f'{output_data_type}',
                       toolbox.
                       mean_peak_prominence(image, significant_peaks,
                                            use_gpu=toolbox.gpu_available,
                                            return_numpy=True))

            if args['detailed']:
                peak_prominence_full = \
                    toolbox.peak_prominence(image,
                                            peak_image=significant_peaks,
                                            kind_of_normalization=1,
                                            use_gpu=toolbox.gpu_available,
                                            return_numpy=True)
                io.imwrite(f'{output_path_name}_peakprominence_detailed'
                           f'{output_data_type}',
                           peak_prominence_full
                           )
                del peak_prominence_full

            tqdm_step.update(1)

        if PEAKWIDTH:
            tqdm_step.set_description('Generating peak width')

            io.imwrite(f'{output_path_name}_peakwidth'
                       f'{output_data_type}',
                       toolbox.mean_peak_width(image, significant_peaks,
                                               use_gpu=toolbox.gpu_available,
                                               return_numpy=True))

            if args['detailed']:
                peak_width_full = \
                    toolbox.peak_width(image, significant_peaks,
                                       use_gpu=toolbox.gpu_available,
                                       return_numpy=False)
                io.imwrite(f'{output_path_name}_peakwidth_detailed'
                           f'{output_data_type}',
                           peak_width_full)
                del peak_width_full

            tqdm_step.update(1)

        if args['no_centroids']:
            tqdm_step.set_description('Generating centroids')

            centroids = toolbox. \
                centroid_correction(image, significant_peaks,
                                    use_gpu=toolbox.gpu_available,
                                    return_numpy=not toolbox.gpu_available)

            if args['detailed']:
                if toolbox.gpu_available:
                    centroids_cpu = centroids.get()
                else:
                    centroids_cpu = centroids
                io.imwrite(f'{output_path_name}_centroid_correction'
                           f'{output_data_type}',
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

            io.imwrite(f'{output_path_name}_peakdistance'
                       f'{output_data_type}', toolbox.
                       mean_peak_distance(significant_peaks, centroids,
                                          use_gpu=toolbox.gpu_available,
                                          return_numpy=True))

            if args['detailed']:
                peak_distance_full = toolbox. \
                    peak_distance(significant_peaks, centroids,
                                  use_gpu=toolbox.gpu_available,
                                  return_numpy=True)
                io.imwrite(f'{output_path_name}_peakdistance_detailed'
                           f'{output_data_type}', peak_distance_full)
                del peak_distance_full

            tqdm_step.update(1)

        if DIRECTION or UNIT_VECTORS:
            tqdm_step.set_description('Generating direction')
            direction = toolbox.direction(significant_peaks, centroids,
                                          correction_angle=args['correctdir'],
                                          use_gpu=toolbox.gpu_available,
                                          return_numpy=True)
            if DIRECTION:
                for dim in range(direction.shape[-1]):
                    io.imwrite(f'{output_path_name}_dir_{dim + 1}'
                               f'{output_data_type}',
                               direction[:, :, dim])
                tqdm_step.update(1)

            if UNIT_VECTORS:
                tqdm_step.set_description('Generating unit vectors')
                UnitX, UnitY = toolbox.unit_vectors(direction,
                                                    use_gpu=
                                                    toolbox.gpu_available,
                                                    return_numpy=True)
                UnitZ = numpy.zeros(UnitX.shape)
                for dim in range(UnitX.shape[-1]):
                    io.imwrite(f'{output_path_name}'
                               f'_dir_{dim + 1}_UnitX.nii',
                               UnitX[:, :, dim])
                    io.imwrite(f'{output_path_name}'
                               f'_dir_{dim + 1}_UnitY.nii',
                               UnitY[:, :, dim])
                    io.imwrite(f'{output_path_name}'
                               f'_dir_{dim + 1}_UnitZ.nii',
                               UnitZ[:, :, dim])

                tqdm_step.update(1)

        if INCLINATION_SIGN:
            tqdm_step.set_description('Generating inclination sign')
            inclination_sign = toolbox.inclination_sign(significant_peaks, centroids,
                                                        correction_angle=args['correctdir'],
                                                        use_gpu=toolbox.gpu_available,
                                                        return_numpy=True)
            io.imwrite(f'{output_path_name}_inclination_sign{output_data_type}', inclination_sign)
            del inclination_sign
            tqdm_step.update(1)

        if OPTIONAL:
            tqdm_step.set_description('Generating optional maps')
            if toolbox.gpu_available:
                image = image.get()
            min_img = numpy.min(image, axis=-1)
            io.imwrite(f'{output_path_name}_min{output_data_type}', min_img)
            del min_img

            max_img = numpy.max(image, axis=-1)
            io.imwrite(f'{output_path_name}_max{output_data_type}', max_img)
            del max_img

            avg_img = numpy.average(image, axis=-1)
            io.imwrite(f'{output_path_name}_avg{output_data_type}', avg_img)
            del avg_img

            non_crossing_direction = toolbox. \
                direction(significant_peaks, centroids,
                          number_of_directions=1,
                          correction_angle=args['correctdir'],
                          use_gpu=toolbox.gpu_available,
                          return_numpy=True)
            io.imwrite(f'{output_path_name}_dir{output_data_type}',
                       non_crossing_direction)
            del non_crossing_direction
            tqdm_step.update(1)

        tqdm_step.reset()
    tqdm_step.close()


if __name__ == "__main__":
    main()
