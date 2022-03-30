from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS

import SLIX.io
from SLIX import toolbox, preparation
from SLIX._logging import get_logger
from matplotlib import pyplot as plt
import os
import tqdm
import numpy
import multiprocessing


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
    optional.add_argument('--prominence_threshold',
                          type=float,
                          default=0.08,
                          help='Change the threshold for prominent peaks. '
                               'Peaks with lower prominences will not be used'
                               ' for further evaluation. (Default: 8%% of'
                               ' total signal amplitude.) Only recommended for'
                               ' experienced users!')
    optional.add_argument('--without_angles',
                          action="store_true",
                          help='Scatterometry measurements typically include '
                               'the measurment angle in their text files. '
                               'Enable this option if you have line profiles '
                               'which do not have angles for each measurement. '
                               'Keep in mind, that the angles will be ignored '
                               'regardless. SLIX will generate the parameters '
                               'based on the number of measurement angles.')
    optional.add_argument('--smoothing',
                          type=str,
                          nargs="*",
                          default="",
                          help='Apply smoothing for each line profile for '
                               'noisy images. Recommended for measurements'
                               ' with less than 5 degree between each image. '
                               'Available options: "fourier" or "savgol"'
                               '. The parameters of those algorithms can be '
                               'set with additional parameters. For example'
                               ' --smoothing fourier 0.25 0.025 or '
                               ' --smoothing savgol 45 3')
    optional.add_argument('--simple',
                          action='store_true',
                          help='Replaces the very detailed output of this '
                               'tool by the average values for '
                               'most of the parameter maps. The line '
                               'profile and filtered line profile will '
                               'still be written completely but all '
                               'other parameter maps are shortened down '
                               'to a single value')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )
    # Return generated parser
    return parser


def read_textfile(path, includes_angles=True):
    profile = numpy.fromfile(path, sep='\t')
    if includes_angles:
        profile = profile[1::2]
    profile = profile[numpy.newaxis, numpy.newaxis, ...]
    return profile


def generate_all_peaks(profile, detailed=False):
    peaks = toolbox.peaks(profile,
                          use_gpu=False)
    if detailed:
        return peaks
    return numpy.sum(peaks)


def generate_significant_peaks(profile, low_prominence, detailed=False):
    significant_peaks = toolbox.significant_peaks(profile,
                                                  low_prominence,
                                                  use_gpu=False)
    if detailed:
        return significant_peaks
    return numpy.sum(significant_peaks)


def generate_direction(significant_peaks, centroids):
    return toolbox.direction(significant_peaks,
                             centroids,
                             use_gpu=False)


def generate_prominence(profile, significant_peaks, detailed=False):
    if detailed:
        return toolbox.peak_prominence(profile,
                                       significant_peaks,
                                       use_gpu=False)
    return toolbox.mean_peak_prominence(profile,
                                        significant_peaks,
                                        use_gpu=False)


def generate_peakwidth(profile, significant_peaks, detailed=False):
    if detailed:
        return toolbox.peak_width(profile,
                                  significant_peaks,
                                  use_gpu=False)
    return toolbox.mean_peak_width(profile,
                                   significant_peaks,
                                   use_gpu=False)


def generate_peakdistance(significant_peaks, centroids, detailed=False):
    if detailed:
        return toolbox.peak_distance(significant_peaks,
                                     centroids,
                                     use_gpu=False)
    return toolbox.mean_peak_distance(significant_peaks,
                                      centroids,
                                      use_gpu=False)


def generate_centroids(profile, significant_peaks, low_prominence):
    return toolbox.centroid_correction(profile,
                                       significant_peaks,
                                       low_prominence,
                                       use_gpu=False)


def create_plot(profile, filtered_profile, significant_peaks, centroids):
    profile = profile.flatten()
    filtered_profile = filtered_profile.flatten()

    profile = (profile - profile.min()) / (profile.max() - profile.min())
    filtered_profile = (filtered_profile - filtered_profile.min()) / \
                       (filtered_profile.max() - filtered_profile.min())

    if not numpy.all(profile == filtered_profile):
        plt.plot(filtered_profile, label='filtered profile')
    plt.plot(profile, label='profile')

    significant_peaks = numpy.argwhere(significant_peaks.flatten()) \
        .flatten()
    centroids = centroids.flatten()[significant_peaks]
    plt.plot(significant_peaks,
             filtered_profile[significant_peaks], 'x',
             label='Peak position')

    centroid_positions = numpy.where(centroids < 0,
                                     # True
                                     filtered_profile[significant_peaks] -
                                     (filtered_profile[significant_peaks] -
                                      filtered_profile[significant_peaks - 1]) *
                                     numpy.abs(centroids),
                                     # False
                                     filtered_profile[significant_peaks] +
                                     (filtered_profile[(significant_peaks + 1) %
                                                       len(filtered_profile)] -
                                      filtered_profile[significant_peaks]) *
                                     centroids)

    plt.plot(significant_peaks + centroids,
             centroid_positions, 'o', label='Corrected peak position')
    plt.legend()


def write_parameter_file(object, output_file):
    with open(output_file + ".csv", 'w') as file:
        for key, value in object.items():
            value = value.flatten()
            file.write(key + ",")
            file.write(",".join([f'{num}' for num in value]))
            file.write("\n")


def generate_filtered_profile(profile, algorithm, first_arg, second_arg):
    if algorithm == "fourier":
        return preparation.low_pass_fourier_smoothing(profile,
                                                      first_arg,
                                                      second_arg)
    elif algorithm == "savgol":
        return preparation.savitzky_golay_smoothing(profile,
                                                    first_arg,
                                                    second_arg)
    else:
        return profile


def subprocess(input_file, detailed, low_prominence, with_angle,
               output_file, algorithm, first_arg, second_arg):
    # Parameters than cannot be created without details and will be shown fully.
    output_parameters = {'profile': read_textfile(input_file, with_angle)}
    output_parameters['filtered'] = generate_filtered_profile(output_parameters['profile'],
                                                              algorithm,
                                                              first_arg,
                                                              second_arg)
    sig_peaks = toolbox.significant_peaks(output_parameters['filtered'],
                                          use_gpu=False)
    output_parameters['centroids'] = generate_centroids(output_parameters['filtered'],
                                                        sig_peaks,
                                                        low_prominence)
    # Parameters than can change their output depending on the detailed parameters
    output_parameters['peaks'] = generate_all_peaks(output_parameters['filtered'],
                                                    detailed)
    significant_peaks = generate_significant_peaks(output_parameters['filtered'],
                                                   low_prominence,
                                                   True)
    output_parameters['significant peaks'] = numpy.sum(significant_peaks) if not detailed else significant_peaks
    output_parameters['prominence'] = generate_prominence(output_parameters['filtered'],
                                                          sig_peaks,
                                                          detailed)
    output_parameters['width'] = generate_peakwidth(output_parameters['filtered'],
                                                    sig_peaks,
                                                    detailed)
    output_parameters['distance'] = generate_peakdistance(sig_peaks,
                                                          output_parameters['centroids'],
                                                          detailed)
    # Direction
    output_parameters['direction'] = generate_direction(sig_peaks,
                                                        output_parameters['centroids'])

    write_parameter_file(output_parameters, output_file)
    create_plot(output_parameters['profile'], output_parameters['filtered'],
                significant_peaks,
                output_parameters['centroids'])
    plt.savefig(output_file + ".png", dpi=300)
    plt.clf()


def main():
    logger = get_logger("SLIXLineplotParameterGenerator")
    parser = create_argument_parser()
    arguments = parser.parse_args()
    args = vars(arguments)

    paths = args['input']
    if not isinstance(paths, list):
        paths = [paths]

    if not SLIX.io.check_output_dir(args['output']):
        exit(1)

    algorithm = ""
    first_val = -1
    second_val = -1
    if args['smoothing']:
        algorithm = args['smoothing'][0]
        if algorithm == "fourier":
            first_val = 0.25
            if len(args['smoothing']) > 1:
                first_val = float(args['smoothing'][1])

            second_val = 0.025
            if len(args['smoothing']) > 2:
                second_val = float(args['smoothing'][2])

        elif algorithm == "savgol":
            first_val = 45
            if len(args['smoothing']) > 1:
                first_val = int(args['smoothing'][1])

            second_val = 2
            if len(args['smoothing']) > 2:
                second_val = int(args['smoothing'][2])

    if len(paths) > 1:
        logger.info('Applying pool workers...')
        args = zip(
            paths,
            [not args['simple'] for _ in paths],
            [args['prominence_threshold'] for _ in paths],
            [not args['without_angles'] for _ in paths],
            [args['output'] + '/' + os.path.splitext(os.path.basename(path))[0] for path in paths],
            [algorithm for _ in paths],
            [first_val for _ in paths],
            [second_val for _ in paths]
        )
        with multiprocessing.Pool(None) as pool:
            pool.starmap(subprocess, args)
    else:
        tqdm_paths = tqdm.tqdm(paths)
        for path in tqdm_paths:
            filename_without_extension = \
                os.path.splitext(os.path.basename(path))[0]
            output_path_name = f'{args["output"]}/{filename_without_extension}'
            tqdm_paths.set_description(filename_without_extension)
            subprocess(path,
                       not args['simple'],
                       args['prominence_threshold'],
                       not args['without_angles'],
                       output_path_name,
                       algorithm, first_val, second_val)


if __name__ == "__main__":
    main()
