from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from SLIX import toolbox
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


def create_plot(profile, significant_peaks, centroids):
    profile = profile.flatten()
    plt.plot(profile)
    significant_peaks = numpy.argwhere(significant_peaks.flatten()) \
        .flatten()
    centroids = centroids.flatten()[significant_peaks]
    plt.plot(significant_peaks,
             profile[significant_peaks], 'x',
             label='Peak position')

    centroid_positions = numpy.where(centroids < 0,
                                     # True
                                     profile[significant_peaks] -
                                     (profile[significant_peaks] -
                                      profile[significant_peaks - 1]) *
                                     numpy.abs(centroids),
                                     # False
                                     profile[significant_peaks] +
                                     (profile[(significant_peaks + 1) %
                                            len(profile)] -
                                      profile[significant_peaks]) *
                                     centroids)

    plt.plot(significant_peaks + centroids,
             centroid_positions, 'o', label='Corrected peak position')
    plt.legend()


def write_parameter_file(object, output_file):
    with open(output_file+".csv", 'w') as file:
        for key, value in object.items():
            value = value.flatten()
            file.write(key+",")
            file.write(",".join([f'{num}' for num in value]))
            file.write("\n")


def subprocess(input_file, detailed, low_prominence, with_angle, output_file):
    output_parameters = {'profile': read_textfile(input_file, with_angle)}
    output_parameters['peaks'] = generate_all_peaks(output_parameters['profile'], detailed)
    output_parameters['significant peaks'] = generate_significant_peaks(output_parameters['profile'], low_prominence, detailed)
    output_parameters['centroids'] = generate_centroids(output_parameters['profile'],  output_parameters['significant peaks'], low_prominence)
    output_parameters['prominence'] = generate_prominence(output_parameters['profile'],  output_parameters['significant peaks'], detailed)
    output_parameters['width'] = generate_peakwidth(output_parameters['profile'],  output_parameters['significant peaks'], detailed)
    output_parameters['distance'] = generate_peakdistance(output_parameters['significant peaks'], output_parameters['centroids'], detailed)
    output_parameters['direction'] = generate_direction(output_parameters['significant peaks'], output_parameters['centroids'])

    write_parameter_file(output_parameters, output_file)
    create_plot(output_parameters['profile'],  output_parameters['significant peaks'], output_parameters['centroids'])
    plt.savefig(output_file+".png", dpi=300)
    plt.clf()


def main():
    parser = create_argument_parser()
    arguments = parser.parse_args()
    args = vars(arguments)

    paths = args['input']
    if not isinstance(paths, list):
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    if len(paths) > 1:
        print('Applying pool workers...')
        args = zip(
            paths,
            [True for _ in paths],
            [args['prominence_threshold'] for _ in paths],
            [not args['without_angles'] for _ in paths],
            [args['output'] + '/' + os.path.splitext(os.path.basename(path))[0] for path in paths]
        )
        with multiprocessing.Pool(None) as pool:
            pool.starmap(subprocess, args)
    else:
        tqdm_paths = tqdm.tqdm(paths)
        for path in tqdm_paths:
            filename_without_extension = \
                os.path.splitext(os.path.basename(path))[0]
            output_path_name = args['output'] + '/' + filename_without_extension
            tqdm_paths.set_description(filename_without_extension)
            subprocess(path, True, args['prominence_threshold'], not args['without_angles'], output_path_name)


if __name__ == "__main__":
    main()
