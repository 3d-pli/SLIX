import Library.ScatterPlotToolbox as toolbox
import numpy
import pandas
import argparse
import os
from matplotlib import pyplot as plt
from PIL import Image

def full_pipeline(PATH, NAME):
    print(PATH)
    roiset = numpy.fromfile(PATH, dtype=numpy.float, sep='\n')
    roiset = numpy.concatenate((roiset, roiset, roiset))

    df = pandas.DataFrame(data=roiset)
    df = df.rolling(window=30).mean()
    roiset_rolled = df.to_numpy()

    z_begin = len(roiset_rolled)//3//2
    z_end = len(roiset_rolled) - z_begin
    roiset_rolled = roiset_rolled[z_begin:z_end]
    roiset_rolled = numpy.swapaxes(roiset_rolled, 1, 0)

    print("Roi finished")
    max_array = toolbox.max_array_from_roiset(roiset_rolled)
    print("Max image finished")
    min_array = toolbox.min_array_from_roiset(roiset_rolled)
    print("Min image finished")
    peak_array = toolbox.peak_array_from_roiset(roiset_rolled)
    print("Peak image finished")
    nc_direction_array = toolbox.non_crossing_direction_array_from_roiset(roiset_rolled)
    print("Non Crossing Direction finished")
    peakwidth_array = toolbox.peakwidth_array_from_roiset(roiset_rolled)
    print("Peakwidth finished")
    direction_array = toolbox.crossing_direction_array_from_roiset(roiset_rolled)
    print("Crossing Directions finished")

    # Generate output parameters for file
    output = 'Max: ' + str(max_array) + '\nMin: ' + str(min_array) + '\nNum_Peaks: ' + str(peak_array) + '\nNon_Crossing_Dir: ' + str(nc_direction_array)\
        + '\nPeakwidth: ' + str(peakwidth_array) + '\nCrossing_Dir: ' + str(direction_array)
    with open(NAME+'_params.txt', 'w') as f:
        f.write(output)
        f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Creation of feature set from scattering image.')
    parser.add_argument('-i', '--input', nargs='*', help=('Input path / files.'))
    parser.add_argument('-o', '--output', help=('Output folder'))
    arguments = parser.parse_args()
    args = vars(arguments)
    
    paths = args['input']
    if not type(paths) is list:
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    for path in paths:
        folder = os.path.dirname(path)
        filename_without_extension = os.path.splitext(os.path.basename(path))[0]
        full_pipeline(path, args['output'] + '/' + filename_without_extension)
