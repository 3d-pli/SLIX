import argparse
from argparse import RawTextHelpFormatter
import sys

class ArgumentParser():
    def __init__(self, args):
        parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, description='Extracting parameters from transmittance and retardation images for creation of inclination images.')

        parser.add_argument('-i', '--input', nargs='*', help=('Input path / files for evaluation.'))
        parser.add_argument('-o', '--output', help=('Output folder to store evaluated parameters. If None is given the parameters are printed to the console instead.'))
        parser.add_argument('--createInclinationImage', dest='inclinationImage', help=('Writes inclination image of estimated parameters if this option is enabled.'), action='store_true')
        parser.add_argument('--createAreaImage', dest='areaImage', help=('Writes colorized images showing the white and gray matter area estimated by this program.'), action='store_true')
        parser.add_argument('--withoutFitgm', dest='fitgm', help=('Use only with --createInclinationImage. Inclination image will be written with fitgm = fitwm'), action='store_false')
        parser.add_argument('--withoutTransmittanceWeighting', dest='imic', help=('Use only with --createInclinationImage. Inclination image will be written which won\'t use lm or lc'), action='store_false')
        parser.add_argument('--saturationCorrection', dest='saturation', help=('Use only with --createInclinationImage. Correct saturation with one of the following methods:\nlocalCorrection,thresh_white[5e-3],thresh_gray[5e-3]\nglobalCorrection,thresh_global[5e-3]\ninpaintLocalCorrection,thresh_white[5e-3],thresh_gray[5e-3],thresh_pixelwise[1e-3]'))
        parser.add_argument('--bins', dest='bins', help=('Change the number of bins for evaluating the parameters. Default = 128'))
        parser.add_argument('--gwSeperatorValue', dest='gwSeperator', help=('Skips the Gray White seperator estimation and uses the given value instead.'))
        parser.set_defaults(inclinationImage=False)
        parser.set_defaults(areaImage=False)
        parser.set_defaults(fitgm=True)
        parser.set_defaults(imic=True)
        parser.set_defaults(saturation='')
        parser.set_defaults(bins=128)
        parser.set_defaults(gwSeperator=None)
        arguments = parser.parse_args()
        self.args = vars(arguments)

        if self.input() == None:
            parser.print_help()
            sys.exit(0)

    def input(self):
        return self.args['input']

    def output(self):
        return self.args['output']

    def inclination(self):
        return self.args['inclinationImage']

    def area(self):
        return self.args['areaImage']

    def use_fitgm(self):
        return self.args['fitgm']

    def use_imic(self):
        return self.args['imic']

    def use_saturation(self):
        if self.args['saturation'] == '':
            return 0, []
        elif 'localCorrection' in self.args['saturation']:
            params = list(map(float, self.args['saturation'].split(',')[1:]))
            if len(params) != 2:
                print('Expected 2 parameters for localCorrection. Got {} instead. Using standard values.'.format(len(params)))
                params = [5e-3, 5e-3]
            return 1, params
        elif 'globalCorrection' in self.args['saturation']:
            params = list(map(float, self.args['saturation'].split(',')[1:]))
            if len(params) != 1:
                print('Expected 1 parameters for globalCorrection. Got {} instead. Using standard values.'.format(len(params)))
                params = [5e-3]
            return 2, params
        elif 'inpaintLocalCorrection' in self.args['saturation']:
            params = list(map(float, self.args['saturation'].split(',')[1:]))
            if len(params) != 3:
                print('Expected 3 parameters for inpaintLocalCorrection. Got {} instead. Using standard values.'.format(len(params)))
                params = [5e-3, 5e-3, 1e-3]
            return 3, params
        return 0

    def bins(self):
        return int(self.args['bins'])
        
    def gwSeperator(self):
        return None if not self.args['gwSeperator'] else float(self.args['gwSeperator'])
