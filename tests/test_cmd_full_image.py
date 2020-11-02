import pytest
import argparse
import shlex
from SLIX import cmd_full_image


class TestCommandFullImage:
    def test_argparse(self):
        argparse = cmd_full_image.create_argument_parser()

        minimal_string = "--input input --output output"
        args = vars(argparse.parse_args(shlex.split(minimal_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == False
        assert args['peaks'] == False
        assert args['peakprominence'] == False
        assert args['peakwidth'] == False
        assert args['peakdistance'] == False
        assert args['optional'] == False

        test_string = minimal_string + " --prominence_threshold 0.56"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.56
        assert args['direction'] == False
        assert args['peaks'] == False
        assert args['peakprominence'] == False
        assert args['peakwidth'] == False
        assert args['peakdistance'] == False
        assert args['optional'] == False

        test_string = minimal_string + " --direction"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == True
        assert args['peaks'] == False
        assert args['peakprominence'] == False
        assert args['peakwidth'] == False
        assert args['peakdistance'] == False
        assert args['optional'] == False

        test_string = minimal_string + " --peaks"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == False
        assert args['peaks'] == True
        assert args['peakprominence'] == False
        assert args['peakwidth'] == False
        assert args['peakdistance'] == False
        assert args['optional'] == False

        test_string = minimal_string + " --peakprominence"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == False
        assert args['peaks'] == False
        assert args['peakprominence'] == True
        assert args['peakwidth'] == False
        assert args['peakdistance'] == False
        assert args['optional'] == False

        test_string = minimal_string + " --peakwidth"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == False
        assert args['peaks'] == False
        assert args['peakprominence'] == False
        assert args['peakwidth'] == True
        assert args['peakdistance'] == False
        assert args['optional'] == False

        test_string = minimal_string + " --peakdistance"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == False
        assert args['peaks'] == False
        assert args['peakprominence'] == False
        assert args['peakwidth'] == False
        assert args['peakdistance'] == True
        assert args['optional'] == False

        test_string = minimal_string + " --optional"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == False
        assert args['peaks'] == False
        assert args['peakprominence'] == False
        assert args['peakwidth'] == False
        assert args['peakdistance'] == False
        assert args['optional'] == True

        test_string = minimal_string + " --direction --peaks " \
                                       "--peakprominence --peakwidth " \
                                       "--peakdistance --optional"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == True
        assert args['peaks'] == True
        assert args['peakprominence'] == True
        assert args['peakwidth'] == True
        assert args['peakdistance'] == True
        assert args['optional'] == True

        test_string = minimal_string + ' --detailed'
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['detailed'] == True

        test_string = minimal_string + ' --with_mask'
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['with_mask'] == True

        test_string = minimal_string + ' --mask_threshold 55'
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['mask_threshold'] == 55

        test_string = minimal_string + ' --disable_gpu'
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['disable_gpu'] == False

    def test_main(self):
        pass