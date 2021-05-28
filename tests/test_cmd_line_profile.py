import csv
import os
import pytest
import shlex
import shutil
from unittest import mock
from SLIX import _cmd


class TestCommandLineProfile:
    def test_argparse(self):
        argparse = _cmd.create_argument_parser_line_profile()

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

    def test_main(self):
        with mock.patch('sys.argv', ['SLIXLineplotParameterGenerator',
                                     '--input',
                                     'examples/90-Stack-1647-1234.txt',
                                     '--output',
                                     'tests/files/output/',
                                     '--optional',
                                     '--with_plots']):
            _cmd.main_line_profile()
        assert os.path.isdir('tests/files/output/')
        assert os.path.isfile('tests/files/output/90-Stack-1647-1234.csv')
        assert os.path.isfile('tests/files/output/90-Stack-1647-1234.png')

        with open('tests/files/output/90-Stack-1647-1234.csv', newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            list_of_attrs = ['High Prominence Peaks',
                             'Low Prominence Peaks',
                             'Mean Prominence',
                             'Mean peak width',
                             'Mean peak distance',
                             'Direction',
                             'Min',
                             'Max',
                             'Avg']
            attr = 0
            for row in reader:
                assert row[0] == list_of_attrs[attr]
                attr += 1

        with mock.patch('sys.argv', ['SLIXLineplotParameterGenerator',
                                     '--input',
                                     'examples/90-Stack-1647-1234.txt',
                                     '--output',
                                     'tests/files/output/second/']):
            _cmd.main_line_profile()
        assert os.path.isdir('tests/files/output/second/')
        assert os.path.isfile('tests/files/output/second/90-Stack-1647-1234.csv')
        assert not os.path.isfile('tests/files/output/second/90-Stack-1647-1234.png')

        with open('tests/files/output/second/90-Stack-1647-1234.csv', newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            list_of_attrs = ['High Prominence Peaks',
                             'Low Prominence Peaks',
                             'Mean Prominence',
                             'Mean peak width',
                             'Mean peak distance',
                             'Direction']
            attr = 0
            for row in reader:
                assert row[0] == list_of_attrs[attr]
                attr += 1


@pytest.fixture(scope="session", autouse=True)
def run_around_tests(request):
    # Code that will run before your test, for example:
    assert os.path.isfile('examples/90-Stack-1647-1234.txt')
    assert os.path.isfile('examples/90-Stack-2481-1524.txt')

    # A test function will be run at this point
    yield

    # Code that will run after your test, for example:
    def remove_test_dir():
        if os.path.isdir('tests/files/output/'):
            shutil.rmtree('tests/files/output/')
    request.addfinalizer(remove_test_dir)