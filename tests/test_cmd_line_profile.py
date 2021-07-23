import csv
import os
import pytest
import shlex
import shutil
from unittest import mock
from SLIX._cmd import LineplotParameterGenerator


class TestCommandLineProfile:
    def test_argparse(self):
        argparse = LineplotParameterGenerator.create_argument_parser()

        minimal_string = "--input input --output output"
        args = vars(argparse.parse_args(shlex.split(minimal_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08

        test_string = minimal_string + " --prominence_threshold 0.56"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.56

    def test_main(self):
        with mock.patch('sys.argv', ['SLIXLineplotParameterGenerator',
                                     '--input',
                                     'examples/90-Stack-1647-1234.txt',
                                     '--output',
                                     'tests/files/output/',
                                     '--without_angles']):
            LineplotParameterGenerator.main()
        assert os.path.isdir('tests/files/output/')
        assert os.path.isfile('tests/files/output/90-Stack-1647-1234.csv')
        assert os.path.isfile('tests/files/output/90-Stack-1647-1234.png')

        list_of_attrs = [
            'profile',
            'filtered',
            'centroids',
            'peaks',
            'significant peaks',
            'prominence',
            'width',
            'distance',
            'direction'
        ]

        with open('tests/files/output/90-Stack-1647-1234.csv', newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            attr = 0
            for row in reader:
                assert row[0] == list_of_attrs[attr]
                attr += 1

        with mock.patch('sys.argv', ['SLIXLineplotParameterGenerator',
                                     '--input',
                                     'examples/90-Stack-1647-1234.txt',
                                     '--output',
                                     'tests/files/output/second/',
                                     '--without_angles',
                                     '--simple']):
            LineplotParameterGenerator.main()
        assert os.path.isdir('tests/files/output/second/')
        assert os.path.isfile('tests/files/output/second/90-Stack-1647-1234.csv')
        assert os.path.isfile('tests/files/output/second/90-Stack-1647-1234.png')

        with open('tests/files/output/second/90-Stack-1647-1234.csv', newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
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