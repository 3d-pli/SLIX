from SLIX._cmd import Cluster
import argparse
import shlex
import shutil
import numpy
import os
import pytest
from unittest import mock


class TestCommandClassification:
    def test_argparse(self):
        parser = Cluster.create_argparse()

        string = "--input input --output output"
        args = vars(parser.parse_args(shlex.split(string)))
        assert args['input'] == 'input'
        assert args['output'] == 'output'
        assert args['output_type'] == 'tiff'
        assert args['all'] == False
        assert args['inclination'] == False
        assert args['crossing'] == False
        assert args['flat'] == False

        string = "--input input --output output --all"
        args = vars(parser.parse_args(shlex.split(string)))
        assert args['input'] == 'input'
        assert args['output'] == 'output'
        assert args['output_type'] == 'tiff'
        assert args['all'] == True
        assert args['inclination'] == False
        assert args['crossing'] == False
        assert args['flat'] == False

        string = "--input input --output output --inclination"
        args = vars(parser.parse_args(shlex.split(string)))
        assert args['input'] == 'input'
        assert args['output'] == 'output'
        assert args['output_type'] == 'tiff'
        assert args['all'] == False
        assert args['inclination'] == True
        assert args['crossing'] == False
        assert args['flat'] == False

        string = "--input input --output output --crossing"
        args = vars(parser.parse_args(shlex.split(string)))
        assert args['input'] == 'input'
        assert args['output'] == 'output'
        assert args['output_type'] == 'tiff'
        assert args['all'] == False
        assert args['inclination'] == False
        assert args['crossing'] == True
        assert args['flat'] == False

        string = "--input input --output output --flat"
        args = vars(parser.parse_args(shlex.split(string)))
        assert args['input'] == 'input'
        assert args['output'] == 'output'
        assert args['output_type'] == 'tiff'
        assert args['all'] == False
        assert args['inclination'] == False
        assert args['crossing'] == False
        assert args['flat'] == True

    def test_crossing(self):
        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/cluster',
                                     '--output',
                                     'tests/output/cluster',
                                     '--crossing']):
            Cluster.main()
            assert os.path.isdir('tests/output/cluster')
            assert os.path.isfile('tests/output/cluster/cluster_crossing_mask.tiff')

    def test_inclination(self):
        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                '--input',
                                'tests/files/cluster',
                                '--output',
                                'tests/output/cluster',
                                '--inclination']):
            Cluster.main()
            assert os.path.isdir('tests/output/cluster')
            assert os.path.isfile('tests/output/cluster/cluster_inclination_mask.tiff')

    def test_flat(self):
        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                '--input',
                                'tests/files/cluster',
                                '--output',
                                'tests/output/cluster',
                                '--flat']):
            Cluster.main()
            assert os.path.isdir('tests/output/cluster')
            assert os.path.isfile('tests/output/cluster/cluster_flat_mask.tiff')

    def test_all(self):
        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                '--input',
                                'tests/files/cluster',
                                '--output',
                                'tests/output/cluster',
                                '--all']):
            Cluster.main()
            assert os.path.isdir('tests/output/cluster')
            assert os.path.isfile('tests/output/cluster/cluster_classification_mask.tiff')


@pytest.fixture(scope="session", autouse=True)
def run_around_tests(request):
    # Code that will run before your test, for example:
    assert os.path.isdir('tests/files/cluster')

    # A test function will be run at this point
    yield

    # Code that will run after your test, for example:
    def remove_test_dir():
        if os.path.isdir('tests/output/'):
            shutil.rmtree('tests/output/')
            # pass
    request.addfinalizer(remove_test_dir)
