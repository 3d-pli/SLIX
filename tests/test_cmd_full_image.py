import pytest
import os
import shutil
import shlex
from SLIX._cmd import ParameterGenerator
from unittest import mock


class TestCommandFullImage:
    def test_argparse(self):
        argparse = ParameterGenerator.create_argument_parser()

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
        assert args['unit_vectors'] == False

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
        assert args['unit_vectors'] == False

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
        assert args['unit_vectors'] == False

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
        assert args['unit_vectors'] == False

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
        assert args['unit_vectors'] == False

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
        assert args['unit_vectors'] == False

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
        assert args['unit_vectors'] == False

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
        assert args['unit_vectors'] == False

        test_string = minimal_string + " --unit_vectors"
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['prominence_threshold'] == 0.08
        assert args['direction'] == False
        assert args['peaks'] == False
        assert args['peakprominence'] == False
        assert args['peakwidth'] == False
        assert args['peakdistance'] == False
        assert args['optional'] == False
        assert args['unit_vectors'] == True

        test_string = minimal_string + " --direction --peaks " \
                                       "--peakprominence --peakwidth " \
                                       "--peakdistance --optional " \
                                       "--unit_vectors"
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
        assert args['unit_vectors'] == True

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

        test_string = minimal_string + ' --disable_gpu'
        args = vars(argparse.parse_args(shlex.split(test_string)))
        assert args['input'] == ['input']
        assert args['output'] == 'output'
        assert args['disable_gpu'] == False

        for strategy in ['strict', 'safe', 'unsafe']:
            test_string = minimal_string + f' --direction_strategy {strategy}'
            args = vars(argparse.parse_args(shlex.split(test_string)))
            assert args['input'] == ['input']
            assert args['output'] == 'output'
            assert args['direction_strategy'] == strategy

    def test_main(self):
        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/gpu',
                                     '--optional',
                                     '--with_mask']):
            ParameterGenerator.main()
            assert os.path.isdir('tests/files/output/gpu')
            assert os.path.isfile('tests/files/output/gpu/demo_high_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_low_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_peakwidth.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_peakprominence.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_peakdistance.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_max.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_min.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_avg.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_dir.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_dir_1.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_dir_2.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_dir_3.tiff')
            assert os.path.isfile('tests/files/output/gpu/demo_background_mask.tiff')

        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/cpu',
                                     '--optional',
                                     '--with_mask',
                                     '--disable_gpu',
                                     '--no_centroids']):
            ParameterGenerator.main()
            assert os.path.isdir('tests/files/output/cpu')
            assert os.path.isfile('tests/files/output/cpu/demo_high_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_low_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_peakwidth.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_peakprominence.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_peakdistance.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_max.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_min.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_avg.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_dir.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_dir_1.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_dir_2.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_dir_3.tiff')
            assert os.path.isfile('tests/files/output/cpu/demo_background_mask.tiff')

        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/single/',
                                     '--direction',
                                     '--no_centroids']):
            ParameterGenerator.main()
            assert os.path.isdir('tests/files/output/single/')
            assert not os.path.isfile('tests/files/output/single/demo_high_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_low_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_peakwidth.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_peakprominence.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_peakdistance.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_max.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_min.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_avg.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_dir.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_dir_1_UnitX.nii')
            assert not os.path.isfile('tests/files/output/single/demo_dir_1_UnitY.nii')
            assert not os.path.isfile('tests/files/output/single/demo_dir_1_UnitZ.nii')
            assert os.path.isfile('tests/files/output/single/demo_dir_1.tiff')
            assert os.path.isfile('tests/files/output/single/demo_dir_2.tiff')
            assert os.path.isfile('tests/files/output/single/demo_dir_3.tiff')
            assert not os.path.isfile('tests/files/output/single/demo_background_mask.tiff')

        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/unit',
                                     '--unit_vectors']):
            ParameterGenerator.main()
            assert os.path.isdir('tests/files/output/unit')
            assert not os.path.isfile('tests/files/output/unit/demo_high_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_low_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_peakwidth.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_peakprominence.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_peakdistance.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_max.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_min.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_avg.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_dir.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_dir_1.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_dir_2.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_dir_3.tiff')
            assert not os.path.isfile('tests/files/output/unit/demo_background_mask.tiff')
            assert os.path.isfile('tests/files/output/unit/demo_dir_1_UnitX.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_2_UnitX.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_3_UnitX.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_1_UnitY.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_2_UnitY.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_3_UnitY.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_1_UnitZ.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_2_UnitZ.nii')
            assert os.path.isfile('tests/files/output/unit/demo_dir_3_UnitZ.nii')

        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/detailed',
                                     '--detailed']):
            ParameterGenerator.main()
            assert os.path.isdir('tests/files/output/detailed')
            assert os.path.isfile('tests/files/output/detailed/demo_all_peaks_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_high_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_high_prominence_peaks_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_low_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_peakwidth.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_peakwidth_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_peakprominence.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_peakprominence_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_peakdistance.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_peakdistance_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_dir_1.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_dir_2.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_dir_3.tiff')
            assert os.path.isfile('tests/files/output/detailed/demo_centroid_correction.tiff')

        for strategy in ['strict', 'safe', 'unsafe']:
            with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                         '--input',
                                         'tests/files/demo.nii',
                                         '--output',
                                         f'tests/files/output/strategy/{strategy}']):
                ParameterGenerator.main()
                assert os.path.isdir(f'tests/files/output/strategy/{strategy}')
                assert os.path.isfile(f'tests/files/output/strategy/{strategy}/demo_dir_1.tiff')
                assert os.path.isfile(f'tests/files/output/strategy/{strategy}/demo_dir_2.tiff')
                assert os.path.isfile(f'tests/files/output/strategy/{strategy}/demo_dir_3.tiff')


@pytest.fixture(scope="session", autouse=True)
def run_around_tests(request):
    # Code that will run before your test, for example:
    assert os.path.isfile('tests/files/demo.nii')
    assert os.path.isfile('tests/files/demo.tiff')
    assert os.path.isfile('tests/files/demo.h5')

    # A test function will be run at this point
    yield

    # Code that will run after your test, for example:
    def remove_test_dir():
        if os.path.isdir('tests/files/output/'):
            shutil.rmtree('tests/files/output/')
            # pass
    request.addfinalizer(remove_test_dir)