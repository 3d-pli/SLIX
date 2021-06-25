import pytest
import os
import shutil
import shlex
from SLIX import _cmd
from unittest import mock


class TestCommandFullImage:
    def test_argparse(self):
        argparse = _cmd.create_argument_parser_full_image()

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

    def test_main(self):
        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/gpu',
                                     '--optional',
                                     '--with_mask']):
            _cmd.main_full_image()
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
            _cmd.main_full_image()
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
                                     'tests/files/output/single/cpu',
                                     '--direction',
                                     '--no_centroids']):
            _cmd.main_full_image()
            assert os.path.isdir('tests/files/output/cpu')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_high_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_low_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_peakwidth.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_peakprominence.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_peakdistance.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_max.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_min.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_avg.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_dir.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_dir_1_UnitX.nii')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_dir_1_UnitY.nii')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_dir_1_UnitZ.nii')
            assert os.path.isfile('tests/files/output/single/cpu/demo_dir_1.tiff')
            assert os.path.isfile('tests/files/output/single/cpu/demo_dir_2.tiff')
            assert os.path.isfile('tests/files/output/single/cpu/demo_dir_3.tiff')
            assert not os.path.isfile('tests/files/output/single/cpu/demo_background_mask.tiff')

        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/unit/cpu',
                                     '--unit_vectors']):
            _cmd.main_full_image()
            assert os.path.isdir('tests/files/output/unit/cpu')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_high_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_low_prominence_peaks.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_peakwidth.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_peakprominence.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_peakdistance.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_max.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_min.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_avg.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_dir.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_dir_1.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_dir_2.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_dir_3.tiff')
            assert not os.path.isfile('tests/files/output/unit/cpu/demo_background_mask.tiff')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_1_UnitX.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_2_UnitX.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_3_UnitX.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_1_UnitY.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_2_UnitY.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_3_UnitY.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_1_UnitZ.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_2_UnitZ.nii')
            assert os.path.isfile('tests/files/output/unit/cpu/demo_dir_3_UnitZ.nii')

        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/detailed/gpu',
                                     '--detailed']):
            _cmd.main_full_image()
            assert os.path.isdir('tests/files/output/detailed/gpu')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_all_peaks_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_high_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_high_prominence_peaks_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_low_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_peakwidth.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_peakwidth_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_peakprominence.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_peakprominence_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_peakdistance.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_peakdistance_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_dir_1.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_dir_2.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_dir_3.tiff')
            assert os.path.isfile('tests/files/output/detailed/gpu/demo_centroid_correction.tiff')

        with mock.patch('sys.argv', ['SLIXParameterGenerator',
                                     '--input',
                                     'tests/files/demo.nii',
                                     '--output',
                                     'tests/files/output/detailed/cpu',
                                     '--disable_gpu',
                                     '--detailed']):
            _cmd.main_full_image()
            assert os.path.isdir('tests/files/output/detailed/cpu')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_all_peaks_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_high_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_high_prominence_peaks_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_low_prominence_peaks.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_peakwidth.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_peakwidth_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_peakprominence.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_peakprominence_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_peakdistance.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_peakdistance_detailed.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_dir_1.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_dir_2.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_dir_3.tiff')
            assert os.path.isfile('tests/files/output/detailed/cpu/demo_centroid_correction.tiff')


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