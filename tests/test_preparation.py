from SLIX import preparation
import numpy
import pytest


class TestPreparation:
    def test_smoothing(self):
        pass

    def test_thinout_plain(self):
        test_arr = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 11)
        test_img = test_arr.reshape((11, 11, 1))
        thinned = preparation.thin_out(test_img, factor=2, strategy='plain')
        assert numpy.all(thinned == 1)

        thinned = preparation.thin_out(test_img, factor=2, strategy='pLaIn')
        assert numpy.all(thinned == 1)

    def test_thinout_median(self):
        test_arr = numpy.array([1, 1, 0] * 4 * 12)
        test_img = test_arr.reshape((12, 12, 1))
        thinned = preparation.thin_out(test_img, factor=3, strategy='median')
        assert numpy.all(thinned == 1)

        thinned = preparation.thin_out(test_img, factor=3, strategy='MeDiAn')
        assert numpy.all(thinned == 1)

    def test_thinout_average(self):
        test_arr = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 12)
        test_img = test_arr.reshape((12, 12, 1))
        thinned = preparation.thin_out(test_img, factor=2, strategy='average')
        assert numpy.all(thinned == 0.5)

        thinned = preparation.thin_out(test_img, factor=2, strategy='AVERage')
        assert numpy.all(thinned == 0.5)

    def test_thinout_error(self):
        test_arr = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 11)
        test_img = test_arr.reshape((11, 11, 1))
        with pytest.raises(ValueError):
            preparation.thin_out(test_img, factor=2, strategy='error')
