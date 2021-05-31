import SLIX

if SLIX.toolbox.gpu_available:
    print(SLIX.toolbox.gpu_available)
    from SLIX.GPU import _toolbox as ntoolbox
    import cupy
    from numba import cuda

    threads_per_block = (1, 1)
    blocks_per_grid = (1, 1)

    class TestNumbaToolboxGPU:
        def test_peak_cleanup(self):
            test_one_peak = cupy.array([0, 1, 0, 0]).reshape((1, 1, 4))

            result = cupy.zeros(test_one_peak.shape, dtype='int8')
            ntoolbox._peaks[blocks_per_grid, threads_per_block](test_one_peak, result)
            cuda.synchronize()
            assert cupy.all(cupy.array([0, 1, 0, 0]) == result)

            test_two_peak = cupy.array([0, 1, 1, 0]).reshape((1, 1, 4))
            result = cupy.zeros(test_two_peak.shape, dtype='int8')
            ntoolbox._peaks[blocks_per_grid, threads_per_block](test_two_peak, result)
            assert cupy.all(cupy.array([0, 1, 0, 0]) == result)

            test_three_peak = cupy.array([0, 1, 1, 1, 0]).reshape((1, 1, 5))
            result = cupy.zeros(test_three_peak.shape, dtype='int8')
            ntoolbox._peaks[blocks_per_grid, threads_per_block](test_three_peak, result)
            assert cupy.all(cupy.array([0, 0, 1, 0, 0]) == result)

            test_double_three_peak = cupy.array([0, 1, 1, 1, 0, 1, 1, 1, 0]).reshape((1, 1, 9))
            result = cupy.zeros(test_double_three_peak.shape, dtype='int8')
            ntoolbox._peaks[blocks_per_grid, threads_per_block](test_double_three_peak, result)
            assert cupy.all(cupy.array([0, 0, 1, 0, 0, 0, 1, 0, 0]) == result)

        def test_prominence(self):
            test_array = cupy.array([0, 0.1, 0.2, 0.4, 0.8, 1, 0.5, 0.7, 0.9, 0.5, 0.3, 0.95, 0], dtype='float32')\
                             .reshape((1, 1, 13))
            peaks = cupy.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype='int8').reshape((1, 1, 13))

            expected_prominence = cupy.array([0, 0, 0, 0, 0, 1, 0, 0, 0.4, 0, 0, 0.65, 0]).reshape((1, 1, 13))
            toolbox_prominence = cupy.zeros(expected_prominence.shape, dtype='float32')
            ntoolbox._prominence[blocks_per_grid, threads_per_block](test_array, peaks, toolbox_prominence)
            print(toolbox_prominence)
            assert cupy.all(cupy.isclose(expected_prominence, toolbox_prominence))

        def test_peakwidth(self):
            test_array = cupy.array([0, 0.1, 0.2, 0.5, 0.8, 1, 0.77, 0.7, 0.66, 0.5, 0.74, 0.98, 0.74], dtype='float32')\
                             .reshape((1, 1, 13))
            peaks = cupy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype='int8').reshape((1, 1, 13))
            prominence = cupy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.48, 0], dtype='float32').reshape((1, 1, 13))
            expected_width = cupy.array([0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 2, 0]).reshape((1, 1, 13))

            toolbox_width = cupy.zeros(expected_width.shape, dtype='float32')
            ntoolbox._peakwidth[blocks_per_grid, threads_per_block](test_array, peaks, prominence, toolbox_width, 0.5)
            assert cupy.all(toolbox_width == expected_width)

        def test_peakdistance(self):
            test_arr = cupy.array(([False, False, True, False, False, False, False, True, False] +
                                   [False] * 15), dtype='int8')\
                           .reshape((1, 1, 24))
            expected_distance = 75
            toolbox_distance = cupy.zeros(test_arr.shape, dtype='float32')
            ntoolbox._peakdistance[blocks_per_grid, threads_per_block]\
                (test_arr,
                 cupy.zeros(test_arr.shape, dtype='float32'),
                 cupy.array([[2]], dtype='int8'),
                 toolbox_distance)
            assert toolbox_distance[0, 0, 2] == expected_distance
            assert toolbox_distance[0, 0, 7] == 360 - expected_distance

        def test_direction(self):
            # Test for one peak
            one_peak_arr = cupy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
                                .reshape((1, 1, 24)).astype('int8')
            expected_direction = cupy.array([45, ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR])
            toolbox_direction = cupy.zeros((1, 1, 3), dtype='float32')
            ntoolbox._direction[blocks_per_grid, threads_per_block]\
                (one_peak_arr,
                 cupy.zeros(one_peak_arr.shape, dtype='float32'),
                 cupy.array([[1]], dtype='int8'),
                 toolbox_direction,
                 0)
            assert cupy.all(expected_direction == toolbox_direction)

            # Test for one direction with 180°+-35° distance
            two_peak_arr = cupy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])\
                                .reshape((1, 1, 24)).astype('int8')
            expected_direction = cupy.array([135, ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR])
            ntoolbox._direction[blocks_per_grid, threads_per_block]\
                (two_peak_arr,
                 cupy.zeros(two_peak_arr.shape, dtype='float32'),
                 cupy.array([[2]], dtype='int8'),
                 toolbox_direction,
                 0)
            assert cupy.all(expected_direction == toolbox_direction)

            # Test for (invalid) two directions with 180°+-35° distance
            four_peak_arr = cupy.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) \
                .reshape((1, 1, 24)).astype('int8')
            expected_direction = cupy.array([ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR, ntoolbox.BACKGROUND_COLOR])
            ntoolbox._direction[blocks_per_grid, threads_per_block] \
                (four_peak_arr,
                 cupy.zeros(four_peak_arr.shape, dtype='float32'),
                 cupy.array([[4]], dtype='int8'),
                 toolbox_direction,
                 0)
            assert cupy.all(expected_direction == toolbox_direction)

        def test_centroid_correction_bases(self):
            # simple test case: one distinct peak
            test_array = cupy.array([0] * 9 + [1] + [0] * 14).reshape((1, 1, 24))
            test_high_peaks = SLIX.toolbox.peaks(test_array)
            test_reverse_peaks = SLIX.toolbox.peaks(-test_array)

            left_bases = cupy.zeros(test_array.shape, dtype='uint8')
            right_bases = cupy.zeros(test_array.shape, dtype='uint8')
            ntoolbox._centroid_correction_bases[blocks_per_grid, threads_per_block]\
                (test_array,
                 test_high_peaks,
                 test_reverse_peaks,
                 left_bases,
                 right_bases)
            assert cupy.sum(left_bases) == 1
            assert cupy.sum(right_bases) == 1

            # simple test case: one distinct peak
            test_array = cupy.array([0] * 8 + [0.95, 1, 0.5] + [0] * 13, dtype='float32').reshape((1, 1, 24))
            test_high_peaks = SLIX.toolbox.peaks(test_array)
            test_reverse_peaks = SLIX.toolbox.peaks(-test_array)

            ntoolbox._centroid_correction_bases[blocks_per_grid, threads_per_block] \
                (test_array,
                 test_high_peaks,
                 test_reverse_peaks,
                 left_bases,
                 right_bases)
            assert cupy.sum(left_bases) == 2
            assert cupy.sum(right_bases) == 1

            # simple test case: centroid is between two measurements
            test_array = cupy.array([0] * 8 + [1, 1] + [0] * 14).reshape((1, 1, 24))
            test_high_peaks = SLIX.toolbox.peaks(test_array)
            test_reverse_peaks = SLIX.toolbox.peaks(-test_array)

            ntoolbox._centroid_correction_bases[blocks_per_grid, threads_per_block] \
                (test_array,
                 test_high_peaks,
                 test_reverse_peaks,
                 left_bases,
                 right_bases)
            assert cupy.sum(left_bases) == 1
            assert cupy.sum(right_bases) == 2

            # more complicated test case: wide peak plateau
            test_array = cupy.array([0] * 8 + [1, 1, 1] + [0] * 13).reshape((1, 1, 24))
            test_high_peaks = SLIX.toolbox.peaks(test_array)
            test_reverse_peaks = SLIX.toolbox.peaks(-test_array)

            ntoolbox._centroid_correction_bases[blocks_per_grid, threads_per_block] \
                (test_array,
                 test_high_peaks,
                 test_reverse_peaks,
                 left_bases,
                 right_bases)

            assert cupy.sum(left_bases) == 2
            assert cupy.sum(right_bases) == 2

        def test_centroid(self):
            image = cupy.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
                         .reshape((1, 1, 24))
            left = cupy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
                        .reshape((1, 1, 24))
            right = cupy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
                         .reshape((1, 1, 24))
            peak = cupy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
                        .reshape((1, 1, 24))
            result_centroid = cupy.zeros(image.shape, dtype='float32')

            ntoolbox._centroid[blocks_per_grid, threads_per_block](image, peak, left, right, result_centroid)
            assert cupy.sum(result_centroid) == 0

