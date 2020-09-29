from numba import cuda
import cupy
import numpy

# DEFAULT PARAMETERS
BACKGROUND_COLOR = -1
MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 2

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.94
TARGET_PROMINENCE = 0.08


@cuda.jit('void(float32[:, :], int8[:, :], float32[:, :])')
def _prominence(image, peak_image, result_image):
    idx = cuda.grid(1)
    sub_image = image[idx]
    sub_peak_array = peak_image[idx]

    for pos in range(len(sub_peak_array)):
        if sub_peak_array[pos] == 1:
            i_min = -len(sub_peak_array) / 2
            i_max = int(len(sub_peak_array) * 1.5)

            i = pos
            left_min = sub_image[pos]
            while i_min <= i and sub_image[i] <= sub_image[pos]:
                if sub_image[i] < left_min:
                    left_min = sub_image[i]
                i -= 1

            i = pos
            right_min = sub_image[pos]
            while i <= i_max and sub_image[i % len(sub_peak_array)] <= sub_image[pos]:
                if sub_image[i % len(sub_peak_array)] < right_min:
                    right_min = sub_image[i % len(sub_peak_array)]
                i += 1

            result_image[idx, pos] = sub_image[pos] - max(left_min, right_min)
        else:
            result_image[idx, pos] = 0


@cuda.jit('void(float32[:, :], int8[:, :], float32[:, :], float32[:, :], float32)')
def _peakwidth(image, peak_image, prominence, result_image, target_height):
    idx = cuda.grid(1)
    sub_image = image[idx]
    sub_peak_array = peak_image[idx]
    sub_prominece = prominence[idx]

    for pos in range(len(sub_peak_array)):
        if sub_peak_array[pos] == 1:
            height = sub_image[pos] - sub_prominece[pos] * target_height
            i_min = -len(sub_peak_array) / 2
            i_max = int(len(sub_peak_array) * 1.5)

            i = int(pos)
            while i_min < i and height < sub_image[i]:
                i -= 1
            left_ip = float(i)
            if sub_image[i] < height:
                # Interpolate if true intersection height is between samples
                left_ip += (height - sub_image[i]) / (sub_image[i + 1] - sub_image[i])

            # Find intersection point on right side
            i = int(pos)
            while i < i_max and height < sub_image[i]:
                i += 1
            right_ip = float(i)
            if sub_image[i] < height:
                # Interpolate if true intersection height is between samples
                right_ip -= (height - sub_image[i]) / (sub_image[i - 1] - sub_image[i])

            result_image[idx, pos] = right_ip - left_ip
        else:
            result_image[idx, pos] = 0


@cuda.jit('void(int8[:, :], int8[:], float32[:, :])')
def _peakdistance(peak_image, number_of_peaks, result_image):
    idx = cuda.grid(1)
    sub_peak_array = peak_image[idx]
    current_pair = 0

    for i in range(len(sub_peak_array)):
        if sub_peak_array[i] == 1:
            if number_of_peaks[idx] == 1:
                result_image[idx, i] = 360.0
                break
            elif number_of_peaks[idx] % 2 == 0:
                left = i * 360.0 / len(sub_peak_array)
                right_side_peak = number_of_peaks[idx]//2
                current_position = i+1
                while right_side_peak > 0 and current_position < len(sub_peak_array):
                    if sub_peak_array[current_position] == 1:
                        right_side_peak = right_side_peak - 1
                    current_position = current_position + 1
                right = (current_position-1) * 360.0 / len(sub_peak_array)
                result_image[idx, i] = right - left
                result_image[idx, current_position-1] = 360 - (right - left)

                current_pair += 1

            if current_pair == number_of_peaks[idx]//2:
                break


@cuda.jit('void(int8[:, :], int8[:], float32[:, :])')
def _direction(peak_array, number_of_peaks, result_image):
    idx = cuda.grid(1)
    sub_peak_array = peak_array[idx]
    num_directions = result_image.shape[-1]

    current_direction = 0

    result_image[idx, :] = BACKGROUND_COLOR
    if number_of_peaks[idx] // 2 <= num_directions:
        for i in range(len(sub_peak_array)):
            if sub_peak_array[i] == 1:
                left = i * 360.0 / len(sub_peak_array)
                if number_of_peaks[idx] == 1:
                    result_image[idx, current_direction] = (270.0 - left) % 180
                    break
                elif number_of_peaks[idx] % 2 == 0:
                    right_side_peak = number_of_peaks[idx]//2
                    current_position = i+1
                    while right_side_peak > 0 and current_position < len(sub_peak_array):
                        if sub_peak_array[current_position] == 1:
                            right_side_peak = right_side_peak - 1
                        current_position = current_position + 1
                    right = (current_position-1) * 360.0 / len(sub_peak_array)
                    if number_of_peaks[idx] == 2 or abs(180 - (right - left)) < 35:
                        result_image[idx, current_direction] = (270.0 - ((left + right) / 2.0)) % 180
                    current_direction += 1

                if current_direction == number_of_peaks[idx]//2:
                    break
