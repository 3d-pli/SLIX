import numpy


def check_valid_input(func):
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, numpy.ndarray):
                assert arg.ndim == 3, "Input must be a 3D array"
        return func(*args, **kwargs)

    return wrapper
