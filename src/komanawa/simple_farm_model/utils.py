"""
created matt_dumont 
on: 9/2/24
"""
import numpy as np


def geclose(array1, array2, atol=1e-8, rtol=1e-5, equal_nan=False):
    """
    Compare two arrays for greater than or equal (close)
    :param array:
    :param atol:
    :param rtol:
    :return:
    """
    return (array1 > array2) | np.isclose(array1, array2, atol=atol, rtol=rtol, equal_nan=equal_nan)


def leclose(array1, array2, atol=1e-8, rtol=1e-5, equal_nan=False):
    """
    Compare two arrays for less than or equal (close)
    :param array:
    :param atol:
    :param rtol:
    :return:
    """
    return (array1 < array2) | np.isclose(array1, array2, atol=atol, rtol=rtol, equal_nan=equal_nan)
