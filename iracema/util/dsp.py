"""
Functions that are commonly used in digital signal processing.
"""

import numpy as np
from scipy import signal


def local_peaks(array):
    """
    Find the local peaks of the `array`.

    Args
    ----
    array : numpy array
        Input array.

    Return
    ------
    values : numpy array
        Values for the local maxima.
    ix : numpy array
        Indexes of the local maxima in the input array.
    """
    # calculate the first derivative
    arr_diff = np.diff(array)
    # search for indexes where the slope changes from positive to negative
    ix = np.nonzero((arr_diff[1:] < 0) & (arr_diff[:-1] >= 0))[0] + 1
    values = array[ix]

    return values, ix


def n_highest_peaks(array, n):
    """
    Find the `n` highest peaks in the given `array`.

    Args
    ----
    array : numpy array
        Input array.
    n : int
        Number of peaks to search.
    """
    # sort the array by its peaks and choose the n highest values
    val, ix = local_peaks(array)
    ix_n = np.argsort(val)[-n:]
    # TODO: to figure out a more efficient way to to this

    return val[ix_n], ix[ix_n]


def hwr(array):
    """
    Half-wave rectifier
    """
    return np.clip(array, 0, None)


def decimate_mean(array, f):
    """
    Decimate array by grouping each `f` samples and taking their mean. The array
    will be padded with zeros at the end if its length is not divisible by `f`.
    """
    if array.ndim > 1:
        raise ValueError("array must be unidimensional")

    # pad the array with zeros at the end before reshaping
    pad = np.zeros((f - array.size % f,), dtype=array.dtype)
    padded_array = np.concatenate((array, pad))
    new_shape = (f, int(padded_array.size/f))
    reshaped_array = np.reshape(padded_array, new_shape)
    decimated_array = np.mean(reshaped_array, axis=0)

    return decimated_array


def but_filter(audio_data, fs, critical_frequency, filter_type='lowpass', filter_order=4):
    """
    Filters the input data using a butterworth digital filter. This is a wrapper
    over ``scipy.signal.butter``.

    Arguments
    ---------
    audio_data: numpy array
        Numpy array containing the data of a time series.
    fs: float
        Sampling frequency.
    critical_frequency: float
        The critical frequency of frequencies.
    filter_type: [‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’]
        The type of filter. filter_order: The order of the filter.
    """
    sos = signal.butter(filter_order,
                        critical_frequency,
                        filter_type,
                        fs=fs,
                        output='sos')
    filtered = signal.sosfilt(sos, audio_data)
    return filtered
