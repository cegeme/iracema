"""
Some aggregation methods for time series.
"""

import numpy as np

from iracema.util.windowing import apply_sliding_window
from iracema.timeseries import TimeSeries


def sliding_window(time_series, window_size, hop_size, function,
                   window_name=None):
    """
    Use a sliding window to aggregate the data from ``time_series`` by applying
    the ``function`` to each analysis window. The content of each window will
    be passed as the first argument to the function. Return the aggregated data
    in an array.

    Args
    ----
    time_series : TimeSeries
        Time series over which the sliding operation must be applied.
    window_size: int
        Size of the window.
    hop_size: int
        Number of samples to be skipped between two successive windowing
        operations.
    window_name : str
        Name of the window function to be used. Options are: {"boxcar",
        "triang", "blackman", "hamming", "hann", "bartlett", "flattop",
        "parzen", "bohman", "blackmanharris", "nuttall", "barthann",
        "no_window", None}.
    """
    new_data = apply_sliding_window(time_series.data, window_size, hop_size,
                                    function, window_name)

    # new sampling frequency for the aggregated time-series
    new_fs = time_series.fs / np.float_(hop_size)

    new_ts = TimeSeries(
        new_fs,
        data=new_data,
        start_time=time_series.start_time)

    return new_ts


def aggregate_features(time_series, func):
    """
    Aggregate the features within each sample from ``time_series``.
    """
    new_data = np.apply_along_axis(func, 0, time_series.data)

    new_ts = TimeSeries(
        time_series.fs,
        data=new_data,
        start_time=time_series.start_time)

    return new_ts


def aggregate_sucessive_samples(time_series, func, padding='zeros'):
    """
    Aggregate consecutive samples in ``time_series``, and generate a new time
    series object.

    Args
    ----
    time_series : TimeSeries
    padding : {'zeros', 'same', 'ones'}
    """
    # TODO: this kind of aggregation could be simplified with shifting
    # operations in time series. The looping can probably be avoided.
    data = time_series.data
    nfeatures = time_series.nfeatures
    nsamples = time_series.nsamples
    dtype = time_series.data.dtype

    if padding == 'zeros':
        padding_array = np.zeros(nfeatures, dtype)
    elif padding == 'ones':
        padding_array = np.ones(nfeatures, dtype)
    else:
        padding_array = data[..., 0]

    new_data = np.empty(nsamples, dtype)

    # a padded array will be used as 'previous sample' for the aggregation
    # of the first sample
    new_data[0] = func(padding_array, data[..., 0])

    for i in range(1, nsamples):
        new_data[i] = func(data[..., i - 1], data[..., i])

    new_ts = TimeSeries(
        time_series.fs,
        data=new_data,
        start_time=time_series.start_time)

    return new_ts
