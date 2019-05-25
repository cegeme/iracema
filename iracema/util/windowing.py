"""
Some useful methods and functions for windowing operations.
"""

import scipy.signal as sig
from numpy import pad, apply_along_axis
from numpy.lib.stride_tricks import as_strided


def apply_sliding_window(x, window_size, hop_size, function, window_name):
    """
    Apply a sliding window with the given parameters to the array `x` and
    aggregate the data within each window using the specified `function`.

    Args
    ----
    x : ndarray
        Array over which the sliding operation must be applied.
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

    Returns
    -------
    y : ndarray
    """
    view = get_sliding_window_view(x, window_size, hop_size)

    if window_name:
        window = get_window_function(window_size, window_name)
        y = apply_along_axis(function, -1, view * window)
    else:
        y = apply_along_axis(function, -1, view)

    return y.T


def get_sliding_window_view(x, window_size, hop_size):
    """
    Generate a view of the input array containing the sliding windows obtained
    for the given parameters.

    This method only creates a view of the sliding windows; it does not apply a
    window function (apodization function) to them.

    Args
    ----
    x : ndarray
        Array over which the sliding operation must be applied.
    window_size: int
        Size of the window.
    hop_size: int
        Number of samples to be skipped between two successive windowing
        operations.

    Returns
    -------
    view : ndarray
    """
    if (x.ndim != 1):
        raise ValueError("`x` must be a unidimensional array")

    pre_padding_size, post_padding_size, num_hops = \
        calculate_sliding_window_parms(window_size, hop_size, x.size)

    # padded version of the array
    x = pad(x, (pre_padding_size, post_padding_size), 'constant',
            constant_values=(0, 0))

    # apply striding tricks to create a windowed view of the array
    view = as_strided(x, shape=(num_hops, window_size),
                      strides=(x.itemsize * hop_size, x.itemsize))

    return view


def get_window_function(window_size, window_name, symmetric=True):
    """
    Get a window function (also known as tapering function or apodization
    function) according to the specified `window_name`.

    This function will return None if the specified `window_name` is also None.

    Args
    ----
    window_name : str
        Name of the window function to be used. Options are: {"boxcar",
        "triang", "blackman", "hamming", "hann", "bartlett", "flattop",
        "parzen", "bohman", "blackmanharris", "nuttall", "barthann",
        "no_window"}.
    """
    # check if the window_name is valid
    possible_windows = {
        "boxcar", "triang", "blackman", "hamming", "hann", "bartlett",
        "flattop", "parzen", "bohman", "blackmanharris", "nuttall",
        "barthann", "no_window"
    }

    if window_name not in possible_windows:
        raise ValueError('invalid window_name: {}'.format(window_name))
    elif window_name is not None:
        return sig.get_window(
            window_name, window_size, fftbins=not symmetric)
    else:
        return None


def calculate_sliding_window_parms(window_size, hop_size, array_size):
    """
    Calculate some parameters that are necessary for applying the sliding
    window over a time series.

    Args
    ----
    window_size: int
        Size of the window.
    hop_size: int
        Number of samples to be skipped between two successive windowing
        operations.
    array_size: int
        Size of the original array (length of the time series) in which the
        sliding window operation will be applied.

    Returns
    -------
    pre_padding_size : int
    post_padding_size : int
    hum_hops : int
    """

    if (hop_size > window_size):
        raise ValueError("the `hop_size` must be <= than `window_size`")
    elif window_size < 2:
        raise ValueError("`the `window_size` must be >= than 2")

    half_window_size = window_size // 2

    pre_padding_size = half_window_size
    num_hops = (pre_padding_size + array_size - 1) // hop_size + 1
    remainder_samples = (pre_padding_size + array_size - 1) % hop_size

    # calculate the size of the post-padding
    post_padding_size = window_size - remainder_samples - 1

    return pre_padding_size, post_padding_size, num_hops
