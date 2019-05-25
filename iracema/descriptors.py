"""
This module contains the implementation of some audio descriptors.
"""

import numpy as np
from scipy.stats import gmean  # pylint: disable=import-error

from .aggregation import (aggregate_features, aggregate_sucessive_samples,
                          sliding_window)
from .util.dsp import hwr


def peak_envelope(time_series, window_size, hop_size):
    """
    Calculate the peak envelope of a time series

    .. math:: PE = max(x(n)), 1 <= n <= L

    Where `x(n)` is the n-th sample of a window of length `L`

    Args
    ----
    time_series : iracema.timeseries.TimeSeries
        An audio time-series object.
    window_size : int
    hop_size : int
    """
    def function(x):
        return np.max(np.abs(x))

    time_series = sliding_window(time_series, window_size, hop_size,
                                 function)
    time_series.label = 'PeakEnvelope'
    time_series.unit = 'amplitude'
    return time_series


def rms(time_series, window_size, hop_size):
    """
    Calculate the root mean square of a time series

    .. math:: RMS = \\sqrt{ \\frac{1}{L} \\sum_{n=1}^{L} x(n)^2 }

    Where `x(n)` is the n-th sample of a window of length `L`

    Args
    ----
    time_series : iracema.timeseries.TimeSeries
        A time-series object. It is usually applied on Audio objects.
    window_size : int
    hop_size : int
    """
    def function(x):
        return np.sqrt(np.mean(x**2))

    time_series = sliding_window(time_series, window_size, hop_size,
                                 function)
    time_series.label = 'RMS'
    time_series.unit = 'amplitude'
    return time_series


def zcr(time_series, window_size, hop_size):
    """
    Calculate the zero-crossing rate of a time series, i.e., the number of
    times the signal crosses the zero axis, per second.

    The zero-crossing rate gives some insight on the noisiness character of a
    sound. In noisy / unvoiced signals, the zero-crossing rate tends to reach
    higher values than in periodic / voiced signals.

    Args
    ----
    time_series : iracema.timeseries.TimeSeries
        A time-series object. It is usually applied on Audio objects.
    window_size : int
    hop_size : int
    """
    # count the number of times the signal changes between successive samples
    def function(x):
        return np.sum(x[1:] * x[:-1] < 0) / window_size * time_series.fs

    time_series = sliding_window(time_series, window_size, hop_size,
                                 function)
    time_series.label = 'ZCR'
    time_series.unit = 'Hz'
    return time_series


def spectral_flatness(fft):
    """
    Calculate the spectral flatness for a given FFT.

    The spectral flatness gives an estimation of the noisiness / sinusoidality
    of an audio signal (for the whole spectrum or for a frequency range). It
    can be used to determine voiced / unvoiced parts of a signal [Park2004]_.

    It is defined as the ratio between the `geometric mean` and the
    `arithmetic mean` of the energy spectrum:

    .. math::
       :nowrap:

       \\begin{eqnarray}
       SFM = 10 log_{10} \\left( \\frac
         {\\left( \\prod_{k=1}^{N} |X(k)| \\right)^\\frac{1}{N}}
         { \\frac{1}{N} \\sum_{k=1}^{N} |X(k)| }
       \\right)
       \\end{eqnarray}

    Where `X(k)` is the result of the FFT for the `k-th` frequency bin.

    Args
    ----
    time_series : iracema.spectral.FFT
        A FFT object

    References
    ----------
    .. [Park2004] Park, T. H. (2004). Towards automatic musical instrument
       timbre recognition. Princeton University.
    """
    def function(X):
        fft_magnitudes = np.abs(X)
        return 10 * np.log10(gmean(fft_magnitudes) / np.mean(fft_magnitudes))

    time_series = aggregate_features(fft, function)
    time_series.label = 'SpectralFlatness'
    time_series.unit = ''
    return time_series


def hfc(fft, method='energy'):
    """
    Calculate the high frequency content for a FFT time-series.

    The HFC _function produces sharp peaks during attacks transients
    [Bello2005]_ and might be a good choice for detecting onsets in percussive
    sounds.

    .. math:: HFC = \sum_{k=1}^{N} |X(k)|^2 \\cdot k

    Alternatively, you can set ``method`` = `'amplitude'` instead of `'energy'`
    (default value):

    .. math:: HFC = \sum_{k=1}^{N} |X(k)| \\cdot k

    Args
    ----
    fft : iracema.spectral.FFT
        FFT time-series.
    method : str
        Method of choice to calculate the HFC.

    References
    ----------
    .. [Bello2005] Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies,
        M., & Sandler, M. B. (2005). A tutorial on onset detection in music
        signals. IEEE Transactions on Speech and Audio Processing, 13(5),
        1035–1046.
    """
    def _func(X):
        N = X.shape[0]
        W = np.arange(1, N + 1)

        if method == 'energy':
            return np.sum(W * np.abs(X)**2) / N
        elif method == 'amplitude':
            return np.sum(W * np.abs(X)) / N
        else:
            ValueError("the argument `method` must be 'energy' or 'amplitude'")

    time_series = aggregate_features(fft, _func)
    time_series.label = 'HFC'
    time_series.unit = ''
    return time_series


def spectral_centroid(fft):
    """
    Calculate the spectral centroid for a FFT time-series.

    The spectral centroid is a well known timbral feature that is used to
    describe the brightness of a sound. It represents the center of gravity
    of the frequency components of a signal [Park2010]_.

    .. math::
       SC = \\frac{\\sum_{k=1}^{N} |X(k)| \\cdot f_k }{\\sum_{k=1}^{N} |X(k)|}

    Where `X(k)` is the result of the FFT for the `k-th` frequency bin.

    Args
    ----
    fft : iracema.spectral.FFT
        A FFT object

    References
    ----------
    .. [Park2010] Park, T. H. (2010). Introduction to digital signal
       processing: Computer musically speaking. World Scientific Publishing
       Co. Pte. Ltd.
    """
    def function(X):
        return __spectral_centroid(X, fft.frequencies)

    time_series = aggregate_features(fft, function)
    time_series.label = 'SpectralCentroid'
    time_series.unit = 'Hz'
    return time_series


def spectral_spread(fft):
    """
    Calculate the spectral spread for a FFT time-series.

    The spectral spread represents the spread of the spectrum around the
    spectral centroid [Peeters2011]_.

    .. math::
       SC = \\sqrt{\\frac{\\sum_{k=1}^{N} |X(k)| \\cdot (f_k - SC)^2 }{\\sum_
       {k=1}^{N} |X (k)|}}

    Where `X(k)` is the result of the FFT for the `k-th` frequency bin and SC
    is the spectral centroid for the frame.

    References
    ----------
    .. [Peeters2011] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N.,
       & McAdams, S. (2011). The timbre toolbox: extracting audio descriptors
       from musical signals, 130(5).

    """
    def function(X):
        return __spectral_spread(X, fft.frequencies)

    time_series = aggregate_features(fft, function)
    time_series.label = 'SpectralSpread'
    time_series.unit = 'Hz'
    return time_series


def __spectral_centroid(X, f):
    """
    Calculate the spectral centroid for a fft frame `X`, being `f` the
    frequency corresponding to its bins.
    """
    abs_X = np.abs(X)
    sum_abs_X = np.sum(abs_X)
    if sum_abs_X == 0:
        return 0
    return np.sum(f * abs_X) / sum_abs_X


def __spectral_spread(X, f):
    """
    Calculate the spectral spread for a fft frame `X`, being `f` the frequency
    corresponding to its bins.
    """
    return np.sqrt(__spectral_centroid(X, (f - __spectral_centroid(X, f))**2))



def spectral_skewness(fft):
    """Spectral Skewness"""
    def _func(X):
        pass


def spectral_kurtosis(fft):
    """Spectral Kurtosis"""
    def _func(X):
        pass


def spectral_flux(fft):
    """
    Calculate the spectral flux for a FFT time-series.

    The spectral flux measures the amount of change between successive
    spectral frames. There are different methods to calculate the spectral
    flux across the literature. For now we have implemented the one proposed
    by [Dixon2006]_.

    .. math:: SF = \\sum_{k=1}^{N} H(|X(t, k)| - |X(t-1, k)|)

    where :math:`H(x) = \\frac{x+|x|}{2}` is the half-wave rectifier _function,
    and `t` is the temporal index of the frame.

    Args
    ----
    fft : iracema.spectral.FFT
        A FFT object

    References
    ----------
    .. [Dixon2006] Dixon, S. (2006). Onset Detection Revisited. In 9th
       International Conference on Digital Audio Effects (pp. 133–137).
       Montreal, Canada.
    """
    def function(X, X_prev):
        return np.sum(hwr(np.abs(X) - np.abs(X_prev)))

    time_series = aggregate_sucessive_samples(fft, function)
    time_series.label = 'SpectralFlux'
    time_series.unit = ''
    return time_series


def spectral_rolloff(fft):
    """Spectral Rolloff"""
    def _func(X):
        pass


def spectral_irregularity(fft):
    """Spectral Irregularity"""
    def _func(X):
        pass


def harmonic_centroid(harmonics):
    """Harmonic Centroid"""
    def _func(X):
        pass


def inharmonicity(fft, harmonics):
    """Inharmonicity"""
    def _func(X):
        pass


def noisiness(fft, harmonics):
    """Noisiness"""
    def _func(X):
        pass


def oer(harmonics):
    """Odd-to-Even Ratio"""
    def _func(X):
        pass
