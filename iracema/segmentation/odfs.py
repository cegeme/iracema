import matplotlib.pyplot as plt
import numpy as np

import iracema.features
import iracema.pitch
import iracema.spectral
from iracema.aggregation import aggregate_sucessive_samples


def odf_adaptative_rms(audio,
                     long_window=4096,
                     short_window=512,
                     hop=512,
                     alpha=0.1,
                     display_plot_rms=False):
    """
    Arguments
    ---------
    audio: Audio
        Audio time series.
    short_window : int
        Length of the short term window for the calculation of the RMS.
    long_window : int
        Length of the long term window for the calculation of the RMS.
    hop:
        Length of the hop for the sliding window.
    alpha:
        Reduction factor for the long term RMS curve.
    display_plot_rms: bool
        Whether of not to plot the RMS curves.

    Return
    ------
    odf: TimeSeries
        Onset detection function.
    """
    rms_long = iracema.features.rms(audio, long_window, hop)
    rms_short = iracema.features.rms(audio, short_window, hop)\
        .pad_like(rms_long)

    rms_long.data = rms_long.data * (1 - alpha)
    rms_diff = (rms_long - rms_short).hwr()

    odf = rms_diff.copy()
    odf.data = np.zeros_like(rms_diff.data)

    last, sum_, pk, ix_pk = 0, 0, 0, 0
    it = np.nditer(rms_diff.data, flags=['f_index'])
    for x in it:
        if x > 0:
            sum_ += x
            if x > pk:
                pk = x
                ix_pk = it.index
        else:
            if last > 0:
                odf.data[ix_pk] = sum_
            sum_ = 0
            pk = 0
        last = x

    if display_plot_rms:
        plt.plot(rms_long.time, rms_long.data, linewidth=0.5, color='r')
        plt.plot(rms_short.time, rms_short.data, linewidth=0.5, color='b')

    return odf


def odf_rms_derivative(audio, window=1024, hop=512):
    """
    Onset detection function based on RMS.

    Arguments
    ---------
    audio : Audio
        Audio object
    window : int
        Window length for computing the RMS.
    hop : int
        Hop length for computing the RMS.

    Return
    ------
    odf: TimeSeries
        Onset detection function.
    """
    rms = iracema.features.rms(audio, window, hop)
    return rms.diff().hwr()  # * rms


def odf_pitch_change(audio,
                     window=1024,
                     hop=512,
                     minf0=120,
                     maxf0=4000,
                     smooth_pitch=True):
    """
    Onset detection function based on Pitch.

    Arguments
    ---------
    audio : Audio
        Audio object
    window : int
        Window length for computing the pitch.
    hop : int
        Hop length for computing the pitch.
    minf0 : int
        Minimum frequency for the pitch detection.
    maxf0 : int
        Maximum frequency for the pitch detection.
    smooth_pitch: bool
        Whether or not the pitch curve should be smoothed.

    Return
    ------
    odf: TimeSeries
        Onset detection function.
    """
    window_mode = 9
    fft = iracema.spectral.fft(audio, window, hop)
    pitch = iracema.pitch.expan(fft, minf0=minf0, maxf0=maxf0)
    if smooth_pitch:
        pitch = iracema.pitch.pitch_filter(pitch)
        pitch = iracema.pitch.pitch_mode(pitch, window=window_mode)

    def ratio_successive(current, last):
        min_denominator = 0.1
        last = np.float(last) + min_denominator
        ratio = current / last
        if ratio >= 1:
            ratio = ratio - 1
        else:
            ratio = 1 - ratio
        return np.abs(ratio)

    return aggregate_sucessive_samples(pitch, ratio_successive)
