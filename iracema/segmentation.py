"""
A couple of different methods for note segmentation.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import iracema.features
import iracema.pitch
import iracema.segment

from iracema.aggregation import aggregate_sucessive_samples
from iracema.plot import plot_waveform_trio_features_and_points


def onsets_adaptive_rms(audio,
                        min_time=None,
                        perc_threshold_pk=0.2,
                        short_window = 512,
                        long_window = 4096,
                        hop=512,
                        alpha=0.1,
                        plot=False):
    """
    Extract the note onsets using the adaptive RMS method.

    Arguments
    ---------
    audio : Audio
        Audio time series.
    min_time : float, optional
        Minimum time (in seconds) between successive onsets.
    perc_threshold_pk : float
        A percentual of the ODF maximum to be defined as a threshold
        for the peak picking.
    short_window : int
        Length of the short term window for the calculation of the RMS.
    long_window : int
        Length of the long term window for the calculation of the RMS.
    hop:
        Length of the hop for the sliding window.
    alpha:
        Reduction factor for the long term RMS curve.
    plot: bool
        Whether of not to plot the results.
   
    Return
    ------
    onsets: PointList
        List of onsets.
    """
    odf = odf_adaptive_rms(audio,
                           long_window=long_window,
                           short_window=short_window,
                           hop=hop,
                           alpha=alpha,
                           plot=plot)

    if min_time:
        min_dist = int(min_time * odf.fs)
        if min_dist == 0:
            min_dist = None
    else:
        min_dist = None

    # peak picking
    threshold = perc_threshold_pk * np.max(odf.data)
    ix_onsets, _ = sig.find_peaks(
        odf.data, height=threshold, distance=min_dist)

    onsets = iracema.segment.PointList([
        iracema.segment.Point(odf, position)
        for position in ix_onsets
    ])

    if plot:
        plot_waveform_trio_features_and_points(audio, odf, onsets)

    return onsets

def odf_adaptive_rms(audio,
                     long_window = 4096,
                     short_window = 512,
                     hop=512,
                     alpha=0.1,
                     plot=False):
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
    plot: bool
        Whether of not to plot the results.

    Return
    ------
    odf: TimeSeries
        Onset detection function.
    """
    rms_long = iracema.features.rms(audio, long_window, hop)
    rms_short = iracema.features.rms(audio, short_window, hop).pad_like(rms_long)

    rms_long.data = rms_long.data*(1-alpha)
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
        
    if plot:
        plt.plot(rms_long.time, rms_long.data, linewidth=0.5, color='r')
        plt.plot(rms_short.time, rms_short.data, linewidth=0.5, color='b')

    return odf

def onsets_rms_derivative(audio,
                          rms=None,
                          min_time=None,
                          perc_threshold_pk=0.2,
                          plot=False):
    """
    Extract note onsets from the ``audio`` time-series using its ``rms``.
    The RMS will be calculated if it's not passed as an argument. The argument
    ``min_time`` can be used to specify the minimum distance (in seconds)
    between two adjacent onsets.

    Args
    ----
    audio : Audio
        Audio object
    rms : iracema.TimeSeries, optional
        Pre-calculated RMS for the audio time-series.
    min_time : float, optional
        Minimum time (in seconds) between successive onsets.
    perc_threshold_pk : float
        A percentual of the ODF maximum to be defined as a threshold
        for the peak picking.
    plot: bool
        Whether of not to plot the results

    Return
    ------
    onsets : list
        List of onset points.
    """
    rms = rms or iracema.features.rms(audio, 2048, 512)
    # TODO: hardcoded parameters should be obtained in a better way

    # handling arguments
    if min_time:
        min_dist = int(min_time * rms.fs)
        if min_dist == 0:
            min_dist = None
    else:
        min_dist = None

    # onset detection function
    onset_df = odf_rms_derivative(rms)

    # peak picking
    threshold = perc_threshold_pk * np.max(onset_df.data)
    ix_onsets, _ = sig.find_peaks(
        onset_df.data, height=threshold, distance=min_dist)

    onsets = iracema.segment.PointList([
        iracema.segment.Point(rms, position)
        for position in ix_onsets
    ])

    if plot:
        plot_waveform_trio_features_and_points(audio, onset_df, onsets)

    return onsets


def onsets_pitch_change(audio,
                        pitch,
                        min_time=None,
                        delta_pitch_ratio=0.04,
                        plot=False):
    """
    Extract note onsets from the ``audio`` time-series using its ``pitch``.
    The argument ``min_time`` can be used to specify the minimum distance (in
    seconds) between two adjacent onsets.

    Args
    ----
    audio : Audio
        Audio object
    pitch : iracema.TimeSeries
        Pre-calculated pitch for the audio time-series.
    min_time : float, optional
        Minimum time (in seconds) between successive onsets.
    plot: bool
        Whether of not to plot the results

    Return
    ------
    onsets : list
        List of onset points.
    """

    # handling arguments
    if min_time:
        min_dist = int(min_time * pitch.fs)
        if min_dist == 0:
            min_dist = None
    else:
        min_dist = None

    # onset detection function
    onset_df = odf_pitch_change(pitch)

    # peak picking
    ix_onsets, _ = sig.find_peaks(
        onset_df.data, height=delta_pitch_ratio, distance=min_dist)

    onsets = iracema.segment.PointList([
        iracema.segment.Point(pitch, position)
        for position in ix_onsets
    ])

    if plot:
        plot_waveform_trio_features_and_points(audio, onset_df, onsets)

    return onsets


def notes_rms_derivative(audio, rms=None, min_time=None, perc_threshold_pk=0.05):
    """
    Extract note segments from the ``audio`` time-series using its ``rms``.
    The RMS will be calculated if it's not passed as an argument. The argument
    ``min_time`` can be used to specify the minimum distance (in seconds)
    between two adjacent onsets.

    Args
    ----
    audio : Audio
        Audio object
    rms : iracema.TimeSeries, optional
        Pre-calculated RMS for the audio time-series.
    min_time : float, optional
        Minimum time (in seconds) between successive onsets.
    perc_threshold_pk : float
        A percentual of the ODF maximum to be defined as a threshold
        for the peak picking.

    Return
    ------
    notes : list
        List of `Note` segments.
    """

    # handling arguments
    if min_time:
        min_dist = int(min_time * pitch.fs)
        if min_dist == 0:
            min_dist = None
    else:
        min_dist = None

    rms = rms or iracema.features.rms(audio, 2048, 512)
    # TODO: hardcoded parameters should be obtained in a better way

    # onset detection function
    onset_df = odf_rms_derivative(rms)

    # peak picking
    threshold = perc_threshold_pk * np.max(onset_df.data)
    ix_onsets, _ = sig.find_peaks(
        onset_df.data, height=threshold, distance=min_dist)

    # map the indexes to the original time-series
    ix_onsets_original = onset_df.map_index_to_original(ix_onsets)

    # offset detection
    ix_offsets = np.empty_like(ix_onsets)
    ix_offsets[-1] = onset_df.nsamples
    ix_offsets[:-1] = ix_onsets[1:] - 1

    # map the indexes to the original time-series
    ix_offsets_original = onset_df.map_index_to_original(ix_offsets)

    notes = [
        Segment(audio, st, end)
        for st, end in zip(ix_onsets_original, ix_offsets_original)
    ]

    return notes


def notes_pitch_variation(audio, pitch, min_time=None):
    """
    Extract note segments from the ``audio`` time-series using its ``pitch``. The
    pitch will be calculated if it's not passed as an argument. The argument
    `min_time` can be used to speficy the minimum distance (in seconds) between
    two adjacent onsets.

    Args
    ----
    audio : Audio
        Audio object
    pitch : TimeSeries, optional
        Pitch calculated from the audio time-series.
    min_time : float, optional
        Minimum time (in seconds) between successive onsets.

    Return
    ------
    notes : list
        List of `Note` segments.

    """
    if min_time:
        min_dist = int(np.ceil(min_time * pitch.fs))
    else:
        min_dist = None

    # onset detection function
    onset_df = odf_pitch_change(pitch)

    ix_onsets, _ = sig.find_peaks(onset_df.data, distance=min_dist)

    # map the indexes to the original time-series
    ix_onsets_original = onset_df.map_index_to_original(ix_onsets)

    # peak picking


def get_notes_list(audio, onsets, offsets):
    """
    Generate a list of note segments using the specified `onsets` and `offsets`
    arrays.

    Args
    ----
    audio : Audio
        Audio object
    onsets : array
        Indexes of the onset occurrences in `audio`.
    offsets : array
        Indexes of the offset occurrences in `audio`.

    Return
    ------
    notes : list
        List of segments.
    """
    if onsets.shape != offsets.shape:
        raise ValueError("the number of onsets and offsets must the same")

    return [Segment(audio, onsets[i], offsets[i]) for i in range(len(onsets))]


def odf_rms_derivative(rms):
    """
    Onset detection function based on RMS.
    """
    return rms.diff().hwr() #* rms


def odf_pitch_change(pitch):
    """
    Onset detection function based on Pitch.
    """
    def ratio_successive(current, last):
        min_denominator = 0.1
        last = np.float(last) + min_denominator
        return abs((current / last) - 1)

    return aggregate_sucessive_samples(pitch, ratio_successive)
