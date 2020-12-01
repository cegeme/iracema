import numpy as np
import scipy.signal as sig

import iracema.features
import iracema.pitch
import iracema.segment

from iracema.plot import waveform_trio_features_and_points
from iracema.segmentation.odfs import (odf_rms_derivative, odf_pitch_change,
                                       odf_adaptive_rms)


def adaptive_rms(audio,
                 short_window=512,
                 long_window=4096,
                 hop=512,
                 alpha=0.1,
                 min_time=None,
                 plot_rms_curves=False,
                 odf_threshold=0.2,
                 plot=False,
                 return_odf_data=False):
    """
    Extract the note onsets using the adaptive RMS method.

    Arguments
    ---------
    audio : Audio
        Audio time series.
    short_window : int
        Length of the short term window for the calculation of the RMS.
    long_window : int
        Length of the long term window for the calculation of the RMS.
    hop:
        Length of the hop for the sliding window.
    alpha:
        Reduction factor for the long term RMS curve.
    plot_rms_curves:
        Whether of not to plot the RMS curves used to calculate the ODF.
    min_time : float
        Minimum time (in seconds) between successive onsets.
    odf_threshold : float
        Ratio of the ODF maxima to be defined as a minimum threshold
        for the peak picking.
    plot: bool
        Whether of not to plot the results.
    return_odf_data: bool
        Whether or not to return the odf data.

    Return
    ------
    onsets: PointList
        List of onsets.
    odf_data: TimeSeries
        Time series containing the onset detection function obtained. This will
        only be returned if the argument `return_odf_data` has been set to
        True.
    """
    onsets, odf_data = extract_from_odf(
        audio,
        odf_adaptive_rms,
        long_window=long_window,
        short_window=short_window,
        hop=hop,
        alpha=alpha,
        plot_rms_curves=plot_rms_curves,
        min_time=min_time,
        odf_threshold=odf_threshold,
        odf_threshold_criteria='relative_to_max',
        plot=plot)

    if return_odf_data:
        return onsets, odf_data
    else:
        return onsets


def rms_derivative(audio,
                   window=1024,
                   hop=512,
                   min_time=None,
                   odf_threshold=0.2,
                   plot=False,
                   return_odf_data=False):
    """
    Extract note onsets from the ``audio`` time-series using its ``rms``.
    The RMS will be calculated if it's not passed as an argument. The argument
    ``min_time`` can be used to specify the minimum distance (in seconds)
    between two adjacent onsets.

    Args
    ----
    audio : Audio
        Audio object
    window : int
        Window length for computing the RMS.
    hop : int
        Hop length for computing the RMS.
    min_time : float, optional
        Minimum time (in seconds) between successive onsets.
    odf_threshold : float
        Minimum threshold for the peak picking in the ODF curve.
    plot: bool
        Whether of not to plot the results
    return_odf_data: bool
        Whether or not to return the odf data

    Return
    ------
    onsets : list
        List of onset points.
    odf_data: TimeSeries
        Time series containing the onset detection function obtained. This will
        only be returned if the argument `return_odf_data` has been set to
        True.
    """
    onsets, odf_data = extract_from_odf(
        audio,
        odf_rms_derivative,
        window=window,
        hop=hop,
        min_time=min_time,
        odf_threshold=odf_threshold,
        plot=plot)

    if return_odf_data:
        return onsets, odf_data
    else:
        return onsets


def pitch_variation(audio,
                    window,
                    hop,
                    minf0=120,
                    maxf0=4000,
                    smooth_pitch=True,
                    min_time=None,
                    odf_threshold=0.04,
                    plot=False,
                    return_odf_data=False):
    """
    Extract note onsets from the ``audio`` time-series using its ``pitch``.
    The argument ``min_time`` can be used to specify the minimum distance (in
    seconds) between two adjacent onsets.

    Args
    ----
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
    min_time : float
        Minimum time (in seconds) between successive onsets.
    odf_threshold : float
        Minimum threshold for the peak picking in the ODF curve
    plot: bool
        Whether of not to plot the results
    return_odf_data: bool
        Whether or not to return the odf data

    Return
    ------
    onsets : list
        List of onset points.
    odf_data: TimeSeries
        Time series containing the onset detection function obtained. This will
        only be returned if the argument `return_odf_data` has been set to
        True.
    """
    odf = odf_pitch_change

    onsets, odf_data = extract_from_odf(
        audio,
        odf,
        min_time=None,
        odf_threshold=odf_threshold,
        hop=hop,
        smooth_pitch=smooth_pitch,
        plot=plot)

    if return_odf_data:
        return onsets, odf_data
    else:
        return onsets


def extract_from_odf(audio,
                     odf,
                     min_time=None,
                     odf_threshold=0.2,
                     odf_threshold_criteria='absolute',
                     plot=False,
                     **parameters):
    """
    Generic method to extract onsets from an ODF (onset detection function).

    Arguments
    ---------
    audio : Audio
        Audio time series.
    odf : function
        Reference to the ODF.
    min_time : float
        Minimum time (in seconds) between successive onsets.
    odf_threshold : float
        Minimum ODF threshold for a peak to be considered as an onset.
    odf_threshold_criteria : string ['absolute', 'relative_to_max']
        Specifies how the argument ``odf_threshold`` will be used: if
        ``'absolute'`` its value will be used directly as the threshold;
        else, if ``'relative_to_max'``, its value will be used to calculate
        the threshold, relative to the maximum value in the ODF curve, e.g.:
        ``odf_threshold``==`0.2` set the threshold to 20% of the maximum value
        of the ODF curve.
    plot : bool
        Whether or not to plot the results.

    Return
    ------
    onsets: PointList
        List of onsets.
    odf_data: TimeSeries
        Time series containing the onset detection function obtained.
    """
    odf_data = odf(audio, **parameters)

    if min_time:
        min_dist = int(min_time * odf_data.fs)
        if min_dist == 0:
            min_dist = None
    else:
        min_dist = None

    if odf_threshold_criteria == 'absolute':
        threshold = odf_threshold
    elif odf_threshold_criteria == 'relative_to_max':
        threshold = odf_threshold * np.max(odf_data.data)
    else:
        raise ValueError(
            ("Invalid value for argument `odf_threshold_criteria`: "
             f"'{odf_threshold_criteria}'")
        )

    ix_onsets, _ = sig.find_peaks(
        odf_data.data, height=threshold, distance=min_dist)

    onsets = iracema.segment.PointList(
        [iracema.segment.Point(odf_data, position) for position in ix_onsets])

    if plot:
        waveform_trio_features_and_points(audio, odf_data, onsets)

    return onsets, odf
