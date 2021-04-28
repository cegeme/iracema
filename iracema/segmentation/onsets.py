"""
Note onset detection methods.
"""
import warnings 
from pathlib import Path

import numpy as np
import scipy.signal as sig
from tensorflow.keras.models import load_model

import iracema.features
import iracema.pitch
import iracema.core.segment
import iracema.core.point
from iracema.plot import waveform_trio_features_and_points
from iracema.segmentation.odfs import (odf_rms_derivative, odf_pitch_change,
                                       odf_adaptative_rms)
from iracema.util.ml import activations_to_points
from .util import convolve_activations, three_sliced_mel_spectrograms


def cnn_model(
    audio,
    instrument='clarinet',
    smooth_odf=True,
    odf_threshold = 0.328,
    display_plot = False,
    return_odf_data = False,
):
    """
    Extract the note onsets using the CNN method.

    Arguments
    ---------
    audio: ir.Audio
        Audio file to be processed.
    instrument: string
        Name of the instrument (currently trained only for clarinet).
    smooth_odf: bool
        If true, the final ODF will be smoothed by convolving it with a hanning
        window of length 5.
    odf_threshold : float
        Minimum threshold for the peak picking in the ODF curve.
    display_plot : bool
        Whether of not to plot the results
    return_odf_data : bool
        Whether or not to return the odf data

    """
    audio_ = audio.copy()
    if float(audio_.fs) != 44100.:
        audio_ = audio_.resample(44100)

    if instrument != 'clarinet':
        warnings.warn(
            "This model is specialized in clarinet recordings. It might not "
            "perform well for other instruments."
        )

    window, hop = 1024, 441
    sliced_spectrogram, frame_fs = three_sliced_mel_spectrograms(
        audio_,
        window,
        hop,
        frames_per_slice = 15,
        n_mels = 80,
        fmin = 27.5,
        fmax = 16000,
        db = True,
    )

    here = Path(__file__).parent
    model_file = here/'clari-onsets.h5'
    model = load_model(model_file)
    y_pred = model.predict(sliced_spectrogram)
    if smooth_odf:
        y_pred[:, 0] = convolve_activations(y_pred[:, 0])

    onsets = activations_to_points(
        y_pred,
        frame_fs,
        encoding='binary',
        threshold=odf_threshold,
        peak_pick=True,
    )
    odf_data = y_pred[:, 0]

    if display_plot:
        waveform_trio_features_and_points(audio_, odf_data, onsets)

    if return_odf_data:
        return onsets, odf_data
    return onsets


def adaptative_rms(
        audio,
        short_window=512,
        long_window=4096,
        hop=512,
        alpha=0.1,
        min_time=None,
        odf_threshold=0.2,
        display_plot=False,
        display_plot_rms=False,
        return_odf_data=False,
):
    """
    Extract the note onsets using the adaptative RMS method.

    Arguments
    ---------
    audio : Audio
        Audio time series.
    short_window : int
        Length of the short term window for the calculation of the RMS.
    long_window : int
        Length of the long term window for the calculation of the RMS.
    hop : int
        Length of the hop for the sliding window.
    alpha : float
        Reduction factor for the long term RMS curve.
    display_plot_rms : bool
        Whether of not to plot the RMS curves used to calculate the ODF.
    min_time : float
        Minimum time (in seconds) between successive onsets.
    odf_threshold : float
        Ratio of the ODF maxima to be defined as a minimum threshold
        for the peak picking.
    display_plot : bool
        Whether of not to plot the results.
    return_odf_data : bool
        Whether or not to return the odf data.

    Return
    ------
    onsets : PointList
        List of onsets.
    odf_data : TimeSeries
        Time series containing the onset detection function obtained. This will
        only be returned if the argument `return_odf_data` has been set to
        True.
    """
    onsets, odf_data = extract_from_odf(
        audio,
        odf_adaptative_rms,
        long_window=long_window,
        short_window=short_window,
        hop=hop,
        alpha=alpha,
        display_plot_rms=display_plot_rms,
        min_time=min_time,
        odf_threshold=odf_threshold,
        odf_threshold_criteria="relative_to_max",
        display_plot=display_plot,
    )

    if return_odf_data:
        return onsets, odf_data
    return onsets


def rms_derivative(
        audio,
        window=1024,
        hop=512,
        min_time=None,
        odf_threshold=0.2,
        display_plot=False,
        return_odf_data=False,
):
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
    display_plot : bool
        Whether of not to plot the results
    return_odf_data : bool
        Whether or not to return the odf data

    Return
    ------
    onsets : list
        List of onset points.
    odf_data : TimeSeries
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
        display_plot=display_plot,
    )

    if return_odf_data:
        return onsets, odf_data
    return onsets


def pitch_variation(
        audio,
        window,
        hop,
        minf0=120,
        maxf0=4000,
        smooth_pitch=True,
        min_time=None,
        odf_threshold=0.04,
        display_plot=False,
        return_odf_data=False,
):
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
    display_plot : bool
        Whether of not to plot the results
    return_odf_data : bool
        Whether or not to return the odf data

    Return
    ------
    onsets : list
        List of onset points.
    odf_data : TimeSeries
        Time series containing the onset detection function obtained. This will
        only be returned if the argument `return_odf_data` has been set to
        True.
    """
    odf = odf_pitch_change

    onsets, odf_data = extract_from_odf(
        audio,
        odf,
        window=window,
        hop=hop,
        min_time=min_time,
        odf_threshold=odf_threshold,
        minf0=minf0,
        maxf0=maxf0,
        smooth_pitch=smooth_pitch,
        display_plot=display_plot,
    )

    if return_odf_data:
        return onsets, odf_data
    return onsets


def extract_from_odf(
        audio,
        odf,
        min_time=None,
        odf_threshold=0.2,
        odf_threshold_criteria="absolute",
        display_plot=False,
        **parameters,
):
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
        ``odf_threshold``==``0.2`` set the threshold to 20% of the maximum
        value of the ODF curve.
    display_plot : bool
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

    if odf_threshold_criteria == "absolute":
        threshold = odf_threshold
    elif odf_threshold_criteria == "relative_to_max":
        threshold = odf_threshold * np.max(odf_data.data)
    else:
        raise ValueError(
            ("Invalid value for argument `odf_threshold_criteria`: "
             f"'{odf_threshold_criteria}'"))

    ixs_onsets, _ = sig.find_peaks(
        odf_data.data, height=threshold, distance=min_dist)

    onsets = iracema.core.point.PointList([
        iracema.core.point.Point.from_sample_index(ix, odf_data)
        for ix in ixs_onsets
    ])

    if display_plot:
        waveform_trio_features_and_points(audio, odf_data, onsets)

    return onsets, odf
