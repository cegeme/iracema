"""
A couple of different methods for note segmentation.
"""

import numpy as np
import scipy.signal as sig

import iracema.features
import iracema.pitch
import iracema.segment
from iracema.aggregation import aggregate_sucessive_samples

from iracema.plot import plot_waveform_trio_features_and_points


def extract_note_onsets_rms(audio, rms=None, min_time=None,
                            perc_threshold_pk=0.2):
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

    Return
    ------
    onsets : list
        List of onset points.
    """
    rms = rms or iracema.features.rms(audio, 2048, 512)
    # TODO: hardcoded parameters should be obtained in a better way

    # handling arguments
    if min_time:
        min_dist = int(min_time * pitch.fs)
        if min_dist == 0:
            min_dist = None
    else:
        min_dist = None

    # onset detection function
    onset_df = detection_function_rms(rms)

    # peak picking
    threshold = perc_threshold_pk * np.max(onset_df.data)
    ix_onsets, _ = sig.find_peaks(
        onset_df.data, height=threshold, distance=min_dist)

    print(ix_onsets)

    onsets = iracema.segment.PointList([
        iracema.segment.Point(rms, position)
        for position in ix_onsets
    ])

    print(onsets.time, onsets.get_values(onset_df))

    plot_waveform_trio_features_and_points(audio, onset_df, onsets)

    return onsets


def extract_note_onsets_pitch(audio, pitch, min_time=None, delta_pitch_ratio=0.03):
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
    onset_df = detection_function_pitch(pitch)

    # peak picking
    ix_onsets, _ = sig.find_peaks(
        onset_df.data, height=delta_pitch_ratio, distance=min_dist)

    onsets = iracema.segment.PointList([
        iracema.segment.Point(pitch, position)
        for position in ix_onsets
    ])

    #plot_waveform_trio_features_and_points(audio, onset_df, onsets)

    return onsets


def segment_notes_rms(audio, rms=None, min_time=None, perc_threshold_pk=0.05):
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
    onset_df = detection_function_rms(rms)

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


def segment_notes_pitch(audio, pitch, min_time=None):
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
    onset_df = detection_function_pitch(pitch)

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


def detection_function_rms(rms):
    """
    Onset detection function based on RMS.
    """
    return rms.diff().hwr() #* rms


def detection_function_pitch(pitch):
    """
    Onset detection function based on Pitch.
    """
    def ratio_successive(current, last):
        return abs((current / last) - 1)

    return aggregate_sucessive_samples(pitch, ratio_successive)
