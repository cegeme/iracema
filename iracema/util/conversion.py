"""
Methods for converting values between different units.
"""
from decimal import Decimal

import numpy as np


def sample_index_to_seconds(sample_index, fs, time_offset=0):
    """
    Convert a sample index to time (in seconds).

    Args
    ----
    sample_index: int
        The index of a sample in a time series.
    fs: float
        Sampling frequency.
    time_offset: float
        Time offset to be added to the result (in seconds).
    """
    sample_index = Decimal(sample_index)
    fs = Decimal(fs)
    time_offset = Decimal(time_offset)
    return sample_index / fs + time_offset


def seconds_to_sample_index(time, fs, time_offset=0):
    """
    Convert time (in seconds) to sample index.

    Args
    ----
    time: float
        Time in seconds.
    fs: float
        Sampling frequency.
    time_offset: float
        Time offset (in seconds) to be subtracted from `time` passed.

    Note
    ----
    The return value will be rounded to an integer number. Therefore, you must
    be careful when doing these conversion operation between sample index and
    time, not to incur in loss of precision.
    """
    return int((time - time_offset) * fs)


def map_sample_index(sample_index, source_fs, source_time_offset, target_fs,
                     target_time_offset):
    """
    Convert a given ``sample_index`` for ``source_fs`` and ``time_offset``
    into the corresponding sample index for ``target_fs`` and
    ``target_time_offset``.

    Args
    ----
    sample_index : int
    source_fs : float
    source_time_offset : float
    target_fs : float
    target_time_offset : float

    Return
    ------
    target_sample_index : int
    """
    seconds = sample_index_to_seconds(
        sample_index, source_fs, time_offset=source_time_offset)

    return seconds_to_sample_index(
        seconds, target_fs, time_offset=target_time_offset)


def amplitude_to_db(amplitude, clip_min=1.e-10):
    """
    Convert amplitude to dB.
    """
    return energy_to_db(amplitude**2, clip_min=clip_min)


def energy_to_db(energy, clip_min=1.e-20):
    """
    Convert energy to dB.
    """
    return 10 * np.log10(np.clip(energy, clip_min, None))


def db_to_amplitude(db):
    """
    Convert dB to amplitude.
    """
    return 10 ** (db / 20)


def db_to_energy(db):
    """
    Convert dB to energy.
    """
    return 10 ** (db / 10)
