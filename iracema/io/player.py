"""
Methods for playing audio.
"""
import sounddevice as sd
import numpy as np


def play(audio_time_series, blocking=False):
    """
    Play audio time series.

    Args
    ----
    audio_time_series: iracema.audio.Audio
    blocking: bool
    """
    sd.default.blocksize = 256
    sd.play(audio_time_series.data, audio_time_series.fs, blocking=blocking)


def play_interval_samples(audio_time_series, from_sample, to_sample,
                          soft_start=True, blocking=False):
    """
    Play audio interval from sample `from_sample` to `to_sample`.

    Args
    ----
    audio_time_series: iracema.audio.Audio
    from_sample: int
    to_sample: int
    soft_start: bool
    blocking: bool
    """
    data_to_play = audio_time_series.data[from_sample:to_sample]
    size_soft_start = int(audio_time_series.fs * 0.005)
    soft_start = np.concatenate(
        (np.linspace(0, 1, size_soft_start, dtype=np.float_),
         np.ones(data_to_play.shape[-1] - size_soft_start)))

    sd.default.blocksize = 256
    sd.play(soft_start * data_to_play, audio_time_series.fs, blocking=blocking)


def play_interval_seconds(audio_time_series, from_seconds, to_seconds,
                          soft_start=True, blocking=False):
    """
    Play audio interval from time `from_seconds` to `to_seconds`.

    Args
    ----
    audio_time_series: iracema.audio.Audio
    from_seconds: int
    to_seconds: int
    soft_start: bool
    blocking: bool
    """
    if from_seconds:
        from_sample = int((from_seconds - audio_time_series.start_time) * audio_time_series.fs)
    else:
        from_sample = None
    if to_seconds:
        to_sample = int((to_seconds - audio_time_series.start_time) * audio_time_series.fs)
    else:
        to_sample = None

    play_interval_samples(audio_time_series, from_sample, to_sample,
                          soft_start=soft_start, blocking=blocking)


def stop():
    """
    Stop playing audio.
    """
    sd.stop()
