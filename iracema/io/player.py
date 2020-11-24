"""
Methods for playing audio.
"""
from pkg_resources import resource_filename
import sounddevice as sd
import numpy as np

import iracema.timeseries


def play(audio_time_series, blocking=False):
    """
    Play audio time series.

    Args
    ----
    audio_time_series: iracema.audio.Audio
    blocking: bool
    """
    return _play_stream(audio_time_series, blocking=blocking)


def play_with_clicks(audio_time_series, points, click_file=None, blocking=False):
    """
    Play audio with clicks in the instants corresponding to the specified
    points. The path to the audio file containing the click sound must be
    provided.

    Args
    ----
    audio_time_series: iracema.timeseries.Audio
    points: iracema.segments.PointList
    click_file: string
    blocking: bool
    """
    if not click_file:
        click_file = resource_filename('iracema', 'assets/audio/Click.wav')
    click_sound = iracema.timeseries.Audio(click_file)
    indexes = points.map_indexes(audio_time_series)
    audio_with_clicks = audio_time_series.copy()
    for i in indexes:
        audio_with_clicks.data[i:i+click_sound.nsamples] += click_sound.data

    return _play_stream(audio_with_clicks, blocking=blocking)


def play_interval_samples(audio_time_series, from_sample, to_sample,
                          soft_start=True, blocking=False):
    """
    Play audio interval from sample `from_sample` to `to_sample`.

    Args
    ----
    audio_time_series: iracema.timeseries.Audio
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

    return _play_stream(soft_start * data_to_play, blocking=blocking)


def play_interval_seconds(audio_time_series, from_seconds, to_seconds,
                          soft_start=True, blocking=False):
    """
    Play audio interval from time `from_seconds` to `to_seconds`.

    Args
    ----
    audio_time_series: iracema.timeseries.Audio
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

def _play_stream(audio_time_series, blocking=False):
    sd.default.blocksize = 256
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            from IPython.display import Audio as IPythonAudio
            return IPythonAudio(data=audio_time_series.data,
                                rate=audio_time_series.fs,
                                autoplay=True)
        else:
            sd.play(audio_time_series.data, audio_time_series.fs, blocking=blocking)
            return None
    except NameError as err:
        sd.play(audio_time_series.data, audio_time_series.fs, blocking=blocking)
        return None


def stop():
    """
    Stop playing audio.
    """
    sd.stop()
