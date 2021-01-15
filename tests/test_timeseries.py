import pytest  # skipcq: PYL-W0611

from iracema import TimeSeries


def test_resample(audio00):
    resampled_audio = audio00.resample(10000)
    assert resampled_audio.fs == 10000
