import pytest

from iracema.audio import Audio


def test_load_audio_from_file(audio01):
    a = audio01
    assert isinstance(a, Audio)
    assert audio01.fs == 44100.
    assert audio01.filename == "01 - Guitar - Stil Got The Blues.wav"
    assert audio01.start_time == 0.
