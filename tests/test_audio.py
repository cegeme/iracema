import pytest

from iracema import Audio


def test_load_audio_from_file(audio01):
    a = audio01
    assert isinstance(a, Audio)
    assert audio01.fs == 44100.
    assert audio01.filename == "01 - Guitar - Stil Got The Blues.wav"
    assert audio01.start_time == 0.

def test_load_audio_from_url(audio_url):
    a = audio_url
    assert isinstance(a, Audio)
    assert audio_url.fs == 44100.
    assert audio_url.filename == "03 - Clarinet - Fast Excerpt.wav"
    assert audio_url.start_time == 0.

def test_audio_play(audio01):
    a = audio01
    a.play()
