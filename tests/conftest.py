import pytest

from iracema import Audio


@pytest.fixture(scope="module")
def audio00():
    return Audio('audio/00 - Flute - Iracema.wav')


@pytest.fixture(scope="module")
def audio01():
    return Audio('audio/01 - Guitar - Stil Got The Blues.wav')


@pytest.fixture(scope="module")
def audio02():
    return Audio('audio/02 - Clarinet - Long Notes.wav')


@pytest.fixture(scope="module")
def audio03():
    return Audio('audio/03 - Clarinet - Fast Excerpt.wav')
