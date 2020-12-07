import pytest

from iracema import Audio


@pytest.fixture(scope="module")
def audio00():
    return Audio.load('audio/iracema-audio/00 - Flute - Iracema.wav')

@pytest.fixture(scope="module")
def audio01():
    return Audio.load('audio/iracema-audio/01 - Guitar - Stil Got The Blues.wav')

@pytest.fixture(scope="module")
def audio02():
    return Audio.load('audio/iracema-audio/02 - Clarinet - Long Notes.wav')

@pytest.fixture(scope="module")
def audio03():
    return Audio.load('audio/iracema-audio/03 - Clarinet - Fast Excerpt.wav')

@pytest.fixture(scope="module")
def audio_url():
    return Audio.load('https://raw.githubusercontent.com/cegeme/iracema-audio/master/03 - Clarinet - Fast Excerpt.wav')

def pytest_addoption(parser):
    parser.addoption(
        "--no-audio-play", action="store_true", default=False, help="do not run tests that play audio"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "audio-play: mark test that plays audio")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-audio-play"):
        skip_audio_play = pytest.mark.skip(reason="need --no-audio-play option to run")
        for item in items:
            if "audio_play" in item.keywords:
                item.add_marker(skip_audio_play)
