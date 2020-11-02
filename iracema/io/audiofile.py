"""
Methods for reading and writing raw audio files.
"""
import re

from os.path import basename, expanduser
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import audioread
import numpy as np
import scipy.io.wavfile

from w3lib.url import canonicalize_url
from w3lib.url import file_uri_to_path

def read(file_location):
    """
    Read WAV audio file from the local file system or download it from URL.

    Arguments
    ---------
    file_location: str
        Path or URL to the file that will be loaded.

    Return
    ------
    data: numpy array
        Audio data.
    fs: int
        Sampling frequency.
    file_name: str
        Base name of the loaded file.
    """
    intmaxabs = 32768.  # maximum value for 16-bit signed integers
    fs = None
    temp_file = None

    if re.match('https://|http://', file_location):
        url = canonicalize_url(file_location)
        response = urlopen(url)
        temp_file = NamedTemporaryFile()
        temp_file.write(response.read())
        file_path = temp_file.name
        file_name = basename(file_uri_to_path(file_location))
    else:
        file_path = expanduser(file_location)
        file_name = basename(file_path)

    with audioread.audio_open(file_path) as input_file:
        fs = input_file.samplerate
        channels = input_file.channels

        # audioread returns buffers containing 16-bit signed integers
        data_int = np.array([], dtype=np.dtype('int16'))
        for frame in input_file:
            frame_int = np.frombuffer(frame, np.dtype('int16'))
            data_int = np.concatenate((data_int, frame_int), axis=0)
        # convert data to float
        data = data_int.astype(
            np.float_, casting='safe')  # pylint: disable=maybe-no-member

        # Conversion to mono (mix both channels)
        if channels > 1:
            data = data.reshape((-1, channels)).T
            data = np.mean(data, axis=0)

        data = data / intmaxabs

    if temp_file:
        temp_file.close()

    return (data, fs, file_name)


def write(filename, data, fs):
    """
    Write the contents of a `data` array to a WAVE file.

    Arguments
    ---------
    filename: str
        Name of the audio file to be loaded.
    data: numpy array
        Data array containing audio samples.
    fs: int
        Sampling frequency.
    """
    scipy.io.wavfile.write(filename, fs, data)
