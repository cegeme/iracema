"""
Methods for reading and writing raw audio files.
"""
import numpy as np
import audioread


def read(filename):
    """
    Read WAV audio file.

    Arguments
    ---------
    filename: str
        Name of the audio file to be loaded.

    Return
    ------
    data: numpy array
        Audio data.
    fs: int
        Sampling frequency.
    """

    intmaxabs = 32768.  # maximum value for 16-bit signed integers
    fs = None

    with audioread.audio_open(filename) as input_file:
        fs = input_file.samplerate
        channels = input_file.channels

        # Audioread returns buffers containing 16-bit signed integers.
        data_int = np.array([], dtype=np.dtype('int16'))

        for frame in input_file:
            frame_int = np.frombuffer(frame, np.dtype('int16'))
            data_int = np.concatenate((data_int, frame_int), axis=0)

        # Convert data to float
        data = data_int.astype(
            np.float_, casting='safe')  # pylint: disable=maybe-no-member

        # Conversion to mono (mix both channels)
        if channels > 1:
            data = data.reshape((-1, channels)).T
            data = np.mean(data, axis=0)

        data = data / intmaxabs

    return (data, fs)


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
    # TODO: Implement method to write wave files.
    pass
