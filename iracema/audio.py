"""
Implementation of audio time series.
"""
from os import path

from .timeseries import TimeSeries
from .io.audiofile import read
from .io import player


class Audio(TimeSeries):
    """
    Stores audio data, which can be loaded from a file or an array.

    Parameters
    ----------
    filename : str, optional
        Name of the audio file to be loaded.
    data : np.array
        Data vector containing the audio data to be loaded.
    fs : int, optional
        Sampling frequency of the data.
    caption : str, optional
        caption for the audio file loaded (optional).

    Examples
    --------
    There are two different ways to initialize an Audio object: from
    audio files or from NumPy arrays.

    To initialize it using an audio file, you just need to pass the
    ``filename`` to be loaded:

    >>> a1 = Audio('clarinet.wav')

    To initialize it using a NumPy array, two arguments are necessary: the
    ``data`` array and the sampling frequency ``fs``:

    >>> a2 = Audio(clarinet_data, 44100)

    There is also the optional parameter ``caption`` which is a
    textual description used for plotting and displaying reports
    about the audio file. In case you load the audio from a file
    and don't specify a caption, the filename will be assigned to
    it.
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor for Audio class.
        """
        self.unit = 'amplitude'
        nargs = len(args)
        caption = kwargs.get('caption', None)

        # one argument: file name
        if nargs == 1:
            filename = args[0]
            filename = path.expanduser(filename)  # expanding ~ to /home/user

            # loading audio file
            data, fs = read(filename)
            self.filename = path.basename(filename)
            self.caption = caption or self.filename

        # two arguments: an array and a sampling frequency
        elif nargs == 2:
            data, fs = args[0], args[1]
            self.filename, self.caption = None, caption

        else:
            raise (TypeError(
                'invalid number of positional arguments: should be '
                '1 or 2'))

        super(Audio, self).__init__(fs, data=data, unit=self.unit)

    def play(self):
        """
        Play audio from Audio object.
        """
        player.play(self)

    def play_from_time(self, from_time):
        """
        Play audio from Audio object start at time ``from_time``.
        """
        player.play_interval_seconds(self, from_time, None)

    def stop(self):
        """
        Stop playing audio.
        """
        player.stop()
