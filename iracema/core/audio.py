"""
Definition of the class Audio.
"""
from iracema.core.timeseries import TimeSeries
from iracema.io.audiofile import read
from iracema.io import player
from iracema.plot import plot_curve


class Audio(TimeSeries):
    """
    Stores audio data, which can be loaded from a file or an array.

    Parameters
    ----------
    file_location : str, optional
        Location from where the file will be loaded. The string might
        contain a path pointing to a local file or an http URL referencing
        a remote file.
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

    To initialize it using an audio file, you just need to pass the location
    from which the file must be loaded:

    >>> a = ir.Audio('03 - Clarinet - Fast Excerpt.wav')

    Alternatively the location can be specified trough an http URL:

    >>> url = (
    ... 'https://raw.githubusercontent.com/cegeme/iracema-audio/master/03 - '
    ... 'Clarinet - Fast Excerpt.wav'
    ... )
    >>> a = ir.Audio(url)

    To initialize it using a NumPy array, two arguments are necessary: the
    ``data`` array and the sampling frequency ``fs``:

    >>> a = ir.Audio(clarinet_data, 44100)  # doctest: +SKIP

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
        self.label = 'waveform'

        nargs = len(args)
        caption = kwargs.get('caption', None)

        # one argument: file name
        if nargs == 1:
            filename = args[0]
            data, fs, basename = read(filename)
            self.filename = basename
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


    def plot(self, linewidth=0.1, alpha=0.9):
        """
        Plot the time series using matplotlib.
        Line width and alpha values can be set as optional parameters.
        """
        return plot_curve(self, linewidth=linewidth, alpha=alpha)


    def play(self):
        """
        Play audio from Audio object.
        """
        return player.play(self)

    def play_from_time(self, from_time):
        """
        Play audio from Audio object start at time ``from_time``.
        """
        return player.play_interval_seconds(self, from_time, None)

    def play_segment(self, segment):
        """
        Play segment from Audio obejct.
        """
        return player.play_interval_seconds(self, segment.start_time, segment.end_time)

    def stop(self):
        """
        Stop playing audio.
        """
        player.stop()

