"""
This module contains the implementation of the class ``Audio``.
"""
from iracema.core.timeseries import TimeSeries
from iracema.io.audiofile import read as _read
from iracema.io import player


class Audio(TimeSeries):
    """
    Class for storing and manipulating audio data.

    .. Hint:: This class is also available at the main package level as
        ``iracema.Audio``.
    """

    def __init__(self, fs, data, start_time=None, caption=None):
        """
        Example
        -------
        There are two ways to instantiate an ``Audio`` object: from audio files
        or from `NumPy`_ arrays. To initialize it using an audio file, you must
        use the method ``Audio.load()``, specifying the location from which the
        file must be loaded:

        >>> ir.Audio.load('02 - Clarinet - Long Notes.wav')
        Audio: (1, 1785244), fs=44100.0, unit=amplitude, label=waveform

        Alternatively the location can be specified trough an http URL:

        >>> url = (
        ... 'https://raw.githubusercontent.com/cegeme/iracema-audio/master/03 - '
        ... 'Clarinet - Fast Excerpt.wav'
        ... )
        >>> ir.Audio.load(url)
        Audio: (1, 390060), fs=44100.0, unit=amplitude, label=waveform

        To initialize the object using a NumPy array, you can use the
        initializer of the class ``Audio()``. In this case, two arguments
        are necessary: the sampling frequency ``fs`` of the data and the
        ``data`` array.

        >>> ir.Audio(44100, clarinet_data)
        Audio: (1, 390060), fs=44100.0, unit=amplitude, label=waveform

        .. _NumPy:
           https://numpy.org/doc/

        Args
        ----
        fs : float
            Sampling frequency for the data.
        data : numpy array
            Data array containing the audio samples.
        start_time : Decimal, optional
            The time in seconds the time series start, relative to the original
            time reference.
        unit : str, optional
            Unit name for plotting the data of the time series.
        caption : str, optional
            Text caption for the time series.
        data : np.array, optional
            Data vector containing the audio data to be loaded.
        fs : int, optional
            Sampling frequency of the data.
        caption : str, optional
            Textual description used for plotting and displaying reports about
            the audio excerpt.
        """
        unit = 'amplitude'
        self.label = 'waveform'
        self.filename, self.caption = None, caption

        super(Audio, self).__init__(
            fs, data=data, unit=unit, start_time=start_time, caption=caption)

    @classmethod
    def load(cls, file_location, caption=None):
        """
        Load an audio file into an ``Audio`` object.

        Parameters
        ----------
        file_location : str
            Location from which the file will be loaded. The string might
            contain a path pointing to a local file or an http URL referencing
            a remote file
        caption : str, optional
            Caption for the audio file loaded (optional). If this argument is
            not provided, the base name of the loaded file will be used.

        Return
        ------
        audio : Audio
            An object of the class ``Audio``, containing the data loaded from
            the specified location.
        """
        data, fs, base_name = _read(file_location)
        caption = caption or base_name
        audio = cls(fs, data, caption=caption)
        audio.filename = base_name
        return audio

    def plot(self, linewidth=0.1, alpha=0.9, **kwargs):
        """
        Plot the time series. The ``linewidth`` and ``alpha`` (opacity) values
        can be set using keyword arguments.
        """
        return super(Audio, self).plot(
            linewidth=linewidth, alpha=alpha, **kwargs)

    def play(self):
        """
        Play audio from Audio object.
        """
        return player.play(self)

    def play_with_clicks(self, points):
        """
        Play audio mixed with click sounds in the instants corresponding
        to ``points``.
        """
        return player.play_with_clicks(self, points)

    def play_from_time(self, from_time):
        """
        Play audio from Audio object start at time ``from_time``.
        """
        return player.play_interval_seconds(self, from_time, None)

    def play_segment(self, segment):
        """
        Play segment from Audio obejct.
        """
        return player.play_interval_seconds(self, segment.start_time,
                                            segment.end_time)

    def stop(self):
        """
        Stop playing audio.
        """
        player.stop()
