"""
This module contains the implementation of the class ``Audio``.
"""
from librosa.effects import time_stretch, pitch_shift
import resampy
import numpy as np

from iracema.core.timeseries import TimeSeries
from iracema.io.audiofile import read as _read
from iracema.io import player
from iracema.util import conversion


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
        self.filepath = None

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
        audio.filepath = file_location
        return audio

    def resample(self, new_fs):
        """
        Resample time series to ``new_fs``.
        """
        if self.start_time != 0:
            raise (NotImplementedError(
                'The method resample is implemented only for time series '
                'objects with start_time equal to 0.'))
        if self.fs == new_fs:
            return self
        new = self.copy()
        new.data = resampy.resample(self.data, float(self.fs), float(new_fs))
        new.fs = new_fs

        return new
    
    def pitch_shift(self, n_steps, **kwargs):
        """
        Shift the pitch of the audio time series and return the new object.

        This method is a wrapper over librosa_'s ``pitch_shift`` method.

        .. _librosa: https://librosa.org
        """
        new = self.copy()
        new.data = pitch_shift(new.data, new.fs, n_steps, **kwargs)

        return new

    def time_stretch(self, rate, **kwargs):
        """
        Time-stretch the audio time series and return the new object.

        This method is a wrapper over librosa_'s ``time_stretch`` method.
        """
        new = self.copy()
        new.data = time_stretch(new.data, rate, **kwargs)

        return new

    def add_noise(self, db=-50.):
        """
        Add noise to the audio time series and return the new object.

        Arguments
        ---------
        db : float
            Peak amplitude (in dB) of the generated noise, with an amplitude
            of 1.0 correspoding to 0 dB.
        """
        scale_factor = conversion.db_to_amplitude(db)
        new = self.copy()
        rand = np.random.uniform(-scale_factor,scale_factor,self.nsamples)
        new.data = self.data + (rand)

        return new

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

    def mix_clicks(self, points):
        """
        Return an audio time series with clicks mixed in the instants
        corresponding to ``points``.
        """
        return player.play_with_clicks(self, points, return_time_series=True)

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

    @staticmethod
    def stop():
        """
        Stop playing audio.
        """
        player.stop()
