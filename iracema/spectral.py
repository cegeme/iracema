"""
Extraction of spectral information.
"""
from decimal import Decimal

import numpy as np
from deprecated.sphinx import deprecated
from librosa.filters import mel
from librosa.core.convert import mel_frequencies
from librosa import cqt, hybrid_cqt, cqt_frequencies

from iracema.util.windowing import apply_sliding_window
from iracema.util import conversion
import iracema.core.timeseries


class STFT(iracema.core.timeseries.TimeSeries):
    "Compute the Short-Time Fourier Transform for the ``time_series``."

    def __init__(self, time_series, window_size, hop_size, fft_len=None):
        """
        Args
        ----
        time_series : TimeSeries
            Time series for applying the STFT.
        window_size : int
        hop_size : int
        fft_len : int
            Length of the FFT. The signal will be zero-padded if ``fft_len`` >
            ``window_size``. The default value is equal to `window_size`.
        """
        if not fft_len:
            fft_len = window_size

        def calculate(x):
            return np.fft.rfft(x, n=fft_len, norm='ortho')

        stft_data = apply_sliding_window(
            time_series.data,
            window_size,
            hop_size,
            calculate,
            window_name='hann')

        new_fs = Decimal(time_series.fs) / Decimal(hop_size)

        super(STFT, self).__init__(
            new_fs,
            data=stft_data,
            start_time=time_series.start_time,
            caption=time_series.caption)

        self.max_frequency = float(time_series.nyquist)
        self.frequencies = np.fft.rfftfreq(fft_len, 1. / float(time_series.fs))

        self.label = 'STFT'
        self.unit = 'complex'

    def magnitude(self, power=2., db=False):
        if float(power) not in (1., 2.):
            raise ValueError(
                'The argument `power` must be equal to 1.0 or 2.0')
        magnitude = np.abs(self.data)**power
        if db:
            if power == 1.0:
                magnitude = conversion.amplitude_to_db(magnitude)
            elif power == 2.0:
                magnitude = conversion.energy_to_db(magnitude)
        return magnitude

    def phase(self):
        return np.angle(self.data)


class Spectrogram(iracema.core.timeseries.TimeSeries):
    "Generate spectrogram for the given `time_series`."

    def __init__(self,
                 time_series,
                 window_size,
                 hop_size,
                 fft_len=None,
                 power=2.,
                 db=False):
        """
        Args
        ----
        time_series : TimeSeries
            Time series for applying the STFT.
        window_size : int
        hop_size : int
        fft_len : int
            Length of the FFT. The signal will be zero-padded if ``fft_len`` >
            ``window_size``. The default value is equal to `window_size`.
        power : float
            Exponent for the spectrogram.
        db : bool
            Whether or not to convert the output values to dB.
        """
        stft = STFT(time_series, window_size, hop_size, fft_len=fft_len)
        data = stft.magnitude(power=power, db=db)

        super(Spectrogram, self).__init__(
            stft.fs,
            data=data,
            start_time=stft.start_time,
            caption=stft.caption)

        self.max_frequency = stft.max_frequency
        self.frequencies = stft.frequencies

        self.label = 'Spectrogram'
        self.unit = 'Magnitude'
        self._power = power
        self._db = db


class MelSpectrogram(iracema.core.timeseries.TimeSeries):
    def __init__(self,
                 time_series,
                 window_size,
                 hop_size,
                 n_mels=256,
                 fft_len=None,
                 power=2.,
                 db=False,
                 fmin=0.,
                 fmax=None,
                 htk=False):
        """
        Compute a mel spectrogram for ``time_series``.

        Args
        ----
        time_series : TimeSeries
            Time series for applying the STFT.
        window_size : int
        hop_size : int
        n_mels : int
            Number of mel-scaled filters/channels.
        fft_len : int
            Length of the FFT. The signal will be zero-padded if ``fft_len`` >
            ``window_size``. The default value is equal to `window_size`.
        power : float
            Exponent for the spectrogram.
        db : bool
            Whether or not to convert the output values to dB.
        fmin : float
            Frequency of the lowest filter.
        fmax : float
            Frequency of the highest filter.
        htk : bool
        """
        if not fft_len:
            fft_len = window_size

        spec = Spectrogram(
            time_series,
            window_size,
            hop_size,
            fft_len=fft_len,
            power=power,
            db=False)

        fmax = fmax or spec.max_frequency
        mel_basis = mel(
            time_series.fs, fft_len, n_mels=n_mels, fmin=fmin, fmax=fmax)
        data = np.dot(mel_basis, spec.data)

        if db:
            if power == 1.0:
                data = conversion.amplitude_to_db(data)
            elif power == 2.0:
                data = conversion.energy_to_db(data)

        super(MelSpectrogram, self).__init__(
            spec.fs,
            data=data,
            start_time=spec.start_time,
            caption=spec.caption)

        self.frequencies = mel_frequencies(
            n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk)
        self.max_frequency = spec.frequencies[-1]
        self.label = 'Mel Spectrogram'
        self.unit = 'Magnitude'
        self._power = power
        self._db = db


class CQT(iracema.core.timeseries.TimeSeries):
    def __init__(self,
                 time_series,
                 hop_size,
                 n_bins=84,
                 bins_per_octave=12,
                 hybrid=True,
                 power=2.,
                 fmin=None,
                 db=False,
                 **kwargs):
        """
        Compute the Constant Q Transform for ``time_series``.

        Args
        ----
        time_series : TimeSeries
            Time series for applying the STFT.
        window_size : int
        hop_size : int
        fft_len : int
            Length of the FFT. The signal will be zero-padded if ``fft_len`` >
            ``window_size``. The default value is equal to `window_size`.
        power : float
            Exponent for the spectrogram.
        db : bool
            Whether or not to convert the output values to dB.
        n_mels : int
            Number of mel-scaled filters/channels.
        fmin : float
            Frequency of the lowest filter.
        fmax : float
            Frequency of the highest filter.
        htk : bool
        """
        new_fs = Decimal(time_series.fs) / Decimal(hop_size)
        if hybrid:
            cqt_func = hybrid_cqt
        else:
            cqt_func = cqt
        data = np.abs(
            cqt_func(time_series.data,
                sr=float(time_series.fs),
                hop_length=hop_size,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                fmin=fmin,
                **kwargs
            )
        )
        if db:
            if power == 1.0:
                data = conversion.amplitude_to_db(data)
            elif power == 2.0:
                data = conversion.energy_to_db(data)
        frequencies = cqt_frequencies(n_bins,
                                      fmin,
                                      bins_per_octave=bins_per_octave,
                                      tuning=kwargs.get('tuning') or 0)
        super(CQT, self).__init__(
            new_fs,
            data=data,
            start_time=time_series.start_time,
            caption=time_series.caption)

        self.frequencies = frequencies
        self.max_frequency = float(frequencies[-1])

        self.label = 'CQT'
        self.unit = 'Magnitude'
        self._power = power
        self._db = db


@deprecated(version='0.2.0', reason='Deprecated method. Use `STFT` instead.')
def fft(*args, **kwargs):
    "Deprecated FFT method."
    return STFT(*args, **kwargs)
