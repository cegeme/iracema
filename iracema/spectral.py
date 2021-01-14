"""
Extraction of spectral information.
"""
from decimal import Decimal

import numpy as np
from deprecated.sphinx import deprecated
from librosa.filters import mel
from librosa.core.convert import mel_frequencies

from iracema.util.windowing import apply_sliding_window
import iracema.core.timeseries


class STFT(iracema.core.timeseries.TimeSeries):
    "Compute the Short-Time Fourier Transform for the ``time_series``."
    def __init__(self, time_series, window_size, hop_size, fft_len=4096):
        """
        Args
        ----
        time_series : TimeSeries
            Time series for applying the STFT.
        window_size : int
        hop_size : int
        fftlen : int
            Length of the FFT. The signal will be zero-padded if ``fftlen`` >
            ``rolling_window.window_size``.
        """
        def calculate(x):
            return np.fft.rfft(x, n=fft_len)

        stft_data = apply_sliding_window(
            time_series.data, window_size, hop_size, calculate, window_name='hann')

        new_fs = Decimal(time_series.fs) / Decimal(hop_size)

        super(STFT, self).__init__(
            new_fs, data=stft_data, start_time=time_series.start_time, caption=time_series.caption)

        self.max_frequency = time_series.nyquist
        self.frequencies = np.fft.rfftfreq(fft_len, 1. / time_series.fs)

        self.label = 'STFT'
        self.unit = ''

    def magnitude(self, power=2.):
        return np.abs(self.data) ** power

    def phase(self):
        return np.angle(self.data)


class Spectrogram(iracema.core.timeseries.TimeSeries):
    "Generate spectrogram for the given `time_series`."
    def __init__(self, time_series, window_size, hop_size, fft_len=4096, power=2.):
        """
        Args
        ----
        time_series : TimeSeries
            Time series for applying the STFT.
        window_size : int
        hop_size : int
        fftlen : int
            Length of the FFT. The signal will be zero-padded if ``fftlen`` >
            ``rolling_window.window_size``.
        power : float
            Exponent for the spectrogram.
        """
        stft = STFT(time_series, window_size, hop_size, fft_len=fft_len)
        data = stft.magnitude(power=power)

        super(Spectrogram, self).__init__(
            stft.fs, data=data, start_time=stft.start_time, caption=stft.caption)

        self.max_frequency = stft.max_frequency
        self.frequencies = stft.frequencies

        self.label = 'Spectrogram'
        self.unit = 'Magnitude'


class MelSpectrogram(iracema.core.timeseries.TimeSeries):
    def __init__(self,
                 time_series,
                 window_size,
                 hop_size,
                 fft_len=4096,
                 power=2.,
                 n_mels=256,
                 fmin=0.,
                 fmax=None):
        """
        Compute a mel spectrogram for ``time_series``.
        """
        spec = Spectrogram(time_series, window_size, hop_size, fft_len=fft_len, power=power)

        fmax = fmax or spec.max_frequency
        mel_basis = mel(
            time_series.fs, fft_len, n_mels=n_mels, fmin=fmin, fmax=fmax)
        data = np.dot(mel_basis, spec.data)

        super(MelSpectrogram, self).__init__(
            spec.fs, data=data, start_time=spec.start_time, caption=spec.caption)

        self.frequencies = mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.max_frequency = spec.frequencies[-1]
        self.label = 'Mel Spectrogram'
        self.label = 'Magnitude'


@deprecated(version='0.2.0', reason='Deprecated method. Use `STFT` instead.')
def fft(*args, **kwargs):
    "Deprecated FFT method."
    return STFT(*args, **kwargs)
