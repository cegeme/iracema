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

    Args
    ----
    time_series : TimeSeries
        Time series to be used in the FFT.
    window_size : int
    hop_size : int
    fftlen : int
        Length of the FFT. The signal will be zero-padded if ``fftlen`` >
        ``rolling_window.window_size``.
    """
    def calculate(x):
        return np.fft.fft(x, n=2 * fft_len)[:int(fft_len)]

    fft_time_series = sliding_window(time_series, window_size,
                                     hop_size, calculate,
                                     window_name='hann')

    fft_time_series.max_frequency = time_series.nyquist
    fft_time_series.frequencies = np.linspace(
        0, fft_time_series.max_frequency, fft_len, dtype=np.float_)


@deprecated(version='0.2.0', reason='Deprecated method. Use `STFT` instead.')
def fft(*args, **kwargs):
    "Deprecated FFT method."
    return STFT(*args, **kwargs)
