"""
Extraction of spectral information.
"""
import numpy as np
from iracema.aggregation import sliding_window


def spectrogram():
    """
    Generate spectrogram for time series.
    """
    pass


def fft(time_series, window_size, hop_size, fft_len=4096):
    """
    Calculate the Fast Fourier Transform for the ``time_series``

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

    fft_time_series.label = 'FFT'
    fft_time_series.unit = 'Magnitude'

    return fft_time_series


def ifft():
    """
    Inverse Fourier Transform.
    """
    pass
