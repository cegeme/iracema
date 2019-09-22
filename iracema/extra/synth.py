"""
Useful methods for synthesizing basic sounds.
"""

import numpy as np

from ..timeseries import Audio


def sinusoid(amplitude, frequency, phase, t, fs, soft_start=False):
    """Generate sine wave according to the arguments provided.

    .. math:: y = a sin(2 \pi f t + \phi)

    Args
    ----
    amplitude : float
        Amplitude of the sinusoid.
    frequency : float
        Frequency of the sinusoid.
    phase : float
        The initial phase shift.
    t : float
        Duration in seconds.
    fs : float
        Sampling frequency.
    soft_start : bool
        If True, a linear smoothing amplitude envelope will be applied to
        the first 0.005 seconds of the generated sinusoid to avoid
        click sounds.
    """

    t = np.linspace(0, t, num=t*fs, dtype=np.float_)
    sine_wave = amplitude * np.sin(2*np.pi*frequency*t + phase)

    if soft_start:
        size_soft_start = int(fs * 0.005)
        soft_start_ar = np.concatenate(
            (np.linspace(0, 1, size_soft_start, dtype=np.float_),
             np.ones(sine_wave.shape[-1] - size_soft_start)))
        sine_wave = soft_start_ar * sine_wave

    ts = Audio(sine_wave, fs)

    return ts

