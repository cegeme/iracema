"""
This module contains the implementation of a harmonic extraction model for
monophonic audio.
"""

import numpy as np

from .timeseries import TimeSeries
from .util.dsp import local_peaks, n_highest_peaks


def extract(fft,
            pitch,
            nharm=16,
            minf0=24,
            maxf0=4200,
           perc_tol=0.04):
    """
    Extract the harmonics of an audio signal, given its `fft` and `pitch`
    curve, using `nharm` harmonics. The pitch detection method will search for
    a a fundamental frequency in the interval between `minf0` and `maxf0`.

    Args
    ----
    fft : FFT
        FFT time series.
    pitch : TimeSeries
        Pitch time series.
    nharm : int
        Number of harmonics to extract.
    minf0 : float
        Minimum fundamental frequency.
    maxf0 : float
        Maximum fundamental frequency.
    min_mag_cand : float
        Minimum amplitude to be candidate, relative to the frame's peak.
    noisiness_thresh : float
        Upper noisiness limit for each frame. If the noisiness exceeds this
        value, the algorithm will consider that the frame has no fundamental
        frequency.
    perc_tol : float
        Frequency tolerance used to search for peaks in the region around the
        expected frequency of each harmonic.

    Returns
    -------
    harmonics : dictionary
        A dictionary containing the results of the harmonics extraction,
        composed of three TimeSeries objects, which can be accessed using the
        following keys: 'frequency', 'magnitude'and 'phase'.
    """
    if minf0 >= maxf0:
        raise ValueError('The parameter maxf0 must be greater than minf0.')
    if nharm < 3:
        raise ValueError('The parameter nharm must be greater than 3.')

    # TODO: handle situation where pitch and fft have different sampling
    # frequencies

    # memory allocation
    harm_mag = np.empty((nharm, fft.nsamples), dtype=np.float_)
    harm_f = np.empty((nharm, fft.nsamples), dtype=np.float_)
    harm_phase = np.empty((nharm, fft.nsamples), dtype=np.float_)

    # run the algorithm for each frame of the fft
    for i, fft_frame in enumerate(np.transpose(fft.data)):
        fft_frame_amp = np.abs(fft_frame)
        fft_frame_phase = np.angle(fft_frame)

        mag, phase, f = frame_harmonics(pitch.data[i], fft_frame_amp,
                                        fft_frame_phase, fft.max_frequency,
                                        nharm, perc_tol)

        harm_mag[:, i] = mag
        harm_phase[:, i] = phase
        harm_f[:, i] = f

    harmonics = {}

    harmonics['frequency'] = TimeSeries(
        fft.fs, data=harm_f, start_time=fft.start_time)
    harmonics['magnitude'] = TimeSeries(
        fft.fs, data=harm_mag, start_time=fft.start_time)
    harmonics['phase'] = TimeSeries(
        fft.fs, data=harm_f, start_time=fft.start_time)

    return harmonics


def frame_harmonics(frame_pitch, fft_frame_mag, fft_frame_phase, fft_max_freq,
                    nharm, perc_tol):
    """
    Extract the harmonics for a fft frame.
    """
    N = fft_frame_mag.shape[-1]

    # calculate the index corresponding to the f0
    ix_f0 = int(round(np.nan_to_num(frame_pitch / fft_max_freq * N)))
    delta = ix_f0 * perc_tol

    # searching for local peaks all over the FFT
    _, ix_pks = local_peaks(fft_frame_mag)
    ix_pks = ix_pks
    # TODO: this could be done more efficiently, the peak searching is
    # being performed twice

    # memory allocation
    cand_mag = np.zeros((nharm))
    cand_phase = np.zeros((nharm))
    ix_cand_harm = np.zeros((nharm))

    # iterate the harmonics
    for j in range(1, nharm):
        ix_f_harm = ix_f0 * (j + 1)

        # check if it is within the boundaries of the FFT
        if ix_f_harm < N:
            # set the tolerance margin to search for the harmonic
            ix_tol_low = int(ix_f_harm - delta)
            ix_tol_hi = int(ix_f_harm + delta)

            # search for the highest peak within the tolerance margin
            val_harm, ix_harm = n_highest_peaks(
                fft_frame_mag[ix_tol_low:ix_tol_hi], 1)
            ix_harm = ix_harm + ix_tol_low

            if ix_harm:
                cand_mag[j] = val_harm
                cand_phase[j] = fft_frame_phase[ix_harm]
                ix_cand_harm[j] = ix_harm
            else:
                cand_mag[j] = fft_frame_mag[(j + 1) * ix_f0]
                cand_phase[j] = fft_frame_phase[(j + 1) * ix_f0]
                ix_cand_harm[j] = (j + 1) * ix_f0
        else:
            # TODO: what if the harmonics are higher than the FFT max
            #       frequency?
            # stop the loop if the frequency index is out of bound
            break

    cand_f = ix_cand_harm * fft_max_freq / N

    return cand_mag, cand_phase, cand_f
