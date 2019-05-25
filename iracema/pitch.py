"""
According to the ANSI standard 1994 [ANSI1994]_, "Pitch is that attribute of
auditory sensation in terms of which sounds may be ordered on a scale extending
from low to high. Pitch depends mainly on the frequency content of the sound
stimulus, but it also depends on the sound pressure and the waveform of the
stimulus."

This module contains the implementation of a few different pitch estimation
methods.

References
----------
.. [ANSI1994] American National Standards Institute, 1994.
"""

import numpy as np
import scipy.signal as sig

from .timeseries import TimeSeries
from .util.dsp import local_peaks, n_highest_peaks, decimate_mean
from .aggregation import aggregate_features


def hps(fft_time_series, minf0, maxf0, n_downsampling=16,
        decimation='discard'):
    """
    Extract the pitch using Harmonic Product Spectrum.

    The Harmonic Product Spectrum measures the maximum coincidence for
    harmonics [Cuadra2001]_. It is based on successive downsampling operations
    on the frequency spectrum of the signal. If the signal contains harmonic
    components, then it should contain energy in the frequency positions
    corresponding to the integer multiples of the fundamental frequency. So by
    down-sampling the spectrum by increasing integer factors
    :math:`(1,2,3,...,R)` it is possible to align the energy of its harmonic
    components with the fundamental frequency of the signal.

    Then we multiply the original spectrum and its downsampled versions. This
    operation will make a strong peak appear in a position that corresponds to
    the fundamental frequency. The HPS calculates the maximum coincidence for
    harmonics, according to the equation:

    .. math:: Y(\\omega) = \\prod_{r=1}^{R} |X(\\omega r)|

    where :math:`X(\omega r)` represents one spectral frame and :math:`R` is
    the number of harmonics to be considered in the calculation. After this
    calculation a simple peak detection algorithm is used to obtain the
    fundamental frequency of the frame.

    This implementation modifies this approach adding an offset of 1 to the
    magnitude spectrum of the signal before applying the product shown in the
    equation above. This makes the algorithm more reliable in situations where
    some harmonics have very little or no energy at all (float arithmetic is
    not reliable when values get too close to zero).

    Also, alternatively to the original approach, it is possible to choose
    between different interpolation methods, using the argument `decimation`.

    Args
    ----
    fft_time_series : iracema.spectral.FFTs
        FFT time series.
    minf0 : float
        Lower frequency limit to search for f0.
    maxf0 : float
        Upper frequency limit to search for f0.
    n_downsampling : int
        Number of downsampling operations
    decimation : 'discard', 'mean' or 'interpolation'
        Type of decimation operation to be performed.

    References
    ----------
    .. [Cuadra2001] De La Cuadra, P. Efficient pitch detection techniques for
       interactive music. In ICMC, page 403–406, 2001.
    """
    if minf0 >= maxf0:
        raise ValueError('The parameter maxf0 must be greater than minf0.')

    N = fft_time_series.data.shape[-2]
    ix_maxf0 = np.int(np.ceil(
        (N * maxf0) /
        fft_time_series.max_frequency))  # max index to search for f0
    ix_minf0 = np.int(np.ceil(
        (N * minf0) /
        fft_time_series.max_frequency))  # min index to search for f0

    def calculate(X):
        """
        Calculate pitch for frame.
        """
        N = X.shape[-1]  # length of the FFT
        S = np.zeros((n_downsampling + 1, N),
                     dtype=np.float_)  # matrix with the decimated FFTs
        X_mag = np.abs(X)
        S[0, :] = X_mag
        for q in range(2, n_downsampling + 2):
            if decimation == 'mean':
                ds = decimate_mean(X_mag, q)
            elif decimation == 'interpolation':
                ds = sig.decimate(X_mag, q)
            elif decimation == 'discard':
                ds = X_mag[::q]
            S[q - 1, :ds.shape[-1]] = ds / q

        S = S + 1
        p_fzero = np.sum(S, axis=0)
        ix_f0 = ix_minf0 + np.argmax(p_fzero[ix_minf0:ix_maxf0])

        return fft_time_series.frequencies[ix_f0]

    pitch_time_series = aggregate_features(
        fft_time_series, calculate)
    pitch_time_series.label = 'Pitch (HPS)'
    pitch_time_series.unit = 'Hz'

    return pitch_time_series


def expan_pitch(fft_time_series,
                minf0=24,
                maxf0=4200,
                nharm=12,
                ncand=5,
                min_mag_cand=0.1,
                noisiness_tresh=0.99,
                perc_tol=0.04):
    """
    Extract the pitch using the Expan pitch detection algorithm.

    Args
    ----
    fft_time_series : iracema.spectral.FFTs
        FFT time series.
    minf0 : float
        Lower frequency limit to search for f0.
    maxf0 : float
        Upper frequency limit to search for f0.
    nharm : int
        Number of harmonics to be considered.
    ncand : int
        Number of f0 candidate components to be used.
    min_mag_cand : float
        Minimum magnitude of the candidate to be chosen as f0.
    noisiness_tresh : float
        Noisiness treshold.
    perc_tol : float
        Tolerance percentage to search for harmonics.
    """
    if minf0 >= maxf0:
        raise ValueError('The parameter maxf0 must be greater than minf0.')
    if nharm < 3:
        raise ValueError('The parameter nharm must be greater than 3.')

    N = fft_time_series.data.shape[-2]
    ix_maxf0 = np.int(np.ceil(
        (N * maxf0) /
        fft_time_series.max_frequency))  # max index to search for f0
    ix_minf0 = np.int(np.ceil(
        (N * minf0) /
        fft_time_series.max_frequency))  # min index to search for f0

    def frame_pitch(fft_frame):
        """
        Extract the pitch and the harmonics for a fft frame.
        """
        N = fft_frame.shape[-1]  # length of the FFTT
        fft_frame_mag = np.abs(fft_frame)

        # searching for f0 candidates
        _, ix_cand = n_highest_peaks(fft_frame_mag[ix_minf0:ix_maxf0], ncand)
        ix_cand = ix_cand + ix_minf0  # TODO: check these indexes

        # searching for local peaks all over the FFT
        _, ix_pks = local_peaks(fft_frame_mag[ix_minf0:])
        ix_pks = ix_pks + ix_minf0
        # TODO: this could probably be done more efficiently, the peak searching
        # is being performed twice

        # update the number of candidates with the number of local peaks found
        n_cand = len(ix_cand)

        # if no peaks were found, return zeroed arrays
        if not ix_cand.any():
            return 0

        # memory allocation
        cand_mag = np.zeros((ncand, nharm))
        ix_cand_harm = np.zeros((ncand, nharm))

        # iterate the candidates
        for i, ix_f0 in enumerate(ix_cand):
            delta = ix_f0 * perc_tol
            cand_mag[i, 0] = fft_frame_mag[ix_f0]

            # test the candidate's magnitude compared to the frame's max
            if cand_mag[i, 0] > min_mag_cand * np.max(fft_frame_mag):
                ix_cand_harm[i, 0] = ix_f0

                # iterate the candidate's potential harmonics
                for j in range(1, nharm):
                    ix_f_harm = ix_f0 * (j + 1)

                    # check if it is within the boundaries of the FFT
                    if ix_f_harm < N:
                        # set the tolerance margin to search for the harmonic
                        ix_tol_low = int(ix_f_harm - delta)
                        ix_tol_hi = int(ix_f_harm + delta)
                        if ix_tol_low < ix_minf0:
                            ix_tol_low = ix_minf0

                        # search for highest peak within the tolerance margin
                        val_harm, ix_harm = n_highest_peaks(
                            fft_frame_mag[ix_tol_low:ix_tol_hi], 1)
                        ix_harm = ix_harm + ix_tol_low

                        if ix_harm:
                            cand_mag[i, j] = val_harm
                            ix_cand_harm[i, j] = ix_harm

                        else:
                            cand_mag[i, j] = fft_frame_mag[(j + 1) * ix_f0]
                            ix_cand_harm[i, j] = (j + 1) * ix_f0
                    else:
                        # stop the loop if the frequency index is out of bound
                        break

        energy_harm = np.zeros(n_cand)

        # calculate the energy of the harmonic components for each candidate
        for i in range(n_cand):
            energy_harm[i] = np.sum(cand_mag[i, :]**2)

        # choose the candidate with the highest harmonic energy
        i = np.argmax(energy_harm)

        # one last test, the noisiness for the winner candidate must be bellow
        # the noisiness threshold
        h_energy = energy_harm[i]
        frame_energy = np.sum(fft_frame_mag**2)
        frame_noisiness = 1 - (h_energy / frame_energy)

        if frame_noisiness < noisiness_tresh:
            return ix_cand_harm[i, 0] * fft_time_series.max_frequency / N
        else:
            return 0

    pitch_time_series = aggregate_features(
        fft_time_series, frame_pitch)

    pitch_time_series.label = 'Pitch (HPS)'
    pitch_time_series.unit = 'Hz'


    return pitch_time_series


def crepe_pitch(audio,
                model_capacity='large',
                min_confidence=0.5,
                viterbi=True):
    """
    Extract the pitch using CREPE pitch tracker.

    This function uses a pitch tracker based on deep convolutional neural
    networks. The model was proposed and trained by [Kim2018]_.

    Args
    ----
    audio : iracema.audio.Audio
        Audio time series.
    step_size : float
        Length of the time steps for the pitch extraction.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the capacity of the model. The value 'full' will use
        the model specified in the paper. The others will reduce the number of
        filters in the convolutional layers, resulting in faster computation,
        at the cost of slightly reduced accuracy.
    min_confidence : float
        Minimum confidence to consider a pitch detection as valid.
    viterbi : bool
        Viterbi smoothing for pitch curve.

    References
    ----------
    .. [Kim2018] Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). CREPE:
       A Convolutional Representation for Pitch Estimation. In Proceedings of
       the IEEE International Conference on Acoustics, Speech, and Signal
       Processing (ICASSP).
    """
    from crepe import predict  # pylint: disable=import-error

    time, frequency, confidence, _ = predict(
        audio.data, audio.fs, viterbi=viterbi, model_capacity=model_capacity)

    frequency[confidence < min_confidence] = np.nan

    step = time[1] - time[0]
    print('step =', step)
    fs = 1 / step
    pitch_time_series = TimeSeries(
        fs, frequency, start_time=audio.start_time, unit='Hz')

    return pitch_time_series


def pitch_filter(pitch_time_series, delta_max=0.04):
    """
    The pitch curve can be noisy, this function tries to smooth it.
    """
    data = pitch_time_series.data
    data_previous = np.concatenate((np.zeros(1), data[0:-1]))
    data_next = np.concatenate((data[1:], np.zeros(1)))

    pitch_filtered = pitch_time_series.copy()

    # find the indexes where the pitch curve is unstable
    # 1st condition: the pitch changes more than delta_max and comes back
    diff_next_prev = np.abs(data_next - data_previous)
    diff_data_prev = np.abs(data - data_previous)
    diff_data_next = np.abs(data - data_next)
    mean_next_prev = (data_next + data_previous) / 2
    indexes_to_interp = ((diff_next_prev < mean_next_prev * delta_max) &
                         ((diff_data_prev > data * delta_max) |
                          (diff_data_next > data * delta_max)))

    for idx in np.where(indexes_to_interp):
        pitch_filtered.data[idx] = (
            pitch_filtered.data[idx + 1] + pitch_filtered.data[idx - 1]) / 2

    # 2nd condition: one isolated point between zeros
    indexes_to_zero = ((data_previous == 0) & (data_next == 0))

    pitch_filtered.data[indexes_to_zero] = 0

    return pitch_filtered
