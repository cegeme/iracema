"""
This module contains the implementation of feature extractors.

References
----------
.. [Bello2005] Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies,
    M., & Sandler, M. B. (2005). A tutorial on onset detection in music
    signals. IEEE Transactions on Speech and Audio Processing, 13(5),
    1035–1046.

.. [Dixon2006] Dixon, S. (2006). Onset Detection Revisited. In 9th
   International Conference on Digital Audio Effects (pp. 133–137).
   Montreal, Canada.

.. [Lerch2012] Lerch, A. (2012). An introduction to audio content analysis:
   Applications in signal processing and music informatics. In An Introduction
   to Audio Content Analysis: Applications in Signal Processing and Music 
   Informatics.

.. [Park2004] Park, T. H. (2004). Towards automatic musical instrument
   timbre recognition. Princeton University.

.. [Park2010] Park, T. H. (2010). Introduction to digital signal
   processing: Computer musically speaking. World Scientific Publishing
   Co. Pte. Ltd.

.. [Peeters2011] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N.,
   & McAdams, S. (2011). The timbre toolbox: extracting audio features
   from musical signals, 130(5).
"""
import numpy as np
from scipy.stats import pearsonr, gmean  # pylint: disable=import-error

from iracema.core.segment import Segment

from iracema.aggregation import (aggregate_features,
                                 aggregate_sucessive_samples,
                                 sliding_window)
from iracema.util.dsp import hwr


def peak_envelope(time_series, window_size, hop_size):
    """
    Calculate the peak envelope of a time series

    The peak envelope consists in the peak absolute values of the
    amplitude within the aggregation window.

    .. math:: \\operatorname{PE} = max(|x(n)|), 1 <= n <= L

    Where `x(n)` is the n-th sample of a window of length `L`.

    Args
    ----
    time_series : iracema.core.timeseries.TimeSeries
        An audio time-series object.
    window_size : int
    hop_size : int
    """
    def function(x):
        return np.max(np.abs(x))

    time_series = sliding_window(time_series, window_size, hop_size, function)
    time_series.label = 'PeakEnvelope'
    time_series.unit = 'amplitude'
    return time_series


def rms(time_series, window_size, hop_size):
    """
    Calculate the root mean square of a time series

    The RMS envelope consists in the root mean square of the amplitude,
    calculated within the aggregation window.

    .. math:: RMS = \\sqrt{ \\frac{1}{L} \\sum_{n=1}^{L} x(n)^2 }

    Where `x(n)` is the n-th sample of a window of length `L`.

    Args
    ----
    time_series : iracema.core.timeseries.TimeSeries
        A time-series object. It is usually applied on Audio objects.
    window_size : int
    hop_size : int
    """
    def function(x):
        return np.sqrt(np.mean(x**2))

    time_series = sliding_window(time_series, window_size, hop_size, function)
    time_series.label = 'RMS'
    time_series.unit = 'amplitude'
    return time_series


def zcr(time_series, window_size, hop_size):
    """
    Calculate the zero-crossing rate of a time series, i.e., the number of
    times the signal crosses the zero axis, per second.

    The zero-crossing rate gives some insight on the noisiness character of a
    sound. In noisy / unvoiced signals, the zero-crossing rate tends to reach
    higher values than in periodic / voiced signals.

    .. math:: \\operatorname{ZC} = \\frac{1}{2 L} \\sum_{n=1}^{L}\\left|\\operatorname{sgn}\\left[x(n)\\right]-\\operatorname{sgn}\\left[x(n-1)\\right]\\right|

    Where

    .. math:: \\operatorname{sgn}\\left[x(n)\\right]=\\left\\{\\begin{array}{c}{1, x(n) \\geq 0} \\\\ {-1, x(n)<0}\\end{array}\\right.

    And `x(n)` is the n-th sample of a window of length `L`.

    Args
    ----
    time_series : iracema.core.timeseries.TimeSeries
        A time-series object. It is usually applied on Audio objects.
    window_size : int
    hop_size : int
    """
    # count the number of times the signal changes between successive samples
    def function(x):
        return np.sum(x[1:] * x[:-1] < 0) / window_size * time_series.fs

    time_series = sliding_window(time_series, window_size, hop_size, function)
    time_series.label = 'ZCR'
    time_series.unit = 'Hz'
    return time_series


def spectral_flatness(stft):
    """
    Calculate the spectral flatness for a given STFT.

    The spectral flatness gives an estimation of the noisiness / sinusoidality
    of an audio signal (for the whole spectrum or for a frequency range). It
    can be used to determine voiced / unvoiced parts of a signal [Park2004]_.

    It is defined as the ratio between the `geometric mean` and the
    `arithmetic mean` of the energy spectrum:

    .. math::
       :nowrap:

       \\begin{eqnarray}
       \\operatorname{SFM} = 10 log_{10} \\left( \\frac
         {\\left( \\prod_{k=1}^{N} |X(k)| \\right)^\\frac{1}{N}}
         { \\frac{1}{N} \\sum_{k=1}^{N} |X(k)| }
       \\right)
       \\end{eqnarray}

    Where `X(k)` is the result of the STFT for the `k-th` frequency bin.

    Args
    ----
    time_series : iracema.spectral.STFT
        A STFT object
    """
    def function(X):
        stft_magnitudes = np.abs(X)
        return 10 * np.log10(gmean(stft_magnitudes) / np.mean(stft_magnitudes))

    time_series = aggregate_features(stft, function)
    time_series.label = 'SpectralFlatness'
    time_series.unit = ''
    return time_series


def hfc(stft, method='energy'):
    """
    Calculate the high frequency content for a STFT time-series.

    The HFC _function produces sharp peaks during attacks or transients
    [Bello2005]_ and might be a good choice for detecting onsets in percussive
    sounds.

    .. math:: \\operatorname{HFC} = \sum_{k=1}^{N} |X(k)|^2 \\cdot k

    Alternatively, you can set ``method`` = `'amplitude'` instead of `'energy'`
    (default value):

    .. math:: \\operatorname{HFC} = \sum_{k=1}^{N} |X(k)| \\cdot k

    Args
    ----
    stft : iracema.spectral.STFT
        STFT time-series.
    method : str
        Method of choice to calculate the HFC.
    """
    def _func(X):
        N = X.shape[0]
        W = np.arange(1, N + 1)

        if method == 'energy':
            return np.sum(W * np.abs(X)**2) / N
        if method == 'amplitude':
            return np.sum(W * np.abs(X)) / N
        ValueError("the argument `method` must be 'energy' or 'amplitude'")

    time_series = aggregate_features(stft, _func)
    time_series.label = 'HFC'
    time_series.unit = ''
    return time_series


def spectral_centroid(stft):
    """
    Calculate the spectral centroid for a STFT time-series.

    The spectral centroid is a well known timbral feature that is used to
    describe the brightness of a sound. It represents the center of gravity
    of the frequency components of a signal [Park2010]_.

    .. math::
       \\operatorname{SC} = \\frac{\\sum_{k=1}^{N} |X(k)| \\cdot f_k }{\\sum_{k=1}^{N} |X(k)|}

    Where `X(k)` is the result of the STFT for the `k-th` frequency bin.

    Args
    ----
    stft : iracema.spectral.STFT
        A STFT object
    """
    def function(X):
        return _spectral_centroid(X, stft.frequencies)

    time_series = aggregate_features(stft, function)
    time_series.label = 'SpectralCentroid'
    time_series.unit = 'Hz'
    return time_series


def spectral_spread(stft):
    """
    Calculate the spectral spread for a STFT time-series.

    The spectral spread represents the spread of the spectrum around the
    spectral centroid [Peeters2011]_, [Lerch2012]_.

    .. math:: \\operatorname{SSp} = \\sqrt{\\frac{\\sum_{k=1}^{N} |X(k)| \\cdot (f_k - SC)^2 }{\\sum_
       {k=1}^{N} |X (k)|}}

    Where `X(k)` is the result of the STFT for the `k-th` frequency bin and SC
    is the spectral centroid for the frame.
    """
    def function(X):
        return _spectral_spread(X, stft.frequencies)

    time_series = aggregate_features(stft, function)
    time_series.label = 'SpectralSpread'
    time_series.unit = 'Hz'
    return time_series


def _spectral_centroid(X, f):
    """
    Calculate the spectral centroid for a stft frame `X`, being `f` the
    frequency corresponding to its bins.
    """
    abs_X = np.abs(X)
    sum_abs_X = np.sum(abs_X)
    if sum_abs_X == 0:
        return 0
    return np.sum(f * abs_X) / sum_abs_X


def _spectral_spread(X, f):
    """
    Calculate the spectral spread for a stft frame `X`, being `f` the frequency
    corresponding to its bins.
    """
    return np.sqrt(_spectral_centroid(X, (f - _spectral_centroid(X, f))**2))


def spectral_skewness(stft):
    """
    Calculate the spectral skewness for an STFT time series
    
    The spectral skewness is a measure of the asymetry of the distribution of
    the spectrum around its mean value, and is calculated from its third order
    moment. It will output negative values when the spectrum has more energy
    bellow the mean value, and positive values when it has more energy above
    the mean. Symmetric distributions will output the value zero [Lerch2012]_.

    .. math::
       \\operatorname{SSk} = \\frac{2 \\cdot \\sum_{k=1}^{N} \\left( |X(k)| - \\mu_{|X|} \\right)^3 }{
       N \\cdot \\sigma_{|X|}^3}

    Where :math:`\\mu_{|X|}` is the mean value of the maginute spectrum and 
    :math:`\\sigma_{|X|}` its standard deviation.
    """
    def _func(X):
        return 2 * np.sum(np.abs(X) - np.mean(X))**3 / (len(X) * np.std(X)**3)

    time_series = aggregate_features(stft, _func)
    time_series.label = 'SpectralSkewness'
    time_series.unit = ''
    return time_series


def spectral_kurtosis(stft):
    """
    Calculate the spectral kurtosis for an STFT time series
    
    The spectral kurtosis is a measure of the flatness of the distribution of
    the spectrum around its mean value. It will output the value 3 for Gaussian
    distributions. Values smaller than 3 represent flatter distributions, while
    values larger than 3 represent peakier distributions [Lerch2012]_.

    .. math::
       \\operatorname{SKu} = \\frac{2 \\cdot \\sum_{k=1}^{N} \\left( |X(k)| - \\mu_{|X|} \\right)^4 }{
       N \\cdot \\sigma_{|X|}^4}

    Where :math:`\\mu_{|X|}` is the mean value of the maginute spectrum and 
    :math:`\\sigma_{|X|}` its standard deviation.
    """
    def _func(X):
        return 2 * np.sum(np.abs(X) - np.mean(X))**4 / (len(X) * np.std(X)**4)

    time_series = aggregate_features(stft, _func)
    time_series.label = 'SpectralKurtosis'
    time_series.unit = ''
    return time_series


def spectral_flux(stft, method='hwrdiff'):
    """
    Calculate the spectral flux for a STFT time-series.

    The spectral flux measures the amount of change between successive
    spectral frames. There are different methods to calculate the spectral
    flux across the literature. For now we have implemented the one proposed
    by [Dixon2006]_.

    .. math:: \\operatorname{SF} = \\sum_{k=1}^{N} H(|X(t, k)| - |X(t-1, k)|)

    where :math:`H(x) = \\frac{x+|x|}{2}` is the half-wave rectifier _function,
    and `t` is the temporal index of the frame.

    Args
    ----
    stft : iracema.spectral.STFT
        A STFT object
    method : str
        'hwrdiff' or 'corr'
    """
    def function_hwrdiff(X, X_prev):
        return np.sum(hwr(np.abs(X) - np.abs(X_prev)))

    def function_corr(X, X_prev):
        r, _ = pearsonr(np.abs(X), np.abs(X_prev))
        return r

    if method=='hwrdiff':
        function = function_hwrdiff
    elif method=='corr':
        function = function_corr
    else:
        raise ValueError('Invalid value for argument `method`.')

    time_series = aggregate_sucessive_samples(stft, function)
    time_series.label = 'SpectralFlux'
    time_series.unit = ''
    return time_series


def harmonic_centroid(harmonics):
    """
    Harmonic Centroid

    The harmonic centroid represents the center of gravity of the amplitudes
    of the harmonic series.

    .. math::
       \\operatorname{HC} = \\frac{\\sum_{k=1}^{H} A(k) \\cdot f_k }{\\sum_{k=1}^{H} A(k)}

    Where :math:`A(h)` represents the amplitude of the h-th harmonic partial.
    """
    def _func(A):
        abs_A = np.abs(A)
        sum_abs_A = np.sum(abs_A)
        if sum_abs_A == 0:
            return 0
        return np.sum(abs_A * np.arange(0, len(A))) / sum_abs_A

    time_series = aggregate_features(harmonics, _func)
    time_series.label = 'HarmonicCentroid'
    time_series.unit = 'Harmonic Number'
    return time_series


def harmonic_energy(harmonics_magnitude):
    """
    Calculate the energy of harmonic partials.

    Harmonic energy is the energy of the harmonic partials of a signal.

    .. math:: \\operatorname{HE} = \\sum_{k=1}^{H} A(k)^2
    """
    def _func(frame):
        return np.sum(frame**2)

    time_series = aggregate_features(harmonics_magnitude, _func)
    time_series.label = 'Harmonic Energy'
    time_series.unit = ''
    return time_series


def spectral_entropy(stft):
    """
    Calculate the spectral entropy for a STFT time series

    The spectral entropy is based on the concept of information entropy from
    Shannon's information theory. It measures the unpredictability of the given
    state of a spectral distribution.

    .. math:: \\operatorname{SEpy} = - \\sum_{k}^{N} P(k) \\cdot \\log_2 P(k)

    Where 

    .. math:: P(i)=\\frac{|X(i)|^2}{\sum_{j}^{N} |X(j)|^2}

    More info at https://www.mathworks.com/help/signal/ref/pentropy.html.
    """
    def function(X):
        N = stft.nfeatures
        P = np.abs(X)**2 / np.sum(np.abs(X)**2)
        H = -(np.sum(P * np.log2(P))) / np.log2(N)
        return H

    time_series = aggregate_features(stft, function)
    time_series.label = 'Spectral Entropy'
    time_series.unit = ''
    return time_series


def spectral_energy(stft):
    """
    Calculate the total energy of an STFT frame.

    Spectral Energy is the total energy of an STFT frame.

    .. math:: \\operatorname{SF} = \\sum_{k=1}^{N} H(|X(t, k)| - |X(t-1, k)|)
    """
    def function(frame):
        return np.sum(np.abs(frame)**2)

    time_series = aggregate_features(stft, function)
    time_series.label = 'Spectral Energy'
    time_series.unit = ''
    return time_series


def noisiness(stft, harmonics_magnitude):
    """
    Calculate the Noisiness for the given STFT and Harmonics time series.

    The Noisiness represent how noisy a signal is (values closer to 1), as
    oposed to harmonic (values close to 0). It is the ratio of the noise
    energy to the total energy of a signal [Peeters2011]_.

    .. math:: \\operatorname{Ns} = \\frac{\\operatorname{SE}-\\operatorname{HE}}{\\operatorname{SE}}
    """
    energy_spectral = spectral_energy(stft)
    energy_harmonic = harmonic_energy(harmonics_magnitude)
    energy_noise = energy_spectral - energy_harmonic

    time_series = energy_noise / energy_spectral
    time_series.label = 'Noisiness'
    time_series.unit = ''

    return time_series


def oer(harmonics):
    """
    Calculate the odd-to-even ratio for the harmonics time series.

    The OER represents the odd-to-even ratio among the harmonics of an audio
    signal. This value will be higher for sounds with predominantly odd
    harmonics, such as the clarinet.
    
    .. math:: \\operatorname{OER}=\\frac{\\sum_{h=1}^{H / 2} A(2 h - 1)^{2}\\left(t_{m}\\right)}{\\sum_{h=1}^{H / 2} A(2 h)^{2}\\left(t_{m}\\right)}

    Where :math:`A(h)` represents the amplitude of the h-th harmonic partial.
    """
    def _func(A):
        odd_energy = np.sum(A[::2])**2
        even_energy = np.sum(A[1::2])**2
        if even_energy==0:
            return 0.
        return odd_energy / even_energy

    time_series = aggregate_features(harmonics, _func)
    time_series.label = 'OER'
    time_series.unit = ''
    return time_series


def local_tempo(onsets, nominal_ioi_durations):
    """
    Calculate the local tempo for a list of note onsets.

    Arguments
    ---------
    onsets : PointList
        List of note onset points.
    nominal_ioi_durations : list
        List containing the nominal durations of the IOIs for the
        execerpt (based on the score).
    
    Return
    ------
    local_tempo : np.array
        Numpy array containing the local tempos for each IOI.
    """
    # calculate IOIs
    iois = np.fromiter(
        [
            float(o1 - o0) for o0, o1 in zip(onsets[0:-1], onsets[1:])
        ],
        dtype=np.float)

    nominal_ioi_durations = np.array(nominal_ioi_durations)
    normalized_ioi_time = iois / nominal_ioi_durations
    local_tempo_ = 60.0 / normalized_ioi_time
    
    return local_tempo_


def legato_index(audio, note_list, window=1024, hop=441):
    """
    Estimate the legato index for the given audio and note list.

    Arguments
    ---------
    audio : Audio
        Audio object.
    note_list : list
        List of dictionaries containing the note envelope points.
    window : int
    hop : int

    Return
    ------
    legato_indexes : np.array
        Numpy array with the calculated legato index for each note.
    """
    rms_ = rms(audio, window, hop)
    legato_indexes = []
    for note_this, note_next in zip(note_list[0:-1], note_list[1:]):

        # legato index
        transition = Segment(note_this['release_start'], note_next['attack_end'])
        rms_transition = rms_[transition]

        min_rms = min(rms_transition.data[0], rms_transition.data[-1])
        max_rms = max(rms_transition.data[0], rms_transition.data[-1])
        length = rms_transition.nsamples

        triangle_area = (max_rms - min_rms) * length / 2
        rectangle_area = min_rms * (length+1)
        total_area = rectangle_area + triangle_area
        sum_rms = np.sum(rms_transition.data)
        legato = np.clip(sum_rms / total_area, 0, 1)
        legato_indexes.append(legato)
        
    return np.array(legato_indexes)
