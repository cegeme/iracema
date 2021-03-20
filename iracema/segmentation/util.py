import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal.windows import get_window
from scipy.signal import convolve

import iracema as ir


def convolve_activations(activations, n_frames=5, window_function='hanning'):
    w = get_window(window_function, n_frames, fftbins=False)
    convolved_activations = convolve(activations, w, mode='same')
    convolved_activations = convolved_activations / np.max(convolved_activations)
    return convolved_activations


def three_sliced_mel_spectrograms(audio,
                                  window,
                                  hop,
                                  medium_window_factor=2,
                                  long_window_factor=4,
                                  frames_per_slice=15,
                                  **kwargs):
    """
    Convert audio into a tensor containing three mel spectrograms. 
    
    The tensor contains slices of the spectrograms. Each mel
    spectrogram correspond to one channel.

    Return
    ------
    data : np.array
        Data array of shape (num_slices, frames_per_slice, frequencies,
        channels)
    fs : float
        Sampling frequency of the `data` array.
    """
    medium_window = window * medium_window_factor
    long_window = window * long_window_factor

    mel_spec_long = (ir.spectral.MelSpectrogram(audio, long_window, hop,
                                                **kwargs))
    mel_spec_medium = (ir.spectral.MelSpectrogram(audio, medium_window, hop,
                                                  **kwargs))
    mel_spec_short = (ir.spectral.MelSpectrogram(audio, window, hop,
                                                 **kwargs))

    pad_len = np.floor_divide(frames_per_slice, 2)

    short_len = mel_spec_short.data.shape[1]
    medium_len = mel_spec_medium.data.shape[1]
    long_len = mel_spec_long.data.shape[1]

    dif_long_short = long_len - short_len
    dif_long_medium = long_len - medium_len

    # padding the extremes of the spectrogram
    mel_spec_short = mel_spec_short.pad(pad_len,
                                        pad_len + dif_long_short,
                                        value='repeat')
    mel_spec_medium = mel_spec_medium.pad(pad_len,
                                          pad_len + dif_long_medium,
                                          value='repeat')
    mel_spec_long = mel_spec_long.pad(pad_len, pad_len, value='repeat')

    data = np.stack(
        [mel_spec_short.data, mel_spec_medium.data, mel_spec_long.data])

    # generating a sliced view of the spectrogram
    nchannels, nfrequencies, ntime = data.shape
    nslices = ntime - 2 * pad_len
    new_shape = (nchannels, nfrequencies, frames_per_slice, nslices)
    strides = (nfrequencies * ntime * data.itemsize, ntime * data.itemsize,
               data.itemsize, data.itemsize)
    data = as_strided(data, shape=new_shape, strides=strides)

    data = data.T
    fs = float(mel_spec_short.fs)

    return data, fs


def get_notes_list(onsets, offsets):
    """
    Generate a list of note segments using the specified `onsets` and `offsets`
    arrays.

    Args
    ----
    audio : Audio
        Audio object
    onsets : array
        Indexes of the onset occurrences in `audio`.
    offsets : array
        Indexes of the offset occurrences in `audio`.

    Return
    ------
    notes : list
        List of segments.
    """
    if onsets.shape != offsets.shape:
        raise ValueError("the number of onsets and offsets must the same")

    return [ir.Segment(onsets[i], offsets[i]) for i in range(len(onsets))]

