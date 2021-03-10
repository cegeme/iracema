import pytest  # skipcq: PYL-W0611

import numpy as np
from iracema import TimeSeries


def test_resample(audio00):
    resampled_audio = audio00.resample(10000)
    assert resampled_audio.fs == 10000

def test_pad(audio00):
    padded_audio = audio00.pad(10, 20, value=0.)
    assert padded_audio.nsamples == audio00.nsamples+30
    assert np.all(padded_audio.data[0:10] == np.zeros(10))
    assert np.all(padded_audio.data[-20:] == np.zeros(20))

def test_pad_spectrogram(spectrogram00):
    padded_spectrogram = spectrogram00.pad(5, 5, value='repeat')
    assert padded_spectrogram.nsamples == spectrogram00.nsamples+10
    assert np.all(padded_spectrogram.data[...,0] == spectrogram00.data[...,0])
    assert np.all(padded_spectrogram.data[...,-1] == spectrogram00.data[...,-1])
