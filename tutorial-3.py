# specifying window and hop sizes
window, hop = 2048, 1024

# calculating the FFT
fft = iracema.spectral.fft(audio, window, hop)

# plotting the spectrogram
iracema.plot.plot_spectrogram(fft)

# calculating the RMS
rms = iracema.features.rms(audio, window, hop)

# plotting the RMS
rms.plot()

# calculating the Peak Envelope
peak = iracema.features.peak_envelope(audio, window, hop)

# plotting the Peak Envelope
peak.plot()