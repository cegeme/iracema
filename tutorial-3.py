# specifying window and hop sizes
window, hop = 2048, 1024

# calculating the STFT
stft = ir.spectral.STFT(audio, window, hop)

# plotting the spectrogram
ir.plot.spectrogram(stft)

# calculating the RMS
rms = ir.features.rms(audio, window, hop)

# plotting the RMS
rms.plot()

# calculating the Peak Envelope
peak = ir.features.peak_envelope(audio, window, hop)

# plotting the Peak Envelope
peak.plot()