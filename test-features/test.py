import iracema as ir

haydn = ir.timeseries.Audio("05 - Trumpet - Haydn.wav")

haydn.play()
#haydn.plot()

window, hop = 2048, 2014
fft = ir.spectral.fft(haydn, window, hop)


skew = ir.features.spectral_skewness(fft)
skew.plot()

kurt = ir.features.spectral_kurtosis(fft)
kurt.plot()
