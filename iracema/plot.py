"""
Data plotting utilities.
"""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import MultiCursor  # pylint: disable=import-error
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=import-error

from iracema.features import rms as rms_, peak_envelope as peak_envelope_

DEFAULT_FIG_SIZE = (9, 9)


def line_plot(time_series, linewidth=1, alpha=0.9, figsize=None, **kwargs):
    """
    Plot the time series using matplotlib.
    Line width and alpha values can be set as optional parameters.
    """
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    plt.plot(
        time_series.time,
        time_series.data,
        label=time_series.label,
        linewidth=linewidth,
        alpha=alpha,
        **kwargs)
    if time_series.label:
        plt.legend(loc='lower right', ncol=2, fontsize='x-small')
    plt.title(time_series.caption)
    plt.ylabel(time_series.unit)
    plt.xlabel('time (s)')
    plt.show()

    return f


def spectrogram(fft, logfft=False, fftlim=(), figsize=None):
    """
    Plot the spectrogram of the audio signal.
    """
    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    ax = f.subplots(1)

    plt.subplots_adjust(hspace=0.05)

    # plotting spectrogram
    _add_spectrogram_to_axes(ax, fft, logfft, fftlim)

    # show the resulting image
    f.show()

    return f


def waveform_spectrogram(audio,
                         fft,
                         logfft=False,
                         rms=None,
                         peak_envelope=None,
                         fftlim=(),
                         figsize=None):
    """
    Plot two graphs: the first one showing curves for the ``audio`` waveform,
    the ``rms`` and the ``peak_envelope``; the second showing the spectrogram
    of the audio signal.
    """
    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    (ax1, ax2) = f.subplots(
        2, sharex=True, gridspec_kw={'height_ratios': [1, 3.5]})

    plt.subplots_adjust(hspace=0.05)

    # plotting curves
    _add_waveform_trio_to_axes(ax1, audio, rms, peak_envelope)

    # plotting spectrogram
    _add_spectrogram_to_axes(ax2, fft, logfft, fftlim)

    # show the resulting image
    f.show()

    return f


def waveform_spectrogram_pitch(audio,
                            fft,
                            pitch,
                            rms=None,
                            peak_envelope=None,
                            logfft=False,
                            fftlim=(),
                            figsize=None):
    """
    Plot two graphs: the first one showing curves for the ``audio`` waveform,
    the ``rms`` and the ``peak_envelope``; the second showing the spectrogram
    of the audio signal and its fundamental frequency  `pitch`.
    """
    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    (ax1, ax2) = f.subplots(
        2, sharex=True, gridspec_kw={'height_ratios': [1, 3.5]})

    plt.subplots_adjust(hspace=0.05)

    # plotting curves
    _add_waveform_trio_to_axes(ax1, audio, rms, peak_envelope)

    # plotting spectrogram
    _add_spectrogram_to_axes(ax2, fft, logfft, fftlim)

    # plotting pitch
    _add_curve_to_axes(ax2, pitch, fmt='r')

    # show the resulting image
    f.show()

    return f


def waveform_spectrogram_harmonics(audio,
                                   fft,
                                   pitch,
                                   harmonics,
                                   rms=None,
                                   peak_envelope=None,
                                   logfft=False,
                                   fftlim=(),
                                   figsize=None):
    """
    Plot two graphs: the first one showing curves for the ``audio`` waveform,
    the ``rms`` and the ``peak_envelope``; the second showing the spectrogram
    of the audio signal and its `harmonics`.
    """
    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    (ax1, ax2) = f.subplots(
        2, sharex=True, gridspec_kw={'height_ratios': [1, 3.5]})

    plt.subplots_adjust(hspace=0.05)

    # plotting curves
    _add_waveform_trio_to_axes(ax1, audio, rms, peak_envelope)

    # plotting spectrogram
    _add_spectrogram_to_axes(ax2, fft, logfft, fftlim)

    # plotting pitch
    _add_curve_to_axes(ax2, pitch)

    # plotting harmonics
    _add_curve_to_axes(ax2, harmonics)

    # show the resulting image
    f.show()

    return f


def waveform_and_notes(audio, notes, figsize=None):
    """
    Plot waveform and note segments.
    """
    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    axes = f.subplots(1)

    _add_waveform_to_axes(axes, audio)
    _add_notes_to_axes(axes, notes)

    f.show()


def waveform_trio(audio, rms=None, peak_envelope=None, figsize=None):
    """
    Plot a graph showing curves for the ``audio`` waveform, the ``rms`` and the
    ``peak_envelope``.
    """
    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    plt.figure(figsize=figsize)

    axes = plt.axes()

    # add waveform trio to first axes
    _add_waveform_trio_to_axes(axes, audio, rms, peak_envelope)

    plt.show()


def waveform_trio_and_features(audio,
                               rms=None,
                               peak_envelope=None,
                               features=(),
                               figsize=None):
    """
    Plot a graph showing curves for the ``audio`` waveform, the ``rms`` and the
    ``peak_envelope``; followed by a series of graphs, one for each time-series
    in the tuple `features`.
    """
    if not features:
        raise ValueError("the features to be plotted were not specified")

    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    axes_list = f.subplots(len(features) + 1, sharex=True)

    plt.subplots_adjust(hspace=0.05)

    # add audio to first axes
    _add_waveform_trio_to_axes(axes_list[0], audio, rms, peak_envelope)

    # add features to the other axes
    for i, feature in enumerate(features, start=1):
        _add_curve_to_axes(axes_list[i], feature, label=feature.label)
        axes_list[i].legend(loc='lower right', fontsize='x-small')

    MultiCursor(f.canvas, axes_list, color='gray', lw=1)

    plt.show()

    return f


def waveform_trio_features_and_points(audio,
                                      feature,
                                      point_list,
                                      rms=None,
                                      peak_envelope=None,
                                      figsize=None):
    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    axes_list = f.subplots(2, sharex=True)

    # add waveform trio to first axes
    _add_waveform_trio_to_axes(axes_list[0], audio, rms, peak_envelope)

    # add feature
    _add_curve_to_axes(axes_list[1], feature, label=feature.label)
    axes_list[1].legend(loc='lower right', fontsize='x-small')

    _add_points_to_axes(axes_list[1], point_list.time,
                        point_list.get_values(feature))

    MultiCursor(f.canvas, axes_list, color='gray', lw=1)

    f.show()


def spectrogram_3d(fft, logfft=False, fftlim=None, figsize=None):
    """
    Plot a 3D spectrogram (experimental feature).
    """
    # handle arguments
    if not fftlim:
        fmin, fmax = 20, fft.max_frequency
    else:
        fmin, fmax = fftlim[0], fftlim[1]
        if fmax > fft.max_frequency:
            warnings.warn(
                "the fmax specified excceeds the maximum frequency of the FFT")
            fmax = fft.max_frequency

    # configuring figure and subplots
    if not figsize:
        figsize = DEFAULT_FIG_SIZE
    f = plt.figure(figsize=figsize)

    ax = Axes3D(f)

    # plotting spectrogram
    lenfft = fft.nfeatures
    bins_per_hz = lenfft / fft.max_frequency
    imax, imin = int(fmax * bins_per_hz), int(fmin * bins_per_hz)

    X, Y = np.meshgrid(fft.time, np.arange(imin, imax))

    fft_abs = (abs(fft.data[imin:imax, :]))

    ax.plot_surface(X, Y, fft_abs**2, rstride=2, cstride=10, cmap='seismic')

    if logfft:
        ax.set_yscale("log", basey=2)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ax2.set(xlabel='time (s)', ylabel='frequency (Hz)')

    f.show()

    return f


def _add_notes_to_axes(axes, notes):
    """
    Add note segments to the given ``axes``.
    """
    for note in notes:
        axes.plot(note.start_time, note.data[0], '^r')
        axes.plot(note.end_time, note.data[-1], 'vr')


def _add_curve_to_axes(axes,
                       time_series,
                       fmt='b',
                       linewidth=1,
                       alpha=0.9,
                       label=None,
                       set_labels=True):
    "Add the curve for the given ``time_series`` to the given ``axes``."

    if set_labels:
        axes.set(ylabel=time_series.unit)
        if time_series.caption:
            axes.set_title(time_series.caption)

    axes.set_xlim(time_series.start_time, time_series.end_time)

    if time_series.nfeatures == 1:
        axes.plot(
            time_series.time,
            time_series.data,
            fmt,
            linewidth=linewidth,
            alpha=alpha,
            label=label)

    else:
        for row in time_series.data:
            axes.plot(
                time_series.time, row, fmt, linewidth=linewidth, alpha=alpha)


def _add_points_to_axes(axes, time, values):
    """
    Plot circle in the specified ``points``x``values`` for the ``axes``.
    """
    axes.plot(time, values, 'or')


def _add_waveform_to_axes(axes, audio):
    """
    Add the waveform for the time-series `audio` to `axes`.
    """

    # adding the curves
    _add_curve_to_axes(axes, audio, linewidth=None, alpha=0.9)


def _add_waveform_trio_to_axes(axes,
                               audio,
                               rms=None,
                               peak_envelope=None,
                               show_legend=True):
    """
    The waveform trio consists in the curves ``audio``, ``rms`` and
    ``peak_evelope``. This method adds them to ``axes``.
    """
    window_size, hop_size = 2048, 512
    rms = rms or rms_(audio, window_size, hop_size)
    peak_envelope =\
        peak_envelope or peak_envelope_(audio, window_size,
                                                   hop_size)
    # adding the curves
    _add_curve_to_axes(axes, audio, linewidth=None, alpha=0.9)
    _add_curve_to_axes(axes, rms, fmt='r', label=rms.label, set_labels=False)
    _add_curve_to_axes(
        axes,
        peak_envelope,
        fmt='k',
        label=peak_envelope.label,
        set_labels=False)

    if show_legend:
        axes.legend(loc='lower right', ncol=2, fontsize='x-small')


def _add_segments_to_axes(axes, segments, start_color='g', end_color='r'):
    "Add two vertical lines to the axes delimiting a segment extent."
    pass


def _add_spectrogram_to_axes(axes, fft, logfft=False, fftlim=()):
    """
    Add a spectrogram image to the axes
    """

    # handling arguments
    if not fftlim:
        fmin, fmax = 20, fft.max_frequency
    else:
        fmin, fmax = fftlim[0], fftlim[1]
        if fmax > fft.max_frequency:
            warnings.warn(
                "the fmax specified excceeds the maximum frequency of the FFT")
            fmax = fft.max_frequency

    # plotting spectrogram
    lenfft = fft.nfeatures
    bins_per_hz = lenfft / fft.max_frequency
    imax, imin = int(fmax * bins_per_hz), int(fmin * bins_per_hz)

    fft_abs = abs(fft.data[imin:imax, :])

    axes.imshow(
        20 * np.log10(fft_abs),
        origin='lower',
        aspect='auto',
        extent=[fft.start_time, fft.end_time, fmin, fmax],
        cmap='viridis')

    if logfft:
        axes.set_yscale("log", basey=2)
        axes.get_yaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())

    axes.set(xlabel='time (s)', ylabel='frequency (Hz)')
