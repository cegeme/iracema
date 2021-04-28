"""
Data plotting utilities.
"""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import MultiCursor  # pylint: disable=import-error

from iracema.features import rms as rms_, peak_envelope as peak_envelope_
from iracema.util import conversion

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


def spectrogram(spec_ts,
                log=False,
                fmin=0.,
                fmax=None,
                figsize=None,
                normalize=True,
                cmap='viridis'):
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
    _add_spectrogram_to_axes(
        ax, spec_ts, log, fmin=fmin, fmax=fmax, normalize=normalize, cmap=cmap)

    # show the resulting image
    f.show()

    return f


def waveform_spectrogram(audio,
                         spec_ts,
                         log=False,
                         rms=None,
                         peak_envelope=None,
                         fmin=0.,
                         fmax=None,
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
    _add_spectrogram_to_axes(ax2, spec_ts, log, fmin=fmin, fmax=fmax)

    # show the resulting image
    f.show()

    return f


def waveform_spectrogram_pitch(audio,
                               spec_ts,
                               pitch,
                               rms=None,
                               peak_envelope=None,
                               log=False,
                               fmin=0.,
                               fmax=None,
                               cmap='viridis',
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
    _add_spectrogram_to_axes(
        ax2, spec_ts, log, fmin=fmin, fmax=fmax, cmap=cmap)

    # plotting pitch
    _add_curve_to_axes(ax2, pitch, fmt='r', ymin=fmin, ymax=fmax)

    # show the resulting image
    f.show()

    return f


def waveform_spectrogram_harmonics(audio,
                                   spec_ts,
                                   harmonics,
                                   rms=None,
                                   peak_envelope=None,
                                   log=False,
                                   fmin=0.,
                                   fmax=None,
                                   normalize=True,
                                   cmap='viridis',
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
    _add_waveform_trio_to_axes(
        ax1,
        audio,
        rms,
        peak_envelope,
    )

    # plotting spectrogram
    _add_spectrogram_to_axes(
        ax2,
        spec_ts,
        log,
        fmin=fmin,
        fmax=fmax,
        normalize=normalize,
        cmap=cmap)

    # plotting harmonics
    _add_curve_to_axes(ax2, harmonics, fmt='r', ymin=fmin, ymax=fmax)

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
                       set_labels=True,
                       ymin=None,
                       ymax=None):
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
    
    axes.set_ylim([ymin, ymax])


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


def _add_spectrogram_to_axes(axes,
                             spec,
                             log=False,
                             fmin=0.,
                             fmax=None,
                             normalize=True,
                             cmap='viridis'):
    """
    Add a spectrogram image to the axes
    """

    # handling arguments
    if not fmax:
        fmax = spec.max_frequency
    else:
        if fmax > spec.max_frequency:
            warnings.warn(
                "the fmax specified excceeds the maximum frequency of the spectrum"
            )
            fmax = spec.max_frequency

    freq_indexes = np.logical_and(spec.frequencies >= fmin,
                                  spec.frequencies <= fmax)

    data = spec.data
    if np.any(np.iscomplex(data)):
        data = np.abs(data)
        power = 1.
        db = False
    else:
        power = spec._power
        db = spec._db

    # if the data is not in dB, convert it
    if not db:
        if power == 1.0:
            data = conversion.amplitude_to_db(data)
        elif power == 2.0:
            data = conversion.energy_to_db(data)

    data = data[freq_indexes, :]

    if isinstance(normalize, bool) or normalize is None:
        if normalize:
            vmin, vmax = np.min(data), np.max(data)
        else:
            vmin, vmax = -96, 0
    elif isinstance(normalize, tuple):
        if len(normalize) != 2:
            raise ValueError(
                "If `normalize` is a tuple, it must contain two values (vmin, vmax)."
            )
        vmin, vmax = normalize

    plt.pcolormesh(
        spec.time,
        spec.frequencies[freq_indexes],
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        shading='nearest')

    if log:
        axes.set_yscale("log", basey=2)
        axes.get_yaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())

    axes.set(xlabel='time (s)', ylabel='frequency (Hz)')
