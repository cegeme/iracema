"""
This module contains the implementation of the class TimeSeries.
"""

import copy as cp
from decimal import Decimal

import numpy as np
import resampy

from iracema.core.segment import Segment
from iracema.util import conversion
from iracema.util.dsp import but_filter
from iracema.plot import line_plot


class TimeSeries:
    """
    Class for storing and manipulating time series objects, which
    model synchronous discrete time series.

    .. Hint:: This class is also available at the main package level as
        ``iracema.TimeSeries``.

    Attributes
    ----------
    data : numpy array
        Data array sampled at ``fs`` Hz.
    time : numpy array
        Numpy data array containing the time of each sample, relative to the
        original time reference.
    fs : float
        Sampling frequency for the data.
    nyquist : float
        Nyquist frequency for the data.
    ts : float
        Sampling period for the data.
    start_time : Decimal
        The time the time series start (in seconds) relative to the original
        time reference.
    duration : float
        Duration of the time series (in seconds).
    end_time : float
        The time the time series end (in seconds) relative to the original time
        reference.
    unit : str
        String containing the name of the unit for the data (for plotting).
    nsamples : int
        Number of samples per time series in the data array.
    nfeatures
        Number of features for each sample in the data array.
    unit
        Unit for the data (for plotting).
    caption
        Text caption.
    label
        Label for the time series.
    """

    # these properties should be overriden by the subclass if they are supposed
    # to be different
    unit = ''
    caption = ''
    label = ''

    def __init__(self, fs, data=None, start_time=None, unit=None, caption=None):
        """
        Args
        ----
        fs : float
            Sampling frequency for the data.
        data : numpy array, optional
            Data array sampled at ``fs`` Hz. If this argument is not provided,
            the method _write_data() must be called after the initialization to
            set the data array.
        start_time : float, optional
            The time in seconds the time series start, relative to the original
            time reference.
        unit : str, optional
            Unit name for plotting the data of the time series.
        caption : str, optional
            Text caption for the time series.
        """
        if fs <= 0:
            raise ValueError(
                "the sampling frequency (fs) must be greater than zero")

        self.data = None
        self.fs = np.float_(fs)
        self.start_time = Decimal(0) if start_time is None else Decimal(start_time)

        if unit:
            self.unit = unit
        if caption:
            self.caption = caption
        if data is not None:
            self._write_data(data)

    @property
    def nsamples(self):  # pylint: disable=missing-docstring
        return self.data.shape[-1]

    @property
    def nfeatures(self):  # pylint: disable=missing-docstring
        if self.data.ndim == 1:
            return 1
        if self.data.ndim == 2:
            return self.data.shape[-2]

    @property
    def duration(self):  # pylint: disable=missing-docstring
        return Decimal(self.nsamples) / Decimal(self.fs)

    @property
    def nyquist(self):  # pylint: disable=missing-docstring
        return Decimal(self.fs) / Decimal(2)

    @property
    def end_time(self):  # pylint: disable=missing-docstring
        return Decimal(self.start_time) + Decimal(self.duration)

    @property
    def ts(self):  # pylint: disable=missing-docstring
        return Decimal(1) / Decimal(self.fs)

    @property
    def time(self):  # pylint: disable=missing-docstring
        start = Decimal(self.start_time)
        step = Decimal(self.duration) / Decimal(self.nsamples)
        
        return [
            start + (t * step)
            for t in range(0, self.nsamples)
        ]

    def copy(self):
        "Return a copy of the time series object (deep copy)."
        return cp.deepcopy(self)

    def normalize(self):
        "Return a normalized copy of the time series."
        normalized_data = self.data / np.max(self.data)

        ts = cp.copy(self)
        ts.data = normalized_data
        return ts

    def diff(self, n=1):
        "Return the n-th discrete difference for the time series"
        nfeatures = self.nfeatures
        dtype = self.data.dtype
        data = np.reshape(self.data, (nfeatures, -1))

        zero_pre_pad = np.zeros((nfeatures, n), dtype)
        padded_data = np.concatenate((zero_pre_pad, data), axis=-1)
        data_diff = np.diff(padded_data, n, axis=-1)

        if nfeatures == 1:
            data_diff.shape = (self.nsamples, )

        ts = cp.copy(self)
        ts.data = data_diff

        return ts

    def zeros_to_nan(self):
        """
        Converts zeros to np.nan in the data array. Returns a new time series.
        """
        ts = self.copy()
        ts.data[self.data == 0] = np.nan
        return ts

    def hwr(self):
        "Return a half-wave rectified copy of the time series."
        rectified_data = np.clip(self.data, 0, None)

        ts = cp.copy(self)
        ts.data = rectified_data

        return ts

    def resample(self, new_fs):
        """
        Resample time series to a new sampling rate.
        """
        if self.start_time != 0:
            raise (NotImplementedError(
                'The method resample is implemented only for time series '
                'objects with start_time equal to 0.'))
        ts = cp.copy(self)
        ts.data = resampy.resample(self.data, self.fs, new_fs)
        ts.fs = new_fs

        return ts

    def pad_like(self, timeseries):
        """
        Pad the end of the current time series to match the length of
        the given time series.
        """
        if self.fs != timeseries.fs:
            ValueError("The sampling rates of both time series must be equal.")
        if self.nsamples > timeseries.nsamples:
            ValueError("The current time series has more samples than the"
                       "given time series.")
        padding_len = timeseries.nsamples - self.nsamples
        padding_array = np.zeros(padding_len)
        new_ts = self.copy()
        new_ts.data = np.concatenate((new_ts.data, padding_array.data))
        return new_ts

    def resample_and_pad_like(self, timeseries):
        """
        Resample and pad the end of the current time series to match
        the given time series.
        """
        new_ts = self.resample(timeseries.fs)
        new_ts = new_ts.pad_like(timeseries)
        return new_ts

    def merge(self, timeseries, unit=None, caption=None, start_time=None):
        """
        Merge two time series. Both time series must have the same length and
        sampling frequency. The attributes ``unit``, ``caption`` and
        ``start_time`` of the resulting time series can be optionally set using
        the method's optional arguments. Otherwise these attributes will be
        equal to the values in the instance on which the method was called
        (``self``).
        """
        new_ts = self.copy()
        if self.fs != timeseries.fs:
            raise ValueError(
                'Incompatible sampling frequencies. Both time series must '
                'have the same sampling frequency.')
        elif self.nsamples != timeseries.nsamples:
            raise ValueError(
                'Incompatible number of samples. Both time series must have '
                'the same number of samples.')
        new_ts.unit = unit
        new_ts.caption = caption
        new_ts.data = np.vstack((self.data, timeseries.data))

        return new_ts

    def filter(self,
               critical_frequency,
               filter_type='low_pass',
               filter_order=4):
        """
        Filters the time series using a butterworth digital filter. This is
        a wrapper over ``scipy.signal.butter``.

        Arguments
        ---------
        critical_frequency: float
            The critical frequency of frequencies.
        filter_type: [‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’]
            The type of filter.
        filter_order:
            The order of the filter.
        """
        audio_filtered = self.copy()
        audio_filtered.data = but_filter(
            self.data,
            self.fs,
            critical_frequency,
            filter_type=filter_type,
            filter_order=filter_order)
        return audio_filtered

    def plot(self, linewidth=1, alpha=0.9, **kwargs):
        "Plot the time series using matplotlib."
        return line_plot(self, linewidth=linewidth, alpha=alpha, **kwargs)

    def time_to_sample_index(self, time):
        """
        Convert time (in seconds) to the correspoding sample index in the time
        series.
        """
        time = time + self.start_time
        index = round(time * self.fs)
        return index

    def __repr__(self):
        """Representation for TimeSeries object."""
        class_name = self.__class__.__name__
        dim = f"({self.nfeatures}, {self.nsamples})"
        other = f"fs={self.fs}, unit={self.unit}, label={self.label}"
        return f"{class_name}: {dim}, {other}"

    def __getitem__(self, sl):
        """
        Get an excerpt from the time series using slices. Return a new
        TimeSeries object.
        """
        if type(sl) == Segment:
            time_offset = sl.start
            sl = sl.generate_slice(self)
        elif (type(sl) == slice):
            index_start = sl.start or sl.stop
            time_offset = conversion.sample_index_to_seconds(
                index_start, self.fs)
        else:
            raise ValueError("invalid value for slicing operation: must be " +
                             "of type `Segment` or a Python slice")

        sliced_data = self.data[sl]
        ts = cp.copy(self)
        ts.data = sliced_data
        ts.start_time += time_offset  # shift start
        return ts

    def get_samples(self, start, stop):
        """
        Get an excerpt from the time series using sample indexes. Return a new
        TimeSeries object.
        """
        return self[..., start:stop]

    # Arithmetic, relational and boolean operations
    def __add__(self, other):
        """Add two time series."""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data + other.data
        return ts

    def __sub__(self, other):
        """Subtract two time series."""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data - other.data
        return ts

    def __mul__(self, other):
        """Multiplicate two time series element-wise."""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data * other.data
        return ts

    def __truediv__(self, other):
        """Divide two time series element-wise."""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data / other.data
        return ts

    def __mod__(self, other):
        """Division remainder for two time series taken element-wise."""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data % other.data
        return ts

    def __lt__(self, other):
        """Less than"""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data < other.data
        return ts

    def __le__(self, other):
        """Less than or equal to"""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data <= other.data
        return ts

    def __gt__(self, other):
        """Greater than"""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data > other.data
        return ts

    def __ge__(self, other):
        """Greater than or equal to"""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data >= other.data
        return ts

    def __eq__(self, other):
        """Equal to"""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data == other.data
        return ts

    def __ne__(self, other):
        """Not equal to"""
        if self.data.shape != other.data.shape:
            raise DimensionalityError("The shape of the time series do not "
                                      "match.")
        ts = self.copy()
        ts.data = self.data != other.data
        return ts

    def __len__(self):
        """Length of the time series -- number of samples."""
        return self.data.shape[-1]

    def _write_data(self, data):
        """
        Write the given data to the data field. This method should be
        implemented by subclasses to control how the data field is initialized.

        Be careful using this method; it modifies the object.
        """
        self.data = data

    def _shift_start(self, seconds):
        """
        Shift the start of the time series by the number of seconds given.

        Be careful using this method; it modifies the object.
        """

        self.start_time += seconds


class DimensionalityError(Exception):
    """
    Exception raised for errors in dimensionality of arays
    """
