"""
Implementation of time series.
"""

import copy as cp

import matplotlib.pyplot as plt
import numpy as np

from .segment import Segment
from .util import conversion


class TimeSeries:
    """
    Base class for time-series objects, which can represent synchronous
    discrete time-series.

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
    start_time : float
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
        Number of samples per time-series in the data array.
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

    def __init__(self, fs, data=None, start_time=0., unit=None):
        """
        """
        if fs <= 0:
            raise ValueError(
                "the sampling frequency (fs) must be greater than zero")

        self.data = None

        self.fs = np.float_(fs)
        self.start_time = 0 if start_time is None else np.float_(start_time)

        if unit:
            self.unit = unit

        if data is not None:
            self._write_data(data)

    @property
    def nsamples(self):  # pylint: disable=missing-docstring
        return self.data.shape[-1]

    @property
    def nfeatures(self):  # pylint: disable=missing-docstring
        if self.data.ndim == 1:
            return 1
        elif self.data.ndim == 2:
            return self.data.shape[-2]

    @property
    def duration(self):  # pylint: disable=missing-docstring
        return self.nsamples / self.fs

    @property
    def nyquist(self):  # pylint: disable=missing-docstring
        return self.fs / 2

    @property
    def end_time(self):  # pylint: disable=missing-docstring
        return self.start_time + self.duration

    @property
    def ts(self):  # pylint: disable=missing-docstring
        return 1 / self.fs

    @property
    def time(self):  # pylint: disable=missing-docstring
        start = self.start_time
        end = self.end_time
        return np.linspace(start, end, self.nsamples)

    def copy(self):
        "Return a copy of the time-series object (deep copy)."
        return cp.deepcopy(self)

    def normalize(self):
        "Return a normalized copy of the time-series."
        normalized_data = self.data / np.max(self.data)

        ts = cp.copy(self)
        ts.data = normalized_data
        return ts

    def diff(self, n=1):
        "Return the n-th discrete difference for the time-series"
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
        Converts zeros to np.nan in the data array. Returns a new time-series.
        """
        ts = self.copy()
        ts.data[self.data == 0] = np.nan
        return ts

    def hwr(self):
        "Return a half-wave rectified copy of the time-series."
        rectified_data = np.clip(self.data, 0, None)

        ts = cp.copy(self)
        ts.data = rectified_data

        return ts

    def plot(self):
        "Plot the time series using matplotlib."
        f = plt.figure(figsize=(15, 9))
        plt.plot(self.time, self.data, label=self.label)
        if self.label:
            plt.legend(loc='lower right', ncol=2, fontsize='x-small')
        plt.title(self.caption)
        plt.ylabel(self.unit)
        plt.xlabel('time (s)')
        plt.show()

        return f

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
        dim = '(nfeatures={}, nsamples={})'.format(self.nfeatures,
                                                   self.nsamples)
        other = 'fs={}, unit={}, label={}'.format(self.fs, self.unit,
                                                  self.label)
        return '{}: {}, {}'.format(class_name, dim, other)

    def __getitem__(self, sl):
        """
        Get an excerpt from the time series using slices. Return a new
        TimeSeries object.
        """
        if type(sl) == Segment:
            time_offset = conversion.sample_index_to_seconds(
                sl.start, sl.fs)
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
    pass
