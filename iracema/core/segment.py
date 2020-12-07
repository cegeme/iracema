"""
This module contain classes used to manipulate segments and slice TimeSeries
objects using them.
"""

from iracema.util import conversion


class Segment:
    """
    The objects generated from this class are used to retrieve excerpts from
    time-series.

    .. Hint:: This class is also available at the main package level as
        ``iracema.Segment``.
    """
    def __init__(self, time_series, start, end, limits_unit='sample_index'):
        """
        The `start` and `end` arguments can be specified in terms of sample
        index or time (in seconds). The argument `limits_unit` must be set
        accordingly.

        Args
        ----
        time_series : TimeSeries
            Original time series related to the segment.
        start : int or float
            Index (or sample number) corresponding to the start of the segment
            in the time-series from which it derived. Alternatively, this
            value can be specified in seconds.
        end : int or float
            Index of the ending sample for the segment. Alternatively, this
            value can be specified in seconds.
        limits_unit : ("sample_index", "seconds")
            If 'sample_index' is passed (default), the arguments `start` and
            `end` must be integers corresponding to sample indexes whitin
            `time_series`. Else, if 'seconds' is passed, both arguments must
            correspond to the time of these limits, in seconds.
        """
        if limits_unit not in ('sample_index', 'seconds'):
            raise ValueError("invalid value for `limits_unit` argument: must"
                             " be 'sample_index' or 'seconds'")

        if end is not None and start > end:
            raise ValueError("end must be > start")

        self.fs = time_series.fs
        self.time_offset = time_series.start_time

        if limits_unit == 'sample_index':
            if type(start) != int or type(end) != int:
                raise ValueError("`start` and `end` must be of type int when"
                                 "`limits_unit`=='sample_index'")
            self.start = start
            self.end = end

        elif limits_unit == 'seconds':
            self.start = conversion.seconds_to_sample_index(start, self.fs)
            self.end = conversion.seconds_to_sample_index(end, self.fs)

    @property
    def nsamples(self):
        return self.end - self.start

    @property
    def start_time(self):
        return conversion.sample_index_to_seconds(self.start, self.fs,
                                                  self.time_offset)

    @property
    def end_time(self):
        return conversion.sample_index_to_seconds(self.end, self.fs,
                                                  self.time_offset)

    def generate_slice(self, time_series):
        """
        Generate a python slice with the sample indexes that correspond to the
        current segment in `time_series`
        """
        new_fs = time_series.fs
        new_time_offset = time_series.start_time

        if self.fs == time_series.fs:
            return slice(self.start, self.end)
        else:
            slice_start = conversion.map_sample_index(
                self.start, self.fs, self.time_offset, new_fs, new_time_offset)
            slice_end = conversion.map_sample_index(
                self.end, self.fs, self.time_offset, new_fs, new_time_offset)

            return slice(slice_start, slice_end)

    def __repr__(self):
        "Overload the representation for the class"
        class_name = self.__class__.__name__
        description = "(start: {}, end: {}), fs: {}".format(
            self.start, self.end, self.fs)
        return '{}: {}'.format(class_name, description)


class SegmentList(list):
    """
    List of segments.

    .. Hint:: This class is also available at the main package level as
        ``iracema.SegmentList``.
    """
    def __init__(self, segment_list):
        """
        Args
        ----
        point_list : list[Segment]
            A list of segments.
        """
        super(SegmentList, self).__init__(segment_list)