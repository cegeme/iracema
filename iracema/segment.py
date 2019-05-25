"""
Contain classes used to extract and manipulate time series segments.
"""
import numpy as np

from .util import conversion


class Segment:
    """
    The objects generated from this class are used to delimit segments from
    time-series.

    The `start` and `end` arguments can be specified in terms of sample index
    or time (in seconds). The argument `limits_unit` must be set accordingly.

    Args
    ----
    time_series : TimeSeries
        Original time series related to the segment.
    start : int or float
        Index (or sample number) corresponding to the start of the segment in
        the time-series from which it devired. Alternatively, this value can
        be specified in seconds.
    end : int or float
        Index of the ending sample for the segment. Alternatively, this value
        can be specified in seconds.
    limits_unit : ("sample_index", "seconds")
        If 'sample_index' is passed (default), the arguments `start` and `end`
        must be integers corresponding to sample indexes whitin `time_series`.
        Else, if 'seconds' is passed, both arguments must correspond to the
        time of these limits, in seconds.
    """
    def __init__(self, time_series, start, end, limits_unit='sample_index'):
        if limits_unit not in ('sample_index', 'seconds'):
            raise ValueError("invalid value for `limits_unit` argument: must" +
                             " be 'sample_index' or 'seconds'")

        if end is not None and start > end:
            raise ValueError("end must be > start")

        self.fs = time_series.fs
        self.time_offset = time_series.start_time

        if limits_unit == 'sample_index':
            if type(start) != int or type(end) != int:
                raise ValueError("`start` and `end` must be of type int when" +
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
            # converting sample index to seconds
            start_seconds = conversion.sample_index_to_seconds(
                self.start, self.fs, self.time_offset)
            end_seconds = conversion.sample_index_to_seconds(
                self.end, self.fs, self.time_offset)

            # converting it back to sample index in a different fs
            new_slice_start = conversion.seconds_to_sample_index(
                start_seconds, new_fs, new_time_offset)
            new_slice_end = conversion.seconds_to_sample_index(
                end_seconds, new_fs, new_time_offset)

            return slice(new_slice_start, new_slice_end)

    def __repr__(self):
        "Overload the representation for the class"
        class_name = self.__class__.__name__
        description = "(start: {}, end: {}), fs: {}".format(
            self.start, self.end, self.fs)
        return '{}: {}'.format(class_name, description)


class SegmentList(list):
    """
    List of segments.
    """

    def __init__(self, segment_list):
        super(SegmentList, self).__init__(segment_list)

    @staticmethod
    def load_from_csv_file(time_series,
                           filename,
                           start_only=False,
                           cls=Segment,
                           limits='seconds'):
        """
        Load segment info from a csv file and create a segment list.
        """
        import csv
        start, end = [], []

        with open(filename, 'r', newline='') as ff:
            for row in csv.reader(ff, delimiter=','):
                start.append(np.float_(row[0]))
                if not start_only:
                    end.append(np.float_(row[1]))

        if start_only:
            # if only the start is supposed to be read from the csv file,
            # then the end of each segment will correpond to the starting
            # sample of the next segment
            end = start.copy()
            end.pop(0)
            end.append(None)

        segments = [
            cls(time_series, st, end, limits) for st, end in zip(start, end)
        ]
        return SegmentList(segments)

    def save_to_csv_file(self, filename, limits='seconds', start_only=False):
        """
        Save segment list to a csv file.
        """
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for note in self:
                if limits == 'seconds':
                    line = [note.start_time, note.end_time]
                else:
                    line = [note.start, note.end]

                if start_only:
                    writer.writerow([line[0]])
                else:
                    writer.writerow(line)
