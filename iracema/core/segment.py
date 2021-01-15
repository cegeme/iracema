"""
This module contain classes used to manipulate segments and slice TimeSeries
objects using them.
"""
from collections.abc import MutableSequence

from iracema.core.point import Point


class Segment:
    """
    The objects generated from this class are used to retrieve excerpts from
    time-series.

    .. Hint:: This class is also available at the main package level as
        ``iracema.Segment``.
    """

    def __init__(self, start, end):
        """
        Segment

        Args
        ----
        start : Point
            Point corresponding to the start of the segment.
        end : Point
            Point corresponding to the end of the segment.
        """
        if end is not None and start > end:
            raise ValueError("end must be > start")

        self.start = Point(start)
        self.end = Point(end)

    def __repr__(self):
        "Overload the representation for the class"
        class_name = self.__class__.__name__
        description = f"({self.start}, {self.end})"
        return f"{class_name}{description}"

    def nsamples(self, time_series):
        return (self.end.map_index(time_series) -
                self.start.map_index(time_series))

    @property
    def duration(self):
        return self.end - self.start

    @property
    def start_time(self):
        return self.start

    @property
    def end_time(self):
        return self.end

    def generate_slice(self, time_series):
        """
        Generate a python slice with the sample indexes that correspond to the
        current segment in `time_series`
        """
        slice_start = self.start.map_index(time_series)
        slice_end = self.end.map_index(time_series)
        return slice(slice_start, slice_end)

    def map_indexes(self, time_series):
        """
        Return a tuple with the indexes of ``time_series`` that correspond
        to the segments in the list.
        """
        return (
            self.start.map_index(time_series),
            self.end.map_index(time_series)
        )


class SegmentList(MutableSequence):
    """
    List of segments.

    .. Hint:: This class is also available at the main package level as
        ``iracema.SegmentList``.
    """

    def __init__(self, segments=None):
        super(SegmentList, self).__init__()
        if (segments is not None):
            self._segments = list(segments)
        else:
            self._segments = []

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SegmentList(self._segments[index])
        return self._segments[index]

    def __setitem__(self, index, item):
        if not isinstance(item, Segment):
            raise ValueError(
                "The list contains an item that is not a ``Segment``")
        self._segments[index] = item

    def __delitem__(self, index):
        self._segments.__delitem__(index)

    def __len__(self):
        return len(self._segments)

    def __repr__(self):
        "Overload the representation for the class"
        return str(list(self))

    def insert(self, index, item):  # skipcq: PYL-W0221
        if not isinstance(item, Segment):
            raise ValueError("The insert item is not a ``Segment``")
        return self._segments.insert(index, item)

    def map_indexes(self, time_series):
        """
        Return an array of tuples with the indexes of ``time_series`` that correspond to
        the segments in the list.
        """
        return [seg.map_indexes(time_series) for seg in self]
