"""
This module contain classes used to manipulate points in TimeSeries objects.
"""
from collections.abc import MutableSequence
import csv
from decimal import Decimal

import numpy as np

import iracema.core.segment


class Point(Decimal):
    """
    A point object represents an instant in a time series, i.e., one specific
    sample index. It is flexible enough to locate samples corresponding to the
    same instant in time series with different sampling rates.

    .. Hint:: This class is also available at the main package level as
        ``iracema.Point``.
    """
    def __repr__(self):
        return f"Point({self})"

    @classmethod
    def from_sample_index(cls, index, time_series):
        time_offset = Decimal(time_series.start_time)
        time = Decimal(int(index)) / Decimal(int(time_series.fs))
        time += time_offset
        return Point(time)

    @property
    def time(self):
        """
        Return the time of the points.
        This method will be deprecated soon.
        """
        return self

    def map_index(self, time_series):
        time = self - Decimal(time_series.start_time)
        index = time * Decimal(int(time_series.fs))
        return int(round(index))

    def get_value(self, time_series):
        index = self.map_index(time_series)
        return time_series.data[..., index]


class PointList(MutableSequence):
    """
    List of points.

    .. Hint:: This class is also available at the main package level as
        ``iracema.PointList``.
    """
    def __init__(self, points=None):
        super(PointList, self).__init__()
        if (points is not None):
            self._points = list(points)
        else:
            self._points = []

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PointList(self._points[index])
        return self._points[index]

    def __setitem__(self, index, item):
        if not isinstance(item, Point):
            raise ValueError(
                "The list contains an item that is not a ``Point``")
        self._points[index] = item

    def __delitem__(self, index):
        self._points.__delitem__(index)

    def __len__(self):
        return len(self._points)

    def __repr__(self):
        return str(list(self))

    def insert(self, index, item):  # skipcq: PYL-W0221
        if not isinstance(item, Point):
            raise ValueError(
                "The insert item is not a ``Point``")
        return self._points.insert(index, item)

    @classmethod
    def load_from_file(cls, file_name):
        """
        Instantiate a list of points, loaded from a CSV file. Each line in the
        file must contain the position of a single point. The position must be
        specified in `seconds`.
        """
        with open(file_name, 'r', newline='') as f:
            points = cls([
                Point(row[0])
                for row in csv.reader(f, delimiter=',')
            ])
        return points

    def save_to_file(self, file_name):
        "Save list of points to CSV file."
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for point in self:
                writer.writerow([point])

    @classmethod
    def from_list_of_indexes(cls, list_indexes, time_series):
        """
        Instantiate a list of points from a list of indexes ``list_indexes``
        and a ``time_series`` object.
        """
        return cls([
            Point.from_sample_index(index, time_series)
            for index in list_indexes
        ])

    @property
    def time(self):
        """
        Return a list with the time of the points.
        This method will be deprecated soon.
        """
        return self

    def map_indexes(self, time_series):
        """
        Return an array with the indexes of ``time_series`` that correspond to
        the points in the list.
        """
        return [point.map_index(time_series) for point in self]

    def get_values(self, time_series):
        """
        Get values from the ``time_series`` corresponding to the points in the
        list.
        """
        return [point.get_value(time_series) for point in self]

    def to_numpy(self):
        """
        Retun a numpy array with the time of the points.
        """
        return np.array([np.float(point) for point in self])

    @classmethod
    def from_numpy(cls, array):
        """
        Instantiate a list of points from a numpy array.
        """
        return cls([Point(p) for p in array])

    def to_segments(self):
        """
        Convert list of points to a list of contiguous segments.
        """
        segments =  iracema.core.segment.SegmentList()
        for pt_start, pt_end in zip(self[0:-1], self[1:]):
            print('-- ', pt_start, pt_end)
            segments.append(iracema.core.segment.Segment(pt_start, pt_end))
        return segments
    
