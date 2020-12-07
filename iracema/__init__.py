"""
iracema is a python package aimed at the extraction of expressive music
information from audio signals
"""
from os.path import dirname as _dirname

import iracema.core.timeseries
import iracema.core.audio
import iracema.core.point
import iracema.core.segment

from iracema.core.timeseries import TimeSeries
from iracema.core.audio import Audio
from iracema.core.point import Point, PointList
from iracema.core.segment import Segment, SegmentList

__version__ = u'0.1.7'

root = _dirname(__spec__.origin)
