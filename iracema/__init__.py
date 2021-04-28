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

import iracema.features
import iracema.pitch
import iracema.harmonics
import iracema.segmentation
import iracema.spectral
import iracema.plot
import iracema.util

import iracema.segmentation.onsets

__version__ = u'0.2.0'

root = _dirname(__spec__.origin)
