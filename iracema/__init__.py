"""
iracema is a python package aimed at the extraction of expressive music
information from audio signals
"""
from .timeseries import TimeSeries, Audio
from .segment import Segment, SegmentList

import iracema.features
import iracema.harmonics
import iracema.pitch
import iracema.plot
import iracema.segmentation
import iracema.spectral

__version__ = u'0.1.5'
