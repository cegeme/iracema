"""
iracema is a python package aimed at the extraction of expressive music
information from audio signals
"""

from .audio import Audio
from .timeseries import TimeSeries
from .segment import Segment, SegmentList

import iracema.features
import iracema.harmonics
import iracema.pitch
import iracema.plot
import iracema.segmentation
import iracema.spectral
