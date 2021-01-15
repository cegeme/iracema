import pytest

import iracema as ir


def test_segment_start(audio01):
    s = ir.Segment(10, 15)
    a1 = audio01[s]
    assert a1.start_time == ir.Point(10)


def test_segment_nsamples(audio01, audio02):
    s = ir.Segment(10, 15)
    a1 = audio01[s]
    a2 = audio02[s]
    assert a1.nsamples == a2.nsamples


def test_segment_list(audio01):
    s1 = ir.Segment(1, 2)
    s2 = ir.Segment(3, 4)
    s3 = ir.Segment(5, 6)
    seg_list = ir.SegmentList(segments=[s1, s2])
    seg_list.append(s3)
    assert seg_list == [s1, s2, s3]
