import pytest  # skipcq: PYL-W0611

import iracema as ir
import numpy as np


def test_numpy_conversion():
    points_array = np.array([0.1, 1.2, 2.5, 4.1])
    points = ir.PointList.from_numpy(points_array)
    new_points_array = points.to_numpy()
    assert np.all(points_array == new_points_array)
