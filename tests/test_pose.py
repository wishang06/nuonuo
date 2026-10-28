import math

import pytest

from nuonuo.pose import head_pose_from_matrix


def _rx(degrees: float) -> list[float]:
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]


def _ry(degrees: float) -> list[float]:
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]


def test_identity_and_chin_up():
    yaw, pitch, roll = head_pose_from_matrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    assert yaw == pitch == roll == 0
    _yaw, pitch, _roll = head_pose_from_matrix(_rx(-20))
    assert pitch == pytest.approx(20, abs=1e-4)
    yaw, _pitch, _roll = head_pose_from_matrix(_ry(-30))
    assert yaw == pytest.approx(30, abs=1e-4)


def test_recorded_landmarker_matrix_and_rejection():
    pose = head_pose_from_matrix([
        0.5567780137062073, 0.034023914486169815, 0.8299639821052551, 0,
        -0.011918997392058372, 0.9993847608566284, -0.03297339752316475, 0,
        -0.8305754661560059, 0.008466490544378757, 0.5568410754203796, 0,
        -1.418548345565796, 6.790719509124756, -39.25355529785156, 1,
    ])
    assert pose is not None
    yaw, pitch, roll = pose
    assert yaw == pytest.approx(56.16, abs=0.05)
    assert pitch == pytest.approx(0.87, abs=0.05)
    assert roll == pytest.approx(-1.23, abs=0.05)
    assert head_pose_from_matrix([1, 0, 0]) is None
    assert head_pose_from_matrix([0.0] * 16) is None
