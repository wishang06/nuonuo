"""Head pose from a MediaPipe facial transformation matrix.

Degrees match the mirrored preview used by BondsAI: positive yaw is the
user's right, positive pitch is chin up, positive roll is toward the
user's right shoulder.
"""

import math

_RAD2DEG = 180.0 / math.pi


def _unit3(x: float, y: float, z: float) -> tuple[float, float, float] | None:
    length = math.hypot(x, y, z)
    if length < 1e-8:
        return None
    return x / length, y / length, z / length


def _rows_from_matrix(data) -> list[list[float]] | None:
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) == 2 and shape[0] >= 3 and shape[1] >= 3:
        return [[float(data[r][c]) for c in range(4)] for r in range(4)]
    try:
        flat = [float(v) for v in list(data)]
    except (TypeError, ValueError):
        return None
    if len(flat) < 16:
        return None
    rows = [[0.0] * 4 for _ in range(4)]
    for col in range(4):
        for row in range(4):
            rows[row][col] = flat[col * 4 + row]
    return rows


def head_pose_from_matrix(data) -> tuple[float, float, float] | None:
    """Return (yaw, pitch, roll) in degrees, or None if the matrix is unusable.

    A flat 16-length sequence is column-major, matching the MediaPipe JS
    matrix. A 4x4 array is treated as row-major.
    """
    rows = _rows_from_matrix(data)
    if rows is None:
        return None
    col_x = _unit3(rows[0][0], rows[1][0], rows[2][0])
    col_y = _unit3(rows[0][1], rows[1][1], rows[2][1])
    col_z = _unit3(rows[0][2], rows[1][2], rows[2][2])
    if col_x is None or col_y is None or col_z is None:
        return None
    m11 = col_x[0]
    m12 = col_y[0]
    m13 = max(-1.0, min(1.0, col_z[0]))
    m22 = col_y[1]
    m23 = col_z[1]
    m32 = col_y[2]
    m33 = col_z[2]
    raw_yaw = math.asin(m13)
    if abs(m13) < 0.9999999:
        raw_pitch = math.atan2(-m23, m33)
        raw_roll = math.atan2(-m12, m11)
    else:
        raw_pitch = math.atan2(m32, m22)
        raw_roll = 0.0
    return -raw_yaw * _RAD2DEG, -raw_pitch * _RAD2DEG, -raw_roll * _RAD2DEG
