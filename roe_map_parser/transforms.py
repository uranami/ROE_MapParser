"""Source coordinate transforms: column vectors, XYZ position, Z up."""
import numpy as np


def quaternion_matrix(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not .99 < np.linalg.norm(q) < 1.01:
        raise ValueError('invalid xyzw quaternion')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler_matrix(value):
    angles = np.asarray(value, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError('invalid XYZ Euler radians')
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx@ry@rz


def instance_matrix(row, kind):
    if kind == 'static':
        rotation = euler_matrix([row['mRotate'+a] for a in 'XYZ'])
    elif kind == 'tree':
        rotation = quaternion_matrix([row['mQuat'+a] for a in 'XYZW'])
    else:
        raise ValueError('unknown instance kind')
    scale = float(row['mScale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid instance scale')
    matrix = np.eye(4)
    matrix[:3, :3] = rotation*scale
    matrix[:3, 3] = [row['mPos'+a] for a in 'XYZ']
    if not np.isfinite(matrix).all():
        raise ValueError('nonfinite instance transform')
    return matrix
