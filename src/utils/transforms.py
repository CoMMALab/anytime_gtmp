"""Transform helpers implemented using brax.math.

These are thin wrappers around brax.math functions. They return JAX arrays
and follow the (w, x, y, z) quaternion ordering used in the repo.

Note: this module requires `brax` to be installed in the active environment.
"""

from importlib import import_module
import jax.numpy as jnp

bm = import_module("brax.math")


def euler_to_quaternion(euler_xyz):
    """Convert Euler angles (roll, pitch, yaw) in radians to a quaternion (w, x, y, z).

    Args:
        euler_xyz: array-like of length 3 or shape (3,) (roll, pitch, yaw)

    Returns:
        jnp.ndarray shape (4,) in (w, x, y, z) order.
    """
    q = bm.euler_to_quat(jnp.asarray(euler_xyz, dtype=jnp.float32))
    return jnp.asarray(q, dtype=jnp.float32)


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix using brax.

    Args:
        quat: array-like length 4 (w, x, y, z)

    Returns:
        jnp.ndarray shape (3,3)
    """
    q = jnp.asarray(quat, dtype=jnp.float32)
    R = bm.quat_to_3x3(q)
    R = jnp.asarray(R, dtype=jnp.float32)
    # If brax returns a 4x4 matrix (homogeneous), take upper-left 3x3
    if R.shape == (4, 4):
        return R[:3, :3]
    return R
