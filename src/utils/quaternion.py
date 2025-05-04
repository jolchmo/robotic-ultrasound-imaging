import numpy as np

# 假设 difference_quat 和 q_log 函数已经正确定义并可用


def difference_quat(q1, q2):
    """Calculates the difference quaternion q1 * inverse(q2)"""
    # This function needs to be correctly implemented elsewhere
    # Example (assuming q is [w, x, y, z]):
    # q2_inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]]) / np.sum(q2**2) # Inverse for unit quaternion is just conjugate / norm^2. If unit, just conjugate.
    # Assuming unit quaternions, inverse is conjugate:
    q2_conj = np.array([q2[0], -q2[1], -q2[2], -q2[3]])
    # Quaternion multiplication (q1 * q2_conj)
    # w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    # x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    # y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    # z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = q1[0]*q2_conj[0] - q1[1]*q2_conj[1] - q1[2]*q2_conj[2] - q1[3]*q2_conj[3]
    x = q1[0]*q2_conj[1] + q1[1]*q2_conj[0] + q1[2]*q2_conj[3] - q1[3]*q2_conj[2]
    y = q1[0]*q2_conj[2] - q1[1]*q2_conj[3] + q1[2]*q2_conj[0] + q1[3]*q2_conj[1]
    z = q1[0]*q2_conj[3] + q1[1]*q2_conj[2] - q1[2]*q2_conj[1] + q1[3]*q2_conj[0]
    return np.array([w, x, y, z])


def q_log(q):
    """Calculates the logarithm map of a quaternion"""
    # This function needs to be correctly implemented elsewhere
    # Example (assuming q is [w, x, y, z]):
    vec = q[1:]
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-6:  # Handle scalar quaternion case
        # If q is [w, 0, 0, 0], log is [log(w), 0, 0, 0]
        # But typically log map is for unit quaternions where w = cos(theta/2)
        # If q is [1, 0, 0, 0], log is [0, 0, 0]
        # If q is [-1, 0, 0, 0], log is [0, pi, 0, 0] or similar based on axis
        # For distance, often normalize q to unit length first.
        # Assuming q is a unit quaternion for distance calculation:
        return np.array([0.0, vec[0], vec[1], vec[2]])  # Or just the vector part [x,y,z] if log map definition is R^3
    else:
        theta = 2 * np.arctan2(vec_norm, q[0])
        return (theta / vec_norm) * vec  # Returns vector part (3D)

# Assuming q_log returns the 3D vector part [x, y, z]
# If q_log returns a 4D vector [0, x, y, z], adjust np.linalg.norm accordingly


def distance_quat(q1, q2):
    """
    Calculates distance metric between two quaternions as defined in eq. (20) in https://ieeexplore.ieee.org/document/6907291

    Args:
        q1 (np.array): quaternion given as (w, x, y, z) where w is the real (scalar part), and (x, y, z) are the complex (vector) part.
                       Assumed to be a unit quaternion.
        q2 (np.array): quaternion given as (w, x, y, z) where w is the real (scalar part), and (x, y, z) are the complex (vector) part.
                       Assumed to be a unit quaternion.

    Returns:
        float: distance metric (angular distance in radians)
    """
    # Ensure quaternions are unit quaternions if required by difference_quat and q_log
    # q1 = q1 / np.linalg.norm(q1)
    # q2 = q2 / np.linalg.norm(q2)

    # q_mult represents the rotation from q2 to q1
    q_mult = difference_quat(q1, q2)  # This calculates q1 * conjugate(q2) assuming unit quaternions

    # This is the corrected line where the ValueError occurred
    # Check if q_mult is the identity quaternion multiplied by -1 (i.e., [-1, 0, 0, 0])
    # which represents the same rotation as the identity quaternion [1, 0, 0, 0]
    # If the difference is [-1, 0, 0, 0], the distance between orientations q1 and q2 is 0.
    # We need np.all() to check if all elements in the vector part are zero.
    if q_mult[0] == -1 and np.all(q_mult[1:] == np.array([0, 0, 0])):
        return 0  # Or perhaps np.pi depending on how distance is defined for opposite quaternions?
        # Based on the formula's intent (distance between orientations), 0 seems correct.

    # Use the logarithm map to get the rotation vector/axis-angle representation
    # Assuming q_log returns the 3D vector part [vx, vy, vz]
    log_q_mult = q_log(q_mult)

    # The distance is 2 times the norm of the logarithm map result
    # The norm of log(q) for a unit quaternion q = [cos(theta/2), v*sin(theta/2)] is theta/2
    # So 2 * norm(log(q)) gives theta, the angle of rotation.
    dist = 2 * np.linalg.norm(log_q_mult)

    # This check handles cases where the calculated angle might be > pi due to the 2*pi periodicity
    # It ensures the distance is the shortest angle between orientations (between 0 and pi)
    if dist > np.pi:
        dist = abs(2 * np.pi - dist)  # Should clamp to the range [0, pi]

    return dist
