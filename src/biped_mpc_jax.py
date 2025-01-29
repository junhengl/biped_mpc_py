import time
import numpy as np
import jax.numpy as jnp
from jaxopt import CvxpyQP

solver = CvxpyQP(
                implicit_diff_solve=True,
                solver='OSQP'
                )

# data class
class MPC:
    def __init__(self):
        self.h = 10
        self.dt = 0.04
        self.x_cmd = np.array([
                                0, 0, 0, 
                                0, 0, 0.55, 
                                0, 0, 0, 
                                0, 0, 0
                            ])  # Command
        # self.Q = np.array([500, 100, 100,  500, 500, 500,  1, 1, 1,   1, 1, 1, 1])  # State weights - walking
        self.Q = np.array([100, 100, 100,  500, 100, 500,  1, 1, 1,   1, 1, 1, 1])  # State weights - standing and height change
        self.R = np.array([1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1]) * 1e-6  # Control input weights
        self.kv = 0.03
        self.kp = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*1000
        self.kd = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*5
        self.swingHeight = 0.1
        self.y_offset = 0.07

class Biped:
    def __init__(self):
        self.m = 10  # Mass
        self.I = np.array([[0.532, 0, 0],
                          [0, 0.5420, 0],
                          [0, 0, 0.0711]])  # Inertia
        self.lt = 0.09  # toe length
        self.lh = 0.05  # heel length
        self.g = 9.81  # Gravity
        self.hip_offset = np.array([-0.005, 0.047, -0.126])
        self.mu = 0.4
        self.f_max = np.array([[250], [250], [250]])
        self.f_min = np.array([[0], [0], [0]])
        self.tau_max =  np.array([[33.5], [33.5], [33.5]])
        self.tau_min = -self.tau_max

mpc = MPC()
biped = Biped()

# numpy static, no need for dynamic jax arrays
def get_contact_sequence(t, mpc):
    # Default contact sequence
    contact = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ]).T
    phase = int(t // mpc.dt)  # Calculate the phase
    k = phase % mpc.h  # Remainder of phase divided by ctrl.mpc.h
    contact = contact[k:k+10, :]
    return contact

def get_reference_trajectory(x_fb):
    x_ref = np.tile(np.append(mpc.x_cmd, 1), (mpc.h, 1)).T
    # x_ref[:12, 0] = x_fb
    for i in range(6):
        for k in range(0, mpc.h):
            if mpc.x_cmd[i + 6] != 0:
                x_ref[i, k] = x_fb[i] + mpc.x_cmd[i + 6] * (k * mpc.dt)
                # x_ref = x_ref.at[i, k].set(x_fb[i] + mpc.x_cmd[i + 6] * (k * mpc.dt))
            else:
                x_ref[i, k] = mpc.x_cmd[i]
                # x_ref = x_ref.at[i, k].set(mpc.x_cmd[i])
    return x_ref

def get_reference_foot_trajectory(x_fb, t, foot, contact):
    foot_des_x_1 = (
        x_fb[3] + x_fb[9] * 1 / 2 * mpc.h / 2 * mpc.dt
        + mpc.kv * (x_fb[3] - mpc.x_cmd[3])
    )
    foot_des_x_2 = (
        x_fb[3] + x_fb[9] * 1 / 2 * mpc.h * mpc.dt
        + mpc.kv * (x_fb[3] - mpc.x_cmd[3])
    )

    foot_des_y_1 = (
        x_fb[4] + x_fb[10] * 1 / 2 * mpc.h / 2 * mpc.dt
        + mpc.kv * (x_fb[4] - mpc.x_cmd[4]) 
    )
    foot_des_y_2 = (
        x_fb[4] + x_fb[10] * 1 / 2 * mpc.h * mpc.dt
        + mpc.kv * (x_fb[4] - mpc.x_cmd[4]) - mpc.y_offset
    )
    foot_des_z = 0
    foot_1 = np.array([foot_des_x_1, foot_des_y_1 + mpc.y_offset, foot_des_z, foot_des_x_1, foot_des_y_1 - mpc.y_offset, foot_des_z])
    foot_2 = np.array([foot_des_x_2, foot_des_y_2 + mpc.y_offset, foot_des_z, foot_des_x_2, foot_des_y_2 - mpc.y_offset, foot_des_z])

    foot = foot.reshape(-1, 1)
    foot_1 = foot_1.reshape(-1, 1)
    foot_2 = foot_2.reshape(-1, 1)

    phase = int(t // mpc.dt)
    k = phase % mpc.h
    kk = k % 5  # 0, 1, 2, 3, 4
    if np.sum(contact[0, :]) == 1:
        foot_vec = np.tile(foot, (1, 5 - kk))
        foot_1_vec = np.tile(foot_1, (1, 5))
        foot_2_vec = np.tile(foot_2, (1, kk))
        foot_ref = np.concatenate((foot_vec, foot_1_vec, foot_2_vec), axis=1)
    else:
        foot_ref = np.tile(foot, (1, mpc.h))

    # foot_ref = np.tile(foot, (1, mpc.h)) # TODO not ideal change this
    return foot_ref

# jax dynamic, need to use jax arrays
def skew(v):
    """Skew-symmetric matrix of a 3D vector v."""
    return jnp.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],    0]
    ])

def eul2rotm(eul):
    """
    Example 3-2-1 intrinsic Euler angles: eul = [roll, pitch, yaw].
    In MATLAB, you used eul2rotm(flip(eul')), which might be [yaw, pitch, roll].
    This is a placeholder. You must adjust for your actual rotation convention.
    """
    # If you want the exact approach:
    #   eul is 3x1 = [r, p, y], but "flip(eul')" => [y, p, r]
    #   Possibly do something with SciPy:
    #       from scipy.spatial.transform import Rotation as R
    #       R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    #
    # For demonstration, let's say your eul = [roll, pitch, yaw]
    cr, cp, cy = jnp.cos(eul)
    sr, sp, sy = jnp.sin(eul)

    # Z-Y-X rotation (roll around X, pitch around Y, yaw around Z)
    Rz = jnp.array([[ cy, -sy,  0 ],
                   [ sy,  cy,  0 ],
                   [  0,   0,  1 ]])
    Ry = jnp.array([[ cp,  0,  sp ],
                   [  0,  1,   0 ],
                   [-sp,  0,  cp ]])
    Rx = jnp.array([[ 1,  0,   0 ],
                   [ 0, cr, -sr ],
                   [ 0, sr,  cr ]])
    # Combined: Rz * Ry * Rx
    return Rz @ Ry @ Rx

def get_simplified_dynamics(mpc, biped, x_ref, foot_ref):
    # Iterate through each step
    # Extract values from x_traj
    yaw = x_ref[2]
    pitch = x_ref[1]
    R = eul2rotm(x_ref[0:3])
    I = R.T @ biped.I @ R

    # Compute Ac matrix
    R_inv = jnp.linalg.inv(jnp.array([
        [jnp.cos(yaw) * jnp.cos(pitch), -jnp.sin(yaw), 0],
        [jnp.sin(yaw) * jnp.cos(pitch), jnp.cos(yaw), 0],
        [-jnp.sin(pitch), 0, 1]
    ]))
    Ac = jnp.block([
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), R_inv @ jnp.eye(3), jnp.zeros((3, 3)), jnp.zeros((3, 1))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.eye(3), jnp.zeros((3, 1))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 1))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.array([[0], [0], [-biped.g]])],
        [jnp.zeros((1, 13))]
    ])

    # Compute Bc matrix
    skew_1 = skew(-x_ref[3:6] + foot_ref[0:3])
    skew_2 = skew(-x_ref[3:6] + foot_ref[3:6])
    Bc = jnp.block([
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        [jnp.linalg.solve(I, skew_1), jnp.linalg.solve(I, skew_2), jnp.linalg.solve(I, jnp.eye(3)), jnp.linalg.solve(I, jnp.eye(3))],
        [jnp.eye(3) / biped.m, jnp.eye(3) / biped.m, jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        [jnp.zeros((1, 12))]
    ])
    A = Ac * mpc.dt + jnp.eye(13)
    B = Bc * mpc. dt
    return A, B

def getLegKinematics(q0, q1, q2, q3, q4, side):
    # Initialize the Jm matrix
    Jm = jnp.zeros((6, 5))
    sin = jnp.sin
    cos = jnp.cos

    # Fill in the matrix entries
    # Jm[0, 0] = sin(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135) + \
    #             cos(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))
    Jm =  Jm.at[0,0].set(sin(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135) + \
                cos(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))))
    # Jm[1, 0] = sin(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))) - \
    #             cos(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135)
    Jm = Jm.at[1,0].set(sin(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))) - \
                cos(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135))

    # Jm[2, 0] = 0.0
    # Jm[3, 0] = 0.0
    # Jm[4, 0] = 0.0
    # Jm[5, 0] = 1.0
    Jm = Jm.at[2,0].set(0.0)
    Jm = Jm.at[3,0].set(0.0)
    Jm = Jm.at[4,0].set(0.0)
    Jm = Jm.at[5,0].set(1.0)

    # Jm[0, 1] = -sin(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))
    Jm =  Jm.at[0,1].set(-sin(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))))
    # Jm[1, 1] = cos(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))
    Jm =  Jm.at[1,1].set(cos(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))))
    # Jm[2, 1] = sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - cos(q1) * (0.018 * side + 0.0025)
    Jm =  Jm.at[2,1].set(sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - cos(q1) * (0.018 * side + 0.0025))
    # Jm[3, 1] = cos(q0)
    Jm =  Jm.at[3,1].set(cos(q0))
    # Jm[4, 1] = sin(q0)
    Jm =  Jm.at[4,1].set(sin(q0))
    # Jm[5, 1] = 0.0
    Jm =  Jm.at[5,1].set(0.0)

    # Jm[0, 2] = sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2)) - \
    #             cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))
    Jm = Jm.at[0,2].set(sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2)) - \
                cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))

    # Jm[1, 2] = -sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - \
    #             cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2))
    Jm = Jm.at[1,2].set(-sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - \
                cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2)))

    # Jm[2, 2] = cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2))
    Jm = Jm.at[2,2].set(cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2)))
    # Jm[3, 2] = -cos(q1) * sin(q0)
    Jm = Jm.at[3,2].set(-cos(q1) * sin(q0))
    # Jm[4, 2] = cos(q0) * cos(q1)
    Jm = Jm.at[4,2].set(cos(q0) * cos(q1))
    # Jm[5, 2] = sin(q1)
    Jm = Jm.at[5,2].set(sin(q1))

    # Jm[0, 3] = sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3)) - \
    #             cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3))
    Jm = Jm.at[0,3].set(sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3)) - \
                cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3)))

    # Jm[1, 3] = -sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3)) - \
    #             cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3))
    Jm = Jm.at[1,3].set(-sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3)) - \
                cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3)))

    # Jm[2, 3] = cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3))
    Jm = Jm.at[2,3].set(cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3)))
    # Jm[3, 3] = -cos(q1) * sin(q0)
    Jm = Jm.at[3,3].set(-cos(q1) * sin(q0))
    # Jm[4, 3] = cos(q0) * cos(q1)
    Jm = Jm.at[4,3].set(cos(q0) * cos(q1))
    # Jm[5, 3] = sin(q1)
    Jm = Jm.at[5,3].set(sin(q1))

    # Jm[0, 4] = 0.04 * sin(q2 + q3 + q4) * sin(q0) * sin(q1) - \
    #             0.04 * cos(q2 + q3 + q4) * cos(q0)
    Jm = Jm.at[0,4].set(0.04 * sin(q2 + q3 + q4) * sin(q0) * sin(q1) - \
                0.04 * cos(q2 + q3 + q4) * cos(q0))
    
    # Jm[1, 4] = -0.04 * cos(q2 + q3 + q4) * sin(q0) - \
    #             0.04 * sin(q2 + q3 + q4) * cos(q0) * sin(q1)
    Jm = Jm.at[1,4].set(-0.04 * cos(q2 + q3 + q4) * sin(q0) - \
                0.04 * sin(q2 + q3 + q4) * cos(q0) * sin(q1))

    # Jm[2, 4] = 0.04 * sin(q2 + q3 + q4) * cos(q1)
    Jm = Jm.at[2,4].set(0.04 * sin(q2 + q3 + q4) * cos(q1))
    # Jm[3, 4] = -cos(q1) * sin(q0)
    Jm = Jm.at[3,4].set(-cos(q1) * sin(q0))
    # Jm[4, 4] = cos(q0) * cos(q1)
    Jm = Jm.at[4,4].set(cos(q0) * cos(q1))
    # Jm[5, 4] = sin(q1)
    Jm = Jm.at[5,4].set(sin(q1))

    Jf = Jm[0:3, :]
    return Jm, Jf

def getFootPositionBody(q0, q1, q2, q3, q4, side):
    # Initialize the pf vector
    # pf = jnp.zeros(3)
    # Fill in the vector entries
    pf = jnp.array([
    - (3 * jnp.cos(q0)) / 200 - \
        (9 * jnp.sin(q4) * (jnp.cos(q3) * (jnp.cos(q0) * jnp.cos(q2) - jnp.sin(q0) * jnp.sin(q1) * jnp.sin(q2)) - 
                            jnp.sin(q3) * (jnp.cos(q0) * jnp.sin(q2) + jnp.cos(q2) * jnp.sin(q0) * jnp.sin(q1)))) / 250 - \
        (11 * jnp.cos(q0) * jnp.sin(q2)) / 50 - \
        ((side) * jnp.sin(q0)) / 50 - \
        (11 * jnp.cos(q3) * (jnp.cos(q0) * jnp.sin(q2) + jnp.cos(q2) * jnp.sin(q0) * jnp.sin(q1))) / 50 - \
        (11 * jnp.sin(q3) * (jnp.cos(q0) * jnp.cos(q2) - jnp.sin(q0) * jnp.sin(q1) * jnp.sin(q2))) / 50 - \
        (9 * jnp.cos(q4) * (jnp.cos(q3) * (jnp.cos(q0) * jnp.sin(q2) + jnp.cos(q2) * jnp.sin(q0) * jnp.sin(q1)) + 
                            jnp.sin(q3) * (jnp.cos(q0) * jnp.cos(q2) - jnp.sin(q0) * jnp.sin(q1) * jnp.sin(q2)))) / 250 - \
        (23 * jnp.cos(q1) * (side) * jnp.sin(q0)) / 1000 - \
        (11 * jnp.cos(q2) * jnp.sin(q0) * jnp.sin(q1)) / 50,

    (jnp.cos(q0) * (side)) / 50 - \
        (9 * jnp.sin(q4) * (jnp.cos(q3) * (jnp.cos(q2) * jnp.sin(q0) + jnp.cos(q0) * jnp.sin(q1) * jnp.sin(q2)) - 
                            jnp.sin(q3) * (jnp.sin(q0) * jnp.sin(q2) - jnp.cos(q0) * jnp.cos(q2) * jnp.sin(q1)))) / 250 - \
        (3 * jnp.sin(q0)) / 200 - \
        (11 * jnp.sin(q0) * jnp.sin(q2)) / 50 - \
        (11 * jnp.cos(q3) * (jnp.sin(q0) * jnp.sin(q2) - jnp.cos(q0) * jnp.cos(q2) * jnp.sin(q1))) / 50 - \
        (11 * jnp.sin(q3) * (jnp.cos(q2) * jnp.sin(q0) + jnp.cos(q0) * jnp.sin(q1) * jnp.sin(q2))) / 50 - \
        (9 * jnp.cos(q4) * (jnp.cos(q3) * (jnp.sin(q0) * jnp.sin(q2) - jnp.cos(q0) * jnp.cos(q2) * jnp.sin(q1)) + 
                            jnp.sin(q3) * (jnp.cos(q2) * jnp.sin(q0) + jnp.cos(q0) * jnp.sin(q1) * jnp.sin(q2)))) / 250 + \
        (23 * jnp.cos(q0) * jnp.cos(q1) * (side)) / 1000 + \
        (11 * jnp.cos(q0) * jnp.cos(q2) * jnp.sin(q1)) / 50,

    (23 * (side) * jnp.sin(q1)) / 1000 - \
        (11 * jnp.cos(q1) * jnp.cos(q2)) / 50 - \
        (9 * jnp.cos(q4) * (jnp.cos(q1) * jnp.cos(q2) * jnp.cos(q3) - jnp.cos(q1) * jnp.sin(q2) * jnp.sin(q3))) / 250 + \
        (9 * jnp.sin(q4) * (jnp.cos(q1) * jnp.cos(q2) * jnp.sin(q3) + jnp.cos(q1) * jnp.cos(q3) * jnp.sin(q2))) / 250 - \
        (11 * jnp.cos(q1) * jnp.cos(q2) * jnp.cos(q3)) / 50 + \
        (11 * jnp.cos(q1) * jnp.sin(q2) * jnp.sin(q3)) / 50 - \
        3.0 / 50.0    
        ])
    return pf

def getFootPositionWorld(x_fb, q):
    R = eul2rotm(x_fb[0:3])
    pf_w = jnp.zeros((6,1))
    # print('pf_w', pf_w)
    for leg in range(2):
        q0 = q[5*leg+0]
        q1 = q[5*leg+1]
        q2 = q[5*leg+2]
        q3 = q[5*leg+3]
        q4 = q[5*leg+4]
        if leg == 0:
            side = 1
        else:
            side = -1
        pf_b = getFootPositionBody(q0, q1, q2, q3, q4, side)
        pf_b = pf_b.reshape(-1,1)
        hip_offset = jnp.array([[biped.hip_offset[0]],  [side* biped.hip_offset[1]], [biped.hip_offset[2]]])
        p_c = x_fb[3:6].reshape(-1,1)
        # pf_w[0+3*leg : 3+3*leg] = p_c + R@(pf_b+hip_offset)
        pf_w = pf_w.at[0+3*leg:3+3*leg].set(p_c + R@pf_b + hip_offset)
        # pf_w.at[0+3*leg].set(p_c[0] + R[0,0]*pf_b[0] + R[0,1]*pf_b[1] + R[0,2]*pf_b[2] + hip_offset[0])

    return pf_w

def swingLegControl(x_fb, t, pf_w, vf_w, side):
    global foot_r, foot_l
    y_offset = mpc.y_offset
    foot_des_x = (
        x_fb[3] + x_fb[9] * 1 / 2 * mpc.h / 2 * mpc.dt
        + mpc.kv * (x_fb[3] - mpc.x_cmd[3])
    )
    foot_des_y = (
        x_fb[4] + x_fb[10] * 1 / 2 * mpc.h / 2 * mpc.dt
        + mpc.kv * (x_fb[4] - mpc.x_cmd[4]) + y_offset*side
    )
    t = jnp.remainder(t, mpc.dt * mpc.h / 2)
    foot_des_z = mpc.swingHeight * jnp.sin(jnp.pi * t / (mpc.dt * mpc.h / 2))
    percent = t / (mpc.dt * mpc.h / 2 )
    # if verbose: print('percent', percent)
    # if t == 0:
    #     foot_l = jnp.zeros([3, 1])
    #     foot_r = jnp.zeros([3, 1])
    if percent == 0.0: 
        if side == 1: # initialize foot position
            foot_l = pf_w
        elif side == -1:
            foot_r = pf_w
    if side == 1:
        foot_i = foot_l
    elif side == -1:
        foot_i = foot_r
    # if verbose: print('foot_i',foot_i)
    foot_des_x = foot_i[0,0] + percent*(foot_des_x - foot_i[0,0])
    foot_des_y = foot_i[1,0] + percent*(foot_des_y - foot_i[1,0])
    # if verbose: print('foot_i', foot_i)
    foot_des = jnp.array([[foot_des_x],[foot_des_y],[foot_des_z]])
    foot_v_des = jnp.zeros((3,1))
    F_swing = mpc.kp@(foot_des - pf_w) + mpc.kd@(foot_v_des - vf_w)
    return F_swing

def lowLevelControl(x_fb, t, pf_w, q, qd, contact, u):
    tau = jnp.zeros((10,1))
    contact = contact[0, 0:2]
    # print('contact', contact)
    R = eul2rotm(x_fb[0:3])
    for leg in range(2):
        q0 = q[5*leg+0]
        q1 = q[5*leg+1]
        q2 = q[5*leg+2]
        q3 = q[5*leg+3]
        q4 = q[5*leg+4]
        if leg == 0:
            side = 1
        else:
            side = -1
        # get Jacobians
        Jm, Jf = getLegKinematics(q0, q1, q2, q3, q4, side)
        # foot velocity in world
        vf_w = R@Jf@qd[5*leg:5*leg+5].reshape(-1,1)
        # swing let force
        F_swing = swingLegControl(x_fb, t, pf_w[3*leg:3*leg+3], vf_w, side)
        # stance mapping
        u_w = -jnp.vstack([ R.T @ u[3*leg:3*leg+3],  R.T @ u[3*leg+6:3*leg+9] ])
        # tau[5*leg:5*leg+5,:] = Jm.T @ u_w * contact[leg] 
        rhs = Jm.T @ u_w * contact[leg]
        # print("rhs:", rhs)
        # print("tau bf:", tau)
        tau = tau.at[5*leg:5*leg+5,:].set(Jm.T @ u_w * contact[leg])
        # print("tau af1:", tau)
        # swing mapping
        # tau[5*leg:5*leg+5,:] += Jf.T @ R.T @ F_swing * -(contact[leg]-1)
        # print(Jf.T @ R.T @ F_swing * -(contact[leg]-1))
        tau = tau.at[5*leg:5*leg+5,:].set(tau[5*leg:5*leg+5,:] + Jf.T @ R.T @ F_swing * -(contact[leg]-1))
        # print("tau af2:", tau)
        # tau[5*leg,:] = 30*(0 - q0) + 1*(0 - qd[5*leg])
        # print(30*(0 - q0) + 1*(0 - qd[5*leg]))
        tau = tau.at[5*leg,:].set(30*(0 - q0) + 1*(0 - qd[5*leg]))
        # print('tau', tau)
    return tau

def solve_mpc(
                # parameters
                Q_cost,
                R_cost,  
                # feedback
                x_fb, 
                foot,
                t, 
                # commands
                contact,
                cmd,
            ):
    
    st = time.time()
    # sync paramters and command 
    mpc.Q = Q_cost
    mpc.R = R_cost
    mpc.x_cmd = cmd

    # get reference
    x_ref = get_reference_trajectory(x_fb) # numpy static, no need for dynamic jax arrays

    # get foot reference
    foot_ref = get_reference_foot_trajectory(x_fb, t, foot, contact) # numpy static, no need for dynamic jax arrays
    print('reference time:', time.time()-st)
    
    # jax dynamic arrays coz participates in optimization
    st = time.time()
    R = eul2rotm(x_fb[0:3])

    # load state matrices for each horizon:
    A_matrices = []
    B_matrices = []
    for k in range(mpc.h):
        A, B = get_simplified_dynamics(mpc, biped, x_ref[:, k], foot_ref[:, k])
        A_matrices.append(A)
        B_matrices.append(B)

    # construct dynamics constraints:
    Aeq_dyn = jnp.zeros((13*mpc.h, 25*mpc.h))
    Beq_dyn = []
    one = jnp.array([1])
    x_0 = jnp.concatenate((x_fb, one), axis=0)
    Beq_0 = jnp.dot(A_matrices[0], x_0)
    Beq_dyn.append(Beq_0)
    for i in range(mpc.h):
        # Aeq_dyn[13*i:13*(i+1),13*i:13*(i+1)] = jnp.eye(13)
        # Aeq_dyn[13*i:13*(i+1),13*mpc.h+12*i:13*mpc.h+12*(i+1)] = -B_matrices[i]
        Aeq_dyn = Aeq_dyn.at[13*i:13*(i+1), 13*i:13*(i+1)].set(jnp.eye(13))
        Aeq_dyn = Aeq_dyn.at[13*i:13*(i+1), 13*mpc.h+12*i:13*mpc.h+12*(i+1)].set(-B_matrices[i])
        if i > 0:
            # Aeq_dyn[13*i:13*(i+1),13*(i-1):13*(i)] = -A_matrices[i]
            Aeq_dyn = Aeq_dyn.at[13*i:13*(i+1), 13*(i-1):13*(i)].set(-A_matrices[i])
            Beq_dyn.append(jnp.zeros(13))

    # zero Mx
    Moment_selection = jnp.array([1, 0, 0])  # Define Moment_selection
    R_foot_R = R  # Replace with actual rotation matrix
    R_foot_L = R  # Replace with actual rotation matrix

    A_M_1 = jnp.block([
        [jnp.zeros((1, 3)), jnp.zeros((1, 3)), Moment_selection @ R_foot_R.T, jnp.zeros((1, 3))],
        [jnp.zeros((1, 3)), jnp.zeros((1, 3)), jnp.zeros((1, 3)), Moment_selection @ R_foot_L.T]
    ])
    A_M_h = jnp.kron(jnp.eye(mpc.h), A_M_1)
    padding = jnp.zeros((2 * mpc.h, 13 * mpc.h))
    A_M = jnp.hstack([padding, A_M_h])
    b_M = jnp.zeros(2 * mpc.h)
    Aeq = jnp.vstack([Aeq_dyn, A_M])
    beq = jnp.hstack([jnp.hstack(Beq_dyn), b_M])

    # construct inequality constraints:
    # Friction pyramid constraints
    A_mu1 = jnp.array([
        [1, 0, -biped.mu, *[0] * 9],
        [0, 1, -biped.mu, *[0] * 9],
        [-1, 0, -biped.mu, *[0] * 9],
        [0, -1, -biped.mu, *[0] * 9],
        [*[0] * 3, 1, 0, -biped.mu, *[0] * 6],
        [*[0] * 3, 0, 1, -biped.mu, *[0] * 6],
        [*[0] * 3, -1, 0, -biped.mu, *[0] * 6],
        [*[0] * 3, 0, -1, -biped.mu, *[0] * 6],
    ])
    A_mu = jnp.kron(jnp.eye(mpc.h), A_mu1)
    A_mu = jnp.hstack([jnp.zeros((A_mu.shape[0], 13*mpc.h)),A_mu])
    b_mu = jnp.zeros((8*mpc.h, 1))

    # force saturations
    A_f1 = jnp.vstack([jnp.eye(12), -jnp.eye(12)])
    A_f = jnp.kron(jnp.eye(mpc.h), A_f1)
    A_f = jnp.hstack([jnp.zeros((A_f.shape[0], 13*mpc.h)),A_f])
    b_f = []
    for k in range(mpc.h):
        col_k = jnp.concatenate([
            contact[k, 0] * biped.f_max,
            contact[k, 1] * biped.f_max,
            contact[k, 0] * biped.tau_max,
            contact[k, 1] * biped.tau_max,
            contact[k, 0] * -biped.f_min,
            contact[k, 1] * -biped.f_min,
            contact[k, 0] * -biped.tau_min,
            contact[k, 1] * -biped.tau_min
        ])
        b_f.append(col_k)
    b_f = jnp.vstack(b_f)

    # Line-foot constraints (preventing toe/heel lift)
    lt = biped.lt - 0.01
    lh = biped.lh - 0.02

    # Construct A_LF1
    A_LF1 = jnp.vstack([
        jnp.hstack([-lh * jnp.array([0, 0, 1]) @ R.T, jnp.zeros(3), jnp.array([0, 1, 0]) @ R.T, jnp.zeros(3)]),
        jnp.hstack([-lt * jnp.array([0, 0, 1]) @ R.T, jnp.zeros(3), -jnp.array([0, 1, 0]) @ R.T, jnp.zeros(3)]),
        jnp.hstack([jnp.zeros(3), -lh * jnp.array([0, 0, 1]) @ R.T, jnp.zeros(3), jnp.array([0, 1, 0]) @ R.T]),
        jnp.hstack([jnp.zeros(3), -lt * jnp.array([0, 0, 1]) @ R.T, jnp.zeros(3), -jnp.array([0, 1, 0]) @ R.T]),
    ])

    # Horizon block expansion
    A_LFh = jnp.kron(jnp.eye(mpc.h), A_LF1)
    padding = jnp.zeros((4 * mpc.h, 13 * mpc.h))
    A_LF = jnp.hstack([padding, A_LFh])

    # Define b_LF
    b_LF = jnp.zeros((4 * mpc.h, 1))

    Aqp = jnp.vstack([A_mu, A_f, A_LF])
    bqp = jnp.vstack([b_mu, b_f, b_LF])


    # Objective function 
    H = 2*jnp.block([
        [jnp.kron(jnp.eye(mpc.h), jnp.diag(mpc.Q)), jnp.zeros((13 * mpc.h, 12 * mpc.h))],
        [jnp.zeros((12 * mpc.h, 13 * mpc.h)), jnp.kron(jnp.eye(mpc.h), jnp.diag(mpc.R))]
    ])
    x_ref_flat = x_ref.T.flatten()
    f = 2*jnp.hstack([
        -jnp.kron(jnp.eye(mpc.h), jnp.diag(mpc.Q)) @ x_ref_flat,
        jnp.zeros(12 * mpc.h)
    ])

    print('setup time:', time.time()-st)
    
    st = time.time()
    # Solve MPC
    solution = solver.run(
                            init_params=None,
                            params_obj=(H, f), 
                            params_eq=(Aeq, beq), 
                            params_ineq=(Aqp, bqp),
                            # solver_opts=solver_opts
                            ).params

    x_opt = solution.primal
    states = x_opt[:13 * mpc.h].reshape((mpc.h,13))
    controls = x_opt[13 * mpc.h:].reshape((mpc.h,12))
    print('solve time:', time.time()-st)

    return states, controls, x_ref