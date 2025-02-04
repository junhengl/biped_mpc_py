import numpy as np
import time
import cvxopt
np.set_printoptions(suppress=True, precision=2)

# Junheng initial update 01/06/2025

## definitions:
# States (13,): euler angles, positions, angular velocity(world frame), linear velocity(world frame), 1
# control input (12,): [force and moment] = [f1; f2; m1; m2] 

# Initialize state feedback and parameters
x_fb = np.array([0, 0, 0, 0, 0, 0.53, 0, 0, 0, 0, 0, 0])  # States: euler angles, positions, angular velocity, linear velocity
foot = np.array([0,-0.1,0, 0,0.1,0])
q = np.array([0,0,-np.pi/4,np.pi/2,-np.pi/4, 0,0,-np.pi/4,np.pi/2,-np.pi/4])
qd = np.zeros((10))
t = 0
gait = 1 # standing = 0; walking = 1;

################## functions #####################

class MPC:
    def __init__(self):
        self.h = 10
        self.dt = 0.04
        self.x_cmd = np.array([0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 0])  # Command
        self.Q = np.array([500, 100, 100, 300, 300, 700, 1, 1, 1, 1, 1, 1, 1])  # State weights
        self.R = np.array([1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1]) * 1e-4  # Control input weights
        self.kv = 0.01
        self.kp = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*500
        self.kd = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*10
        self.swingHeight = 0.1

class Biped:
    def __init__(self):
        self.m = 12  # Mass
        self.I = np.array([[0.932, 0, 0],
                          [0, 0.9420, 0],
                          [0, 0, 0.0711]])  # Inertia
        self.lt = 0.09  # toe length
        self.lh = 0.05  # heel length
        self.g = 9.81  # Gravity
        self.hip_offset = np.array([-0.005, 0.047, -0.126])
        self.mu = 0.5
        self.f_max = np.array([[500], [500], [500]])
        self.f_min = np.array([[0], [0], [0]])
        self.tau_max =  np.array([[0], [67], [33.5]])
        self.tau_min = -self.tau_max

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

def get_reference_trajectory(x_fb, mpc):
    x_ref = np.tile(np.append(mpc.x_cmd, 1), (mpc.h, 1)).T
    x_ref[:12, 0] = x_fb
    for i in range(6):
        for k in range(1, mpc.h):
            if mpc.x_cmd[i + 6] != 0:
                x_ref[i, k] = x_fb[i] + mpc.x_cmd[i + 6] * (k * mpc.dt)
            else:
                x_ref[i, k] = mpc.x_cmd[i]
    return x_ref

def get_reference_foot_trajectory(x_fb, t, foot, mpc, contact):
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
        x_fb[10] + x_fb[10] * 1 / 2 * mpc.h * mpc.dt
        + mpc.kv * (x_fb[4] - mpc.x_cmd[4])
    )

    foot_des_z = 0
    foot_1 = np.array([foot_des_x_1, foot_des_y_1, foot_des_z, foot_des_x_1, foot_des_y_1, foot_des_z])
    foot_2 = np.array([foot_des_x_2, foot_des_y_2, foot_des_z, foot_des_x_2, foot_des_y_2, foot_des_z])

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
    return foot_ref

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
    cr, cp, cy = np.cos(eul)
    sr, sp, sy = np.sin(eul)

    # Z-Y-X rotation (roll around X, pitch around Y, yaw around Z)
    Rz = np.array([[ cy, -sy,  0 ],
                   [ sy,  cy,  0 ],
                   [  0,   0,  1 ]])
    Ry = np.array([[ cp,  0,  sp ],
                   [  0,  1,   0 ],
                   [-sp,  0,  cp ]])
    Rx = np.array([[ 1,  0,   0 ],
                   [ 0, cr, -sr ],
                   [ 0, sr,  cr ]])
    # Combined: Rz * Ry * Rx
    return Rz @ Ry @ Rx

def skew(v):
    """Skew-symmetric matrix of a 3D vector v."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],    0]
    ])

def get_simplified_dynamics(mpc, biped, x_ref, foot_ref):
    # Iterate through each step
    # Extract values from x_traj
    roll = x_ref[2]
    pitch = x_ref[1]
    yaw = x_ref[0]
    from scipy.spatial.transform import Rotation as R
    euler_angles = [yaw, pitch, roll]  # [yaw, pitch, roll]
    Rot = R.from_euler('zyx', euler_angles).as_matrix()  # Rotation matrix
    I = Rot.T @ biped.I @ Rot

    # Compute Ac matrix
    R_inv = np.linalg.inv(np.array([
        [np.cos(yaw) * np.cos(pitch), -np.sin(yaw), 0],
        [np.sin(yaw) * np.cos(pitch), np.cos(yaw), 0],
        [-np.sin(pitch), 0, 1]
    ]))
    Ac = np.block([
        [np.zeros((3, 3)), np.zeros((3, 3)), R_inv @ np.eye(3), np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.array([[0], [0], [-biped.g]])],
        [np.zeros((1, 13))]
    ])

    # Compute Bc matrix
    skew_1 = skew(-x_ref[3:6] + foot_ref[0:3])
    skew_2 = skew(-x_ref[3:6] + foot_ref[3:6])
    Bc = np.block([
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        [np.linalg.solve(I, skew_1), np.linalg.solve(I, skew_2), np.linalg.solve(I, np.eye(3)), np.linalg.solve(I, np.eye(3))],
        [np.eye(3) / biped.m, np.eye(3) / biped.m, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((1, 12))]
    ])
    A = Ac * mpc.dt + np.eye(13)
    B = Bc * mpc. dt
    return A, B

def solve_mpc(x_fb, t, foot, mpc, biped, contact):
    x_ref = get_reference_trajectory(x_fb, mpc)
    foot_ref = get_reference_foot_trajectory(x_fb, t, foot, mpc, contact)
    print("state reference: \n", x_ref)
    print("contact sequence: \n", contact)
    print("foot reference: \n", foot_ref)
    R = eul2rotm(x_fb[0:3])
    # load state matrices for each horizon:
    A_matrices = []
    B_matrices = []
    for k in range(mpc.h):
        A, B = get_simplified_dynamics(mpc, biped, x_ref[:, k], foot_ref[:, k])
        A_matrices.append(A)
        B_matrices.append(B)

    # construct dynamics constraints:
    Aeq_dyn = np.zeros((13*mpc.h, 25*mpc.h))
    Beq_dyn = []
    one = np.array([1])
    x_0 = np.concatenate((x_fb, one), axis=0)
    Beq_0 = np.dot(A_matrices[0], x_0)
    Beq_dyn.append(Beq_0)
    for i in range(mpc.h):
        Aeq_dyn[13*i:13*(i+1),13*i:13*(i+1)] = np.eye(13)
        Aeq_dyn[13*i:13*(i+1),13*mpc.h+12*i:13*mpc.h+12*(i+1)] = -B_matrices[i]
        if i > 0:
            Aeq_dyn[13*i:13*(i+1),13*(i-1):13*(i)] = -A_matrices[i]
            Beq_dyn.append(np.zeros(13))
    Aeq = Aeq_dyn
    beq = np.hstack(Beq_dyn)

    # construct inequality constraints:
    # Friction pyramid constraints
    A_mu1 = np.array([
        [1, 0, -biped.mu, *[0] * 9],
        [0, 1, -biped.mu, *[0] * 9],
        [-1, 0, -biped.mu, *[0] * 9],
        [0, -1, -biped.mu, *[0] * 9],
        [*[0] * 3, 1, 0, -biped.mu, *[0] * 6],
        [*[0] * 3, 0, 1, -biped.mu, *[0] * 6],
        [*[0] * 3, -1, 0, -biped.mu, *[0] * 6],
        [*[0] * 3, 0, -1, -biped.mu, *[0] * 6],
    ])
    A_mu = np.kron(np.eye(mpc.h), A_mu1)
    A_mu = np.hstack([np.zeros((A_mu.shape[0], 13*mpc.h)),A_mu])
    b_mu = np.zeros((8*mpc.h, 1))

    # force saturations
    A_f1 = np.vstack([np.eye(12), -np.eye(12)])
    A_f = np.kron(np.eye(mpc.h), A_f1)
    A_f = np.hstack([np.zeros((A_f.shape[0], 13*mpc.h)),A_f])
    b_f = []
    for k in range(mpc.h):
        col_k = np.concatenate([
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
    b_f = np.vstack(b_f)

    # Line-foot constraints (preventing toe/heel lift)
    lt = biped.lt - 0.01
    lh = biped.lh - 0.02
   
    # Construct A_LF1
    A_LF1 = np.vstack([
        np.hstack([-lh * np.array([0, 0, 1]) @ R.T, np.zeros(3), np.array([0, 1, 0]) @ R.T, np.zeros(3)]),
        np.hstack([-lt * np.array([0, 0, 1]) @ R.T, np.zeros(3), -np.array([0, 1, 0]) @ R.T, np.zeros(3)]),
        np.hstack([np.zeros(3), -lh * np.array([0, 0, 1]) @ R.T, np.zeros(3), np.array([0, 1, 0]) @ R.T]),
        np.hstack([np.zeros(3), -lt * np.array([0, 0, 1]) @ R.T, np.zeros(3), -np.array([0, 1, 0]) @ R.T]),
    ])

    # Horizon block expansion
    A_LFh = np.kron(np.eye(mpc.h), A_LF1)
    padding = np.zeros((4 * mpc.h, 13 * mpc.h))
    A_LF = np.hstack([padding, A_LFh])
    
    # Define b_LF
    b_LF = np.zeros((4 * mpc.h, 1))

    Aqp = np.vstack([A_mu, A_f, A_LF])
    bqp = np.vstack([b_mu, b_f, b_LF])


    # Objective function 
    H = 2*np.block([
        [np.kron(np.eye(mpc.h), np.diag(mpc.Q)), np.zeros((13 * mpc.h, 12 * mpc.h))],
        [np.zeros((12 * mpc.h, 13 * mpc.h)), np.kron(np.eye(mpc.h), np.diag(mpc.R))]
    ])
    x_ref_flat = x_ref.T.flatten()
    f = 2*np.hstack([
        -np.kron(np.eye(mpc.h), np.diag(mpc.Q)) @ x_ref_flat,
        np.zeros(12 * mpc.h)
    ])

    # Convert to cvxopt format
    H_cvx = cvxopt.matrix(H)
    f_cvx = cvxopt.matrix(f)
    Aeq_cvx = cvxopt.matrix(Aeq)
    beq_cvx = cvxopt.matrix(beq)
    Aqp_cvx = cvxopt.matrix(Aqp)
    bqp_cvx = cvxopt.matrix(bqp)

    # Solve QP using cvxopt
    solution = cvxopt.solvers.qp(H_cvx, f_cvx, G=Aqp_cvx, h=bqp_cvx, A=Aeq_cvx, b=beq_cvx)

    # Extract states and controls from the solution
    x_opt = np.array(solution['x']).flatten()
    states = x_opt[:13 * mpc.h].reshape((mpc.h,13))
    controls = x_opt[13 * mpc.h:].reshape((mpc.h,12))

    return states, controls

def getLegKinematics(q0, q1, q2, q3, q4, side):
    # Initialize the Jm matrix
    Jm = np.zeros((6, 5))
    sin = np.sin
    cos = np.cos

    # Fill in the matrix entries
    Jm[0, 0] = sin(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135) + \
                cos(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))

    Jm[1, 0] = sin(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))) - \
                cos(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135)

    Jm[2, 0] = 0.0
    Jm[3, 0] = 0.0
    Jm[4, 0] = 0.0
    Jm[5, 0] = 1.0

    Jm[0, 1] = -sin(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))
    Jm[1, 1] = cos(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))
    Jm[2, 1] = sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - cos(q1) * (0.018 * side + 0.0025)
    Jm[3, 1] = cos(q0)
    Jm[4, 1] = sin(q0)
    Jm[5, 1] = 0.0

    Jm[0, 2] = sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2)) - \
                cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))

    Jm[1, 2] = -sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - \
                cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2))

    Jm[2, 2] = cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2))
    Jm[3, 2] = -cos(q1) * sin(q0)
    Jm[4, 2] = cos(q0) * cos(q1)
    Jm[5, 2] = sin(q1)

    Jm[0, 3] = sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3)) - \
                cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3))

    Jm[1, 3] = -sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3)) - \
                cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3))

    Jm[2, 3] = cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3))
    Jm[3, 3] = -cos(q1) * sin(q0)
    Jm[4, 3] = cos(q0) * cos(q1)
    Jm[5, 3] = sin(q1)

    Jm[0, 4] = 0.04 * sin(q2 + q3 + q4) * sin(q0) * sin(q1) - \
                0.04 * cos(q2 + q3 + q4) * cos(q0)

    Jm[1, 4] = -0.04 * cos(q2 + q3 + q4) * sin(q0) - \
                0.04 * sin(q2 + q3 + q4) * cos(q0) * sin(q1)

    Jm[2, 4] = 0.04 * sin(q2 + q3 + q4) * cos(q1)
    Jm[3, 4] = -cos(q1) * sin(q0)
    Jm[4, 4] = cos(q0) * cos(q1)
    Jm[5, 4] = sin(q1)

    Jf = Jm[0:3, :]
    return Jm, Jf

def getFootPositionBody(q0, q1, q2, q3, q4, side):
    # Initialize the pf vector
    pf = np.zeros(3)

    # Fill in the vector entries
    pf[0] = - (3 * np.cos(q0)) / 200 - \
        (9 * np.sin(q4) * (np.cos(q3) * (np.cos(q0) * np.cos(q2) - np.sin(q0) * np.sin(q1) * np.sin(q2)) - 
                            np.sin(q3) * (np.cos(q0) * np.sin(q2) + np.cos(q2) * np.sin(q0) * np.sin(q1)))) / 250 - \
        (11 * np.cos(q0) * np.sin(q2)) / 50 - \
        ((side) * np.sin(q0)) / 50 - \
        (11 * np.cos(q3) * (np.cos(q0) * np.sin(q2) + np.cos(q2) * np.sin(q0) * np.sin(q1))) / 50 - \
        (11 * np.sin(q3) * (np.cos(q0) * np.cos(q2) - np.sin(q0) * np.sin(q1) * np.sin(q2))) / 50 - \
        (9 * np.cos(q4) * (np.cos(q3) * (np.cos(q0) * np.sin(q2) + np.cos(q2) * np.sin(q0) * np.sin(q1)) + 
                            np.sin(q3) * (np.cos(q0) * np.cos(q2) - np.sin(q0) * np.sin(q1) * np.sin(q2)))) / 250 - \
        (23 * np.cos(q1) * (side) * np.sin(q0)) / 1000 - \
        (11 * np.cos(q2) * np.sin(q0) * np.sin(q1)) / 50

    pf[1] = (np.cos(q0) * (side)) / 50 - \
        (9 * np.sin(q4) * (np.cos(q3) * (np.cos(q2) * np.sin(q0) + np.cos(q0) * np.sin(q1) * np.sin(q2)) - 
                            np.sin(q3) * (np.sin(q0) * np.sin(q2) - np.cos(q0) * np.cos(q2) * np.sin(q1)))) / 250 - \
        (3 * np.sin(q0)) / 200 - \
        (11 * np.sin(q0) * np.sin(q2)) / 50 - \
        (11 * np.cos(q3) * (np.sin(q0) * np.sin(q2) - np.cos(q0) * np.cos(q2) * np.sin(q1))) / 50 - \
        (11 * np.sin(q3) * (np.cos(q2) * np.sin(q0) + np.cos(q0) * np.sin(q1) * np.sin(q2))) / 50 - \
        (9 * np.cos(q4) * (np.cos(q3) * (np.sin(q0) * np.sin(q2) - np.cos(q0) * np.cos(q2) * np.sin(q1)) + 
                            np.sin(q3) * (np.cos(q2) * np.sin(q0) + np.cos(q0) * np.sin(q1) * np.sin(q2)))) / 250 + \
        (23 * np.cos(q0) * np.cos(q1) * (side)) / 1000 + \
        (11 * np.cos(q0) * np.cos(q2) * np.sin(q1)) / 50

    pf[2] = (23 * (side) * np.sin(q1)) / 1000 - \
        (11 * np.cos(q1) * np.cos(q2)) / 50 - \
        (9 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3))) / 250 + \
        (9 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2))) / 250 - \
        (11 * np.cos(q1) * np.cos(q2) * np.cos(q3)) / 50 + \
        (11 * np.cos(q1) * np.sin(q2) * np.sin(q3)) / 50 - \
        3.0 / 50.0    

    return pf

def getFootPositionWorld(x_fb, q, biped):
    R = eul2rotm(x_fb[0:3])
    pf_w = np.zeros((6,1))
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
        hip_offset = np.array([[biped.hip_offset[0]],  [side* biped.hip_offset[1]], [biped.hip_offset[2]]])
        p_c = x_fb[3:6].reshape(-1,1)
        pf_w[0+3*leg : 3+3*leg] = p_c + R.T@(pf_b+hip_offset)
    return pf_w

def swingLegControl(x_fb, t, pf_w, vf_w, mpc, side):
    y_offset = 0.04
    foot_des_x = (
        x_fb[3] + x_fb[9] * 1 / 2 * mpc.h / 2 * mpc.dt
        + mpc.kv * (x_fb[3] - mpc.x_cmd[3])
    )
    foot_des_y = (
        x_fb[4] + x_fb[10] * 1 / 2 * mpc.h / 2 * mpc.dt
        + mpc.kv * (x_fb[4] - mpc.x_cmd[4]) + y_offset*side
    )
    t = np.remainder(t, mpc.dt * mpc.h / 2)
    foot_des_z = mpc.swingHeight * np.sin(np.pi * t / (mpc.dt * mpc.h / 2))

    foot_des = np.array([[foot_des_x],[foot_des_y],[foot_des_z]])
    foot_v_des = np.zeros((3,1))
    F_swing = mpc.kp@(foot_des - pf_w) + mpc.kd@(foot_v_des - vf_w)
    return F_swing

def lowLevelControl(x_fb, t, pf_w, q, qd, mpc, biped, contact, u):
    tau = np.zeros((10,1))
    contact = contact[0, 0:2]
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
        vf_w = R.T@Jf@qd[5*leg:5*leg+5].reshape(-1,1)
        # swing let force
        F_swing = swingLegControl(x_fb, t, pf_w[3*leg:3*leg+3], vf_w, mpc, side)
        # stance mapping
        u_w = -np.vstack([ R.T @ u[3*leg:3*leg+3],  R.T @ u[3*leg+6:3*leg+9] ])
        tau[5*leg:5*leg+5,:] = Jm.T @ u_w * contact[leg] 
        # swing mapping
        tau[5*leg:5*leg+5,:] += Jf.T @ F_swing * -(contact[leg]-1)

    return tau


############################## Main Script ###################################

mpc = MPC()
biped = Biped()
# forward kinematics
pf_w = getFootPositionWorld(x_fb, q, biped)
foot = pf_w.reshape(-1)
# contact sequence generation
if gait == 1:
    contact = get_contact_sequence(t, mpc)
elif gait == 0:
    contact = np.ones((mpc.h, 2))
# run MPC
start_time = time.time()
states, controls = solve_mpc(x_fb, t, foot, mpc, biped, contact)
end_time = time.time()
print(f"MPC Function execution time: {end_time - start_time} seconds")
print("States: \n", states)
print("Controls: \n", controls)
# low level force-to-torque
u0 = controls[0, :].reshape(-1,1)
tau = lowLevelControl(x_fb, t, pf_w, q, qd, mpc, biped, contact, u0)
print("Torques: \n", tau)