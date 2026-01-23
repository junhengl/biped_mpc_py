"""
Integration Example: Using Parallel MPC with Real Biped Data
Demonstrates how to use the parallel solver with actual MPC matrices from the biped controller.
Uses the exact matrix generation procedure from mpc.py but parallelized for batch processing.
"""

import numpy as np
import sys
import time

sys.path.insert(0, '/home/junhengl/biped_mpc_py/src')

from mpc import MPC, Biped, get_reference_trajectory, get_reference_foot_trajectory, get_simplified_dynamics, eul2rotm
from mpc_parallel import ParallelMPC

try:
    from cvxopt import matrix, solvers
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False


def build_mpc_matrices_for_environment(mpc: MPC, biped: Biped, x_fb: np.ndarray, 
                                       t: float, foot: np.ndarray, contact: np.ndarray) -> tuple:
    """
    Build complete QP matrices for a single environment using actual MPC formulation.
    Mimics the solve_mpc() function from mpc.py but isolated for batch processing.
    
    Args:
        mpc: MPC parameters
        biped: Biped parameters
        x_fb: Current state (13,)
        t: Current time
        foot: Foot position (6,)
        contact: Contact sequence (h, 2)
        
    Returns:
        H: Hessian (250, 250)
        f: Linear term (250,)
        Aeq: Equality constraint matrix (n_eq, 250)
        beq: Equality constraint RHS (n_eq,)
    """
    # Get reference trajectory and foot trajectory
    x_ref = get_reference_trajectory(x_fb[:12], mpc)
    foot_ref = get_reference_foot_trajectory(x_fb[:12], t, foot, mpc, contact)
    
    # Get rotation matrix from state
    R = eul2rotm(x_fb[0:3])
    
    # Build dynamics matrices A and B for each horizon step
    A_matrices = []
    B_matrices = []
    for k in range(mpc.h):
        A, B = get_simplified_dynamics(mpc, biped, x_ref[:, k], foot_ref[:, k])
        A_matrices.append(A)
        B_matrices.append(B)
    
    # Reshape reference to vector form
    y = np.reshape(x_ref.T, (13 * mpc.h, 1))
    
    # Build dynamics constraints
    Aeq_dyn = np.zeros((13*mpc.h, 25*mpc.h))
    Beq_dyn = []
    
    # Initial state condition (use x_fb as is)
    x_0 = x_fb.reshape(-1, 1)  # (13, 1)
    Beq_0 = np.dot(A_matrices[0], x_0)
    Beq_dyn.append(Beq_0.flatten())
    
    for i in range(mpc.h):
        Aeq_dyn[13*i:13*(i+1), 13*i:13*(i+1)] = np.eye(13)
        Aeq_dyn[13*i:13*(i+1), 13*mpc.h+12*i:13*mpc.h+12*(i+1)] = -B_matrices[i]
        if i > 0:
            Aeq_dyn[13*i:13*(i+1), 13*(i-1):13*i] = -A_matrices[i]
            Beq_dyn.append(np.zeros(13))
    
    # Zero moment constraint (Mx = 0)
    Moment_selection = np.array([1, 0, 0])
    R_foot_R = R
    R_foot_L = R
    
    A_M_1 = np.block([
        [np.zeros((1, 3)), np.zeros((1, 3)), Moment_selection @ R_foot_R.T, np.zeros((1, 3))],
        [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), Moment_selection @ R_foot_L.T]
    ])
    A_M_h = np.kron(np.eye(mpc.h), A_M_1)
    padding = np.zeros((2 * mpc.h, 13 * mpc.h))
    A_M = np.hstack([padding, A_M_h])
    b_M = np.zeros(2 * mpc.h)
    
    # Combine all equality constraints
    Aeq = np.vstack([Aeq_dyn, A_M])
    beq = np.hstack([np.hstack(Beq_dyn), b_M.reshape(-1,)])
    
    # Objective function: minimize 0.5*x'Hx + f'x
    H = 2 * np.block([
        [np.kron(np.eye(mpc.h), np.diag(mpc.Q)), np.zeros((13 * mpc.h, 12 * mpc.h))],
        [np.zeros((12 * mpc.h, 13 * mpc.h)), np.kron(np.eye(mpc.h), np.diag(mpc.R))]
    ])
    
    x_ref_flat = x_ref.T.flatten()
    f = 2 * np.hstack([
        -np.kron(np.eye(mpc.h), np.diag(mpc.Q)) @ x_ref_flat,
        np.zeros(12 * mpc.h)
    ])
    
    return H, f, Aeq, beq


def build_batch_f_beq_from_environments(mpc: MPC, biped: Biped, batch_data: dict, Aeq: np.ndarray) -> tuple:
    """
    Build batch of f and beq vectors for different environments.
    H is fixed across environments; only f and beq vary.
    
    Args:
        mpc: MPC parameters
        biped: Biped parameters
        batch_data: dict with 'x_fb', 't', 'foot', 'contact' for each environment
        Aeq: Fixed equality constraint matrix
        
    Returns:
        f_batch: (batch_size, 250) linear terms
        beq_batch: (batch_size, n_eq) constraint RHS
    """
    batch_size = len(batch_data['x_fb_batch'])
    n_vars = 25 * mpc.h
    n_eq = Aeq.shape[0]
    
    f_batch = np.zeros((batch_size, n_vars))
    beq_batch = np.zeros((batch_size, n_eq))
    
    for i in range(batch_size):
        x_fb = batch_data['x_fb_batch'][i]
        t = batch_data['t_batch'][i]
        foot = batch_data['foot_batch'][i]
        contact = batch_data['contact_batch'][i]
        
        # Get reference trajectory
        x_ref = get_reference_trajectory(x_fb[:12], mpc)
        R = eul2rotm(x_fb[0:3])
        foot_ref = get_reference_foot_trajectory(x_fb[:12], t, foot, mpc, contact)
        
        # Build A matrices for this environment
        A_matrices = []
        B_matrices = []
        for k in range(mpc.h):
            A, B = get_simplified_dynamics(mpc, biped, x_ref[:, k], foot_ref[:, k])
            A_matrices.append(A)
            B_matrices.append(B)
        
        # Build f for this environment
        x_ref_flat = x_ref.T.flatten()
        f_batch[i] = 2 * np.hstack([
            -np.kron(np.eye(mpc.h), np.diag(mpc.Q)) @ x_ref_flat,
            np.zeros(12 * mpc.h)
        ])
        
        # Build beq for this environment
        Beq_dyn = []
        x_0 = x_fb.reshape(-1, 1)  # (13, 1)
        Beq_0 = np.dot(A_matrices[0], x_0)
        Beq_dyn.append(Beq_0.flatten())
        
        for j in range(mpc.h):
            if j > 0:
                Beq_dyn.append(np.zeros(13))
        
        # Zero moment constraints
        b_M = np.zeros(2 * mpc.h)
        
        beq_batch[i] = np.hstack([np.hstack(Beq_dyn), b_M.reshape(-1,)])
    
    return f_batch, beq_batch


def generate_dummy_environments(n_envs: int, seed: int = 42) -> dict:
    """
    Generate dummy initial conditions for multiple environments
    
    Args:
        n_envs: Number of environments
        seed: Random seed
        
    Returns:
        dict with batch data:
            - x_fb_batch: (n_envs, 13) states
            - t_batch: (n_envs,) times
            - foot_batch: (n_envs, 6) foot positions
            - contact_batch: (n_envs, h, 2) contact states
    """
    np.random.seed(seed)
    
    mpc = MPC()
    
    # Generate random states around standing position
    x_fb_batch = np.zeros((n_envs, 13))
    x_fb_batch[:, :3] = np.random.randn(n_envs, 3) * 0.1  # Small orientation perturbations
    x_fb_batch[:, 3:6] = np.random.randn(n_envs, 3) * 0.05  # Small position variations
    x_fb_batch[:, 5] = 0.55  # Keep height constant
    x_fb_batch[:, 6:] = np.random.randn(n_envs, 7) * 0.01  # Small velocity perturbations
    
    # Random times
    t_batch = np.random.rand(n_envs) * 1.0  # 0 to 1 second
    
    # Foot positions
    foot_batch = np.zeros((n_envs, 6))
    foot_batch[:, 1] = -0.1  # Left foot y offset
    foot_batch[:, 4] = 0.1   # Right foot y offset
    foot_batch += np.random.randn(n_envs, 6) * 0.02
    
    # Contact states (alternating gait pattern)
    contact_batch = np.ones((n_envs, mpc.h, 2))
    for i in range(n_envs):
        phase = int(t_batch[i] // mpc.dt) % (2 * mpc.h)
        if phase < mpc.h:
            contact_batch[i, :, 0] = 1  # Left foot
            contact_batch[i, :, 1] = 0  # Right foot swing
        else:
            contact_batch[i, :, 0] = 0  # Left foot swing
            contact_batch[i, :, 1] = 1  # Right foot
    
    return {
        'x_fb_batch': x_fb_batch,
        't_batch': t_batch,
        'foot_batch': foot_batch,
        'contact_batch': contact_batch,
    }


def example_real_mpc_matrices():
    """
    Use actual MPC matrix generation with parallel solving
    """
    
    print("\n" + "="*70)
    print("PARALLEL MPC WITH REAL MATRIX GENERATION")
    print("="*70)
    
    # Initialize MPC and Biped
    mpc = MPC()
    biped = Biped()
    
    print(f"\nMPC Configuration:")
    print(f"  Horizon: {mpc.h} steps")
    print(f"  dt: {mpc.dt} seconds")
    print(f"  Q shape: {np.diag(mpc.Q).shape}")
    print(f"  R shape: {np.diag(mpc.R).shape}")
    
    # Build matrices for first environment (reference environment)
    print(f"\nGenerating reference environment...")
    dummy_envs = generate_dummy_environments(1)
    x_fb_ref = dummy_envs['x_fb_batch'][0]
    t_ref = dummy_envs['t_batch'][0]
    foot_ref = dummy_envs['foot_batch'][0]
    contact_ref = dummy_envs['contact_batch'][0]
    
    print(f"  State: {x_fb_ref[3:6]} (position)")
    print(f"  Time: {t_ref:.3f} s")
    
    # Build matrices using actual MPC method
    print(f"\nBuilding QP matrices using actual MPC formulation...")
    start_build = time.time()
    
    H, f_ref, Aeq, beq_ref = build_mpc_matrices_for_environment(mpc, biped, x_fb_ref, t_ref, foot_ref, contact_ref)
    
    build_time = time.time() - start_build
    print(f"  H shape: {H.shape}")
    print(f"  Aeq shape: {Aeq.shape}")
    print(f"  Build time: {build_time:.4f} seconds")
    
    # Initialize parallel solver
    print(f"\nInitializing parallel solver...")
    batch_size = 64
    solver = ParallelMPC(batch_size=batch_size, device='cuda')
    solver.setup_problem_matrices(H, f_ref, Aeq, beq_ref)
    
    # Generate batch of environments
    print(f"\nGenerating batch of {batch_size} environments...")
    batch_data = generate_dummy_environments(batch_size, seed=123)
    
    # Build batch f and beq
    print(f"Building batch f and beq vectors...")
    start_batch_build = time.time()
    f_batch, beq_batch = build_batch_f_beq_from_environments(mpc, biped, batch_data, Aeq)
    batch_build_time = time.time() - start_batch_build
    
    print(f"  Batch build time: {batch_build_time:.4f} seconds")
    print(f"  f_batch range: [{f_batch.min():.2e}, {f_batch.max():.2e}]")
    print(f"  beq_batch range: [{beq_batch.min():.2e}, {beq_batch.max():.2e}]")
    
    # Solve batch
    print(f"\nSolving batch of {batch_size} environments...")
    
    start_solve = time.time()
    try:
        states, controls, solve_time = solver.solve_batch_optimized(f_batch, beq_batch)
        total_time = time.time() - start_solve
        
        print(f"  ✓ Batch solved successfully")
        print(f"  GPU compute time: {solve_time*1000:.2f} ms")
        print(f"  Total time (with overhead): {total_time*1000:.2f} ms")
        print(f"  Throughput: {batch_size/total_time:.0f} env/s")
        
    except Exception as e:
        print(f"  ✗ Error solving batch: {e}")
        return None
    
    # Analyze output
    print(f"\nOutput Analysis:")
    print(f"  States shape: {states.shape} (batch, horizon, state_dim)")
    print(f"  Controls shape: {controls.shape} (batch, horizon, control_dim)")
    
    # Sample trajectories
    print(f"\nSample Trajectory (first environment):")
    print(f"  Initial COM position: {states[0, 0, 3:6]}")
    print(f"  Final COM position: {states[0, -1, 3:6]}")
    print(f"  Initial control magnitude: {np.linalg.norm(controls[0, 0]):.2f}")
    print(f"  Final control magnitude: {np.linalg.norm(controls[0, -1]):.2f}")
    
    # Statistics
    print(f"\nTrajectory Statistics (all environments):")
    print(f"  Mean COM y displacement: {(states[:, -1, 4] - states[:, 0, 4]).mean():.4f}")
    print(f"  Std COM y displacement: {(states[:, -1, 4] - states[:, 0, 4]).std():.4f}")
    print(f"  Mean control magnitude: {np.linalg.norm(controls, axis=-1).mean():.2f}")
    print(f"  Max control magnitude: {np.linalg.norm(controls, axis=-1).max():.2f}")
    
    print("\n" + "="*70 + "\n")
    
    return solver, states, controls, mpc, biped


def example_comparison_real_vs_sequential():
    """
    Compare real matrix generation with sequential vs parallel solving
    """
    
    print("\n" + "="*70)
    print("REAL MPC: SEQUENTIAL vs PARALLEL COMPARISON")
    print("="*70)
    
    mpc = MPC()
    biped = Biped()
    
    # Generate reference environment
    dummy_envs = generate_dummy_environments(1)
    x_fb_ref = dummy_envs['x_fb_batch'][0]
    t_ref = dummy_envs['t_batch'][0]
    foot_ref = dummy_envs['foot_batch'][0]
    contact_ref = dummy_envs['contact_batch'][0]
    
    # Build reference matrices
    H, f_ref, Aeq, beq_ref = build_mpc_matrices_for_environment(mpc, biped, x_fb_ref, t_ref, foot_ref, contact_ref)
    
    # Test case: 100 environments
    n_test = 4096
    batch_data = generate_dummy_environments(n_test, seed=456)
    
    # Sequential solving (simulate)
    print(f"\n1. Sequential Solving ({n_test} environments):")
    
    start_seq = time.time()
    
    for i in range(n_test):
        x_fb = batch_data['x_fb_batch'][i]
        t = batch_data['t_batch'][i]
        foot = batch_data['foot_batch'][i]
        contact = batch_data['contact_batch'][i]
        
        # This simulates building matrices for each environment individually
        x_ref = get_reference_trajectory(x_fb[:12], mpc)
        foot_ref_i = get_reference_foot_trajectory(x_fb[:12], t, foot, mpc, contact)
        
        for k in range(mpc.h):
            A, B = get_simplified_dynamics(mpc, biped, x_ref[:, k], foot_ref_i[:, k])
    
    time_seq = time.time() - start_seq
    print(f"  Time for matrix prep: {time_seq*1000:.2f} ms")
    print(f"  Throughput: {n_test/time_seq:.0f} env/s")
    
    # Parallel solving
    print(f"\n2. Parallel Solving ({n_test} environments, batch=32):")
    
    solver = ParallelMPC(batch_size=32, device='cuda')
    solver.setup_problem_matrices(H, f_ref, Aeq, beq_ref)
    
    start_par = time.time()
    
    all_states = []
    for i in range(0, n_test, 32):
        batch_idx_end = min(i + 32, n_test)
        batch_size_actual = batch_idx_end - i
        
        batch_data_slice = {
            'x_fb_batch': batch_data['x_fb_batch'][i:batch_idx_end],
            't_batch': batch_data['t_batch'][i:batch_idx_end],
            'foot_batch': batch_data['foot_batch'][i:batch_idx_end],
            'contact_batch': batch_data['contact_batch'][i:batch_idx_end],
        }
        
        # Build f_batch and beq_batch
        f_batch, beq_batch = build_batch_f_beq_from_environments(mpc, biped, batch_data_slice, Aeq)
        
        # Pad to batch_size=32 if necessary
        if batch_size_actual < 32:
            n_pad = 32 - batch_size_actual
            # Repeat the last element to pad
            f_pad = np.tile(f_batch[-1:], (n_pad, 1))
            beq_pad = np.tile(beq_batch[-1:], (n_pad, 1))
            f_batch = np.vstack([f_batch, f_pad])
            beq_batch = np.vstack([beq_batch, beq_pad])
        
        assert f_batch.shape[0] == 32, f"f_batch has {f_batch.shape[0]} rows, expected 32"
        
        states, controls, _ = solver.solve_batch_optimized(f_batch, beq_batch)
        all_states.append(states[:batch_size_actual])
    
    time_par = time.time() - start_par
    print(f"  Total time: {time_par*1000:.2f} ms")
    print(f"  Throughput: {n_test/time_par:.0f} env/s")
    
    # Sequential solving (actual QP solve, not just matrix generation)
    if CVXOPT_AVAILABLE:
        print(f"\n3. Sequential Solving with Actual QP:")
        
        n_seq = min(100, n_test)
        n_seq = n_test
        
        # Build f and beq for sequential test
        batch_data_seq = {
            'x_fb_batch': batch_data['x_fb_batch'][:n_seq],
            't_batch': batch_data['t_batch'][:n_seq],
            'foot_batch': batch_data['foot_batch'][:n_seq],
            'contact_batch': batch_data['contact_batch'][:n_seq],
        }
        
        f_batch_seq, beq_batch_seq = build_batch_f_beq_from_environments(mpc, biped, batch_data_seq, Aeq)

        # Suppress cvxopt output
        solvers.options['show_progress'] = False
        
        start_seq_solve = time.time()
        
        H_cvx = matrix(H)
        Aeq_cvx = matrix(Aeq)
        
        for i in range(n_seq):
            f_cvx = matrix(f_batch_seq[i])
            beq_cvx = matrix(beq_batch_seq[i])
            
            try:
                sol = solvers.qp(2*H_cvx, f_cvx, A=Aeq_cvx, b=beq_cvx)
            except:
                pass  # Solver failed, continue
        
        time_seq_solve = time.time() - start_seq_solve
        print(f"  Time to solve {n_seq} environments: {time_seq_solve*1000:.2f} ms")
        print(f"  Throughput: {n_seq/time_seq_solve:.0f} env/s")
        print(f"  Per-environment time: {time_seq_solve/n_seq*1000:.2f} ms")
    else:
        print(f"\n3. Sequential Solving with Actual QP: cvxopt not available")
        print(f"  Install with: pip install cvxopt")
        time_seq_solve = None
    
    # Speedup comparison
    print(f"\n4. Performance Summary ({n_test} environments):")
    print(f"  Matrix generation only: {time_seq*1000:.2f} ms = {n_test/time_seq:.0f} env/s")
    print(f"  Parallel batch solving: {time_par*1000:.2f} ms = {n_test/time_par:.0f} env/s")
    if time_seq_solve is not None:
        print(f"  Sequential QP solving ({n_seq} envs): {time_seq_solve*1000:.2f} ms = {n_seq/time_seq_solve:.0f} env/s")
        print(f"\n  Parallel vs Sequential QP speedup: {(time_seq_solve/n_seq) / (time_par/n_test) * n_seq:.1f}x")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n\nRunning Parallel MPC Integration Examples with Real Matrices...")
    print("="*70)
    
    # Example 1: Real matrix generation with parallel solving
    result = example_real_mpc_matrices()
    
    # Example 2: Comparison
    example_comparison_real_vs_sequential()
    
    print("All integration examples completed!")
