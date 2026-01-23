"""
Example: Parallel MPC Solver with 4096 Environments
Demonstrates setting up and solving a large batch of biped MPC problems on GPU
"""

import numpy as np
import torch
import time
import sys

sys.path.insert(0, '/home/junhengl/biped_mpc_py/src')

from mpc import MPC, Biped, get_reference_trajectory, get_reference_foot_trajectory, get_simplified_dynamics, eul2rotm, skew
from mpc_parallel import ParallelMPC


def build_qp_matrices_dummy():
    """
    Build dummy QP matrices for parallel solver demonstration.
    Creates problem with correct dimensions but simplified structure.
    
    Returns:
        H: Hessian (250 x 250)
        f: Linear term (250,)
        Aeq: Equality constraint matrix
        beq: Equality constraint RHS
    """
    mpc = MPC()
    n_vars = 25 * mpc.h  # 250
    n_eq = 13 * mpc.h + 2 * mpc.h  # 150
    
    # Dummy H: symmetric positive definite
    H = np.eye(n_vars) * 2.0
    
    # Dummy f: linear terms
    f = np.ones(n_vars) * 0.1
    
    # Dummy Aeq: equality constraints (150 x 250)
    Aeq = np.random.randn(n_eq, n_vars) * 0.01
    
    # Make constraint matrix have full rank
    for i in range(min(n_eq, n_vars)):
        Aeq[i, i] = 1.0
    
    # Dummy beq: constraint RHS
    beq = np.zeros(n_eq)
    
    return H, f, Aeq, beq


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
    x_fb_batch[:, 3:6] = np.random.randn(n_envs, 3) * 0.05  # Small position variations around COM
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


def setup_parallel_mpc_4096():
    """
    Setup and benchmark parallel MPC with 4096 environments
    """
    
    print("\n" + "="*70)
    print("PARALLEL MPC SOLVER - 4096 ENVIRONMENTS BENCHMARK")
    print("="*70)
    
    # Device info
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {device_name}")
        print(f"Total memory: {total_memory:.1f} GB")
    else:
        device = 'cpu'
        print("\nWarning: CUDA not available, using CPU (will be slow)")
    
    # Parameters
    n_envs = 4096
    batch_size = 256  # Process 256 environments at a time
    n_batches = (n_envs + batch_size - 1) // batch_size
    
    print(f"\nConfiguration:")
    print(f"  Total environments: {n_envs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {n_batches}")
    print(f"  Device: {device}")
    
    # Initialize MPC
    mpc = MPC()
    biped = Biped()
    print(f"\nMPC parameters:")
    print(f"  Horizon: {mpc.h} steps")
    print(f"  dt: {mpc.dt} seconds")
    print(f"  Problem size: {25*mpc.h} variables")
    
    # Build problem matrices (dummy simplified version)
    print(f"\nBuilding QP matrices...")
    start_build = time.time()
    
    H, f_template, Aeq, beq_template = build_qp_matrices_dummy()
    
    build_time = time.time() - start_build
    print(f"  H shape: {H.shape}")
    print(f"  Aeq shape: {Aeq.shape}")
    print(f"  Build time: {build_time:.4f} seconds")
    
    # Initialize parallel solver
    print(f"\nInitializing parallel solver...")
    start_init = time.time()
    
    parallel_mpc = ParallelMPC(batch_size=batch_size, device=device)
    parallel_mpc.setup_problem_matrices(H, f_template, Aeq, beq_template)
    
    init_time = time.time() - start_init
    print(f"  Initialization time: {init_time:.4f} seconds")
    
    # Generate all dummy environments
    print(f"\nGenerating {n_envs} dummy environments...")
    start_gen = time.time()
    
    all_envs = generate_dummy_environments(n_envs)
    x_fb_all = all_envs['x_fb_batch']
    t_all = all_envs['t_batch']
    foot_all = all_envs['foot_batch']
    contact_all = all_envs['contact_batch']
    
    gen_time = time.time() - start_gen
    print(f"  Generation time: {gen_time:.4f} seconds")
    
    # Solve all batches
    print(f"\nSolving {n_envs} environments in {n_batches} batches of {batch_size}...")
    
    all_states = []
    all_controls = []
    
    start_solve = time.time()
    total_solve_time = 0
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_envs)
        actual_batch_size = end_idx - start_idx
        
        # Get batch data
        x_fb_batch = x_fb_all[start_idx:end_idx]
        t_batch = t_all[start_idx:end_idx]
        foot_batch = foot_all[start_idx:end_idx]
        contact_batch = contact_all[start_idx:end_idx]
        
        # Pad to batch_size if necessary
        if actual_batch_size < batch_size:
            pad_size = batch_size - actual_batch_size
            x_fb_batch = np.vstack([x_fb_batch, x_fb_batch[:pad_size]])
            t_batch = np.hstack([t_batch, t_batch[:pad_size]])
            foot_batch = np.vstack([foot_batch, foot_batch[:pad_size]])
            contact_batch = np.vstack([contact_batch, contact_batch[:pad_size]])
        
        # Build f and beq for batch
        f_batch = np.zeros((batch_size, parallel_mpc.n_vars))
        beq_batch = np.zeros((batch_size, parallel_mpc.n_eq))
        
        for i in range(batch_size):
            # Create dummy f and beq variations for each environment
            f_batch[i] = f_template + np.random.randn(parallel_mpc.n_vars) * 0.01
            beq_batch[i] = beq_template + np.random.randn(parallel_mpc.n_eq) * 0.001
        
        # Solve batch
        states, controls, solve_time = parallel_mpc.solve_batch_optimized(f_batch, beq_batch)
        total_solve_time += solve_time
        
        # Keep only actual batch size results
        all_states.append(states[:actual_batch_size])
        all_controls.append(controls[:actual_batch_size])
        
        # Progress
        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            elapsed = time.time() - start_solve
            envs_processed = min((batch_idx + 1) * batch_size, n_envs)
            throughput = envs_processed / elapsed
            print(f"    Batch {batch_idx + 1}/{n_batches}: {envs_processed}/{n_envs} environments "
                  f"({throughput:.0f} env/s)")
    
    total_time = time.time() - start_solve
    
    # Results
    print(f"\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    
    print(f"Total solving time (GPU compute): {total_solve_time:.4f} seconds")
    print(f"Total elapsed time (with overhead): {total_time:.4f} seconds")
    print(f"Average time per batch: {total_solve_time / n_batches:.6f} seconds")
    print(f"Average time per environment: {total_solve_time / n_envs * 1000:.3f} ms")
    print(f"Throughput: {n_envs / total_time:.0f} environments/second")
    
    # Memory usage
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    # Output shapes
    print(f"\nOutput shapes:")
    print(f"  States: {np.vstack(all_states).shape} (n_envs, horizon, 13)")
    print(f"  Controls: {np.vstack(all_controls).shape} (n_envs, horizon, 12)")
    
    # Sample output
    print(f"\nSample output (first environment):")
    sample_states = all_states[0][0]
    sample_controls = all_controls[0][0]
    print(f"  Initial state: {sample_states[0, 3:6]}")
    print(f"  Initial control: {sample_controls[0, 0:6]}")
    print(f"  Final state: {sample_states[-1, 3:6]}")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70 + "\n")
    
    return {
        'parallel_mpc': parallel_mpc,
        'mpc': mpc,
        'biped': biped,
        'all_states': all_states,
        'all_controls': all_controls,
        'timings': {
            'build': build_time,
            'init': init_time,
            'gen': gen_time,
            'solve': total_solve_time,
            'total': total_time,
        }
    }


if __name__ == "__main__":
    results = setup_parallel_mpc_4096()
    
    print("\nExample: Accessing results")
    print("-" * 70)
    print(f"Total environments solved: 4096")
    print(f"Total solving time: {results['timings']['solve']:.4f} seconds")
    print(f"Throughput: {4096 / results['timings']['total']:.0f} env/s")
    print(f"\nTo use the solver in your code:")
    print("  1. solver = results['parallel_mpc']")
    print("  2. states, controls, time = solver.solve_batch_optimized(f_batch, beq_batch)")
