"""
Simple Example: Parallel MPC Solver Usage
Demonstrates basic setup and solving with the parallel MPC solver
"""

import numpy as np
import torch
import sys

sys.path.insert(0, '/home/junhengl/biped_mpc_py/src')

from mpc_parallel import ParallelMPC


def example_basic_usage():
    """
    Simple example: Create a parallel MPC solver and solve a small batch of problems
    """
    
    print("\n" + "="*70)
    print("SIMPLE PARALLEL MPC EXAMPLE")
    print("="*70)
    
    # Problem dimensions (from biped MPC)
    n_vars = 250  # 25 variables * 10 horizon steps
    n_eq = 150    # 13*10 dynamics + 2*10 moment constraints
    batch_size = 32
    
    print(f"\nProblem Configuration:")
    print(f"  Variables: {n_vars}")
    print(f"  Equality constraints: {n_eq}")
    print(f"  Batch size: {batch_size}")
    
    # Step 1: Create dummy QP problem matrices
    print(f"\n1. Creating problem matrices...")
    H = np.eye(n_vars) * 2.0  # Hessian (symmetric positive definite)
    f_template = np.ones(n_vars) * 0.1  # Linear term
    Aeq = np.random.randn(n_eq, n_vars) * 0.01  # Equality constraints
    beq_template = np.zeros(n_eq)  # Constraint RHS
    
    # Make Aeq well-posed
    for i in range(min(n_eq, n_vars)):
        Aeq[i, i] = 1.0
    
    print(f"  H shape: {H.shape}")
    print(f"  Aeq shape: {Aeq.shape}")
    
    # Step 2: Initialize parallel MPC solver
    print(f"\n2. Initializing parallel solver...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    solver = ParallelMPC(batch_size=batch_size, device=device)
    solver.setup_problem_matrices(H, f_template, Aeq, beq_template)
    
    # Step 3: Create a batch of problems
    print(f"\n3. Creating batch of {batch_size} problems...")
    
    # Vary f and beq slightly for each environment
    f_batch = np.tile(f_template, (batch_size, 1)) + np.random.randn(batch_size, n_vars) * 0.01
    beq_batch = np.tile(beq_template, (batch_size, 1)) + np.random.randn(batch_size, n_eq) * 0.001
    
    print(f"  f_batch shape: {f_batch.shape}")
    print(f"  beq_batch shape: {beq_batch.shape}")
    
    # Step 4: Solve batch
    print(f"\n4. Solving batch...")
    
    states, controls, solve_time = solver.solve_batch_optimized(f_batch, beq_batch)
    
    print(f"  Solve time: {solve_time*1000:.2f} ms")
    print(f"  Time per environment: {solve_time/batch_size*1000:.3f} ms")
    print(f"  Throughput: {batch_size/solve_time:.0f} env/s")
    
    # Step 5: Extract results
    print(f"\n5. Results:")
    print(f"  States shape: {states.shape}")
    print(f"  Controls shape: {controls.shape}")
    
    # States should be (batch_size, horizon, 13)
    # Controls should be (batch_size, horizon, 12)
    print(f"\n  First environment:")
    print(f"    State shape: {states[0].shape}")
    print(f"    Control shape: {controls[0].shape}")
    print(f"    Initial state: {states[0, 0]}")
    print(f"    Initial control: {controls[0, 0]}")
    
    # Step 6: Solve another batch
    print(f"\n6. Solving another batch (with different f and beq)...")
    
    f_batch_2 = np.tile(f_template, (batch_size, 1)) + np.random.randn(batch_size, n_vars) * 0.02
    beq_batch_2 = np.tile(beq_template, (batch_size, 1)) + np.random.randn(batch_size, n_eq) * 0.002
    
    states_2, controls_2, solve_time_2 = solver.solve_batch_optimized(f_batch_2, beq_batch_2)
    
    print(f"  Solve time: {solve_time_2*1000:.2f} ms")
    print(f"  Second batch first environment initial state: {states_2[0, 0]}")
    
    # Step 7: GPU Memory info
    if device == 'cuda':
        print(f"\n7. GPU Memory Usage:")
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved: {reserved:.1f} MB")
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70 + "\n")
    
    return solver, states, controls


def example_multiple_batches():
    """
    Example: Solve multiple batches of problems sequentially
    """
    
    print("\n" + "="*70)
    print("SOLVING MULTIPLE BATCHES SEQUENTIALLY")
    print("="*70)
    
    # Setup
    n_vars = 250
    n_eq = 150
    batch_size = 64
    num_batches = 5
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Total environments: {batch_size * num_batches}")
    
    # Create solver
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = ParallelMPC(batch_size=batch_size, device=device)
    
    # Setup problem
    H = np.eye(n_vars) * 2.0
    f_template = np.ones(n_vars) * 0.1
    Aeq = np.random.randn(n_eq, n_vars) * 0.01
    for i in range(min(n_eq, n_vars)):
        Aeq[i, i] = 1.0
    beq_template = np.zeros(n_eq)
    
    solver.setup_problem_matrices(H, f_template, Aeq, beq_template)
    
    # Solve multiple batches
    all_states = []
    all_controls = []
    total_time = 0
    
    print(f"\nSolving batches...")
    for batch_idx in range(num_batches):
        # Create batch
        f_batch = np.tile(f_template, (batch_size, 1)) + np.random.randn(batch_size, n_vars) * 0.01
        beq_batch = np.tile(beq_template, (batch_size, 1)) + np.random.randn(batch_size, n_eq) * 0.001
        
        # Solve
        states, controls, solve_time = solver.solve_batch_optimized(f_batch, beq_batch)
        
        all_states.append(states)
        all_controls.append(controls)
        total_time += solve_time
        
        print(f"  Batch {batch_idx + 1}/{num_batches}: {solve_time*1000:.2f} ms")
    
    # Aggregate results
    all_states_agg = np.vstack(all_states)
    all_controls_agg = np.vstack(all_controls)
    
    print(f"\nResults:")
    print(f"  Total states: {all_states_agg.shape}")
    print(f"  Total controls: {all_controls_agg.shape}")
    print(f"  Total time: {total_time*1000:.2f} ms")
    print(f"  Throughput: {batch_size * num_batches / total_time:.0f} env/s")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run examples
    print("\n\nRunning Parallel MPC Solver Examples...")
    print("="*70)
    
    # Example 1: Basic usage
    solver, states, controls = example_basic_usage()
    
    # Example 2: Multiple batches
    example_multiple_batches()
    
    print("All examples completed successfully!")
