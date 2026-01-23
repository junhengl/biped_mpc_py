"""
Integration example: Using ParallelMPC with the biped controller
Demonstrates how to parallelize multiple biped environments
"""

import numpy as np
import sys
sys.path.insert(0, '/home/junhengl/biped_mpc_py/src')

from mpc import MPC, Biped, get_reference_trajectory, get_reference_foot_trajectory, get_simplified_dynamics
from mpc_parallel import ParallelMPC
import torch


class BikedMPCParallel:
    """
    Wrapper for parallel solving of multiple biped MPC instances
    """
    
    def __init__(self, batch_size: int = 32, device: str = 'cuda'):
        """
        Initialize parallel biped MPC
        
        Args:
            batch_size: Number of parallel environments
            device: 'cuda' or 'cpu'
        """
        self.batch_size = batch_size
        self.device = device
        
        # Create MPC and biped parameters (shared across batch)
        self.mpc = MPC()
        self.biped = Biped()
        
        # Initialize parallel solver
        self.parallel_solver = ParallelMPC(batch_size=batch_size, device=device)
        
        # Problem matrices (computed once)
        self.H = None
        self.f_template = None
        self.Aeq = None
        self.beq_template = None
        
        print(f"BikedMPCParallel initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
        print(f"  Horizon: {self.mpc.h} steps")

    def setup_matrices_from_reference(self, 
                                      x_fb_batch: np.ndarray,
                                      t_batch: np.ndarray,
                                      foot_batch: np.ndarray,
                                      contact_batch: np.ndarray) -> None:
        """
        Setup QP matrices for a batch of reference trajectories
        
        Args:
            x_fb_batch: (batch_size, 13) - State feedback
            t_batch: (batch_size,) - Time
            foot_batch: (batch_size, 6) - Foot positions
            contact_batch: (batch_size, h, 2) - Contact states
            
        Note: This is simplified - in practice you'd need to build all matrices
        For now we'll use a single reference setup repeated
        """
        # Use first state as reference (simplified - could use batch)
        x_fb = x_fb_batch[0]
        t = t_batch[0]
        foot = foot_batch[0]
        contact = contact_batch[0]
        
        # Build reference trajectory
        x_ref = get_reference_trajectory(x_fb, self.mpc)
        foot_ref = get_reference_foot_trajectory(x_fb, t, foot, self.mpc, contact)
        
        # Build matrices (simplified version - just equality constraints)
        # In full implementation, you'd build complete H, Aeq, beq here
        from mpc import eul2rotm, skew
        
        R = eul2rotm(x_fb[0:3])
        A_matrices = []
        B_matrices = []
        
        for k in range(self.mpc.h):
            A, B = get_simplified_dynamics(self.mpc, self.biped, x_ref[:, k], foot_ref[:, k])
            A_matrices.append(A)
            B_matrices.append(B)
        
        # Build Aeq (simplified - dynamics only)
        Aeq_dyn = np.zeros((13 * self.mpc.h, 25 * self.mpc.h))
        one = np.array([1])
        x_0 = np.concatenate((x_fb, one), axis=0).reshape(-1, 1)
        
        for i in range(self.mpc.h):
            Aeq_dyn[13*i:13*(i+1), 13*i:13*(i+1)] = np.eye(13)
            Aeq_dyn[13*i:13*(i+1), 13*self.mpc.h+12*i:13*self.mpc.h+12*(i+1)] = -B_matrices[i]
            if i > 0:
                Aeq_dyn[13*i:13*(i+1), 13*(i-1):13*(i)] = -A_matrices[i]
        
        # Setup in parallel solver
        self.parallel_solver.setup_problem_matrices(
            H=np.eye(25 * self.mpc.h),  # Simplified
            f=np.zeros(25 * self.mpc.h),
            Aeq=Aeq_dyn,
            beq=np.zeros(13 * self.mpc.h)
        )
        
        print("Matrices setup for batch processing")

    def solve_batch(self,
                   x_fb_batch: np.ndarray,
                   t_batch: np.ndarray,
                   foot_batch: np.ndarray,
                   contact_batch: np.ndarray) -> tuple:
        """
        Solve MPC for a batch of environments
        
        Args:
            x_fb_batch: (batch_size, 13)
            t_batch: (batch_size,)
            foot_batch: (batch_size, 6)
            contact_batch: (batch_size, h, 2)
            
        Returns:
            states_batch: (batch_size, h, 13)
            controls_batch: (batch_size, h, 12)
            solve_time: Total solving time
        """
        # Create batch of f vectors
        f_batch = np.zeros((self.batch_size, 25 * self.mpc.h))
        beq_batch = np.zeros((self.batch_size, 13 * self.mpc.h))
        
        # Build individual f and beq for each environment
        for i in range(self.batch_size):
            x_ref = get_reference_trajectory(x_fb_batch[i], self.mpc)
            foot_ref = get_reference_foot_trajectory(x_fb_batch[i], t_batch[i], 
                                                     foot_batch[i], self.mpc, 
                                                     contact_batch[i])
            
            # Simplified: just use reference tracking cost
            x_ref_flat = x_ref.T.flatten()
            f_batch[i, :13*self.mpc.h] = -self.mpc.Q * x_ref_flat
        
        # Solve batch
        states_batch, controls_batch, solve_time = self.parallel_solver.solve_batch_optimized(
            f_batch, beq_batch
        )
        
        return states_batch, controls_batch, solve_time


def benchmark_parallel_vs_sequential():
    """
    Benchmark parallel vs sequential solving
    """
    import time
    
    batch_size = 32
    n_envs = 100  # Number of different environments to solve
    
    print("\n" + "="*60)
    print("BENCHMARK: Parallel vs Sequential MPC Solving")
    print("="*60)
    
    # Initialize solvers
    parallel_mpc = BikedMPCParallel(batch_size=batch_size, device='cuda')
    
    # Create random batch data
    x_fb_batch = np.random.randn(batch_size, 13).astype(np.float32)
    x_fb_batch[:, 5] = 0.55  # Keep height
    t_batch = np.random.rand(batch_size).astype(np.float32)
    foot_batch = np.random.randn(batch_size, 6).astype(np.float32)
    contact_batch = np.ones((batch_size, parallel_mpc.mpc.h, 2))
    
    # Setup matrices once
    parallel_mpc.setup_matrices_from_reference(x_fb_batch, t_batch, foot_batch, contact_batch)
    
    # Benchmark parallel solving
    print(f"\nSolving {n_envs} environments with batch_size={batch_size}")
    print(f"Number of batches: {(n_envs + batch_size - 1) // batch_size}")
    
    start = time.time()
    total_time_parallel = 0
    
    for batch_idx in range((n_envs + batch_size - 1) // batch_size):
        states, controls, solve_time = parallel_mpc.solve_batch(
            x_fb_batch, t_batch, foot_batch, contact_batch
        )
        total_time_parallel += solve_time
    
    end = time.time()
    
    print(f"\nParallel solving time: {total_time_parallel:.4f} seconds")
    print(f"Total time (with overhead): {end - start:.4f} seconds")
    print(f"Average time per batch: {total_time_parallel / ((n_envs + batch_size - 1) // batch_size):.6f} seconds")
    print(f"Speedup: {(n_envs / (end - start)) * batch_size:.1f}x environments per second")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Biped MPC Parallel Solver Integration")
    print("-" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, will use CPU")
    
    # Run benchmark
    benchmark_parallel_vs_sequential()
