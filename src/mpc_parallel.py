import numpy as np
import torch
import time
from typing import Tuple, Optional

"""
Parallel MPC Solver using PyTorch and CUDA
Solves the analytical QP formulation for multiple environments simultaneously
"""

class ParallelMPC:
    def __init__(self, 
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 dtype: torch.dtype = torch.float32):
        """
        Initialize parallel MPC solver
        
        Args:
            batch_size: Number of parallel environments
            device: 'cuda' or 'cpu'
            dtype: torch.float32 or torch.float64
        """
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        # MPC parameters
        self.h = 10  # Horizon
        self.dt = 0.04
        
        # Problem dimensions
        self.n_states = 13 * self.h  # State variables
        self.n_controls = 12 * self.h  # Control variables
        self.n_vars = self.n_states + self.n_controls
        self.n_eq = 13 * self.h + 2 * self.h  # Dynamics + moment constraints
        
        print(f"ParallelMPC initialized on {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Problem size: {self.n_vars} variables, {self.n_eq} equality constraints")

    def setup_problem_matrices(self, 
                              H: np.ndarray,
                              f: np.ndarray,
                              Aeq: np.ndarray,
                              beq: np.ndarray) -> None:
        """
        Setup QP problem matrices as batch tensors
        
        Args:
            H: Hessian matrix (n_vars x n_vars)
            f: Linear term (n_vars,)
            Aeq: Equality constraint matrix (n_eq x n_vars)
            beq: Equality constraint RHS (n_eq,)
        """
        # Convert to tensors and move to device
        self.H = torch.from_numpy(H).to(dtype=self.dtype, device=self.device)
        self.f = torch.from_numpy(f).to(dtype=self.dtype, device=self.device)
        self.Aeq = torch.from_numpy(Aeq).to(dtype=self.dtype, device=self.device)
        self.beq = torch.from_numpy(beq).to(dtype=self.dtype, device=self.device)
        
        # Precompute H_inv for efficiency
        self.H_inv = torch.linalg.inv(self.H)
        
        print("Problem matrices loaded and moved to device")

    def solve_batch(self, 
                   f_batch: np.ndarray,
                   beq_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve analytical QP for a batch of different RHS vectors
        
        Args:
            f_batch: (batch_size, n_vars) - Linear terms for each environment
            beq_batch: (batch_size, n_eq) - Equality RHS for each environment
            
        Returns:
            states_batch: (batch_size, h, 13) - Predicted states for each environment
            controls_batch: (batch_size, h, 12) - Predicted controls for each environment
        """
        # Convert to tensors
        f_batch_t = torch.from_numpy(f_batch).to(dtype=self.dtype, device=self.device)
        beq_batch_t = torch.from_numpy(beq_batch).to(dtype=self.dtype, device=self.device)
        
        start_time = time.time()
        
        # Batch analytical solution: x = -H_inv @ f - H_inv @ Aeq.T @ (Aeq @ H_inv @ Aeq.T)^-1 @ (Aeq @ H_inv @ f + beq)
        
        # Precomputed part: Aeq @ H_inv @ Aeq.T (same for all)
        # temp = Aeq @ H_inv @ Aeq.T
        # temp_inv = (temp)^-1
        # These are computed once (they don't depend on f or beq)
        if not hasattr(self, 'temp_inv'):
            temp = self.Aeq @ self.H_inv @ self.Aeq.T
            self.temp_inv = torch.linalg.inv(temp)
        
        # For each sample in batch:
        # term1 = H_inv @ f_batch  (batch matrix-vector product)
        # term2 = Aeq @ H_inv @ f_batch  (batch matrix-vector product)
        # term3 = (Aeq @ H_inv @ Aeq.T)^-1 @ (term2 + beq)  (batch matrix-vector product)
        # x = -(term1 + H_inv @ Aeq.T @ term3)
        
        # Batch operations
        term1 = torch.matmul(f_batch_t, self.H_inv.T)  # (batch_size, n_vars)
        term2 = torch.matmul(f_batch_t, self.H_inv.T)  # (batch_size, n_vars)
        term2_eq = torch.matmul(term2, self.Aeq.T)  # (batch_size, n_eq)
        term2_eq_rhs = term2_eq + beq_batch_t  # (batch_size, n_eq)
        
        # Solve batch system: temp_inv @ term2_eq_rhs
        term3 = torch.matmul(term2_eq_rhs, self.temp_inv.T)  # (batch_size, n_eq)
        
        # Final: x = -(term1 + H_inv @ Aeq.T @ term3)
        H_inv_Aeq_T = torch.matmul(self.H_inv, self.Aeq.T)  # (n_vars, n_eq)
        term4 = torch.matmul(term3, H_inv_Aeq_T.T)  # (batch_size, n_vars)
        x_opt = -(term1 + term4)
        
        solve_time = time.time() - start_time
        
        # Extract states and controls
        states_batch = x_opt[:, :13*self.h].reshape(self.batch_size, self.h, 13)
        controls_batch = x_opt[:, 13*self.h:].reshape(self.batch_size, self.h, 12)
        
        return states_batch.cpu().numpy(), controls_batch.cpu().numpy(), solve_time

    def solve_batch_optimized(self,
                             f_batch: np.ndarray,
                             beq_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Optimized batch solving with better matrix layout
        
        Args:
            f_batch: (batch_size, n_vars)
            beq_batch: (batch_size, n_eq)
            
        Returns:
            states_batch: (batch_size, h, 13)
            controls_batch: (batch_size, h, 12)
            solve_time: Computation time in seconds
        """
        f_batch_t = torch.from_numpy(f_batch).to(dtype=self.dtype, device=self.device)  # (batch, n_vars)
        beq_batch_t = torch.from_numpy(beq_batch).to(dtype=self.dtype, device=self.device)  # (batch, n_eq)
        
        start_time = time.time()
        
        # Precomputed inverse (done once in setup_problem_matrices)
        if not hasattr(self, 'temp_inv'):
            temp = self.Aeq @ self.H_inv @ self.Aeq.T
            self.temp_inv = torch.linalg.inv(temp)
        
        # Analytical QP solution formula:
        # x = -H_inv @ (f + Aeq.T @ temp_inv @ (Aeq @ H_inv @ f + beq))
        
        # Step 1: H_inv @ f for batch
        H_inv_f = torch.matmul(f_batch_t, self.H_inv.T)  # (batch, n_vars) @ (n_vars, n_vars) = (batch, n_vars)
        
        # Step 2: Aeq @ H_inv @ f for batch
        # Aeq: (n_eq, n_vars), H_inv: (n_vars, n_vars), f: (batch, n_vars)
        # Aeq @ H_inv: (n_eq, n_vars)
        # (Aeq @ H_inv) @ f.T = (n_eq, n_vars) @ (n_vars, batch) = (n_eq, batch)
        # Transpose to (batch, n_eq)
        Aeq_H_inv = self.Aeq @ self.H_inv  # (n_eq, n_vars)
        # For batch multiply: f @ (Aeq @ H_inv).T = (batch, n_vars) @ (n_vars, n_eq)
        Aeq_H_inv_f = torch.matmul(f_batch_t, Aeq_H_inv.T)  # (batch, n_eq)
        
        # Step 3: RHS = Aeq @ H_inv @ f + beq
        rhs = Aeq_H_inv_f + beq_batch_t  # (batch, n_eq)
        
        # Step 4: temp_inv @ rhs for batch
        # temp_inv: (n_eq, n_eq), rhs: (batch, n_eq)
        # rhs @ temp_inv.T = (batch, n_eq) @ (n_eq, n_eq) = (batch, n_eq)
        correction = torch.matmul(rhs, self.temp_inv.T)  # (batch, n_eq)
        
        # Step 5: Aeq.T @ correction for batch
        # Aeq.T: (n_vars, n_eq), correction: (batch, n_eq)
        # correction @ Aeq = (batch, n_eq) @ (n_eq, n_vars) = (batch, n_vars)
        correction_term = torch.matmul(correction, self.Aeq)  # (batch, n_vars)
        
        # Step 6: x = -(H_inv_f + H_inv @ correction_term)
        # H_inv: (n_vars, n_vars), correction_term: (batch, n_vars)
        # correction_term @ H_inv.T = (batch, n_vars) @ (n_vars, n_vars) = (batch, n_vars)
        x_opt = -(H_inv_f + torch.matmul(correction_term, self.H_inv.T))  # (batch, n_vars)
        
        solve_time = time.time() - start_time
        
        # Reshape to states and controls
        # x_opt has shape (batch_size, n_vars) where n_vars = 13*h + 12*h = 250
        n_state_vars = 13 * self.h
        n_control_vars = 12 * self.h
        
        # Debug: check actual shape
        # print(f"DEBUG: x_opt shape = {x_opt.shape}, expected = ({self.batch_size}, {n_state_vars + n_control_vars})")
        
        # Extract states and controls
        states_batch_flat = x_opt[:, :n_state_vars]  # (batch_size, 130)
        controls_batch_flat = x_opt[:, n_state_vars:n_state_vars+n_control_vars]  # (batch_size, 120)
        
        # print(f"DEBUG: states shape = {states_batch_flat.shape}, controls shape = {controls_batch_flat.shape}")
        
        # Reshape to (batch_size, h, state_dim) and (batch_size, h, control_dim)
        states_batch = states_batch_flat.reshape(self.batch_size, self.h, 13)  # (batch_size, 10, 13)
        controls_batch = controls_batch_flat.reshape(self.batch_size, self.h, 12)  # (batch_size, 10, 12)
        
        return states_batch.cpu().numpy(), controls_batch.cpu().numpy(), solve_time


def create_random_batch(batch_size: int, 
                       n_vars: int,
                       n_eq: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create random batch data for testing
    
    Args:
        batch_size: Number of environments
        n_vars: Number of variables
        n_eq: Number of equality constraints
        
    Returns:
        f_batch: (batch_size, n_vars)
        beq_batch: (batch_size, n_eq)
    """
    f_batch = np.random.randn(batch_size, n_vars).astype(np.float32)
    beq_batch = np.random.randn(batch_size, n_eq).astype(np.float32)
    
    return f_batch, beq_batch


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/junhengl/biped_mpc_py/src')
    from mpc import MPC, Biped, solve_mpc
    
    # Setup single environment to get problem matrices
    mpc = MPC()
    biped = Biped()
    
    # Get example matrices from one solve
    x_fb = np.array([0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 0])
    foot = np.array([0, -0.1, 0, 0, 0.1, 0])
    contact = np.ones((mpc.h, 2))
    t = 0.0
    
    # Note: This imports solve_mpc which builds the problem
    # We'll extract the matrices from there
    print("Setting up parallel MPC solver...")
    
    # Initialize parallel solver
    batch_size = 32
    parallel_mpc = ParallelMPC(batch_size=batch_size, 
                               device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random batch for testing
    f_batch, beq_batch = create_random_batch(batch_size, 
                                             parallel_mpc.n_vars,
                                             parallel_mpc.n_eq)
    
    print(f"\nTesting batch shapes:")
    print(f"  f_batch: {f_batch.shape}")
    print(f"  beq_batch: {beq_batch.shape}")
    
    print("\nTo use this solver:")
    print("1. Set up problem matrices once: parallel_mpc.setup_problem_matrices(H, f, Aeq, beq)")
    print("2. Solve batches: states, controls, time = parallel_mpc.solve_batch_optimized(f_batch, beq_batch)")
