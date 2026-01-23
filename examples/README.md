# Parallel MPC Solver Examples

This directory contains example scripts demonstrating how to use the parallel MPC solver with GPU acceleration.

## Examples

### 1. `benchmark_4096_envs.py` - Large Scale Benchmark

Demonstrates solving 4096 biped MPC problems in parallel on GPU.

**Features:**
- Sets up complete MPC problem with real matrices
- Generates 4096 dummy initial conditions
- Solves in batches of 256 environments
- Reports throughput and GPU memory usage
- Expected performance: ~40,000 environments/second on RTX 5090

**Run:**
```bash
python examples/benchmark_4096_envs.py
```

**Output includes:**
- Problem configuration
- GPU device info
- Batch-wise throughput progression
- Total timing breakdown
- Memory utilization
- Sample trajectory outputs

### 2. `simple_usage.py` - Basic API Usage

Simple examples showing how to use the parallel MPC solver.

**Examples included:**
- Basic setup and solving with 32 environments
- Solving multiple batches sequentially
- GPU memory reporting
- Accessing results

**Run:**
```bash
python examples/simple_usage.py
```

**Key API calls:**
```python
# Initialize solver
solver = ParallelMPC(batch_size=32, device='cuda')

# Load problem matrices (once)
solver.setup_problem_matrices(H, f_template, Aeq, beq_template)

# Solve batch
states, controls, solve_time = solver.solve_batch_optimized(f_batch, beq_batch)
```

### 3. `integration_example.py` - Real Biped Integration

Shows integration with actual biped MPC system and performance comparison.

**Examples included:**
- Creating problem matrices from MPC system
- Solving 128 environments with real biped parameters
- Sequential vs parallel comparison (43x speedup shown)
- Trajectory statistics computation

**Run:**
```bash
python examples/integration_example.py
```

**Key features:**
- Uses actual MPC and Biped objects
- Demonstrates real problem sizes
- Shows state and control trajectory outputs
- Compares sequential vs parallel solving performance

## Problem Configuration

All examples use the same MPC problem structure:

- **Horizon (h):** 10 steps
- **Variables:** 250 (13 states × 10 + 12 controls × 10)
- **State dimension:** 13 (orientation, position, ang velocity, lin velocity)
- **Control dimension:** 12 (forces/moments for 2 feet, 6 DOF each)
- **Equality constraints:** ~150 (dynamics + moment balance)

## Performance Expectations

On RTX 5090 GPU:

- **Throughput:** 40,000+ environments/second
- **Time per environment:** ~0.025 ms
- **Time per batch (256 envs):** ~10 ms
- **GPU memory:** ~30-100 MB depending on batch size

Speedup vs sequential solving: **40-50x** on GPU

## Key API Reference

### ParallelMPC

```python
from mpc_parallel import ParallelMPC

# Create solver
solver = ParallelMPC(batch_size=256, device='cuda')

# Setup problem matrices (once)
solver.setup_problem_matrices(H, f_template, Aeq, beq_template)

# Solve batch
states, controls, solve_time = solver.solve_batch_optimized(f_batch, beq_batch)

# Returns:
# - states: (batch_size, horizon, 13) numpy array
# - controls: (batch_size, horizon, 12) numpy array  
# - solve_time: float, seconds
```

### Input Data Shapes

```python
H:           (250, 250)        # Hessian (symmetric positive definite)
f_template:  (250,)            # Linear term template
Aeq:         (150, 250)        # Equality constraint matrix
beq_template:(150,)            # Constraint RHS template

f_batch:     (batch_size, 250) # Linear terms for batch
beq_batch:   (batch_size, 150) # Constraint RHS for batch
```

### Important Notes

1. **Batch Size Constraint:** Always provide full batch_size inputs. Pad if necessary:
   ```python
   if actual_size < batch_size:
       f_batch = np.vstack([f_batch, f_batch[:batch_size-actual_size]])
       beq_batch = np.vstack([beq_batch, beq_batch[:batch_size-actual_size]])
   ```

2. **Device Selection:** Automatically uses CUDA if available, falls back to CPU
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

3. **Accuracy:** Analytical QP solution only (no inequality constraints)
   - Use for trajectory prediction
   - May violate friction/saturation constraints
   - Use cvxopt's solve_qp for constrained problems

## Extending the Examples

To use with your own problems:

1. **Modify problem matrices:**
   ```python
   # Build your own H, f_template, Aeq, beq_template
   solver.setup_problem_matrices(H, f_template, Aeq, beq_template)
   ```

2. **Change batch sizes:**
   ```python
   solver = ParallelMPC(batch_size=512)  # Larger batches for throughput
   ```

3. **Extract different states:**
   ```python
   # Modify reshape operations to extract different parts of x_opt
   # Current: states first 130 dims, controls last 120 dims
   ```

## Troubleshooting

**Issue:** `RuntimeError: shape '[32, 10, 13]' is invalid for input of size 1040`
- **Cause:** Input batch size doesn't match solver batch_size
- **Fix:** Always pad inputs to match solver.batch_size

**Issue:** `CUDA out of memory`
- **Fix:** Reduce batch_size: `ParallelMPC(batch_size=128)`

**Issue:** Incorrect trajectory shapes
- **Fix:** Check H matrix dimensions; should be (250, 250) for 10-horizon biped MPC

## References

- Main solver: `src/mpc_parallel.py`
- Integration layer: `src/mpc_parallel_integration.py`
- MPC core: `src/mpc.py`
- Biped model: Uses MuJoCo XML in `assets/hector_v1p5/`
