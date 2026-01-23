# Parallel MPC Solver - Quick Start Guide

## Installation

Ensure you have the required packages installed:

```bash
pip install numpy torch cvxopt mujoco
```

## 30-Second Quick Start

```python
import numpy as np
from mpc_parallel import ParallelMPC

# 1. Initialize solver
solver = ParallelMPC(batch_size=256, device='cuda')

# 2. Setup problem (once per configuration)
H = np.eye(250) * 2.0  # Hessian
f = np.ones(250) * 0.1  # Linear term
Aeq = np.eye(150, 250)  # Equality constraints
beq = np.zeros(150)  # Constraint RHS

solver.setup_problem_matrices(H, f, Aeq, beq)

# 3. Create batch of problems
batch_size = 256
f_batch = np.tile(f, (batch_size, 1)) + np.random.randn(batch_size, 250) * 0.01
beq_batch = np.tile(beq, (batch_size, 1)) + np.random.randn(batch_size, 150) * 0.001

# 4. Solve
states, controls, solve_time = solver.solve_batch_optimized(f_batch, beq_batch)

# 5. Access results
print(f"States shape: {states.shape}")  # (256, 10, 13)
print(f"Controls shape: {controls.shape}")  # (256, 10, 12)
print(f"Solve time: {solve_time*1000:.2f} ms")  # ~10-15 ms for 256 envs
```

## Key Concepts

### Problem Structure

The solver handles constrained QP problems of the form:

```
minimize: 0.5 * x.T @ H @ x + f.T @ x
subject to: Aeq @ x = beq
```

For biped MPC:
- **x**: [x₁, x₂, ..., x₁₀, u₁, u₂, ..., u₁₀] (250 variables)
  - xᵢ: 13-dim state (orientation, position, velocities)
  - uᵢ: 12-dim control (forces/moments for 2 feet)
- **H**: 250×250 cost matrix (precomputed from Q, R weights)
- **f**: 250-dim linear term (depends on reference trajectory)
- **Aeq**: 150×250 dynamics constraint matrix
- **beq**: 150-dim constraint RHS (depends on current state)

### Batch Processing

The solver processes multiple environments in parallel:

```
Inputs:
  f_batch:     (batch_size, 250)  - Multiple linear terms
  beq_batch:   (batch_size, 150)  - Multiple constraint RHS

Outputs:
  states:      (batch_size, 10, 13)  - State trajectories
  controls:    (batch_size, 10, 12)  - Control trajectories
```

Each problem is solved independently with different f and beq values, but the same H and Aeq matrices.

## Performance Metrics

On NVIDIA RTX 5090:

| Batch Size | Time (ms) | Throughput (env/s) |
|-----------|-----------|-------------------|
| 32        | 1-2       | ~15,000           |
| 128       | 3-5       | ~25,000           |
| 256       | 10-15     | ~35,000           |
| 4096      | 120-150   | ~30,000           |

**Speedup vs Sequential:** 40-50x faster than solving individual problems

## Common Usage Patterns

### Pattern 1: Single Batch

```python
solver = ParallelMPC(batch_size=256)
solver.setup_problem_matrices(H, f, Aeq, beq)

# Solve once
states, controls, time = solver.solve_batch_optimized(f_batch, beq_batch)
```

### Pattern 2: Multiple Batches (e.g., online MPC loop)

```python
solver = ParallelMPC(batch_size=32)
solver.setup_problem_matrices(H, f, Aeq, beq)

for t in range(num_steps):
    # Update f_batch based on current state observations
    f_batch = generate_batch_objectives()
    beq_batch = generate_batch_constraints()
    
    states, controls, _ = solver.solve_batch_optimized(f_batch, beq_batch)
    
    # Use control trajectories
    apply_controls(controls[:, 0])  # Apply first step
```

### Pattern 3: Multi-Batch Large-Scale

```python
solver = ParallelMPC(batch_size=256)
solver.setup_problem_matrices(H, f, Aeq, beq)

# Solve 10,000 environments in batches
all_states = []
all_controls = []

for i in range(0, 10000, 256):
    f_batch = generate_batch(256, i)
    beq_batch = generate_batch_constraints(256, i)
    
    states, controls, _ = solver.solve_batch_optimized(f_batch, beq_batch)
    all_states.append(states)
    all_controls.append(controls)

# Concatenate results
all_states = np.vstack(all_states)  # (10000, 10, 13)
```

## Extracting Information

### State Trajectories

```python
# states shape: (batch_size, horizon=10, state_dim=13)
# state = [roll, pitch, yaw, x, y, z, p, q, r, vx, vy, vz, ?]

for i in range(batch_size):
    com_trajectory = states[i, :, 3:6]  # Position for trajectory i
    velocity_trajectory = states[i, :, 9:12]  # Velocities
```

### Control Trajectories

```python
# controls shape: (batch_size, horizon=10, control_dim=12)
# control = [Fx_L, Fy_L, Fz_L, Mx_L, My_L, Mz_L, Fx_R, Fy_R, Fz_R, Mx_R, My_R, Mz_R]

for i in range(batch_size):
    left_foot_forces = controls[i, :, 0:3]   # Left foot forces
    right_foot_forces = controls[i, :, 6:9]  # Right foot forces
    left_foot_moments = controls[i, :, 3:6]  # Left foot moments
    right_foot_moments = controls[i, :, 9:12] # Right foot moments
```

## Advanced Options

### Device Selection

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver = ParallelMPC(batch_size=256, device=device)
```

### Data Types

```python
# Default: float32
solver = ParallelMPC(batch_size=256, dtype=torch.float32)

# High precision: float64
solver = ParallelMPC(batch_size=256, dtype=torch.float64)
```

### Batch Size Selection

- **Small (32-64):** For interactive/real-time applications
- **Medium (128-256):** Balanced throughput and latency
- **Large (512-4096):** For offline batch processing

## Limitations

1. **Equality constraints only:** No inequality constraints (friction, saturation)
   - Use full cvxopt solver for constrained problems
   
2. **Fixed problem structure:** Once H and Aeq are set, they cannot change
   - f and beq can be arbitrary for each solve
   
3. **Batch size requirement:** Always provide full batch_size inputs
   - Pad with dummy data if needed

## Troubleshooting

**Q: GPU out of memory**
- A: Reduce batch_size: `ParallelMPC(batch_size=128)`

**Q: Wrong output shapes**
- A: Check H and Aeq dimensions match problem size

**Q: Slow on GPU**
- A: Increase batch_size for better GPU utilization

**Q: Accuracy issues**
- A: Verify H is positive definite: `np.linalg.eigvals(H).min() > 0`

## Running Examples

```bash
# Large-scale benchmark (4096 envs)
python examples/benchmark_4096_envs.py

# Simple API tutorial
python examples/simple_usage.py

# Biped integration example
python examples/integration_example.py
```

## Next Steps

1. Review `examples/README.md` for detailed examples
2. Check `src/mpc_parallel.py` for API documentation
3. See `src/mpc.py` for MPC problem formulation
4. Experiment with different batch sizes for your hardware

## Support

For issues or questions, check:
- Example scripts in `examples/`
- API documentation in `src/mpc_parallel.py`
- MPC formulation in `src/mpc.py`
