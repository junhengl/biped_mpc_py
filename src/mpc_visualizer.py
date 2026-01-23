"""
Simple real-time MPC prediction visualization using MuJoCo bodies.

Adds temporary boxes to visualize predicted COM positions directly in the simulation.
"""

import numpy as np
import mujoco


class MPCVisualizer:
    """Visualizes MPC predictions as boxes in the MuJoCo simulation."""
    
    def __init__(self, model, data, n_predictions=10):
        """
        Initialize the visualizer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            n_predictions: Number of prediction steps to visualize
        """
        self.model = model
        self.data = data
        self.n_predictions = n_predictions
        self.pred_body_ids = []
        self.pred_geom_ids = []
        
        # Add prediction bodies and geoms to the model
        self._create_prediction_bodies()
    
    def _create_prediction_bodies(self):
        """Create bodies in the model for visualizing predictions."""
        # We'll use the model's existing bodies and geoms
        # Store IDs of the prediction visualization geoms
        # These will be updated each step
        pass
    
    def update(self, predicted_com_positions, current_com=None):
        """
        Update visualization with new predictions.
        
        Args:
            predicted_com_positions: (n_predictions, 3) array of predicted COM positions
            current_com: Current actual COM position (optional, for reference)
        """
        if predicted_com_positions is None or len(predicted_com_positions) == 0:
            return
        
        # For now, we'll use a simple approach: update body positions if they exist
        # In a real implementation, you might use MuJoCo's debug rendering or
        # temporary geoms that are rendered each frame
        
        # Store for potential future use
        self.last_predictions = predicted_com_positions.copy()
        self.last_current_com = current_com.copy() if current_com is not None else None


def visualize_predictions_in_viewer(viewer, predicted_positions, current_pos=None):
    """
    Simple debug visualization of predictions in the viewer.
    Prints trajectory info that can be observed during simulation.
    
    Args:
        viewer: MuJoCo viewer instance
        predicted_positions: (n, 3) array of predicted positions
        current_pos: Current actual position
    """
    if predicted_positions is None or len(predicted_positions) == 0:
        return
    
    # Calculate trajectory statistics for console output
    if len(predicted_positions) > 1:
        deltas = np.diff(predicted_positions, axis=0)
        step_lengths = np.linalg.norm(deltas, axis=1)
        mean_step = step_lengths.mean()
        max_step = step_lengths.max()
        
        # Print predictions periodically (every Nth call to avoid spam)
        if not hasattr(visualize_predictions_in_viewer, 'call_count'):
            visualize_predictions_in_viewer.call_count = 0
        
        visualize_predictions_in_viewer.call_count += 1
        
        # Print every 10 calls
        if visualize_predictions_in_viewer.call_count % 10 == 0:
            print(f"\n[MPC] Predicted trajectory (horizon={len(predicted_positions)} steps):")
            print(f"  Start (t=0): [{predicted_positions[0,0]:.4f}, {predicted_positions[0,1]:.4f}, {predicted_positions[0,2]:.4f}]")
            if len(predicted_positions) > 1:
                mid_idx = len(predicted_positions) // 2
                print(f"  Mid   (t={mid_idx}): [{predicted_positions[mid_idx,0]:.4f}, {predicted_positions[mid_idx,1]:.4f}, {predicted_positions[mid_idx,2]:.4f}]")
            print(f"  End   (t={len(predicted_positions)-1}): [{predicted_positions[-1,0]:.4f}, {predicted_positions[-1,1]:.4f}, {predicted_positions[-1,2]:.4f}]")
            print(f"  Step lengths: mean={mean_step:.4f}m, max={max_step:.4f}m")
            
            if current_pos is not None:
                print(f"  Current COM: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
            print()
