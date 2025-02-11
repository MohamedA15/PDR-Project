import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_rmse(estimated_trajectory_file, ground_truth_file):
    """
    Calculate RMSE between the estimated trajectory and the ground truth.

    Parameters:
        estimated_trajectory_file (str): Path to the estimated trajectory CSV file.
                                         Must contain two columns: 'x', 'y'.
        ground_truth_file (str): Path to the ground truth CSV file.
                                 Must contain two columns: 'x', 'y'.

    Returns:
        float: The RMSE value.
    """
    # Load estimated trajectory
    estimated_trajectory = pd.read_csv(estimated_trajectory_file)
    if not {'x', 'y'}.issubset(estimated_trajectory.columns):
        raise ValueError("Estimated trajectory file must have 'x' and 'y' columns.")
    estimated_positions = estimated_trajectory[['x', 'y']].values

    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_file)
    if not {'x', 'y'}.issubset(ground_truth.columns):
        raise ValueError("Ground truth file must have 'x' and 'y' columns.")
    true_positions = ground_truth[['x', 'y']].values

    # Ensure alignment
    min_length = min(len(estimated_positions), len(true_positions))
    aligned_estimated = estimated_positions[:min_length]
    aligned_true = true_positions[:min_length]

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.sum((aligned_estimated - aligned_true)**2, axis=1)))

    # Print RMSE
    print(f"Root Mean Square Error (RMSE): {rmse:.4f} meters")

    # Optional: Plot trajectories
    plt.figure(figsize=(8, 6))
    plt.plot(aligned_true[:, 0], aligned_true[:, 1], label="Ground Truth", color="green")
    plt.plot(aligned_estimated[:, 0], aligned_estimated[:, 1], label="Estimated Trajectory", color="red", linestyle="--")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    return rmse

# Example usage
if __name__ == "__main__":
    estimated_file = "estimated_trajectory.csv"  # Replace with your file path
    ground_truth_file = "ground_truth.csv"       # Replace with your file path
    calculate_rmse(estimated_file, ground_truth_file)
