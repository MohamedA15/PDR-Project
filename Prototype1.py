import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from calculate_rmse import calculate_rmse  # Import RMSE function

# Select route and ground truth
print("Select a route file:")
print("1. route1.csv\n2. route2.csv\n3. route3.csv")
route_choice = int(input("Enter the number of the route file: "))
filename = f"route{route_choice}.csv"

print("Select a ground truth file:")
print("1. ground_truth1.csv\n2. ground_truth2.csv\n3. ground_truth3.csv")
ground_truth_choice = int(input("Enter the number of the ground truth file: "))
ground_truth_file = f"ground_truth{ground_truth_choice}.csv"

required_columns = ['time', 'ax', 'ay', 'az', 'wz']

# Load and preprocess data
df = pd.read_csv(filename).dropna(subset=required_columns)
df.columns = df.columns.str.strip()
if not all(col in df.columns for col in required_columns):
    raise ValueError("Error: Missing required columns")
if not np.issubdtype(df['time'].dtype, np.number):
    raise ValueError("Error: 'time' column is not numeric")

true_positions = pd.read_csv(ground_truth_file).values

timestamps, x_axis, y_axis, z_axis, z_gyro = (
    df[col].values for col in required_columns)

# Calibration and visualization
static_samples = 800
means = [np.mean(axis[:static_samples]) for axis in [x_axis, y_axis, z_axis]]
x_calib, y_calib, z_calib = [axis - mean for axis, mean in zip([x_axis, y_axis, z_axis], means)]
plt.plot(x_calib, label="X_Calibrated")
plt.plot(y_calib, label="Y_Calibrated")
plt.plot(z_calib, label="Z_Calibrated")
plt.legend()
plt.title("Calibrated Accelerometer Data")
plt.show()

# Calculate magnitude of acceleration
accel_magnitude = np.linalg.norm([x_calib, y_calib, z_calib], axis=0)
plt.plot(accel_magnitude, label="Raw Magnitude")
plt.legend()
plt.title("Acceleration Magnitude")
plt.show()

# Step detection
steps, _ = signal.find_peaks(accel_magnitude, height=0.2, distance=5)
plt.plot(accel_magnitude, label="Raw Acceleration")
plt.plot(steps, accel_magnitude[steps], "x", label="Detected Steps", color="red")
plt.legend()
plt.title("Step Detection")
plt.show()

# Heading calculation
def calculate_heading(z_gyro, timestamps):
    return np.concatenate(([0], np.cumsum(z_gyro[:-1] * np.diff(timestamps))))

heading = calculate_heading(z_gyro, timestamps)
motion_vectors = np.array([np.cos(heading), np.sin(heading)]).T

# Compute trajectory
step_length, cur_position = 50.0, np.array([0.0, 0.0])  # Further increase step length to worsen drift
trajectory = [cur_position]
for step in steps:
    cur_position = cur_position + step_length * motion_vectors[step]
    trajectory.append(cur_position)

trajectory = np.array(trajectory)

# Compare with ground truth
min_length = min(len(trajectory), len(true_positions))
aligned_trajectory, aligned_truth = trajectory[:min_length], true_positions[:min_length]
drift = np.linalg.norm(aligned_trajectory - aligned_truth, axis=1)

plt.plot(aligned_truth[:, 0], aligned_truth[:, 1], label="True Trajectory", color="green")
plt.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], label="Estimated Trajectory", color="red")
plt.legend()
plt.title("PDR Trajectory vs Ground Truth")
plt.show()

plt.plot(drift, label="Drift", color="blue")
plt.legend()
plt.title("Drift Per Interval")
plt.show()

# Display drift at each interval
print("\nDrift at each interval (meters):")
for i, d in enumerate(drift):
    print(f"Step {i+1}: Drift = {d:.3f} meters")

# Save Data & Compute RMSE
pd.DataFrame(aligned_trajectory, columns=['x', 'y']).to_csv("estimated_trajectory.csv", index=False)
pd.DataFrame(aligned_truth, columns=['x', 'y']).to_csv("ground_truth.csv", index=False)
rmse = calculate_rmse("estimated_trajectory.csv", "ground_truth.csv")
print(f"\nRoot Mean Square Error (RMSE): {rmse:.3f} meters")

