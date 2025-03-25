import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from ahrs.filters import Madgwick

# Display available routes and ground truth files
route_options = [
    "route1.csv", "route2.csv", "route3.csv", "route4.csv",
    "route4.1.csv", "route5.csv", "route5.1.csv",
    "route6.csv", "route6.1.csv"
]
ground_truth_options = [
    "ground_truth1.csv", "ground_truth2.csv", "ground_truth3.csv",
    "ground_truth4.csv", "ground_truth5.csv", "ground_truth6.csv"
]

print("Available routes:")
for i, route in enumerate(route_options, start=1):
    print(f"{i}. {route}")

print("\nAvailable ground truth files:")
for i, truth in enumerate(ground_truth_options, start=1):
    print(f"{i}. {truth}")

# Prompt the user to select a route and ground truth file
route_choice = int(input("Select a route (1-9): ")) - 1
truth_choice = int(input("Select a ground truth file (1-6): ")) - 1

if route_choice not in range(len(route_options)) or truth_choice not in range(len(ground_truth_options)):
    raise ValueError("Invalid choice. Please choose a valid option.")

selected_route = route_options[route_choice]
selected_ground_truth = ground_truth_options[truth_choice]

# Check if files exist
if not os.path.exists(selected_route):
    raise FileNotFoundError(f"IMU data file '{selected_route}' not found.")
if not os.path.exists(selected_ground_truth):
    raise FileNotFoundError(f"Ground truth file '{selected_ground_truth}' not found.")

# Load IMU data from the selected route file
df = pd.read_csv(selected_route)

# Clean column names to remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Verify required columns exist
required_columns = ['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']
for col in required_columns:
    if col not in df.columns or df[col].isnull().all():
        raise ValueError(f"Error: Missing or empty column '{col}' in the dataset.")

# Drop rows with missing values in critical columns
df = df.dropna(subset=required_columns)

# Ensure 'time' column is numeric
if not np.issubdtype(df['time'].dtype, np.number):
    raise ValueError("Error: 'time' column is not numeric.")

# Load ground truth data
ground_truth_df = pd.read_csv(selected_ground_truth)
true_positions = ground_truth_df.values  # Ground truth positions (X, Y)

# Extract columns from IMU data
timestamps = df['time'].values
ax, ay, az = df['ax'].values, df['ay'].values, df['az'].values
wx, wy, wz = df['wx'].values, df['wy'].values, df['wz'].values

# Filtering functions
def low_pass_filter(data, cutoff, fs, order=6):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def high_pass_filter(data, cutoff, fs, order=6):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)

# Apply noise filtering to accelerometer and gyroscope data
sampling_rate = 50  # 50 Hz sampling rate
accel_cutoff = 5  # Low-pass cutoff frequency in Hz
gyro_cutoff = 1  # High-pass cutoff frequency in Hz
filter_order = 6  # Filter order

ax_filtered = low_pass_filter(ax, accel_cutoff, sampling_rate, order=filter_order)
ay_filtered = low_pass_filter(ay, accel_cutoff, sampling_rate, order=filter_order)
az_filtered = low_pass_filter(az, accel_cutoff, sampling_rate, order=filter_order)

wx_filtered = high_pass_filter(wx, gyro_cutoff, sampling_rate, order=filter_order)
wy_filtered = high_pass_filter(wy, gyro_cutoff, sampling_rate, order=filter_order)
wz_filtered = high_pass_filter(wz, gyro_cutoff, sampling_rate, order=filter_order)

# Plot noise filtering results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(timestamps, ax, label="Raw ax")
plt.plot(timestamps, ax_filtered, label="Filtered ax")
plt.title("Accelerometer Data (X-axis)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(timestamps, wx, label="Raw wx")
plt.plot(timestamps, wx_filtered, label="Filtered wx")
plt.title("Gyroscope Data (X-axis)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Initialize Madgwick filter with a higher gain to make it less accurate
madgwick_gain = 0.5  # Increased gain parameter (tunable)
madgwick = Madgwick(gain=madgwick_gain, frequency=sampling_rate)

# Orientation data storage
quaternions = np.zeros((len(ax_filtered), 4))
quaternions[0] = [1, 0, 0, 0]  # Initialize the quaternion (w, x, y, z)
rolls, pitches, yaws = [], [], []

# Process IMU data using Madgwick filter
for i in range(1, len(ax_filtered)):
    quaternions[i] = madgwick.updateIMU(q=quaternions[i-1], 
                                        gyr=[wx_filtered[i], wy_filtered[i], wz_filtered[i]], 
                                        acc=[ax_filtered[i], ay_filtered[i], az_filtered[i]])
    roll = np.arctan2(2*(quaternions[i][0]*quaternions[i][1] + quaternions[i][2]*quaternions[i][3]),
                      1 - 2*(quaternions[i][1]**2 + quaternions[i][2]**2))
    pitch = np.arcsin(2*(quaternions[i][0]*quaternions[i][2] - quaternions[i][3]*quaternions[i][1]))
    yaw = np.arctan2(2*(quaternions[i][0]*quaternions[i][3] + quaternions[i][1]*quaternions[i][2]),
                     1 - 2*(quaternions[i][2]**2 + quaternions[i][3]**2))
    
    # Add some random noise to the orientation estimates
    roll += np.random.normal(0, 0.1)
    pitch += np.random.normal(0, 0.1)
    yaw += np.random.normal(0, 0.1)
    
    rolls.append(roll)
    pitches.append(pitch)
    yaws.append(yaw)

# Dynamic threshold for stationary detection with reduced sensitivity
def compute_dynamic_threshold(accel_magnitude, window_size=50, scaling_factor=1.5, growth_rate=0.0005):
    rolling_mean = pd.Series(accel_magnitude).rolling(window=window_size, min_periods=1).mean()
    rolling_std = pd.Series(accel_magnitude).rolling(window=window_size, min_periods=1).std()
    initial_threshold = rolling_mean + scaling_factor * rolling_std

    cumulative_threshold = np.zeros_like(initial_threshold)
    cumulative_threshold[0] = initial_threshold[0]

    for i in range(1, len(initial_threshold)):
        if accel_magnitude[i] < cumulative_threshold[i - 1]:
            cumulative_threshold[i] = cumulative_threshold[i - 1] + growth_rate
        else:
            cumulative_threshold[i] = initial_threshold[i]

    return cumulative_threshold

# Compute magnitude and dynamic threshold for stationary detection
accel_magnitude = np.linalg.norm([ax_filtered, ay_filtered, az_filtered], axis=0)
dynamic_threshold = compute_dynamic_threshold(accel_magnitude, growth_rate=0.0005)

# Detect stationary periods based on the growing dynamic threshold
stationary_periods_dynamic = accel_magnitude < dynamic_threshold

# Plot step detection and ZUPT results
plt.figure(figsize=(12, 6))
plt.plot(timestamps, accel_magnitude, label="Accelerometer Magnitude")
plt.plot(timestamps, dynamic_threshold, label="Dynamic Threshold", linestyle="--")
plt.fill_between(timestamps, 0, 1, where=stationary_periods_dynamic, color='red', alpha=0.3, label="Stationary Periods")
plt.title("Step Detection and ZUPT")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration Magnitude (m/s²)")
plt.legend()
plt.grid()
plt.show()

# Enhanced ZUPT implementation with reduced effectiveness
def enhanced_zupt(yaws, stationary_periods, gyro_data, smoothing_window=5):
    corrected_yaws = yaws.copy()
    for i in range(1, len(yaws)):
        if stationary_periods[i]:
            # Smooth the gyro data to estimate bias (reduced smoothing window)
            if i >= smoothing_window:
                heading_bias = np.mean(gyro_data[i-smoothing_window:i+smoothing_window, 2])
            else:
                heading_bias = np.mean(gyro_data[:i+smoothing_window, 2])
            corrected_yaws[i:] -= heading_bias
    return corrected_yaws

gyro_data = np.vstack((wx_filtered, wy_filtered, wz_filtered)).T
corrected_yaws = enhanced_zupt(yaws, stationary_periods_dynamic, gyro_data)

# Trajectory estimation with intentional errors (scaled up)
def estimate_trajectory(yaws, accel_magnitude, timestamps, stationary_periods):
    trajectory = np.zeros((len(yaws), 2))  # X, Y positions
    velocity = np.zeros(2)  # X, Y velocities
    dt = np.diff(timestamps, prepend=timestamps[0])

    for i in range(1, len(yaws)):
        if stationary_periods[i]:
            velocity = np.zeros(2)  # Reset velocity during stationary periods
        else:
            step_length = 0.5 * accel_magnitude[i]  # Dynamic step length estimation
            # Add scaled-up random noise to velocity
            velocity += np.array([np.cos(yaws[i]), np.sin(yaws[i])]) * step_length * dt[i] + np.random.normal(0, 0.5, size=2)
        trajectory[i] = trajectory[i-1] + velocity * dt[i]
    return trajectory

# Generate trajectory with intentional drift
trajectory = estimate_trajectory(corrected_yaws, accel_magnitude, timestamps, stationary_periods_dynamic)

# Plot trajectories
if len(trajectory) > 0 and len(true_positions) > 0:
    min_length = min(len(trajectory), len(true_positions))
    aligned_trajectory = trajectory[:min_length]
    aligned_true_positions = true_positions[:min_length]

    plt.figure(figsize=(8, 6))
    plt.plot(aligned_true_positions[:, 0], aligned_true_positions[:, 1], label="Ground Truth", color="green")
    plt.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], label="Estimated Trajectory", color="red", linestyle="--")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Trajectory vs Ground Truth")
    plt.legend()
    plt.grid()
    plt.show()

    # Compute drift
    drift = np.linalg.norm(aligned_trajectory - aligned_true_positions, axis=1)
    print("Drift at each step (meters):", drift)

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(drift)), drift, label="Drift", color="blue")
    plt.xlabel("Step Index")
    plt.ylabel("Drift (meters)")
    plt.title("Drift Over Steps")
    plt.legend()
    plt.grid()
    plt.show()