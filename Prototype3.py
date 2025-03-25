import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import imufusion  # IMU fusion library
from scipy.signal import butter, filtfilt

# List of available routes and ground truth files
route_options = [
    "route1.csv", "route2.csv", "route3.csv", "route4.csv",
    "route4.1.csv", "route5.csv", "route5.1.csv",
    "route6.csv", "route6.1.csv"
]
ground_truth_options = [
    "ground_truth1.csv", "ground_truth2.csv", "ground_truth3.csv",
    "ground_truth4.csv", "ground_truth5.csv", "ground_truth6.csv"
]

# Function to select files
def select_file(options, name):
    print(f"\nAvailable {name}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    while True:
        try:
            choice = int(input(f"Select {name} (Enter number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Select route and ground truth files
filename = select_file(route_options, "route file")
ground_truth_file = select_file(ground_truth_options, "ground truth file")

# Load dataset
required_columns = ['time', 'ax', 'ay', 'az', 'wz']
df = pd.read_csv(filename).dropna(subset=required_columns)
df.columns = df.columns.str.strip()

# Validate data
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns")
if not np.issubdtype(df['time'].dtype, np.number):
    raise ValueError("'time' column is not numeric")

# Load ground truth positions
true_positions = pd.read_csv(ground_truth_file).values

# Extract necessary columns
timestamps, x_axis, y_axis, z_axis, z_gyro = (df[col].values for col in required_columns)

# **Filtering Functions**
def butter_lowpass_filter(data, cutoff=0.25, fs=30, order=8):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Apply filtering
x_lowpass = butter_lowpass_filter(x_axis)
y_lowpass = butter_lowpass_filter(y_axis)
z_lowpass = butter_lowpass_filter(z_axis)
z_gyro_filtered = butter_lowpass_filter(z_gyro)

# Compute acceleration magnitudes
accel_magnitude = np.linalg.norm([x_lowpass, y_lowpass, z_lowpass], axis=0)

#  Zero Velocity Update (ZUPT)
def detect_stationary_periods(accel_magnitude, gyro, window_size=10, accel_alpha=0.1, gyro_alpha=0.1):
    """
    Detect stationary periods using adaptive thresholding and smoothing.
    
    Parameters:
        accel_magnitude (np.array): Magnitude of acceleration.
        gyro (np.array): Gyroscope data (z-axis).
        window_size (int): Size of the moving average window.
        accel_alpha (float): Exponential smoothing factor for acceleration threshold.
        gyro_alpha (float): Exponential smoothing factor for gyroscope threshold.
    
    Returns:
        np.array: Boolean array indicating stationary periods.
    """
    # Adaptive thresholds for acceleration and gyroscope
    accel_threshold = np.mean(accel_magnitude) - accel_alpha * np.std(accel_magnitude)
    gyro_threshold = np.mean(np.abs(gyro)) - gyro_alpha * np.std(np.abs(gyro))

    # Detect stationary periods
    stationary = (accel_magnitude < accel_threshold) & (np.abs(gyro) < gyro_threshold)
    
    # Smooth the stationary periods using a moving average filter
    smoothed_stationary = np.convolve(stationary, np.ones(window_size) / window_size, mode='same') > 0.5
    return smoothed_stationary

stationary_periods = detect_stationary_periods(accel_magnitude, z_gyro_filtered)

# Step detection with adaptive thresholding
step_threshold = np.mean(accel_magnitude) + 0.4 * np.std(accel_magnitude)
steps, _ = signal.find_peaks(accel_magnitude, height=step_threshold, distance=8)

# Remove steps detected during stationary periods
steps = np.array([step for step in steps if not stationary_periods[step]])

#  IMU Fusion Heading Estimation**
ahrs = imufusion.Ahrs()
yaws = []

for i in range(len(z_gyro_filtered)):
    # Gyroscope data
    gyro_sample = np.array([0, 0, z_gyro_filtered[i]])
    
    # Accelerometer data
    accel_sample = np.array([x_lowpass[i], y_lowpass[i], z_lowpass[i]])
    
    # Update AHRS (without magnetometer)
    ahrs.update_no_magnetometer(gyro_sample, accel_sample, 1 / 100)
    
    # Extract yaw from quaternion
    _, _, yaw = ahrs.quaternion.to_euler()
    yaws.append(yaw)

# Recursive smoothing for heading stabilization**
def recursive_smoothing(data, alpha=0.15):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

heading = np.unwrap(yaws)
heading = recursive_smoothing(heading)

# Motion vector calculation**
motion_vectors = np.array([np.cos(heading), np.sin(heading)]).T

# **Step length estimation**
step_lengths = np.clip(20 + np.abs(z_lowpass[steps]) * 3.5, 18, 24)

# **Position tracking**
cur_position = np.array([0.0, 0.0])
trajectory = [cur_position]

for i, step in enumerate(steps):
    step_length = step_lengths[i]
    if stationary_periods[step]:  
        step_length *= 0.1  # Reduce step length during stationary periods
    
    cur_position = cur_position + step_length * motion_vectors[step]
    trajectory.append(cur_position)

trajectory = np.array(trajectory)

# **Compute drift at each step**
min_length = min(len(trajectory), len(true_positions))
aligned_trajectory = trajectory[:min_length]
aligned_truth = true_positions[:min_length]

drift = np.linalg.norm(aligned_trajectory - aligned_truth, axis=1)

# Plots

# Noise Filtering Graph
plt.figure(figsize=(12, 6))

# Plot raw acceleration
plt.plot(accel_magnitude, label="Raw Acceleration", linestyle='dotted', color='gray', alpha=0.7)

# Plot low-pass filtered acceleration
plt.plot(x_lowpass, label="Low-Pass Filtered", color='blue', linewidth=2)

# Plot high-pass filtered acceleration
plt.plot(z_gyro_filtered, label="High-Pass Filtered", color='red', linewidth=2)

# Add labels and title
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Acceleration Magnitude", fontsize=12)
plt.title("Comparison of Low-Pass and High-Pass Filtering Effects on Acceleration Data", fontsize=14)

# Add legend
plt.legend(loc='upper right', fontsize=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()

# ZUPT Step Detection Graph
plt.figure(figsize=(10, 5))
plt.plot(accel_magnitude, label="Filtered Acceleration", color='blue', linewidth=2)
plt.scatter(steps, accel_magnitude[steps], color='red', label="Detected Steps", s=50)
plt.scatter(np.where(stationary_periods), accel_magnitude[stationary_periods], color='purple', marker='x', label="Stationary Periods", s=50)
plt.xlabel("Time Steps")
plt.ylabel("Acceleration Magnitude")
plt.title("ZUPT Detection with Step Marking")
plt.legend()
plt.grid(True)
plt.show()

# Drift Per Interval Graph
plt.figure(figsize=(10, 5))
plt.plot(drift, label="Drift per Interval", color='blue', linewidth=2)
plt.xlabel("Step Interval")
plt.ylabel("Drift (meters)")
plt.title("Drift Per Interval")
plt.legend()
plt.grid(True)
plt.show()

# Estimated vs Ground Truth Trajectory Graph
plt.figure(figsize=(10, 5))
plt.plot(aligned_truth[:, 0], aligned_truth[:, 1], label="Ground Truth", color="green", linewidth=2)
plt.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], label="Estimated Trajectory", color="red", linestyle="--", linewidth=2)
plt.legend()
plt.title("Estimated vs Ground Truth Trajectory")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.grid(True)
plt.show()

# **Print Drift Results**
print("\nTotal Drift (Cumulative Error at Each Interval):")
for i, d in enumerate(drift):
    print(f"Interval {i+1}:  {d:.3f} meters")

print("\nDrift Change Between Intervals:")
drift_intervals = np.diff(drift)
for i, d in enumerate(drift_intervals):
    change_label = "Increase" if d > 0 else "Correction detected"
    print(f"Interval {i+1} → Interval {i+2}:  {d:.3f} meters ({change_label})")