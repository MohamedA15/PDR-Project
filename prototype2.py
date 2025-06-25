import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import imufusion 
from scipy.signal import butter, filtfilt

# Route and ground truth options
route_options = [
    "route1.csv", "route2.csv", "route3.csv", "route4.csv",
    "route4.1.csv", "route5.csv", "route5.1.csv",
    "route6.csv", "route6.1.csv", "route7.csv", "route8.csv", "route9.csv"
]

ground_truth_options = [
    "ground_truth1.csv", "ground_truth2.csv", "ground_truth3.csv",
    "ground_truth4.csv", "ground_truth5.csv", "ground_truth6.csv", "ground_truth7.csv", "ground_truth8.csv", "ground_truth9.csv"
]

# Function to Get Valid User Input
def get_valid_input(prompt, valid_range):
    """Ensures user enters a valid number within the given range"""
    while True:
        try:
            choice = int(input(prompt))
            if 1 <= choice <= len(valid_range):
                return choice
            else:
                print(f"Invalid choice! Please enter a number between 1 and {len(valid_range)}.")
        except ValueError:
            print("Invalid input! Please enter a valid number.")

# User Input: Select Route & Ground Truth
print("Available route files:")
for i, route in enumerate(route_options, 1):
    print(f"{i}. {route}")
route_choice = get_valid_input("Select a route file: ", range(1, len(route_options)+1))
filename = route_options[route_choice-1]

print("\nAvailable ground truth files:")
for i, truth in enumerate(ground_truth_options, 1):
    print(f"{i}. {truth}")
ground_truth_choice = get_valid_input("Select a ground truth file: ", range(1, len(ground_truth_options)+1))
ground_truth_file = ground_truth_options[ground_truth_choice-1]

required_columns = ['time', 'ax', 'ay', 'az', 'wz']

# Load and Preprocess Data
df = pd.read_csv(filename).dropna(subset=required_columns)
df.columns = df.columns.str.strip()
if not all(col in df.columns for col in required_columns):
    raise ValueError("Error: Missing required columns")
if not np.issubdtype(df['time'].dtype, np.number):
    raise ValueError("Error: 'time' column is not numeric")

true_positions = pd.read_csv(ground_truth_file).values

timestamps, x_axis, y_axis, z_axis, z_gyro = (
    df[col].values for col in required_columns)

# Noise Filtering Functions
def butter_lowpass_filter(data, cutoff=5, fs=30, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff=0.1, fs=30, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Apply Low-Pass and High-Pass Filtering
x_axis_lowpass = butter_lowpass_filter(x_axis)
y_axis_lowpass = butter_lowpass_filter(y_axis)
z_axis_lowpass = butter_lowpass_filter(z_axis)
z_gyro_lowpass = butter_lowpass_filter(z_gyro)

x_axis_highpass = butter_highpass_filter(x_axis)
y_axis_highpass = butter_highpass_filter(y_axis)
z_axis_highpass = butter_highpass_filter(z_axis)
z_gyro_highpass = butter_highpass_filter(z_gyro)

# Compute Acceleration Magnitude
accel_magnitude = np.linalg.norm([x_axis_lowpass, y_axis_lowpass, z_axis_lowpass], axis=0)

# Step Detection
step_threshold = 0.2  # Threshold for detecting steps
steps, _ = signal.find_peaks(accel_magnitude, height=step_threshold, distance=5)

# Stationary detection (simple threshold-based)
stationary = accel_magnitude < 0.1  # Adjust threshold as needed

# VISUALIZATIONS 

# Visualize Noise Filtering Effects
plt.figure(figsize=(12, 10))

# Low-Pass Filter Plot
plt.subplot(3, 1, 1)
plt.plot(x_axis, label="Raw X-Axis", alpha=0.5)
plt.plot(x_axis_lowpass, label="Low-Pass Filtered X-Axis", linestyle="--")
plt.legend()
plt.title("Low-Pass Filtering Effect on Accelerometer Data")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration")
plt.grid(True)

# High-Pass Filter Plot
plt.subplot(3, 1, 2)
plt.plot(x_axis, label="Raw X-Axis", alpha=0.5)
plt.plot(x_axis_highpass, label="High-Pass Filtered X-Axis", linestyle="--")
plt.legend()
plt.title("High-Pass Filtering Effect on Accelerometer Data")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration")
plt.grid(True)

# Comparison of Low-Pass & High-Pass Filtering
plt.subplot(3, 1, 3)
plt.plot(x_axis_lowpass, label="Low-Pass Filtered")
plt.plot(x_axis_highpass, label="High-Pass Filtered")
plt.legend()
plt.title("Comparison of Low-Pass & High-Pass Filtering")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration")
plt.grid(True)

plt.tight_layout()
plt.show()

# Step Detection Plot
plt.figure(figsize=(10, 5))
plt.plot(accel_magnitude, label="Filtered Acceleration Magnitude")
plt.plot(steps, accel_magnitude[steps], "x", label="Detected Steps", color="red")
plt.legend()
plt.title("Step Detection After Noise Filtering")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration Magnitude")
plt.grid(True)
plt.show()

# AHRS Initialization (IMUFUSION)
print("Initializing AHRS...")
ahrs = imufusion.Ahrs()
ahrs.settings = imufusion.Settings(
    imufusion.CONVENTION_NED, 0.5, 1000, 10, 30, 1000
)

# Compute Orientation Using AHRS
yaws = []
for i in range(len(z_gyro_lowpass)):  
    gyro_sample = np.array([0, 0, z_gyro_lowpass[i]], dtype=float)
    accel_sample = np.array([x_axis_lowpass[i], y_axis_lowpass[i], z_axis_lowpass[i]], dtype=float)
    
    ahrs.update_no_magnetometer(gyro_sample, accel_sample, 1 / 100)
    _, _, yaw = ahrs.quaternion.to_euler()
    
    yaws.append(yaw)

# Compute Heading
def calculate_heading(yaws):
    return np.unwrap(yaws)

heading = calculate_heading(yaws)
motion_vectors = np.array([np.cos(heading), np.sin(heading)]).T

# Compute Trajectory Using Step Detection
step_length, cur_position = 30.0, np.array([0.0, 0.0])
trajectory = [cur_position]

for step in steps:
    cur_position = cur_position + step_length * motion_vectors[step]
    trajectory.append(cur_position)

trajectory = np.array(trajectory)

# Compare with Ground Truth
min_length = min(len(trajectory), len(true_positions))
aligned_trajectory, aligned_truth = trajectory[:min_length], true_positions[:min_length]
drift = np.linalg.norm(aligned_trajectory - aligned_truth, axis=1)

# Drift Plot
plt.figure(figsize=(10, 5))
plt.plot(drift, label="Drift", color="blue")
plt.legend()
plt.title("Accumulated Drift Over Time")
plt.xlabel("Step Interval")
plt.ylabel("Drift (meters)")
plt.grid(True)
plt.show()

# Trajectory Comparison Plot
plt.figure(figsize=(10, 5))
plt.plot(aligned_truth[:, 0], aligned_truth[:, 1], label="Ground Truth", color="green")
plt.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], label="Estimated Trajectory", color="red", linestyle="--")
plt.legend()
plt.title("Estimated vs True Trajectory (PDR vs Ground Truth)")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.grid(True)
plt.show()

# PERFORMANCE METRICS 

def calculate_rmse(estimated, ground_truth):
    """Calculate Root Mean Square Error between two trajectories"""
    return np.sqrt(np.mean(np.sum((estimated - ground_truth)**2, axis=1)))

def calculate_mae(estimated, ground_truth):
    """Calculate Mean Absolute Error between two trajectories"""
    return np.mean(np.linalg.norm(estimated - ground_truth, axis=1))

rmse = calculate_rmse(aligned_trajectory, aligned_truth)
mae = calculate_mae(aligned_trajectory, aligned_truth)

def evaluate_noise_reduction(raw_data, filtered_data):
    raw_var = np.var(raw_data)
    filtered_var = np.var(filtered_data)
    reduction = 100 * (raw_var - filtered_var) / raw_var
    return raw_var, filtered_var, reduction

def heading_stability_metric(heading_data):
    changes = np.abs(np.diff(heading_data))
    return np.mean(changes)

def evaluate_step_detection(steps, timestamps):
    step_times = timestamps[steps]
    intervals = np.diff(step_times)
    step_freq = len(steps) / (timestamps[-1] - timestamps[0])
    return step_freq, np.mean(intervals), np.std(intervals)

heading_mac = heading_stability_metric(heading)
step_freq, avg_interval, std_interval = evaluate_step_detection(steps, timestamps)
duration_sec = timestamps[-1] - timestamps[0]

print("\n=== PERFORMANCE METRICS ===")
print(f"RMSE: {rmse:.3f} meters")
print(f"MAE: {mae:.3f} meters")
print(f"Max Drift: {np.max(drift):.3f} meters")
print(f"Average Drift: {np.mean(drift):.3f} meters")
print(f"Steps Detected: {len(steps)}")
print(f"Stationary Periods: {np.sum(stationary)} samples")

print("\n=== NOISE REDUCTION ===")
for axis, raw, filt in zip(['AX', 'AY', 'AZ'], [x_axis, y_axis, z_axis], [x_axis_lowpass, y_axis_lowpass, z_axis_lowpass]):
    var_red = 100 * (np.var(raw) - np.var(filt)) / np.var(raw)
    print(f"{axis}: Variance Reduction = {var_red:.1f}%")

print("\n--- Noise Filtering Evaluation (Variance Reduction) ---")
print("\n[ Accelerometer Axes ]")
for axis_name, raw, filtered in zip(['AX', 'AY', 'AZ'], [x_axis, y_axis, z_axis], [x_axis_lowpass, y_axis_lowpass, z_axis_lowpass]):
    raw_var, filtered_var, reduction = evaluate_noise_reduction(raw, filtered)
    print(f"{axis_name} - Raw Var: {raw_var:.6f}, Filtered Var: {filtered_var:.6f}, Reduction: {reduction:.2f}%")

print("\n[ Gyroscope Axes ]")
for axis_name, raw, filtered in zip(['WZ'], [z_gyro], [z_gyro_lowpass]):
    raw_var, filtered_var, reduction = evaluate_noise_reduction(raw, filtered)
    print(f"{axis_name} - Raw Var: {raw_var:.6f}, Filtered Var: {filtered_var:.6f}, Reduction: {reduction:.2f}%")

print("\n Heading Evaluation ")
print(f"Heading Stability (Mean Absolute Change): {heading_mac:.6f} radians")

print("\n--- Step Detection Evaluation ---")
print(f"Total Steps Detected: {len(steps)}")
print(f"Step Frequency: {step_freq:.4f} steps/sec")
print(f"Average Step Interval: {avg_interval:.4f} sec")
print(f"Step Interval Std Dev: {std_interval:.4f} sec")
print(f"Total IMU samples: {len(x_axis)}")
print(f"Recording duration: {duration_sec:.2f} seconds")

print("\nDrift at each interval (meters):")
for i, d in enumerate(drift):
    print(f"Interval {i+1}: {d:.3f} meters")

print("\nDrift Change Between Intervals:")
drift_intervals = np.diff(drift)
for i, d in enumerate(drift_intervals):
    change_label = "Increase" if d > 0 else "Decrease"
    print(f"Interval {i+1} → Interval {i+2}: {d:.3f} meters ({change_label})")