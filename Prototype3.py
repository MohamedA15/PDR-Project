import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import imufusion
from scipy.signal import butter, filtfilt
from calculate_rmse import calculate_rmse  # Import RMSE function

# Function to Get Valid User Input
def get_valid_input(prompt, valid_range):
    """Ensures user enters a valid number within the given range"""
    while True:
        try:
            choice = int(input(prompt))
            if choice in valid_range:
                return choice
            else:
                print(f"Invalid choice! Please enter a number between {valid_range[0]} and {valid_range[-1]}.")
        except ValueError:
            print("Invalid input! Please enter a valid number.")

# User Input: Select Route & Ground Truth
route_choice = get_valid_input("Select a route file (1/2/3): ", [1, 2, 3])
filename = f"route{route_choice}.csv"

ground_truth_choice = get_valid_input("Select a ground truth file (1/2/3): ", [1, 2, 3])
ground_truth_file = f"ground_truth{ground_truth_choice}.csv"

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

# Apply Low-Pass Filtering
x_axis_lowpass = butter_lowpass_filter(x_axis)
y_axis_lowpass = butter_lowpass_filter(y_axis)
z_axis_lowpass = butter_lowpass_filter(z_axis)
z_gyro_lowpass = butter_lowpass_filter(z_gyro)

# Compute Acceleration Magnitude
accel_magnitude = np.linalg.norm([x_axis_lowpass, y_axis_lowpass, z_axis_lowpass], axis=0)

# ZUPT: Detect Stationary Periods
def detect_stationary_periods(accel_magnitude, gyro, accel_threshold=0.1, gyro_threshold=0.1):
    stationary = (accel_magnitude < accel_threshold) & (np.abs(gyro) < gyro_threshold)
    return stationary

stationary_periods = detect_stationary_periods(accel_magnitude, z_gyro_lowpass)

# Step Detection with ZUPT Filter
step_threshold = 0.2
steps, _ = signal.find_peaks(accel_magnitude, height=step_threshold, distance=5)

# Filter Steps Based on ZUPT
steps = [step for step in steps if not stationary_periods[step]]

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

# Compute RMSE and Display
rmse = calculate_rmse("estimated_trajectory.csv", "ground_truth.csv")
print(f"RMSE: {rmse:.3f} meters")

# Plots
plt.figure(figsize=(10, 5))
plt.plot(accel_magnitude, label="Filtered Acceleration Magnitude")
plt.plot(steps, accel_magnitude[steps], "x", label="Detected Steps", color="red")
plt.legend()
plt.title("Step Detection After ZUPT Filtering")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration Magnitude")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(drift, label="Drift", color="blue")
plt.legend()
plt.title("Accumulated Drift Over Time")
plt.xlabel("Step Interval")
plt.ylabel("Drift (meters)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(aligned_truth[:, 0], aligned_truth[:, 1], label="Ground Truth", color="green")
plt.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], label="Estimated Trajectory", color="red", linestyle="--")
plt.legend()
plt.title("Estimated vs True Trajectory (PDR vs Ground Truth)")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.grid(True)
plt.show()