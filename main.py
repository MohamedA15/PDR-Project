import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from ahrs.filters import Mahony
from scipy.optimize import least_squares

# File selection
route_options = ["route1.csv", "route2.csv", "route3.csv", "route4.csv", "route4.1.csv", 
                "route5.csv", "route5.1.csv", "route6.csv", "route6.1.csv", "route7.csv", "route8.csv", "route9.csv"]
ground_truth_options = ["ground_truth1.csv", "ground_truth2.csv", "ground_truth3.csv", 
                       "ground_truth4.csv", "ground_truth5.csv", "ground_truth6.csv", "ground_truth7.csv", "ground_truth8.csv", "ground_truth9.csv"]

def select_file(options, name):
    print(f"\nAvailable {name}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input(f"Select {name} (Enter number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

filename = select_file(route_options, "route file")
ground_truth_file = select_file(ground_truth_options, "ground truth file")

# Data loading
required_columns = ['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']
df = pd.read_csv(filename).dropna(subset=required_columns)
df.columns = df.columns.str.strip()

if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns")
if not np.issubdtype(df['time'].dtype, np.number):
    raise ValueError("'time' column is not numeric")

true_positions = pd.read_csv(ground_truth_file).values
timestamps, ax, ay, az, wx, wy, wz = (df[col].values for col in required_columns)

# KALMAN FILTER 
def kalman_filter(data, process_noise=1e-4, measurement_noise=1e-2, adaptive=True):
    # Start with the first data point as the initial state
    state = np.array([data[0]])

    # Initial uncertainty (covariance) in the estimate
    P = np.array([[1.0]])

    # Process noise: how uncertain we are about changes in the system
    Q = np.array([[process_noise]])

    # Measurement noise: how noisy or unreliable the sensor is
    R = np.array([[measurement_noise]])

    # Measurement model H: maps the true state to the measured space (identity for 1D)
    H = np.array([[1.0]])

    # Array to store filtered output
    filtered = np.zeros_like(data)
    filtered[0] = state[0]

    # Parameters for adaptive measurement noise tuning
    window_size = 20
    innovations = []  # Store recent prediction errors

    for i in range(1, len(data)):
        #  Prediction step 
        state_pred = state                  
        P_pred = P + Q                     

        #  Measurement update step 
        z = data[i]                         
        y = (z - H @ state_pred).item()   
        S = H @ P_pred @ H.T + R            
        K = P_pred @ H.T / S               


        innovations.append(y)
        if adaptive and i > window_size:
            recent_innovations = np.array(innovations[-window_size:])
            estimated_R = np.var(recent_innovations)  
            R = np.array([[estimated_R * 0.9]])       

        #  Update the state estimate 
        state = state_pred + K * y                  
        P = (np.eye(1) - K @ H) @ P_pred             

        # Store filtered result
        filtered[i] = state.item()

    return filtered


# Apply Kalman filter to accelerometer and gyroscope data
ax_filtered = kalman_filter(ax)
ay_filtered = kalman_filter(ay)
az_filtered = kalman_filter(az)
wx_filtered = kalman_filter(wx, process_noise=1e-5, measurement_noise=1e-3)
wy_filtered = kalman_filter(wy, process_noise=1e-5, measurement_noise=1e-3)
wz_filtered = kalman_filter(wz, process_noise=1e-5, measurement_noise=1e-3)

# ZERO VELOCITY DETECTION 
def detect_stationary_periods(accel_mag, gyro_mag, accel_thresh=0.2, gyro_thresh=0.1, min_duration=10, window_size=15):
    # Smooth the squared acceleration and gyroscope magnitude using a moving average
    accel_var = np.convolve(accel_mag**2, np.ones(window_size) / window_size, mode='same')
    gyro_var = np.convolve(gyro_mag**2, np.ones(window_size) / window_size, mode='same')

    # Initialize the output array for stationary detection
    stationary = np.zeros_like(accel_mag, dtype=bool)

    # Variables to track when a stationary segment starts and whether we're currently in one
    in_stationary = False
    stationary_start = 0

    # Iterate through all time steps to detect low-motion segments
    for i in range(len(accel_mag)):
        if not in_stationary:
            # If motion is below both thresholds, start a potential stationary period
            if accel_var[i] < accel_thresh and gyro_var[i] < gyro_thresh:
                in_stationary = True
                stationary_start = i
        else:
            # If motion goes above 1.5× threshold, the stationary period ends
            if accel_var[i] > accel_thresh * 1.5 or gyro_var[i] > gyro_thresh * 1.5:
                # Mark the segment as stationary only if it was long enough
                if (i - stationary_start) >= min_duration:
                    stationary[stationary_start:i] = True
                in_stationary = False

    # If we end the loop while still in a stationary segment, finalize it
    if in_stationary and (len(accel_mag) - stationary_start) >= min_duration:
        stationary[stationary_start:] = True

    return stationary


accel_magnitude = np.linalg.norm([ax_filtered, ay_filtered, az_filtered], axis=0)
gyro_magnitude = np.linalg.norm([wx_filtered, wy_filtered, wz_filtered], axis=0)
stationary = detect_stationary_periods(accel_magnitude, gyro_magnitude)

# STEP DETECTION 
def detect_steps(accel_mag, stationary, min_peak_height =1.2, min_peak_distance=20):
    # Create dynamic threshold based on moving average + variance bump
    window = 100
    threshold = np.convolve(accel_mag, np.ones(window)/window, mode='same') + 0.5 * np.std(accel_mag)

    # Peak detection to identify step impacts
    steps, _ = signal.find_peaks(accel_mag, height=threshold, distance=min_peak_distance)

    # Remove any steps during stationary moments
    return np.array([step for step in steps if not stationary[step]])

steps = detect_steps(accel_magnitude, stationary)

# MAHONY FILTER FOR HEADING ESTIMATION
mahony = Mahony(gain=0.1, frequency=1/np.mean(np.diff(timestamps)))
quaternions = np.zeros((len(ax_filtered), 4))
quaternions[0] = [1, 0, 0, 0]
yaws = []

for i in range(1, len(ax_filtered)):
    quaternions[i] = mahony.updateIMU(
        q=quaternions[i-1],
        gyr=[wx_filtered[i], wy_filtered[i], wz_filtered[i]],
        acc=[ax_filtered[i], ay_filtered[i], az_filtered[i]]
    )
    # Extract yaw (rotation around vertical axis)
    yaw = np.arctan2(
        2*(quaternions[i][0]*quaternions[i][3] + quaternions[i][1]*quaternions[i][2]),
        1 - 2*(quaternions[i][2]**2 + quaternions[i][3]**2)
    )
    yaws.append(yaw)

# Smooth the heading to avoid jitter
yaws = [0.0] + yaws
heading = np.unwrap(yaws)
alpha = 0.1
smoothed_heading = np.zeros_like(heading)
smoothed_heading[0] = heading[0]
for i in range(1, len(heading)):
    smoothed_heading[i] = alpha * heading[i] + (1 - alpha) * smoothed_heading[i-1]

# ZUPT TRAJECTORY ESTIMATION
def estimate_trajectory_with_zupt(steps, heading, accel_mag, stationary, timestamps):
    velocity = np.zeros(len(timestamps))
    position = np.zeros((len(timestamps), 2))
    
    base_step_length = 0.7
    scaling_factor = 0.2

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]

        if stationary[i]:
            velocity[i] = 0.9 * velocity[i-1]  # slow it down during stationary
        elif i in steps:
            step_length = base_step_length + scaling_factor * (accel_mag[i] - np.mean(accel_mag))
            velocity[i] = step_length / dt
        else:
            velocity[i] = velocity[i-1]

        step_vector = velocity[i] * dt * np.array([np.cos(heading[i]), np.sin(heading[i])])
        position[i] = position[i-1] + step_vector

    return position[steps]

trajectory = estimate_trajectory_with_zupt(steps, smoothed_heading, accel_magnitude, stationary, timestamps)

# LEAST-SQUARES DRIFT CORRECTION
def correct_drift(estimated, ground_truth):
    def error(params):
        scale, theta, dx, dy = params
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        transformed = scale * estimated @ R + np.array([dx, dy])
        return np.linalg.norm(transformed - ground_truth[:len(transformed)], axis=1)

    initial_params = [1.0, 0.0, 0.0, 0.0]
    result = least_squares(error, initial_params)
    scale, theta, dx, dy = result.x

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return scale * estimated @ R + np.array([dx, dy])

min_length = min(len(trajectory), len(true_positions))
corrected_trajectory = correct_drift(trajectory[:min_length], true_positions[:min_length])

# Align trajectories
min_length = min(len(trajectory), len(true_positions))
corrected_trajectory = correct_drift(trajectory[:min_length], true_positions[:min_length])

# EVALUATION METRICS 
def calculate_rmse(true, pred):
    return np.sqrt(np.mean(np.linalg.norm(true - pred, axis=1)**2))

def calculate_mae(true, pred):
    return np.mean(np.linalg.norm(true - pred, axis=1))

rmse = calculate_rmse(true_positions[:min_length], corrected_trajectory)
mae = calculate_mae(true_positions[:min_length], corrected_trajectory)
drift = np.linalg.norm(corrected_trajectory - true_positions[:min_length], axis=1)

# Visulisation

# Plot  Noise Filtering Comparison
# Visulisation 
plt.figure(figsize=(12, 6))
plt.plot(np.linalg.norm([ax, ay, az], axis=0), label="Raw Acceleration", linestyle='dotted', color='gray')
plt.plot(accel_magnitude, label="Kalman Filtered", color='blue')
plt.title("Comparison of Raw and Kalman Filtered Acceleration Data")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration Magnitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot  Step Detection
plt.figure(figsize=(12, 6))
plt.plot(accel_magnitude, label="Acceleration")
plt.scatter(steps, accel_magnitude[steps], color='red', label="Steps")
plt.scatter(np.where(stationary), accel_magnitude[stationary], color='green', marker='x', label="ZUPT")
plt.title("Step Detection with ZUPT")
plt.xlabel("Samples")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)

# Plot Trajectory Comparison
plt.figure(figsize=(12, 6))
plt.plot(true_positions[:min_length, 0], true_positions[:min_length, 1], label="Ground Truth", linewidth=2)
plt.plot(corrected_trajectory[:, 0], corrected_trajectory[:, 1], '--', label="Estimated")
plt.title(f"Trajectory Comparison\nRMSE: {rmse:.2f}m, MAE: {mae:.2f}m")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 4 Drift Over Time
plt.figure(figsize=(12, 6))
plt.plot(drift)
plt.title("Drift per interval")
plt.xlabel("Interval Number")
plt.ylabel("Error (m)")
plt.grid(True)

plt.tight_layout()
plt.show()

# PERFORMANCE METRICS
print("\n PERFORMANCE METRICS ")
print(f"RMSE: {rmse:.3f} meters")
print(f"MAE: {mae:.3f} meters")
print(f"Max Drift: {np.max(drift):.3f} meters")
print(f"Average Drift: {np.mean(drift):.3f} meters")
print(f"Steps Detected: {len(steps)}")
print(f"Stationary Periods: {np.sum(stationary)} samples")

# **Print Drift Results**
print("\nTotal Drift (Cumulative Error at Each Interval):")
for i, d in enumerate(drift):
    print(f"Interval {i+1}:  {d:.3f} meters")

print("\nDrift Change Between Intervals:")
drift_intervals = np.diff(drift)
for i, d in enumerate(drift_intervals):
    print(f"Interval {i+1} → Interval {i+2}:  {d:.3f} meters")

print("\n NOISE REDUCTION ")
for axis, raw, filt in zip(['AX', 'AY', 'AZ'], [ax, ay, az], [ax_filtered, ay_filtered, az_filtered]):
    var_red = 100 * (np.var(raw) - np.var(filt)) / np.var(raw)
    print(f"{axis}: Variance Reduction = {var_red:.1f}%")

print("\n Noise Filtering Evaluation (Variance Reduction) ")

def evaluate_noise_reduction(raw_data, filtered_data):
    raw_var = np.var(raw_data)
    filtered_var = np.var(filtered_data)
    reduction = 100 * (raw_var - filtered_var) / raw_var
    return raw_var, filtered_var, reduction

# Accelerometer axes
print("\n[ Accelerometer Axes ]")
for axis_name, raw, filtered in zip(['AX', 'AY', 'AZ'], [ax, ay, az], [ax_filtered, ay_filtered, az_filtered]):
    raw_var, filtered_var, reduction = evaluate_noise_reduction(raw, filtered)
    print(f"{axis_name} - Raw Var: {raw_var:.6f}, Filtered Var: {filtered_var:.6f}, Reduction: {reduction:.2f}%")

# Gyroscope axes
print("\n[ Gyroscope Axes ]")
for axis_name, raw, filtered in zip(['WX', 'WY', 'WZ'], [wx, wy, wz], [wx_filtered, wy_filtered, wz_filtered]):
    raw_var, filtered_var, reduction = evaluate_noise_reduction(raw, filtered)
    print(f"{axis_name} - Raw Var: {raw_var:.6f}, Filtered Var: {filtered_var:.6f}, Reduction: {reduction:.2f}%")

print("\n- Heading Evaluation (Mahony Filter Stability) ")

def heading_stability_metric(heading_data):
    changes = np.abs(np.diff(heading_data))
    return np.mean(changes)

heading_mac = heading_stability_metric(heading)
print(f"Heading Stability (Mean Absolute Change): {heading_mac:.6f} radians")

print("\n Step Detection Evaluation ")

def evaluate_step_detection(steps, timestamps):
    step_times = timestamps[steps]
    intervals = np.diff(step_times)
    step_freq = len(steps) / (timestamps[-1] - timestamps[0])
    return step_freq, np.mean(intervals), np.std(intervals)

step_freq, avg_interval, std_interval = evaluate_step_detection(steps, timestamps)
print(f"Total Steps Detected: {len(steps)}")
print(f"Step Frequency: {step_freq:.4f} steps/sec")
print(f"Average Step Interval: {avg_interval:.4f} sec")
print(f"Step Interval Std Dev: {std_interval:.4f} sec")
print(f"Total IMU samples: {len(ax)}")
duration_sec = timestamps[-1] - timestamps[0]
print(f"Recording duration: {duration_sec:.2f} seconds")
