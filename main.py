import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from ahrs.filters import Mahony  
from scipy.optimize import least_squares

# The list of available routes and ground truth files to select
route_options = ["route1.csv", "route2.csv", "route3.csv", "route4.csv", "route4.1.csv", "route5.csv", "route5.1.csv", "route6.csv", "route6.1.csv"]
ground_truth_options = ["ground_truth1.csv", "ground_truth2.csv", "ground_truth3.csv", "ground_truth4.csv", "ground_truth5.csv", "ground_truth6.csv"]

# pick a file
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

# Select the route and ground truth files
filename = select_file(route_options, "route file")
ground_truth_file = select_file(ground_truth_options, "ground truth file")

# Load dataset
required_columns = ['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']
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
timestamps, ax, ay, az, wx, wy, wz = (df[col].values for col in required_columns)

# Kalman Filter for Noise Reduction
def kalman_filter_noise_reduction(data, process_noise=0.01, measurement_noise=0.1):
    """
    Apply a 1D Kalman filter to reduce noise in the sensor data.
    """
    # Initialise Kalman filter state and covariance
    state = np.array([data[0]])  
    P = np.array([[1.0]])
    Q = np.array([[process_noise]]) 
    R = np.array([[measurement_noise]]) 
    H = np.array([[1.0]]) 

    filtered_data = np.zeros_like(data)
    filtered_data[0] = state[0]

    for i in range(1, len(data)):
        # Predict the steps
        state_pred = state  # stay constant
        P_pred = P + Q

        # Update the steps 
        z = data[i]  
        y = z - H @ state_pred 
        S = H @ P_pred @ H.T + R 
        K = P_pred @ H.T / S  # Kalman gain

        state = state_pred + K * y  # Updated the state
        P = (np.eye(1) - K @ H) @ P_pred  # Updated the covariance

        filtered_data[i] = state[0]

    return filtered_data

# Apply Kalman filter to sensor data
ax_filtered = kalman_filter_noise_reduction(ax)
ay_filtered = kalman_filter_noise_reduction(ay)
az_filtered = kalman_filter_noise_reduction(az)
wx_filtered = kalman_filter_noise_reduction(wx)
wy_filtered = kalman_filter_noise_reduction(wy)
wz_filtered = kalman_filter_noise_reduction(wz)

# Compute the acceleration magnitudes
accel_magnitude = np.linalg.norm([ax_filtered, ay_filtered, az_filtered], axis=0)

# ZUPT Detection
def detect_stationary_periods(accel_magnitude, gyro, window_size=10, accel_alpha=0.1, gyro_alpha=0.1, hysteresis=0.1):
    """
    Detect stationary periods using adaptive thresholding
    """
    # Moving variance for acceleration and gyroscope
    accel_var = np.convolve(accel_magnitude ** 2, np.ones(window_size) / window_size, mode='same')
    gyro_var = np.convolve(gyro ** 2, np.ones(window_size) / window_size, mode='same')
    
    # Adaptive thresholds with hysteresis
    accel_threshold = np.mean(accel_var) - accel_alpha * np.std(accel_var)
    gyro_threshold = np.mean(gyro_var) - gyro_alpha * np.std(gyro_var)
    
    # Detect stationary periods with hysteresis
    stationary = np.zeros_like(accel_magnitude, dtype=bool)
    for i in range(1, len(accel_magnitude)):
        if (accel_var[i] < accel_threshold + hysteresis) and (gyro_var[i] < gyro_threshold + hysteresis):
            stationary[i] = True
        elif (accel_var[i] > accel_threshold - hysteresis) or (gyro_var[i] > gyro_threshold - hysteresis):
            stationary[i] = False
        else:
            stationary[i] = stationary[i - 1]  # Maintain previous state
    
    return stationary

stationary_periods = detect_stationary_periods(accel_magnitude, wz_filtered)

#  Step Detection
def detect_steps(accel_magnitude, stationary_periods, window_size=10, step_alpha=0.5, min_step_interval=10):
    """
    Detect steps using dynamic thresholding and minimum step interval.
    """
    # Dynamic threshold based on local variance
    local_var = np.convolve(accel_magnitude ** 2, np.ones(window_size) / window_size, mode='same')
    step_threshold = np.mean(local_var) + step_alpha * np.std(local_var)
    
    # Find peaks (steps)
    steps, _ = signal.find_peaks(accel_magnitude, height=step_threshold, distance=min_step_interval)
    
    # Remove steps detected during stationary periods
    steps = np.array([step for step in steps if not stationary_periods[step]])
    return steps

steps = detect_steps(accel_magnitude, stationary_periods)

# Mahony Filter for Orientation Estimation
mahony = Mahony(gain=0.1, frequency=30)  # Adjust frequency as needed
quaternions = np.zeros((len(ax_filtered), 4))
quaternions[0] = [1, 0, 0, 0]  # Initialize quaternion
yaws = []

for i in range(1, len(ax_filtered)):
    quaternions[i] = mahony.updateIMU(q=quaternions[i - 1],
                                      gyr=[wx_filtered[i], wy_filtered[i], wz_filtered[i]],
                                      acc=[ax_filtered[i], ay_filtered[i], az_filtered[i]])
    # Extract yaw from quaternion
    yaw = np.arctan2(2 * (quaternions[i][0] * quaternions[i][3] + quaternions[i][1] * quaternions[i][2]),
                      1 - 2 * (quaternions[i][2] ** 2 + quaternions[i][3] ** 2))
    yaws.append(yaw)

# Recursive smoothing for heading stabilization
def recursive_smoothing(data, alpha=0.15):
    """
    Apply recursive smoothing to stabilize the heading.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

heading = np.unwrap(yaws)
heading = recursive_smoothing(heading)

#Trajectory Estimation**
cur_position = np.array([0.0, 0.0])
trajectory = [cur_position]

for i, step in enumerate(steps):
    step_length = 0.75 + 0.2 * (accel_magnitude[step] - np.mean(accel_magnitude))  # Dynamic step length
    step_vector = step_length * np.array([np.cos(heading[step]), np.sin(heading[step])])
    
    # Apply ZUPT during stationary periods
    zupt = stationary_periods[step]
    
    # Update current position
    cur_position = cur_position + step_vector
    trajectory.append(cur_position)

trajectory = np.array(trajectory)

# Drift Correction**
def correct_drift(estimated_trajectory, ground_truth):
    """
    Correct drift using least-squares optimization.
    """
    def error_function(params):
        scale, theta = params
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        transformed_trajectory = scale * estimated_trajectory @ rotation_matrix
        return np.linalg.norm(transformed_trajectory - ground_truth, axis=1)

    # Initial guess for scale and rotation angle
    initial_params = [1.0, 0.0]
    result = least_squares(error_function, initial_params)
    scale, theta = result.x

    # Apply correction
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
    corrected_trajectory = scale * estimated_trajectory @ rotation_matrix
    return corrected_trajectory

# Align trajectories
min_length = min(len(trajectory), len(true_positions))
aligned_trajectory = trajectory[:min_length]
aligned_truth = true_positions[:min_length]
corrected_trajectory = correct_drift(aligned_trajectory, aligned_truth)

# RMSE and MAE Calculation**
def calculate_rmse(true, pred):
    """
    Calculate Root Mean Square Error (RMSE) between true and predicted values.
    """
    return np.sqrt(np.mean(np.linalg.norm(true - pred, axis=1) ** 2))

def calculate_mae(true, pred):
    """
    Calculate Mean Absolute Error (MAE) between true and predicted values.
    """
    return np.mean(np.linalg.norm(true - pred, axis=1))

# Calculate RMSE and MAE
rmse = calculate_rmse(aligned_truth, corrected_trajectory)
mae = calculate_mae(aligned_truth, corrected_trajectory)

# Print RMSE and MAE results
print(f"\nRoot Mean Square Error (RMSE): {rmse:.3f} meters")
print(f"Mean Absolute Error (MAE): {mae:.3f} meters")

#Compute Drift
drift = np.linalg.norm(corrected_trajectory - aligned_truth, axis=1)

# Plots
# Noise Filtering Graph
plt.figure(figsize=(12, 6))
plt.plot(accel_magnitude, label="Raw Acceleration", linestyle='dotted', color='gray', alpha=0.7)
plt.plot(ax_filtered, label="Kalman Filtered", color='blue', linewidth=2)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Acceleration Magnitude", fontsize=12)
plt.title("Comparison of Raw and Kalman Filtered Acceleration Data", fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
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

# Print ground truth and corrected trajectory for verification
print("Ground Truth:\n", aligned_truth)
print("Estimated Trajectory:\n", corrected_trajectory)

# Estimated vs Ground Truth Trajectory Graph
plt.figure(figsize=(10, 5))
plt.plot(aligned_truth[:, 0], aligned_truth[:, 1], label="Ground Truth", color="green", linewidth=2)
plt.plot(corrected_trajectory[:, 0], corrected_trajectory[:, 1], label="Corrected Trajectory", color="red", linestyle="--", linewidth=2)
plt.legend()
plt.title("Estimated vs Ground Truth Trajectory")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.grid(True)
plt.axis('equal')  
plt.show()

# Print Drift Results
print("\nTotal Drift (Cumulative Error at Each Interval):")
for i, d in enumerate(drift):
    print(f"Interval {i+1}:  {d:.3f} meters")

print("\nDrift Change Between Intervals:")
drift_intervals = np.diff(drift)
for i, d in enumerate(drift_intervals):
    change_label = "Increase" if d > 0 else "Correction detected"
    print(f"Interval {i+1} → Interval {i+2}:  {d:.3f} meters ({change_label})")

# Additional Drift Reduction Techniques

def alternative_step_length_model(accel_magnitude, step):
    """
    Alternative step-length model based on acceleration magnitude.
    """
    return 0.5 + 0.3 * (accel_magnitude[step] - np.mean(accel_magnitude))

# Apply alternative step-length model
trajectory_alt = [cur_position]
for i, step in enumerate(steps):
    step_length = alternative_step_length_model(accel_magnitude, step)
    step_vector = step_length * np.array([np.cos(heading[step]), np.sin(heading[step])])
    cur_position = cur_position + step_vector
    trajectory_alt.append(cur_position)

trajectory_alt = np.array(trajectory_alt)

# Correct drift for alternative trajectory
corrected_trajectory_alt = correct_drift(trajectory_alt[:min_length], aligned_truth)

# Calculate RMSE and MAE for alternative trajectory
rmse_alt = calculate_rmse(aligned_truth, corrected_trajectory_alt)
mae_alt = calculate_mae(aligned_truth, corrected_trajectory_alt)

print(f"\nAlternative Step-Length Model RMSE: {rmse_alt:.3f} meters")
print(f"Alternative Step-Length Model MAE: {mae_alt:.3f} meters")
# Compute drift per step, drift per second, and cumulative drift separately.

# Ensure steps and timestamps are aligned
step_timestamps = timestamps[steps]  # Timestamps corresponding to detected steps

# Drift per step (difference between consecutive positions in the corrected trajectory)
# Use the steps array to index into the corrected_trajectory
drift_per_step = np.linalg.norm(corrected_trajectory[1:] - corrected_trajectory[:-1], axis=1)

# Time intervals between consecutive steps
time_intervals = np.diff(step_timestamps)

# Ensure drift_per_step and time_intervals have the same length
if len(drift_per_step) != len(time_intervals):
    # Trim the longer array to match the shorter one
    min_length = min(len(drift_per_step), len(time_intervals))
    drift_per_step = drift_per_step[:min_length]
    time_intervals = time_intervals[:min_length]

# Drift per second (drift per step divided by time interval)
drift_per_second = drift_per_step / time_intervals

# Cumulative drift (cumulative sum of drift per step)
cumulative_drift = np.cumsum(drift_per_step)

# Plot drift breakdown
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(drift_per_step, label="Drift per Step", color='blue', linewidth=2)
plt.xlabel("Step Interval")
plt.ylabel("Drift (meters)")
plt.title("Drift Per Step")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(drift_per_second, label="Drift per Second", color='green', linewidth=2)
plt.xlabel("Step Interval")
plt.ylabel("Drift (meters/second)")
plt.title("Drift Per Second")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(cumulative_drift, label="Cumulative Drift", color='red', linewidth=2)
plt.xlabel("Step Interval")
plt.ylabel("Cumulative Drift (meters)")
plt.title("Cumulative Drift")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()