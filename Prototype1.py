import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

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

# Select route
print("Select a route file:")
for i, route in enumerate(route_options, 1):
    print(f"{i}. {route}")
route_choice = int(input("Enter the number of the route file: "))
filename = route_options[route_choice - 1]

# Select ground truth
print("\nSelect a ground truth file:")
for i, truth in enumerate(ground_truth_options, 1):
    print(f"{i}. {truth}")
ground_truth_choice = int(input("Enter the number of the ground truth file: "))
ground_truth_file = ground_truth_options[ground_truth_choice - 1]

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

# Calibration
static_samples = 800
means = [np.mean(axis[:static_samples]) for axis in [x_axis, y_axis, z_axis]]
x_calib, y_calib, z_calib = [axis - mean for axis, mean in zip([x_axis, y_axis, z_axis], means)]

# Plot calibrated accelerometer
plt.plot(x_calib, label="X_Calibrated")
plt.plot(y_calib, label="Y_Calibrated")
plt.plot(z_calib, label="Z_Calibrated")
plt.legend()
plt.title("Calibrated Accelerometer Data")
plt.show()

# Magnitude
accel_magnitude_raw = np.linalg.norm([x_calib, y_calib, z_calib], axis=0)
accel_magnitude = signal.savgol_filter(accel_magnitude_raw, window_length=11, polyorder=4)

# Plot magnitude
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(accel_magnitude_raw, label="Raw Magnitude")
plt.title("Raw Acceleration Magnitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(accel_magnitude, label="Smoothed Magnitude", color='orange')
plt.title("Smoothed Acceleration Magnitude (Savitzky-Golay Filter)")
plt.legend()
plt.tight_layout()
plt.show()

# Step detection
steps, _ = signal.find_peaks(accel_magnitude, height=0.1, distance=5)
plt.plot(accel_magnitude, label="Smoothed Acceleration")
plt.plot(steps, accel_magnitude[steps], "x", label="Detected Steps", color="red")
plt.legend()
plt.title("Step Detection")
plt.show()

# Heading
def calculate_heading(z_gyro, timestamps):
    return np.concatenate(([0], np.cumsum(z_gyro[:-1] * np.diff(timestamps))))

heading = calculate_heading(z_gyro, timestamps)
motion_vectors = np.array([np.cos(heading), np.sin(heading)]).T

# Trajectory
step_length = 50.0
cur_position = np.array([0.0, 0.0])
trajectory = [cur_position]
for step in steps:
    cur_position = cur_position + step_length * motion_vectors[step]
    trajectory.append(cur_position)
trajectory = np.array(trajectory)

# Align with ground truth
min_length = min(len(trajectory), len(true_positions))
aligned_trajectory = trajectory[:min_length]
aligned_truth = true_positions[:min_length]
drift = np.linalg.norm(aligned_trajectory - aligned_truth, axis=1)

# Trajectory plot
plt.plot(aligned_truth[:, 0], aligned_truth[:, 1], label="True Trajectory", color="green")
plt.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], label="Estimated Trajectory", color="red")
plt.legend()
plt.title("PDR Trajectory vs Ground Truth")
plt.show()

# Drift plot
plt.plot(drift, label="Drift", color="blue")
plt.title("Drift per Interval")
plt.legend()
plt.show()

# Define raw and filtered variables
ax, ay, az = x_calib, y_calib, z_calib
wz = z_gyro
wx = np.zeros_like(wz)
wy = np.zeros_like(wz)

ax_filtered = signal.savgol_filter(ax, 11, 4)
ay_filtered = signal.savgol_filter(ay, 11, 4)
az_filtered = signal.savgol_filter(az, 11, 4)
wx_filtered = signal.savgol_filter(wx, 11, 4)
wy_filtered = signal.savgol_filter(wy, 11, 4)
wz_filtered = signal.savgol_filter(wz, 11, 4)

# Stationary detection
stationary = accel_magnitude < 0.1

# METRICS 

# Calculate RMSE and MAE
rmse = np.sqrt(np.mean(np.square(drift)))
mae = np.mean(np.abs(drift))

#  METRICS 
print("\n PERFORMANCE METRICS ")
print(f"RMSE: {rmse:.3f} meters")
print(f"MAE: {mae:.3f} meters")
print(f"Max Drift: {np.max(drift):.3f} meters")
print(f"Average Drift: {np.mean(drift):.3f} meters")


print("\nTotal Drift (Cumulative Error at Each Interval):")
for i, d in enumerate(drift):
    print(f"Interval {i+1}:  {d:.3f} meters")
    
print("\nDrift Change Between Intervals:")
drift_intervals = np.diff(drift)
for i, d in enumerate(drift_intervals):
    change_label = "Increase" if d > 0 else "Decrease"
    print(f"Interval {i+1} → Interval {i+2}:  {d:.3f} meters ({change_label})")

# Noise Reduction 
print("\nNOISE REDUCTION ")
for axis, raw, filt in zip(['AX', 'AY', 'AZ'], [ax, ay, az], [ax_filtered, ay_filtered, az_filtered]):
    var_red = 100 * (np.var(raw) - np.var(filt)) / np.var(raw)
    print(f"{axis}: Variance Reduction = {var_red:.1f}%")

print("\n- Noise Filtering Evaluation (Variance Reduction) ")

def evaluate_noise_reduction(raw_data, filtered_data):
    raw_var = np.var(raw_data)
    filtered_var = np.var(filtered_data)
    reduction = 100 * (raw_var - filtered_var) / raw_var
    return raw_var, filtered_var, reduction

print("\n[ Accelerometer Axes ]")
for axis_name, raw, filtered in zip(['AX', 'AY', 'AZ'], [ax, ay, az], [ax_filtered, ay_filtered, az_filtered]):
    raw_var, filtered_var, reduction = evaluate_noise_reduction(raw, filtered)
    print(f"{axis_name} - Raw Var: {raw_var:.6f}, Filtered Var: {filtered_var:.6f}, Reduction: {reduction:.2f}%")

print("\n[ Gyroscope Axes ]")
for axis_name, raw, filtered in zip(['WX', 'WY', 'WZ'], [wx, wy, wz], [wx_filtered, wy_filtered, wz_filtered]):
    raw_var, filtered_var, reduction = evaluate_noise_reduction(raw, filtered)
    print(f"{axis_name} - Raw Var: {raw_var:.6f}, Filtered Var: {filtered_var:.6f}, Reduction: {reduction:.2f}%")


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
