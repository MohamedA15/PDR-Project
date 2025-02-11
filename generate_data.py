import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from ahrs.filters import Madgwick
from calculate_rmse import calculate_rmse  # Import RMSE function
from kalman_filter import kalman_filter_step  # Import Kalman Filter function

# Display available routes for the user to choose
def display_routes(routes):
    print("Available routes:")
    for i, route in enumerate(routes, start=1):
        print(f"{i}. {route}")

# Prompt the user to select a route
def select_route(routes):
    display_routes(routes)
    route_choice = int(input("Select a route (1/2/3): ")) - 1
    if route_choice not in range(len(routes)):
        raise ValueError("Invalid route choice. Please choose a valid option.")
    return routes[route_choice], f"ground_truth{route_choice + 1}.csv"

# Check if files exist
def check_files_exist(*files):
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File '{file}' not found.")

# Load and clean IMU data
def load_and_clean_data(file, required_columns):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    for col in required_columns:
        if col not in df.columns or df[col].isnull().all():
            raise ValueError(f"Error: Missing or empty column '{col}' in the dataset.")
    return df.dropna(subset=required_columns)

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

# Apply filters to IMU data
def filter_imu_data(ax, ay, az, wx, wy, wz, accel_cutoff, fs, filter_order):
    ax_filtered = low_pass_filter(ax, accel_cutoff, fs, order=filter_order)
    ay_filtered = low_pass_filter(ay, accel_cutoff, fs, order=filter_order)
    az_filtered = low_pass_filter(az, accel_cutoff, fs, order=filter_order)

    wx_filtered = high_pass_filter(wx, accel_cutoff, fs, order=filter_order)
    wy_filtered = high_pass_filter(wy, accel_cutoff, fs, order=filter_order)
    wz_filtered = high_pass_filter(wz, accel_cutoff, fs, order=filter_order)

    return ax_filtered, ay_filtered, az_filtered, wx_filtered, wy_filtered, wz_filtered

# Compute steps based on peaks in acceleration magnitude
def compute_steps(accel_magnitude):
    peaks, _ = signal.find_peaks(accel_magnitude, height=np.percentile(accel_magnitude, 90))
    return len(peaks)

# Apply ZUPT to the state vector
def apply_zupt(state, stationary_periods, i):
    if stationary_periods[i]:
        state[2] = 0  # Reset velocity_x
        state[3] = 0  # Reset velocity_y
    return state

# Calculate trajectory using Kalman Filter and ZUPT
def calculate_trajectory(yaws, accel_magnitude, true_positions):
    state_size = 5
    sigma_a, sigma_v, sigma_omega = 0.03, 0.003, 0.003
    Q = np.diag([sigma_a, sigma_a, sigma_v, sigma_v, sigma_omega])
    R = np.diag([sigma_v, sigma_v]) ** 2
    H = np.eye(2, 5)

    state = np.zeros(state_size)
    P = np.eye(state_size)
    trajectory = []
    cur_position = np.zeros(2)

    stationary_periods_dynamic = accel_magnitude < np.percentile(accel_magnitude, 20)

    for i, yaw in enumerate(yaws):
        state = apply_zupt(state, stationary_periods_dynamic, i)
        step_length = 0.5 * max(accel_magnitude)
        step_vector = step_length * np.array([np.cos(yaw), np.sin(yaw)])
        z = cur_position
        state, P = kalman_filter_step(state, P, z, step_vector, yaw, Q, R, H)
        cur_position = state[:2]
        trajectory.append(cur_position.copy())

    trajectory = np.array(trajectory)

    # Align trajectory and ground truth lengths
    min_length = min(len(trajectory), len(true_positions))
    aligned_trajectory = trajectory[:min_length]
    aligned_true_positions = true_positions[:min_length]

    # Calculate Drift
    drift = np.linalg.norm(aligned_trajectory - aligned_true_positions, axis=1)
    return aligned_trajectory, aligned_true_positions, drift

# Plot functions
def plot_trajectory_vs_ground_truth(aligned_trajectory, aligned_true_positions):
    plt.figure(figsize=(8, 6))
    plt.plot(aligned_true_positions[:, 0], aligned_true_positions[:, 1], label="Ground Truth", color="green")
    plt.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], label="Estimated Trajectory", linestyle="--", color="red")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Trajectory vs Ground Truth")
    plt.legend()
    plt.grid()
    plt.show()

def plot_drift_over_intervals(drift):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(drift)), drift, label="Drift", color="blue")
    plt.xlabel("Interval Index")
    plt.ylabel("Drift (meters)")
    plt.title("Drift Over Intervals")
    plt.legend()
    plt.grid()
    plt.show()

def print_drift_intervals(drift):
    print("Drift at intervals (meters):")
    for i, value in enumerate(drift):
        print(f"Interval {i + 1}: {value:.2f} meters")

# Main script execution
def main():
    routes = ["route1", "route2", "route3"]
    while True:
        selected_route, ground_truth_file = select_route(routes)

        imu_file = f"{selected_route}.csv"
        check_files_exist(imu_file, ground_truth_file)

        required_columns = ['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']
        df = load_and_clean_data(imu_file, required_columns)

        ground_truth_df = pd.read_csv(ground_truth_file)
        true_positions = ground_truth_df.values

        timestamps = df['time'].values
        ax, ay, az = df['ax'].values, df['ay'].values, df['az'].values
        wx, wy, wz = df['wx'].values, df['wy'].values, df['wz'].values

        sampling_rate = 30
        accel_cutoff = 5
        filter_order = 6

        ax_filtered, ay_filtered, az_filtered, wx_filtered, wy_filtered, wz_filtered = filter_imu_data(
            ax, ay, az, wx, wy, wz, accel_cutoff, sampling_rate, filter_order)

        accel_magnitude = np.linalg.norm([ax_filtered, ay_filtered, az_filtered], axis=0)
        estimated_steps = compute_steps(accel_magnitude)
        print(f"Estimated steps taken: {estimated_steps}")

        madgwick = Madgwick(gain=0.1, frequency=sampling_rate)
        quaternions = np.zeros((len(ax_filtered), 4))
        quaternions[0] = [1, 0, 0, 0]
        yaws = []

        for i in range(1, len(ax_filtered)):
            quaternions[i] = madgwick.updateIMU(q=quaternions[i-1],
                                                gyr=[wx_filtered[i], wy_filtered[i], wz_filtered[i]],
                                                acc=[ax_filtered[i], ay_filtered[i], az_filtered[i]])
            yaws.append(np.arctan2(2 * (quaternions[i][0] * quaternions[i][3] + quaternions[i][1] * quaternions[i][2]),
                                   1 - 2 * (quaternions[i][2] ** 2 + quaternions[i][3] ** 2)))

        aligned_trajectory, aligned_true_positions, drift = calculate_trajectory(yaws, accel_magnitude, true_positions)

        plot_trajectory_vs_ground_truth(aligned_trajectory, aligned_true_positions)
        plot_drift_over_intervals(drift)
        print_drift_intervals(drift)

        estimated_trajectory_file = "estimated_trajectory.csv"
        ground_truth_file = "ground_truth.csv"
        pd.DataFrame(aligned_trajectory, columns=['x', 'y']).to_csv(estimated_trajectory_file, index=False)
        pd.DataFrame(aligned_true_positions, columns=['x', 'y']).to_csv(ground_truth_file, index=False)

       # Compute RMSE from Saved Files
        rmse = calculate_rmse("estimated_trajectory.csv", "ground_truth.csv")

        # Ask the user if they want to select another route or exit
        repeat = input("Would you like to select another route? (yes/no): ").strip().lower()
        if repeat != 'yes':
            print("Exiting the program. Goodbye!")
            break

if __name__ == "__main__":
    main()