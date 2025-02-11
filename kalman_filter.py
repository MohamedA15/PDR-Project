import numpy as np

def kalman_filter_step(state, P, z, step_vector, corrected_yaw, Q, R, H):
    """
    Perform a single step of the Kalman Filter.

    Parameters:
        state (np.array): Current state vector [x, y, velocity_x, velocity_y, orientation].
        P (np.array): Current state covariance matrix.
        z (np.array): Observation vector (e.g., current position).
        step_vector (np.array): Predicted step displacement vector based on step length and orientation.
        corrected_yaw (float): Corrected yaw value for orientation after ZUPT or bias correction.
        Q (np.array): Process noise covariance matrix, representing uncertainties in system dynamics.
        R (np.array): Measurement noise covariance matrix, representing uncertainties in measurements.
        H (np.array): Observation matrix mapping the state space to the observation space.

    Returns:
        state (np.array): Updated state vector after prediction and correction.
        P (np.array): Updated state covariance matrix after incorporating observations.

    Details:
    The Kalman Filter operates in two main steps:
    1. Prediction Step:
        - Updates the state vector based on the motion model.
        - Updates the state covariance matrix to reflect increased uncertainty.
    2. Correction Step:
        - Incorporates observations to refine the state estimate.
        - Reduces uncertainty in the state covariance matrix.

    The function combines these steps to iteratively estimate the system's state with improved accuracy.
    """
    # Determine the size of the state vector
    state_size = len(state)

    # State transition matrix (F): Models the dynamics of the system
    F = np.eye(state_size)
    F[0, 2] = 1  # Update x position with velocity_x
    F[1, 3] = 1  # Update y position with velocity_y

    # Prediction step
    # Predict the next state based on the step vector and corrected yaw
    state[:2] += step_vector  # Update x and y positions
    state[4] = corrected_yaw  # Update orientation

    # Predict the next state covariance matrix
    P = F @ P @ F.T + Q

    # Observation residual (innovation): Difference between observed and predicted positions
    y = z - state[:2]

    # Compute the Kalman gain (K): Balances the confidence in the prediction vs observation
    S = H @ P @ H.T + R  # Innovation covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

    # Correction step
    # Update the state estimate with the weighted observation residual
    state += K @ y

    # Update the state covariance matrix
    I = np.eye(state_size)  # Identity matrix
    P = (I - K @ H) @ P

    return state, P

def initialize_kalman_filter(state_size, process_noise, measurement_noise):
    """
    Initialize the Kalman Filter with default values.

    Parameters:
        state_size (int): Size of the state vector.
        process_noise (float): Standard deviation of process noise.
        measurement_noise (float): Standard deviation of measurement noise.

    Returns:
        state (np.array): Initial state vector.
        P (np.array): Initial state covariance matrix.
        Q (np.array): Process noise covariance matrix.
        R (np.array): Measurement noise covariance matrix.
        H (np.array): Observation matrix.
    """
    # Initial state vector (all zeros)
    state = np.zeros(state_size)

    # Initial state covariance matrix (identity matrix scaled)
    P = np.eye(state_size) * 0.1

    # Process noise covariance matrix (diagonal matrix)
    Q = np.diag([process_noise] * state_size)

    # Measurement noise covariance matrix (diagonal matrix)
    R = np.diag([measurement_noise] * 2)  # Only for x and y positions

    # Observation matrix (maps state to observations)
    H = np.eye(2, state_size)  # Only observes x and y

    return state, P, Q, R, H

def kalman_filter_usage_example():
    """
    Example usage of the Kalman Filter functions.

    This function demonstrates how to initialize and apply the Kalman Filter to a simple scenario.
    """
    # Initialize Kalman Filter parameters
    state_size = 5  # [x, y, velocity_x, velocity_y, orientation]
    process_noise = 0.1
    measurement_noise = 0.05

    state, P, Q, R, H = initialize_kalman_filter(state_size, process_noise, measurement_noise)

    # Example observation and step vector
    z = np.array([1.0, 2.0])  # Observed position
    step_vector = np.array([0.1, 0.2])  # Predicted displacement
    corrected_yaw = 0.5  # Corrected orientation (in radians)

    # Apply a single Kalman Filter step
    updated_state, updated_P = kalman_filter_step(state, P, z, step_vector, corrected_yaw, Q, R, H)

    # Print results
    print("Updated State:", updated_state)
    print("Updated Covariance Matrix:", updated_P)


