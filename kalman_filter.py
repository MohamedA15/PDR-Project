import numpy as np

def kalman_filter_step(state, P, z, step_vector, corrected_yaw, Q, R, H, zupt=False):
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
        zupt (bool): If True, apply Zero Velocity Update (ZUPT) to reset velocity.

    Returns:
        state (np.array): Updated state vector after prediction and correction.
        P (np.array): Updated state covariance matrix after incorporating observations.
    """
    state_size = len(state)

    # State transition matrix (F): Models the dynamics of the system
    F = np.eye(state_size)
    F[0, 2] = 1  # Update x position with velocity_x
    F[1, 3] = 1  # Update y position with velocity_y

    # Prediction step
    state[:2] += step_vector  # Update x and y positions
    state[4] = corrected_yaw  # Update orientation

    # Predict the next state covariance matrix
    P = F @ P @ F.T + Q

    # Apply ZUPT (Zero Velocity Update)
    if zupt:
        state[2:4] = 0  # Reset velocity_x and velocity_y

    # Observation residual (innovation): Difference between observed and predicted positions
    y = z - state[:2]

    # Compute the Kalman gain (K): Balances the confidence in the prediction vs observation
    S = H @ P @ H.T + R  # Innovation covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

    # Correction step
    state += K @ y  # Update state estimate
    P = (np.eye(state_size) - K @ H) @ P  # Update state covariance matrix

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

def adaptive_process_noise(Q, accel_magnitude, gyro_magnitude):
    """
    Adapt the process noise covariance matrix (Q) based on acceleration and gyroscope magnitudes.

    Parameters:
        Q (np.array): Current process noise covariance matrix.
        accel_magnitude (float): Magnitude of acceleration.
        gyro_magnitude (float): Magnitude of gyroscope data.

    Returns:
        Q (np.array): Updated process noise covariance matrix.
    """
    # Increase process noise during high dynamics
    dynamic_factor = np.sqrt(accel_magnitude**2 + gyro_magnitude**2)
    Q = Q * (1 + dynamic_factor)
    return Q

def adaptive_measurement_noise(R, position_uncertainty):
    """
    Adapt the measurement noise covariance matrix (R) based on position uncertainty.

    Parameters:
        R (np.array): Current measurement noise covariance matrix.
        position_uncertainty (float): Uncertainty in position measurements.

    Returns:
        R (np.array): Updated measurement noise covariance matrix.
    """
    # Increase measurement noise during high uncertainty
    R = R * (1 + position_uncertainty)
    return R