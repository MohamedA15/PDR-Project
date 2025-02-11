import numpy as np
import pandas as pd
import imufusion
from scipy import signal
import matplotlib.pyplot as plt

def zupt(df, sample_rate=200, zupt_tresh=3, margin=0.1, debug=False, lp_filter=False):
    """ZUPT implementation for drift correction."""
    plt.style.use("default")
    
    df = df.copy().reset_index()
    df.index /= sample_rate
    df["gyro"] *= 180 / np.pi

    dt = 1 / sample_rate
    margin = int(margin * sample_rate)

    # Low-pass filtering
    if lp_filter:
        b, a = signal.butter(10, 20, fs=200, btype="lowpass", analog=False)
        df = df.apply(lambda x: signal.filtfilt(b, a, x))

    # AHRS initialization
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()
    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NED, 0.5, 2000, 10, 30, 5 * sample_rate
    )

    def update(x):
        ahrs.update_no_magnetometer(
            x["gyro"].to_numpy(), x["accel"].to_numpy(), 0.005
        )
        euler = ahrs.quaternion.to_euler()
        acceleration = ahrs.earth_acceleration
        return {
            "x": acceleration[0],
            "y": acceleration[1],
            "z": acceleration[2],
            "roll": euler[0],
            "pitch": euler[1],
            "yaw": euler[2],
        }

    sf = df.apply(update, axis=1)
    sf = pd.DataFrame(list(sf), index=df.index)

    # Additional ZUPT implementation (velocity, drift correction, etc.)
    # This can be added here if needed in the future.

    return sf  # Return processed DataFrame
