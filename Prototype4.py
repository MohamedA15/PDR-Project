import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from ahrs.filters import Madgwick
import pywt
from scipy.stats import median_abs_deviation as mad
from scipy.interpolate import interp1d

#  File selection
route_options = [f"route{i}.csv" for i in range(1,10)] + [f"route{i}.1.csv" for i in range(4,7)]
ground_truth_options = [f"ground_truth{i}.csv" for i in range(1,10)]

print("Available routes:")
for i, route in enumerate(route_options, start=1):
    print(f"{i}. {route}")

print("\nAvailable ground truth files:")
for i, truth in enumerate(ground_truth_options, start=1):
    print(f"{i}. {truth}")

route_choice = int(input("Select a route (1-10): ")) - 1
truth_choice = int(input("Select a ground truth file (1-7): ")) - 1

selected_route = route_options[route_choice]
selected_ground_truth = ground_truth_options[truth_choice]


# Load and clean data

df = pd.read_csv(selected_route)
df.columns = df.columns.str.strip()
ground_truth = pd.read_csv(selected_ground_truth).values

required_cols = ['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']
df = df.dropna(subset=required_cols)
timestamps = df['time'].values
accel = df[['ax','ay','az']].values
gyro  = df[['wx','wy','wz']].values
mag   = df[['Bx','By','Bz']].values

def wavelet_denoise(data, wavelet='db6', level=2):
    # If the data length is odd, pad it so wavelet transform can be applied properly
    if len(data) % 2 != 0:
        data = np.pad(data, (0,1), mode='edge')[:len(data)]

    # Break the signal down into wavelet coefficients
    coeffs = pywt.wavedec(data, wavelet, mode='per', level=level)

    # Estimate noise level using Median Absolute Deviation on the detail coefficients
    sigma = mad(coeffs[-level])

    # Compute a threshold based on the estimated noise
    thresh = sigma * np.sqrt(2*np.log(len(data)))

    # Apply soft thresholding to all detail coefficients (ignore the first one which is approximation)
    coeffs[1:] = [pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:]]

    # Reconstruct the signal from the thresholded coefficients
    den = pywt.waverec(coeffs, wavelet, mode='per')

    # Return the denoised signal (same length as input)
    return den[:len(data)]


# Sensor filtering

def apply_sensor_filtering(accel, gyro, mag, fs=60):
    af = np.zeros_like(accel)
    gf = np.zeros_like(gyro)
    mf = np.zeros_like(mag)
    for i in range(3):
        a = wavelet_denoise(accel[:,i])
        a = signal.medfilt(a, kernel_size=7)
        af[:,i] = signal.sosfiltfilt(
            signal.butter(4,6,btype='low',fs=fs,output='sos'), a
        )
        g = wavelet_denoise(gyro[:,i])
        gf[:,i] = signal.sosfiltfilt(
            signal.butter(4,1.5,btype='high',fs=fs,output='sos'), g
        )
        m = wavelet_denoise(mag[:,i])
        mf[:,i] = signal.sosfiltfilt(
            signal.butter(4,6,btype='low',fs=fs,output='sos'), m
        )
    mf -= np.mean(mf, axis=0)
    return af, gf, mf

accel_filt, gyro_filt, mag_filt = apply_sensor_filtering(accel, gyro, mag)


# Orientation via Madgwick
madgwick = Madgwick(frequency=30, gain=0.010)

# Preallocate a quaternion array for storing orientation estimates
quats = np.zeros((len(accel_filt), 4))

# Start with an identity quaternion (no rotation)
quats[0] = [1, 0, 0, 0]

# Loop through IMU data and update orientation using the Madgwick filter

for i in range(1, len(accel_filt)):
    quats[i] = madgwick.updateMARG(
        q=quats[i-1],             
        gyr=gyro_filt[i],         
        acc=accel_filt[i],        
        mag=mag_filt[i]           
    )

# Convert quaternion orientation into Euler angles (roll, pitch, yaw)
def quat_to_euler(q):
    # Roll (rotation around x-axis)
    roll  = np.arctan2(2 * (q[0]*q[1] + q[2]*q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
    
    # Pitch (rotation around y-axis)
    pitch = np.arcsin(2 * (q[0]*q[2] - q[3]*q[1]))
    
    # Yaw (rotation around z-axis, i.e. heading)
    yaw   = np.arctan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
    
    return roll, pitch, yaw

# Apply quaternion-to-Euler conversion to the entire set of orientation estimates
euler = np.array([quat_to_euler(q) for q in quats])



#  Step detection

def improved_step_detection(accel_mag, timestamps, yaw, fs=50):
    sos = signal.butter(4, [1,3], btype='bandpass', fs=fs, output='sos')
    f = signal.sosfiltfilt(sos, accel_mag)
    h = np.percentile(f,75)
    peaks, props = signal.find_peaks(f, height=h, distance=fs//2)
    sf = len(peaks)/(len(accel_mag)/fs)
    base = 0.7
    L = base*(1+0.1*(sf-1.8))
    lengths = np.full(len(peaks), L)
    if 'peak_heights' in props:
        rh = props['peak_heights']/np.max(props['peak_heights'])
        lengths *= (0.8+0.4*rh)
    return peaks, timestamps[peaks], yaw[peaks], lengths

accel_mag = np.linalg.norm(accel_filt - np.mean(accel_filt,axis=0), axis=1)
steps, step_times, step_yaws, step_lengths = improved_step_detection(
    accel_mag, timestamps, euler[:,2]
)


#Stationary detection

def detect_stationary(accel, gyro, mag, window=15):
    a_var = pd.Series(np.linalg.norm(accel,axis=1)).rolling(window,center=True).var().fillna(0)
    g_var = pd.Series(np.linalg.norm(gyro,axis=1)).rolling(window,center=True).var().fillna(0)
    m_var = pd.Series(np.linalg.norm(mag,axis=1)).rolling(window,center=True).var().fillna(0)
    return (a_var<0.05)&(g_var<0.01)&(m_var<5)

stationary = detect_stationary(accel_filt, gyro_filt, mag_filt)

#  Trajectory estimation

def estimate_trajectory(steps, yaws, times, lengths, stationary):
    pos = np.zeros((len(steps)+1,2))
    vel = np.zeros(2)
    bias = 0.0
    for i in range(1,len(steps)):
        dt = times[i]-times[i-1]
        L  = lengths[i-1]
        if stationary[steps[i]]:
            vel *= 0.2
            if 1 <= i < len(steps)-1:
                mag_h = np.arctan2(mag_filt[steps[i],1], mag_filt[steps[i],0])
                bias = 0.1*(mag_h - yaws[i])
        else:
            yc = yaws[i] + bias
            d  = np.array([np.cos(yc), np.sin(yc)])
            vel = d*(L/max(0.1, dt))
        pos[i] = pos[i-1] + vel*dt
    return pos[:len(steps)]

trajectory = estimate_trajectory(steps, step_yaws, step_times, step_lengths, stationary)


#Drift calculation

def calculate_drift(est, gt):
    n = len(est)
    idx = np.linspace(0, len(gt)-1, n)
    fx = interp1d(np.arange(len(gt)), gt[:,0], fill_value="extrapolate")
    fy = interp1d(np.arange(len(gt)), gt[:,1], fill_value="extrapolate")
    gt_i = np.vstack((fx(idx), fy(idx))).T
    drift = np.linalg.norm(est - gt_i, axis=1)
    return drift, gt_i

drift_vals, gt_interp = calculate_drift(trajectory, ground_truth)

# 10. Performance metrics
def evaluate_noise(raw, filt):
    rv = np.var(raw)
    fv = np.var(filt)
    rd = 100*(rv-fv)/rv if rv!=0 else 0
    return rv, fv, rd

def heading_metrics(h):
    ch = np.abs(np.diff(h))
    return np.mean(ch), np.std(ch)

def step_metrics(steps, times):
    if len(steps)<2: return 0,0,0
    itv = np.diff(times[steps])
    freq = len(steps)/(times[-1]-times[0]) if times[-1]!=times[0] else 0
    return freq, np.mean(itv), np.std(itv)

def trajectory_metrics(est, gt):
    e = np.linalg.norm(est-gt,axis=1)
    return {
        'RMSE': np.sqrt(np.mean(e**2)),
        'MAE': np.mean(e),
        'MaxError': np.max(e),
        'FinalError': e[-1],
        'AvgError': np.mean(e),
        'MedianError': np.median(e)
    }

traj_m = trajectory_metrics(trajectory, gt_interp)
step_f, mean_int, std_int = step_metrics(steps, timestamps)
head_m, head_s = heading_metrics(euler[:,2])
noise_m = [evaluate_noise(accel[:,i], accel_filt[:,i]) for i in range(3)] + \
          [evaluate_noise(gyro[:,i], gyro_filt[:,i]) for i in range(3)]
duration = timestamps[-1] - timestamps[0]
zupt_eff = (np.mean(accel_mag[stationary]), np.mean(accel_mag[~stationary]))

# Compute drift at fixed intervals and changes
num_intervals = 6
interval_length = len(drift_vals) // num_intervals
interval_drifts = [
    drift_vals[(i+1)*interval_length - 1]
    for i in range(num_intervals)
]
drift_changes = [
    interval_drifts[i] - interval_drifts[i-1]
    for i in range(1, num_intervals)
]


# Plotting

plt.figure(figsize=(10,5))
plt.plot(ground_truth[:,0], ground_truth[:,1], 'g-', label='Ground Truth')
plt.plot(trajectory[:,0], trajectory[:,1], 'r--', label='Estimated')
plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.title('Estimated vs Ground Truth Trajectory')
plt.legend(); plt.grid(True); plt.axis('equal')

# Sensor data panels
plt.figure(figsize=(12,10))
plt.subplot(3,1,1)
plt.plot(timestamps, accel_filt)
plt.title('Filtered Accelerometer'); plt.ylabel('m/s²')
plt.legend(['X','Y','Z'])
plt.subplot(3,1,2)
plt.plot(timestamps, gyro_filt)
plt.title('Filtered Gyroscope'); plt.ylabel('rad/s')
plt.legend(['X','Y','Z'])
plt.subplot(3,1,3)
plt.plot(timestamps, mag_filt)
plt.title('Filtered Magnetometer'); plt.ylabel('μT'); plt.xlabel('Time (s)')
plt.legend(['X','Y','Z'])
plt.tight_layout()
plt.show()

# Drift per step
plt.figure(figsize=(10,5))
plt.plot(drift_vals, linewidth=2)
plt.xlabel('Step Index'); plt.ylabel('Drift (m)')
plt.title('Drift per Step Interval')
plt.grid(True)
plt.show()


#metrics

print("\n COMPREHENSIVE PERFORMANCE METRICS ")

print("\n Trajectory Accuracy ")
print(f"RMSE:                 {traj_m['RMSE']:.3f} m")
print(f"MAE:                  {traj_m['MAE']:.3f} m")
print(f"Max Error:            {traj_m['MaxError']:.3f} m")
print(f"Final Position Error: {traj_m['FinalError']:.3f} m")
print(f"Average Error:        {traj_m['AvgError']:.3f} m")
print(f"Median Error:         {traj_m['MedianError']:.3f} m")

print("\n Motion Detection ")
print(f"Steps Detected:       {len(steps)}")
print(f"Step Frequency:       {step_f:.2f} steps/s")
print(f"Avg Step Interval:    {mean_int:.3f} ± {std_int:.3f} s")
print(f"Stationary Samples:   {np.sum(stationary)}")
print(f"ZUPT Effectiveness:   {zupt_eff[0]:.3f} m/s² (stationary) vs {zupt_eff[1]:.3f} m/s² (moving)")

head_m, head_s = heading_metrics(euler[:,2])
print(f"Heading Stability (Mean Absolute Change): {head_m:.6f} radians")


print("\n Noise Reduction ")
axes = ['AX','AY','AZ','WX','WY','WZ']
print(f"{'Axis':<4} {'RawVar':>8} {'FiltVar':>10} {'Red%':>8}")
for ax, nm in zip(axes, noise_m):
    print(f"{ax:<4} {nm[0]:8.6f} {nm[1]:10.6f} {nm[2]:8.1f}%")

print("\n System Performance ")
print(f"Total Samples:        {len(timestamps)}")
print(f"Duration:             {duration:.2f} s")
print(f"Processing Rate:      {len(timestamps)/duration:.2f} Hz")

print("\n DRIFT ANALYSIS ")
print("\nTotal Drift (Cumulative Error at Each Interval):")
for i, d in enumerate(interval_drifts, start=1):
    print(f"Interval {i}: {d:.3f} meters")

print("\nDrift Change Between Intervals:")
for i, change in enumerate(drift_changes, start=1):
    print(f"Interval {i} → Interval {i+1}: {change:.3f} meters")