# Pedestrian Dead Reckoning (PDR) - Final Year Project

This project presents a complete indoor navigation system using Pedestrian Dead Reckoning (PDR), developed as part of my final year Computer Science degree.

## 📌 Project Title
**Minimising Drift in Indoor Navigation using Pedestrian Dead Reckoning (PDR)**

##  Overview
This project explores how various filtering and correction techniques can reduce drift in PDR systems using smartphone sensor data. The system processes accelerometer and gyroscope inputs from a Samsung A55 to estimate a user's trajectory indoors, without GPS.

##  Models Implemented

| Model | Description |
|-------|-------------|
| **Model 1** | Baseline PDR using raw acceleration and integration. |
| **Model 2** | ZUPT-based correction using thresholded stationary detection. |
| **Model 3** | Kalman Filtering for noise reduction on IMU signals. |
| **Model 4** | Orientation estimation using Madgwick filter + ZUPT. |
| **Model 5 (Main) ** | Final model with Mahony filter, Kalman filtering, ZUPT, dynamic step detection, and least-squares drift correction. |

## 📊 Evaluation Metrics
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Max/Avg Drift
- Noise Reduction (%)
- Heading Stability (MAC)
- Step Frequency

## 🧪 Testing
- Tested on **9 different indoor routes**
- Ground truth measured manually for accuracy
- Sensor data collected using a smartphone (Samsung A55)

## 🏆 Results
- **Model 5** showed the lowest average drift overall
- **Model 4** had the most stable heading estimation
- Dynamic parameter tuning significantly improved accuracy

## 📂 Structure
- `/data/` – IMU datasets (accelerometer, gyroscope)
- `/models/` – Python implementations of all 5 models
- `/results/` – CSVs, metrics, and graphs
- `/report/` – Final year project report in PDF and LaTeX

## 📌 Technologies Used
- Python (NumPy, SciPy, Matplotlib, Pandas)
- Git & GitHub
- LaTeX (for report)
- Smartphone IMU sensors
