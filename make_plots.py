from scipy.signal import resample_poly, correlate, correlate2d, savgol_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.fftpack import fft, fftfreq, fftshift
from prep_data import *
# %%
def plot_spectral_content(data, sr):
    data_f = fft(data)
    freqs = fftfreq(max(data.shape), 1/sr)
    p_inds = freqs >= 0
    plt.plot(freqs[p_inds], (1 / max(data.shape)) * np.abs(data_f[p_inds]))
    plt.show()
# %%
sensor_loc = r"E:\data\18-06-26_data_for_hackton\Vesper Data\rat1_try2 - rat4_try1.csv"
sensor_data = pd.read_csv(sensor_loc)
times = np.array(sensor_data.iloc[:, 0])
sensor_data = sensor_data.iloc[:,1:]
sensor_data['time'] = [datetime.fromordinal(t.astype(int)) + timedelta(days=t%1) - timedelta(days = 366) for t in times]

camera_loc = r"E:\data\18-06-26_data_for_hackton\Tracking Data\rat1_try2.csv"
camera_data = pd.read_csv(camera_loc)

data_loc = 'alldata.mat'
start_ind = 800
data = load_synced_data(data_loc, start_ind)
target = make_target(data)
features = make_features(data)
# %% Raw data
plt.figure(1)
plt.subplot(121)
plt.plot(camera_data[['X1','Y1','Z1']])
plt.title('Tracker Data')
plt.legend(['X', 'Y', 'Z'])
plt.ylabel('Location')
plt.xlabel('Sample #')
plt.subplot(122)
plt.plot(sensor_data[['x_acc','y_acc','z_acc']])
plt.title('Accelerometer data')
plt.ylabel('Acceleration')
plt.xlabel('Sample #')
plt.show()
# %% Divergence
fig, ax11 = plt.subplots()
ax11.plot(sensor_data.x_acc, '-b')
ax11.set_xlabel('Sample #')
ax11.set_ylabel('Acceleration', color='b')
ax11.set_title('Sensor X-Axis data')
ax11.tick_params('y', colors='b')

ax21 = ax11.twinx()
ax21.plot(sensor_data.x_acc.cumsum(), '-r')
ax21.set_ylabel('Velocity', color='r')
ax21.tick_params('y', colors='r')
# %% Sync dance
plt.figure(3)
plt.subplot(211)
plt.title('Syncing')
plt.plot(camera_data.TIME, camera_data['Z1'])
plt.xlabel('Time (s)')
plt.subplot(212)
plt.plot(sensor_data.time[75000:250000], sensor_data.z_acc[75000:250000])
plt.xlabel('Day + TOD')
# %%
fig, ax12 = plt.subplots()
ax12.plot(features.acc_vec_magnitude, '-b')
ax12.set_xlabel('Sample #')
ax12.set_ylabel('Acceleration magnitude', color='b')
ax12.set_title('Vector Data')
ax12.tick_params('y', colors='b')

ax22 = ax12.twinx()
ax22.plot(target.magnitude, '-r')
ax22.set_ylabel('Velocity magnitude', color='r')
ax22.tick_params('y', colors='r')
# %%
fig, ax2 = plt.subplots(sharex=True)
x = real_data.x_acc
sr = 50
bx = butter_bandpass(x, 1, 5, sr)
plt.figure()
plt.subplot(221)
plt.plot(x)
plt.title('Original x')
plt.subplot(222)
plot_spectral_content(x, 50)
plt.title('X power spectrum')
plt.subplot(223)
plt.plot(bx)
plt.title('X filtered between 1-5Hz')
plt.xlabel('Sample #')
plt.subplot(224)
plot_spectral_content(bx, 50)
plt.title('X filtered spectral content')
plt.xlabel('Frequency (Hz)')
# %%
plt.figure()
plt.subplot(121)
plt.plot(x.cumsum())
plt.title('X derived velocity')
plt.ylabel('Velocity')
plt.subplot(122)
plt.plot(bx.cumsum())
plt.title('Filtered x derived velocity')
plt.ylabel('Velocity')