# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:11:05 2021

@author: robin
"""
#%% Load config file
from utils.utils import load_config
from calibration.intrinsic import calibrate_intrinsic
from calibration.extrinsic import calibrate_extrinsic

config = load_config('config_pop_1217.toml')
config['paths_to_2d_data'] = paths_to_save
config['calibration']['calib_video_path'] = calib_path
# config['output_video_path'] = 
#%% Calibration
calibrate_intrinsic(config)
calibrate_extrinsic(config)

#%% 3D reconstruction
from triangulation.triangulate import reconstruct_3d_ransac
recovery = reconstruct_3d_ransac(config, 2)

#%% Import 3D data
import pandas as pd
import numpy as np
import os
from numpy.linalg import norm
from matplotlib import pyplot as plt

cams = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
postfix = 'DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_180000_full.pickle'
iteration = 19
session_folder = 'pop_0811_1'
calib_folder = 'calib_0811_1'

# os_path = '/media/minyoungpark/T7 Touch/'
os_path = 'F:/'

calib_path = os.path.join(os_path, calib_folder)
pickle_folder =  os.path.join(os_path, session_folder, 'iteration-'+str(iteration))

pickle_paths = [os.path.join(pickle_folder, cam+postfix) for cam in cams]
paths_to_save = [os.path.join(pickle_folder, cam+'.csv') for cam in cams]

df_origin = pd.read_csv(os.path.join(pickle_folder,'output_3d_data_raw.csv'))

joints = ['CMC_thumb', 'MCP_thumb', 'IP_thumb', 'Tip_thumb',
          'MCP1', 'PIP1', 'Dip1', 'Tip1',
          'MCP2', 'PIP2', 'Dip2', 'Tip2', 
          'MCP3', 'PIP3', 'Dip3', 'Tip3', 
          'MCP4', 'PIP4', 'Dip4', 'Tip4', 
          'Wrist']

scheme = [
  ["Wrist", "CMC_thumb"],
  ["CMC_thumb", "MCP_thumb"],
  ["MCP_thumb", "IP_thumb"],
  ["IP_thumb", "Tip_thumb"],
  ["Wrist", "MCP1"],
  ["Wrist", "MCP2"],
  ["Wrist", "MCP3"],
  ["Wrist", "MCP4"],
  ["MCP1", "PIP1"],
  ["MCP2", "PIP2"],
  ["MCP3", "PIP3"],
  ["MCP4", "PIP4"],
  ["PIP1", "Dip1"],
  ["PIP2", "Dip2"],
  ["PIP3", "Dip3"],
  ["PIP4", "Dip4"],
  ["Dip1", "Tip1"],
  ["Dip2", "Tip2"],
  ["Dip3", "Tip3"],
  ["Dip4", "Tip4"]]

fingers = [[
  ["Wrist", "CMC_thumb"],
  ["CMC_thumb", "MCP_thumb"],
  ["MCP_thumb", "IP_thumb"],
  ["IP_thumb", "Tip_thumb"]],
    
  [["Wrist", "MCP1"],
  ["MCP1", "PIP1"],
  ["PIP1", "Dip1"],
  ["Dip1", "Tip1"]],
    
  [["Wrist", "MCP2"],
  ["MCP2", "PIP2"],
  ["PIP2", "Dip2"],
  ["Dip2", "Tip2"]],

  [["Wrist", "MCP3"],
  ["MCP3", "PIP3"],
  ["PIP3", "Dip3"],
  ["Dip3", "Tip3"]],
    
  [["Wrist", "MCP4"],
  ["MCP4", "PIP4"],
  ["PIP4", "Dip4"],
  ["Dip4", "Tip4"]]]


fig1 = plt.figure()
joint_angles = []
connection_lengths = []
bins = np.arange(0,100,2)
dist_meds = []

for i, finger in enumerate(fingers):
    finger_meds = []
    for j, connection in enumerate(finger):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df_origin[poi1+'_x'], 
                               df_origin[poi1+'_y'], 
                               df_origin[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df_origin[poi2+'_x'], 
                               df_origin[poi2+'_y'], 
                               df_origin[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        connection_lengths.append(dist)
        dist_med = np.median(dist[np.isfinite(dist)])
        finger_meds.append(dist_med*1.5)
        dist_std = np.std(dist[np.isfinite(dist)])
        nan_prop = np.sum(np.isnan(dist))/len(dist)
        ax = fig1.add_subplot(5,4,i*4+j+1)
        ax.hist(dist, bins, density=True)
        ax.set_title(poi1+' to '+poi2 + '\n' +
                     f'Med: {dist_med:.1f} Std: {dist_std:.1f} (mm) NAN prop: {nan_prop:.3f}')
        ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 0.3])
    dist_meds.append(finger_meds)
        
left = 0.05  # the left side of the subplots of the figure
right = 0.99   # the right side of the subplots of the figure
bottom = 0.03  # the bottom of the subplots of the figure
top = 0.94     # the top of the subplots of the figure
wspace = 0.12  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.54 # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig1.subplots_adjust(left, bottom, right, top, wspace, hspace)

#%% LPF Speed
# LPF speed vs time
# double peak
import more_itertools as mit
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt, freqz, freqs, lfilter

def butter_lowpass_filter(data):    
    n = len(data)  # total number of samples
    fs = 30       # sample rate, Hz
    T = n/fs         # Sample Period
    cutoff = 8      # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 4       # sin wave can be approx represented as quadratic

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    # zero phase
    # filtfilt acausal filter
    y = filtfilt(b, a, data)
    output = {'y': y,
              'b': b,
              'a': a}
    return output

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

fig = plt.figure()
t = np.arange(20*30+20, (30+20)*30+20)
speed_thr = [0.6, 0.75, 0.9, 1.2]
joint_speeds = []

for i, joint in enumerate(joints):
    joint_coord = np.stack([df_origin[joint+'_x'], 
                            df_origin[joint+'_y'], 
                            df_origin[joint+'_z']], axis=1)
    joint_speed = norm(np.diff(joint_coord, axis=0), axis=1)/(1/30)/1000
    joint_speeds.append(joint_speed)
    speed_sub = joint_speed[t].copy()
    
    nans_consec = [i for i in range(len(speed_sub)) if np.isnan(speed_sub[i])]
    nans_groups = [list(group) for group in mit.consecutive_groups(nans_consec)]
    
    for group in nans_groups:
        if len(group) <= 5:
            if group[-1]+1 < len(speed_sub):
                group_expand = [group[0]-1] + group + [group[-1]+1]
                speed_sub_sub = speed_sub[group_expand].copy()
                nans, fun = nan_helper(speed_sub_sub)
                
                speed_sub_sub[nans]= np.interp(fun(nans), fun(~nans), speed_sub_sub[~nans])
                
                speed_sub[group_expand] = speed_sub_sub
            
    where_finite = [i for i in range(len(speed_sub)) if np.isfinite(speed_sub[i])]
    finite_groups = [list(group) for group in mit.consecutive_groups(where_finite)]
    
    for group in finite_groups:
        # if len(group) > 1:
        speed_sub_sub = speed_sub[group].copy()
        lpf_s = butter_lowpass_filter(speed_sub_sub)
        speed_sub[group] = lpf_s['y']
    
    if joint.startswith('MCP') or joint.startswith('Wrist') or joint.startswith('CMC') or \
        joint.startswith('IP') or joint.startswith('Tip_thumb'):
        y = speed_thr[0]
    elif joint.startswith('PIP'):
        y = speed_thr[1]
    elif joint.startswith('Dip'):
        y = speed_thr[2]
    elif joint.startswith('Tip'):
        y = speed_thr[3]
    
    ax = fig.add_subplot(6,4,i+1)
    ax.plot(t, speed_sub)
    ax.hlines(y, t[0], t[-1], colors='r')
    ax.set_title(joint)
    
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

#%% New speed cut
fig = plt.figure()
df_speed2 = df_origin.copy()
bins = np.arange(0,1.5,0.02)
for i, (joint, joint_speed) in enumerate(zip(joints, joint_speeds)):
    # speed_thr = [0.6, 0.75, 0.9, 1.2]
    if joint.startswith('MCP') or joint.startswith('Wrist') or joint.startswith('CMC') or \
        joint.startswith('IP') or joint.startswith('Tip_thumb'):
        y = speed_thr[0]
    elif joint.startswith('PIP'):
        y = speed_thr[1]
    elif joint.startswith('Dip'):
        y = speed_thr[2]
    elif joint.startswith('Tip'):
        y = speed_thr[3]
    
    above_thr = np.squeeze(np.argwhere(joint_speed > y) + 1)
    df_speed2[joint+'_x'][above_thr] = np.nan
    df_speed2[joint+'_y'][above_thr] = np.nan
    df_speed2[joint+'_z'][above_thr] = np.nan
    
    joint_coord = np.stack([df_speed2[joint+'_x'], 
                            df_speed2[joint+'_y'], 
                            df_speed2[joint+'_z']], axis=1)
    
    speed = norm(np.diff(joint_coord, axis=0), axis=1)/(1/30)/1000
    
    ax = fig.add_subplot(6,4,i+1)
    ax.hist(speed, bins)
    ax.set_title(joint)
    ax.set_ylim([0,3000])
    ax.set_xlim([0,1.5])
    
left = 0.05  # the left side of the subplots of the figure
right = 0.99   # the right side of the subplots of the figure
bottom = 0.03  # the bottom of the subplots of the figure
top = 0.97     # the top of the subplots of the figure
wspace = 0.12  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.42  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

df_speed2.to_csv(os.path.join(pickle_folder, 'output_3d_data_speed.csv'), index=False)

#%% Compute median connection lengths 
# , and remove connection outliers
# df = df_speed.copy()
df = df_speed2.copy()
dist_meds = []
for i, finger in enumerate(fingers):
    finger_meds = []
    for j, connection in enumerate(finger):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df[poi1+'_x'], df[poi1+'_y'], df[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df[poi2+'_x'], df[poi2+'_y'], df[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        dist_med = np.median(dist[np.isfinite(dist)])
        finger_meds.append(dist_med)
    dist_meds.append(finger_meds)


def remove_finger_connection2(df_origin, finger, lowers, uppers):
    df = df_origin.copy()
    start = 1 # MCP's
    for j in range(start, len(finger)):
        lower = lowers[j]
        upper = uppers[j]
        poi1 = finger[j][0]
        poi2 = finger[j][1]
        poi1_coord = np.stack([df[poi1+'_x'], df[poi1+'_y'], df[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df[poi2+'_x'], df[poi2+'_y'], df[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        
        df[poi2+'_x'][(dist < lower) | (dist > upper)] = np.nan
        df[poi2+'_y'][(dist < lower) | (dist > upper)] = np.nan
        df[poi2+'_z'][(dist < lower) | (dist > upper)] = np.nan
        
    for j in range(0, j):
        lower = lowers[j]
        upper = uppers[j]
        poi1 = finger[j][0]
        poi2 = finger[j][1]
        poi1_coord = np.stack([df[poi1+'_x'], df[poi1+'_y'], df[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df[poi2+'_x'], df[poi2+'_y'], df[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        
        df[poi1+'_x'][(dist < lower) | (dist > upper)] = np.nan
        df[poi1+'_y'][(dist < lower) | (dist > upper)] = np.nan
        df[poi1+'_z'][(dist < lower) | (dist > upper)] = np.nan
    
    return df


left = 0.05  # the left side of the subplots of the figure
right = 0.99   # the right side of the subplots of the figure
bottom = 0.03  # the bottom of the subplots of the figure
top = 0.94     # the top of the subplots of the figure
wspace = 0.12  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.54 # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height

fig = plt.figure()
bins = np.arange(0, 100, 2)
for i, finger in enumerate(fingers):
    for j, connection in enumerate(finger):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df[poi1+'_x'],
                               df[poi1+'_y'], 
                               df[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df[poi2+'_x'], 
                               df[poi2+'_y'], 
                               df[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        dist_mean = np.mean(dist[np.isfinite(dist)])
        dist_med = np.median(dist[np.isfinite(dist)])
        dist_std = np.std(dist[np.isfinite(dist)])
        nan_prop = np.sum(np.isnan(dist))/len(dist)
        ax = fig.add_subplot(5,4,i*4+j+1)
        ax.hist(dist, bins, density=True)
        # ax.hist(dist, bins)
        ax.set_title(poi1+' to '+poi2 + '\n' +
                     f'Mean: {dist_mean:.1f} Std: {dist_std:.1f} (mm) NAN prop: {nan_prop:.3f}')
        ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        # ax.axvline(x=dist_meds_real[poi1+'-'+poi2][2], ymin=0, ymax=1, 
        #            color='tab:olive', linewidth=3, alpha=0.6)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 0.3])
        
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

fig4 = plt.figure()
dist_means = []
dist_stds = []
where_outliers = {}
dist_meds2 = []
bins = np.arange(0, 100, 2)
for i, (finger, uppers) in enumerate(zip(fingers, dist_meds)):
    df = remove_finger_connection2(df, finger, np.array(uppers)*0.6, np.array(uppers)*1.4)
    # where_outliers.append(where_outlier)
    finger_means = []
    finger_meds = []
    finger_stds = []
    for j, connection in enumerate(finger):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df[poi1+'_x'],
                               df[poi1+'_y'], 
                               df[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df[poi2+'_x'], 
                               df[poi2+'_y'], 
                               df[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        # dist[np.where(dist > 1.5*dist_med)] = np.nan
        # dist[np.where(dist < 0.5*dist_med)] = np.nan
        dist_mean = np.mean(dist[np.isfinite(dist)])
        dist_med = np.median(dist[np.isfinite(dist)])
        finger_means.append(dist_mean)
        finger_meds.append(dist_med)
        dist_std = np.std(dist[np.isfinite(dist)])
        finger_stds.append(dist_std)
        nan_prop = np.sum(np.isnan(dist))/len(dist)
        ax = fig4.add_subplot(5,4,i*4+j+1)
        ax.hist(dist, bins, density=True)
        ax.set_title(poi1+' to '+poi2 + '\n' +
                     f'Med: {dist_med:.1f} Std: {dist_std:.1f} (mm) NAN prop: {nan_prop:.3f}')
        ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 0.3])
        
    dist_means.append(finger_means)
    dist_meds2.append(finger_meds)
    dist_stds.append(finger_stds)
        
fig4.subplots_adjust(left, bottom, right, top, wspace, hspace)

df.to_csv(os.path.join(pickle_folder, 'output_3d_data_out1.csv'), index=False)
# Comparison between measured lengths after removing outliers

dist_meds_real = {'Wrist-CMC_thumb': [23.77, 20.83, 23.26],
 'CMC_thumb-MCP_thumb': [18.73, 24.96, 22.56],
 'MCP_thumb-IP_thumb': [14.25, 18.23, 17.72],
 'IP_thumb-Tip_thumb': [11.66, 9.87, 10.53],
 'Wrist-MCP1': [43.86, 42.38, 35.63],
 'Wrist-MCP2': [42.81, 39.57, 35.35],
 'Wrist-MCP3': [42.81, 37.46, 36.32],
 'Wrist-MCP4': [40.50, 40.22, 36.89],
 'MCP1-PIP1': [26.32, 27.40, 25.19],
 'MCP2-PIP2': [30.07, 33.44, 30.00],
 'MCP3-PIP3': [30.27, 32.70, 28.53],
 'MCP4-PIP4': [21.95, 24.52, 24.44],
 'PIP1-Dip1': [18.01, 16.92, 13.58],
 'PIP2-Dip2': [23.08, 20.38, 18.19],
 'PIP3-Dip3': [21.98, 21.16, 20.16],
 'PIP4-Dip4': [18.29, 18.25, 14.78],
 'Dip1-Tip1': [11.54, 12.82, 11.55],
 'Dip2-Tip2': [16.18, 15.39, 12.98],
 'Dip3-Tip3': [14.72, 13.34, 14.15],
 'Dip4-Tip4': [11.17, 11.64, 10.57]}

dist_meds3 = []
fig5 = plt.figure()
# where_outliers = {}
for i, (finger, med, std) in enumerate(zip(fingers, dist_meds2, dist_stds)):
    df = remove_finger_connection2(df, finger, np.array(med)-2.0*np.array(std), 
                                                  np.array(med)+2.0*np.array(std))
    # df = remove_finger_connection2(df, finger, np.array(med)*0.75, 
    #                                               np.array(med)*1.25)
    finger_meds = []
    for j, connection in enumerate(finger):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df[poi1+'_x'],
                               df[poi1+'_y'], 
                               df[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df[poi2+'_x'], 
                               df[poi2+'_y'], 
                               df[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        # dist[np.where(dist > 1.5*dist_med)] = np.nan
        # dist[np.where(dist < 0.5*dist_med)] = np.nan
        dist_mean = np.mean(dist[np.isfinite(dist)])
        dist_med = np.median(dist[np.isfinite(dist)])
        finger_meds.append(dist_med)
        dist_std = np.std(dist[np.isfinite(dist)])
        nan_prop = np.sum(np.isnan(dist))/len(dist)
        ax = fig5.add_subplot(5,4,i*4+j+1)
        ax.hist(dist, bins, density=True)
        ax.set_title(poi1+' to '+poi2 + '\n' +
                     f'Med: {dist_med:.1f} Std: {dist_std:.1f} (mm) NAN prop: {nan_prop:.3f}')
        ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        # ax.axvline(x=dist_meds_real[poi1+'-'+poi2][0], ymin=0, ymax=1, 
        #            color='tab:orange', linewidth=3)
        # ax.axvline(x=dist_meds_real[poi1+'-'+poi2][1], ymin=0, ymax=1, 
        #            color='tab:red', linewidth=3)
        ax.axvline(x=dist_meds_real[poi1+'-'+poi2][2], ymin=0, ymax=1, 
                   color='tab:olive', linewidth=3, alpha=0.6)
        # ax.axvline(x=np.mean(dist_meds_real[poi1+'-'+poi2]), ymin=0, ymax=1, 
        #            color='tab:pink', linewidth=3)
        ax.set_xlim([0, 100])
        # ax.legend(['Measure 1', 'Measure 2', 'Measure 3'])
        ax.set_ylim([0, 0.3])
    dist_meds3.append(finger_meds)
    
fig5.subplots_adjust(left, bottom, right, top, wspace, hspace)

df.to_csv(os.path.join(pickle_folder, 'output_3d_data_out2.csv'), index=False)

#%% 3D filter
import os
import smooth
import more_itertools as mit
from matplotlib import pyplot as plt
from scipy.signal import butter,filtfilt, freqz, freqs, welch, periodogram, savgol_filter

from scipy.interpolate import CubicSpline

def butter_lowpass_filter(data):    
    n = len(data)  # total number of samples
    fs = 30       # sample rate, Hz
    T = n/fs         # Sample Period
    cutoff = 8      # desired cutoff frequency of the filter
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 4       # sin wave can be approx represented as quadratic

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # z, p, k = butter(order, normal_cutoff, btype='low', analog=False)
    
    y = filtfilt(b, a, data)
    output = {'y': y,
              'b': b,
              'a': a}
    return output


def filt_3d(interp_type, filt_type, nan_win, xyz):
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.
    
        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """
    
        return np.isnan(y), lambda z: z.nonzero()[0]
    
    
    nans_consec = [i for i in range(len(xyz)) if np.isnan(xyz[i][0])]
    nans_groups = [list(group) for group in mit.consecutive_groups(nans_consec)]
    
    xyz_filt = xyz.copy()
    
    for nans_group in nans_groups:
        if len(nans_group) <= nan_win:
            if (nans_group[0]-2 > 0) and (nans_group[-1]+2 < len(xyz)):
                expand_indices = [nans_group[0]-2] + [nans_group[0]-1] + \
                nans_group + [nans_group[-1]+1] + [nans_group[-1]+2]
                
                frames = np.arange(len(expand_indices))
                xyz_trim = xyz[expand_indices, :].copy()
                
                x = xyz[expand_indices, 0].copy()
                y = xyz[expand_indices, 1].copy()
                z = xyz[expand_indices, 2].copy()
                
                if interp_type == 'linear':
                    nans, x_fun = nan_helper(x)
                    _, y_fun = nan_helper(y)
                    _, z_fun = nan_helper(z)
                    
                    x[nans]= np.interp(x_fun(nans), x_fun(~nans), x[~nans])
                    y[nans]= np.interp(y_fun(nans), y_fun(~nans), y[~nans])
                    z[nans]= np.interp(z_fun(nans), z_fun(~nans), z[~nans])
                    
                    xyz_lin = np.stack([x,y,z], axis=1)
                    xyz_filt[nans_group] = xyz_lin[2:-2]
                    
                elif interp_type == 'spline':
                    nans, _ = nan_helper(x)
                    cs = CubicSpline(frames[~nans], [x[~nans], y[~nans], z[~nans]], axis=1)
                    xyz_trim[nans] = cs(frames[nans]).T
                    xyz_filt[nans_group] = xyz_trim[2:-2]
        
    # xyz = smooth.smooth(xyz, best_tol, best_sigR, keepOriginal=True)
    
    if filt_type == None:
        return xyz_filt
    else:
        where_finite = np.arange(len(xyz_filt))
        where_finite = where_finite[np.isfinite(xyz_filt[:, 0])]
        finite_groups = [list(group) for group in mit.consecutive_groups(where_finite)]
        
        for group in finite_groups:
            x = xyz_filt[group, 0].copy()
            y = xyz_filt[group, 1].copy()
            z = xyz_filt[group, 2].copy()
            
            if filt_type == 'savgol' and len(y) > 11:
                sg_xyz = savgol_filter(np.stack([x, y, z]), 7, 5)
                xyz[group] = sg_xyz.T
            elif filt_type == 'lpf' and len(y) > 18:
                lpf_x = butter_lowpass_filter(x)
                lpf_y = butter_lowpass_filter(y)
                lpf_z = butter_lowpass_filter(z)
                xyz_filt[group, 0] = lpf_x['y']
                xyz_filt[group, 1] = lpf_y['y']
                xyz_filt[group, 2] = lpf_z['y']                
        return xyz_filt

joints = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']


path = os.path.join(pickle_folder,'output_3d_data_out2.csv')
df = pd.read_csv(path)
df_filt = df.copy()
df_interp = df.copy()
df_sg = df.copy()

for joint in joints:
    x = df[joint+'_x'].copy()
    y = df[joint+'_y'].copy()
    z = df[joint+'_z'].copy()
    coords = np.stack([x,y,z]).T
    xyz_interp = filt_3d('linear', None, 4, coords)
    df_interp[joint+'_x'] = xyz_interp[:, 0]
    df_interp[joint+'_y'] = xyz_interp[:, 1]
    df_interp[joint+'_z'] = xyz_interp[:, 2]

for joint in joints:
    x = df[joint+'_x'].copy()
    y = df[joint+'_y'].copy()
    z = df[joint+'_z'].copy()
    coords = np.stack([x,y,z]).T
    xyz_filt = filt_3d('linear', 'lpf', 4, coords)
    df_filt[joint+'_x'] = xyz_filt[:, 0]
    df_filt[joint+'_y'] = xyz_filt[:, 1]
    df_filt[joint+'_z'] = xyz_filt[:, 2]

for joint in joints:
    x = df[joint+'_x'].copy()
    y = df[joint+'_y'].copy()
    z = df[joint+'_z'].copy()
    coords = np.stack([x,y,z]).T
    xyz_filt = filt_3d('linear', 'savgol', 4, coords)
    df_sg[joint+'_x'] = xyz_filt[:, 0]
    df_sg[joint+'_y'] = xyz_filt[:, 1]
    df_sg[joint+'_z'] = xyz_filt[:, 2]
    
df_interp.to_csv(os.path.join(pickle_folder, 'output_3d_data_interp.csv'), index=False)
df_filt.to_csv(os.path.join(pickle_folder, 'output_3d_data_lpf.csv'), index=False)
df_sg.to_csv(os.path.join(pickle_folder, 'output_3d_data_sg.csv'), index=False)

fig = plt.figure()
bins = np.arange(0, 100, 2)
dist_meds4 = []
for i, finger in enumerate(fingers):
    finger_meds = []
    for j, connection in enumerate(finger):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df_interp[poi1+'_x'],
                               df_interp[poi1+'_y'], 
                               df_interp[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df_interp[poi2+'_x'], 
                               df_interp[poi2+'_y'], 
                               df_interp[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        
        dist_mean = np.mean(dist[np.isfinite(dist)])
        dist_med = np.median(dist[np.isfinite(dist)])
        dist_std = np.std(dist[np.isfinite(dist)])
        nan_prop = np.sum(np.isnan(dist))/len(dist)
        finger_meds.append(dist_med)
        ax = fig.add_subplot(5,4,i*4+j+1)
        ax.hist(dist, bins)
        ax.set_title(poi1+' to '+poi2 + '\n' +
                     f'Mean: {dist_mean:.1f} Std: {dist_std:.1f} (mm) NAN prop: {nan_prop:.3f}')
        # ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        # ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        # ax.axvline(x=dist_meds_real[poi1+'-'+poi2][-1]-2, ymin=0, ymax=1, 
        #            color='r', linewidth=0.5)
        # ax.axvline(x=dist_meds_real[poi1+'-'+poi2][-1]+2, ymin=0, ymax=1, 
        #            color='r', linewidth=0.5)
        ax.set_xlim([0, 100])
    dist_meds4.append(finger_meds)
        
left = 0.05  # the left side of the subplots of the figure
right = 0.99   # the right side of the subplots of the figure
bottom = 0.03  # the bottom of the subplots of the figure
top = 0.97     # the top of the subplots of the figure
wspace = 0.12  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.42  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

#%% Additional cut
path_interp = os.path.join(pickle_folder, 'output_3d_data_interp.csv')
df_interp_cut = pd.read_csv(path)
path = os.path.join(pickle_folder, 'output_3d_data_lpf.csv')
df_filt_cut = pd.read_csv(path)

fig = plt.figure()

for i, (finger, meds) in enumerate(zip(fingers, dist_meds4)):
    df_interp_cut = remove_finger_connection2(df_interp_cut, finger, 
                                              np.array(meds)*0.6, np.array(meds)*1.4)
    df_filt_cut = remove_finger_connection2(df_filt_cut, finger, 
                                            np.array(meds)*0.6, np.array(meds)*1.4)
    for j, connection in enumerate(finger):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df_interp_cut[poi1+'_x'],
                               df_interp_cut[poi1+'_y'], 
                               df_interp_cut[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df_interp_cut[poi2+'_x'], 
                               df_interp_cut[poi2+'_y'], 
                               df_interp_cut[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        dist_mean = np.mean(dist[np.isfinite(dist)])
        dist_med = np.median(dist[np.isfinite(dist)])
        dist_std = np.std(dist[np.isfinite(dist)])
        nan_prop = np.sum(np.isnan(dist))/len(dist)
        ax = fig.add_subplot(5,4,i*4+j+1)
        ax.hist(dist, bins, density=True)
        ax.set_title(poi1+' to ' + poi2 + '\n' +
                     f'Med: {dist_med:.1f} '
                     'Std: {dist_std:.1f} (mm) '\
                     'Prop. missing: {nan_prop:.3f}')
        ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.axvline(x=dist_meds_real[poi1+'-'+poi2][2], ymin=0, ymax=1, 
                   color='tab:olive', linewidth=3, alpha=0.6)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 0.3])
        
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)
df_interp_cut.to_csv(os.path.join(pickle_folder, 'output_3d_data_interp_cut.csv'), index=False)
df_filt_cut.to_csv(os.path.join(pickle_folder, 'output_3d_data_lpf_cut.csv'), index=False)