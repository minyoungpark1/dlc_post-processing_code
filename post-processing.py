#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:59:49 2021

@author: minyoungpark
"""
#%% Import

import re
import os
import pandas as pd
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt

cams = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
postfix = 'DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_180000_full.pickle'
iteration = 14
# session_folder = 'pop_0811_1'
# calib_folder = 'calib_0811'
# session_folder = 'groot_0218'
# calib_folder = 'calib_0218'
# session_folder = 'groot_0216'
# calib_folder = 'calib_0216'
session_folder = 'pop_1217'
calib_folder = 'calib_1217'

# os_path = '/media/minyoungpark/T7 Touch/'
# os_path = 'F:/'
os_path = '/Volumes/T7 Touch/'
calib_path = os.path.join(os_path, calib_folder)
pickle_folder =  os.path.join(os_path, session_folder, 'iteration-'+str(iteration))
pickle_paths = [os.path.join(pickle_folder, cam+postfix) for cam in cams]
paths_to_save_raw_csv = [os.path.join(pickle_folder, cam+'.csv') for cam in cams]

joints = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']

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

# Measured connection length in mm
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

#%% Convert pickle to csv
from utils.data_utils import convertPickleToCSV

convertPickleToCSV(postfix, joints, pickle_paths, paths_to_save_raw_csv)

#%% Create labeled videos
import os
import cv2
import pandas as pd 
from tqdm import trange

vidpath = 'F:/pop_1217/cam_0.avi'
datapath = 'F:/pop_1217/iteration-19/cam_0.csv'

data_2d = pd.read_csv(datapath, header=[2,3], index_col=0)
cap = cv2.VideoCapture(vidpath) 
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_num/fps
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(os.path.dirname(vidpath), 'output.avi'),fourcc, 30.0, (2048,1536))


colorclass = plt.cm.ScalarMappable(cmap='jet')
C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
colors = C[:, :3]

# for i in trange(frame_num):
#     ret, frame = cap.read()
    
#     out.write(frame)
    
# cap.release()
# out.release()

# %% Interpolate 2D
from utils.data_utils import interpDroppedData2

csv_paths = [os.path.join(pickle_folder, cam+'.csv') for cam in cams]
timestamp_paths = [os.path.join(os_path, session_folder, cam+'_logfile.txt') for cam in cams]
paths_to_save_interp_csv = [os.path.join(pickle_folder, cam+'_interp.csv') for cam in cams]

# interpDroppedData2(postfix, joints, csv_paths, timestamp_paths, paths_to_save_interp_csv)
# 
#%% First extract images and manually delete repetitive images
from utils.vis_utils import extract_frames

img_format = 'png'

# Path to folder where videos are located
vidfolder = '/media/minyoungpark/T7 Touch/pop_1217/'

vidnames = ['cam_0.avi']
vidpaths = [os.path.join(vidfolder, vidname) for vidname in vidnames]

# Path to parent folder where you want to save extracted images
output_folder = '/media/minyoungpark/T7 Touch/for_raquel/'
# Full path to save images
ext = '0218'
folders = [vid.split('.')[0] + '_' + ext for vid in vidnames]
paths_to_save_imgs1 = [os.path.join(output_folder, folder) for folder in folders]

# Manually find time intervals (in seconds) to extract and label
times = [(11, 14),
         (16, 18),
         (20, 22),
         (24, 26),
         (28, 32),
         (36, 39),
         (43, 45),
         (67, 69),
         (73, 75),
         (103, 107)]
every_n_frames = 5
# for vidpath, path_to_save in zip(vidpaths, paths_to_save_imgs1):
#     extract_frames(vidpath, times, every_n_frames, path_to_save, img_format)
    
#%% Extract images with corresponding frame counts after deleting repetitive images
from utils.vis_utils import extract_specific_frames

vidnames = ['cam_1.avi',
            'cam_2.avi', 'cam_3.avi']
vidpaths = [os.path.join(vidfolder, vidname) for vidname in vidnames]

# Full path to save images
folders = [vid.split('.')[0] + '_' + ext for vid in vidnames]
paths_to_save_imgs2 = [os.path.join(output_folder, folder) for folder in folders]

# Read remaining images after you manually deleted repetitive images
images_folder = paths_to_save_imgs1[0]
image_indices = [int(re.findall(r'\d+', file)[0]) 
                 for file in os.listdir(images_folder) if file.endswith('.'+img_format)]


# for vidpath, path_to_save in zip(vidpaths, paths_to_save_imgs2):
#     extract_specific_frames(vidpath, image_indices, path_to_save, img_format)
    

#%% Load config file
from utils.utils import load_config
from calibration.intrinsic import calibrate_intrinsic
from calibration.extrinsic import calibrate_extrinsic

config = load_config('config_pop.toml')
config['paths_to_2d_data'] = paths_to_save_interp_csv
config['calibration']['calib_video_path'] = calib_path
config['output_video_path'] = pickle_folder
config['triangulation']['reconstruction_output_path'] = pickle_folder

#%% Calibration
calibrate_intrinsic(config)
calibrate_extrinsic(config)

#%% 3D reconstruction
from triangulation.triangulate import reconstruct_3d_ransac
recovery = reconstruct_3d_ransac(config, 2)

#%% Outlier removal
from post_processing.outlier_removal import outlier_speed_removal, \
    outlier_connection_removal
from post_processing.filtering import filter_3d

# Representative speed threshold for points of interest in m/s
# Speed of  PIP:  0.75 m/s
#           DIP:  0.9  m/s
#           Tip:  1.2  m/s
#           rest: 0.6  m/s
speed_thr = [0.6, 0.75, 0.9, 1.2]
# This function removes any outlier points that passed over the speed threshold
# This function will save "output_3d_data_speed.csv" inside "pickle_folder"
outlier_speed_removal(config, joints, speed_thr)

# This function 1) removes any connection that is outside of (0.6*med, 1.4*med) range,
#               2) and then (med-2*std, med+2*std)
# This function will save "output_3d_data_out1.csv" and "output_3d_data_out2.csv"
outlier_connection_removal(config, fingers)

# This function perfrom 1) interpolation, 2) filter, and 3) remove outlier connection
# There are two kinds of interpolation methods: linear and cubic spline
# Interpolate at most 4 missing consecutive data
# There are three kinds of filter you can use: lpf, savgol, None
# lpf: 8 Hz zero-phase low pass filter (output_3d_data_lpf.csv)
# savgol: Savitzky–Golay filter with 5th order and window size of 7 (output_3d_data_savgol.csv)
# interp: Doesn't apply filter, it just perform interpolation (output_3d_data_None.csv)
interp_type = 'linear'
filt_type = 'lpf'
filter_3d(config, joints, fingers, interp_type, filt_type)

#%% Generate 3D video   
from utils.vis_utils import generate_three_dim_video
generate_three_dim_video(config)

#%% Connection length

path = os.path.join(pickle_folder,'output_3d_data_interp.csv')
df = pd.read_csv(path)
left = 0.02  # the left side of the subplots of the figure
right = 0.99   # the right side of the subplots of the figure
bottom = 0.03  # the bottom of the subplots of the figure
top = 0.94     # the top of the subplots of the figure
wspace = 0.12  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.54 # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height


fig = plt.figure(figsize=(16, 9))
bins = np.arange(0, 100, 2)
dist_meds = []
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
        dist_med = np.median(dist[np.isfinite(dist)])
        dist_meds.append(dist_med)
        dist_std = np.std(dist[np.isfinite(dist)])
        nan_prop = np.sum(np.isnan(dist))/len(dist)
        ax = fig.add_subplot(5,4,i*4+j+1)
        ax.hist(dist, bins, density=True)
        # ax.hist(dist, bins)
        ax.set_title(poi1+' to '+poi2 + '\n' +
                     f'Med: {dist_med:.1f} Std: {dist_std:.1f} (mm) NAN prop: {nan_prop:.3f}')
        ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
        # ax.axvline(x=dist_meds_real[poi1+'-'+poi2][2], ymin=0, ymax=1, color='tab:olive', linewidth=3, alpha=0.6)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 0.3])
        
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

#%% 
iteration = 19
session_folder = 'pop_1217'
pickle_folder =  os.path.join(os_path, session_folder, 'iteration-'+str(iteration))
import scipy.io as sio
df = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_interp_cut.csv'))
testSave = {name: col.values for name, col in df.items()}
saveDict = {'testDict': testSave}
sio.savemat(os.path.join(pickle_folder, 'output_3d_data_interp.mat'), saveDict)

dist_meds_real = {'Wrist-CMC_thumb': [23.26],
                  'CMC_thumb-MCP_thumb': [22.56],
                  'MCP_thumb-IP_thumb': [17.72],
                  'IP_thumb-Tip_thumb': [10.53],
                  'Wrist-MCP1': [35.63],
                  'Wrist-MCP2': [35.35],
                  'Wrist-MCP3': [36.32],
                  'Wrist-MCP4': [36.89],
                  'MCP1-PIP1': [25.19],
                  'MCP2-PIP2': [30.00],
                  'MCP3-PIP3': [28.53],
                  'MCP4-PIP4': [24.44],
                  'PIP1-Dip1': [13.58],
                  'PIP2-Dip2': [18.19],
                  'PIP3-Dip3': [20.16],
                  'PIP4-Dip4': [14.78],
                  'Dip1-Tip1': [11.55],
                  'Dip2-Tip2': [12.98],
                  'Dip3-Tip3': [14.15],
                  'Dip4-Tip4': [10.57]}