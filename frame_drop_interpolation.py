# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:51:24 2021

@author: robin
"""
#%% Import

import os
import pandas as pd
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt

cams = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
postfix = 'DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_180000_full.pickle'
iteration = 19
session_folder = 'pop_1103'
calib_folder = 'calib_1103'

# os_path = '/media/minyoungpark/T7 Touch/'
os_path = 'F:/'
calib_path = os.path.join(os_path, calib_folder)
pickle_folder =  os.path.join(os_path, session_folder, 'iteration-'+str(iteration))
pickle_paths = [os.path.join(pickle_folder, cam+postfix) for cam in cams]
paths_to_save = [os.path.join(pickle_folder, cam+'.csv') for cam in cams]

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

#%% Convert pickle to csv
from utils.data_utils import convertPickleToCSV

convertPickleToCSV(postfix, joints, pickle_paths, paths_to_save)

# %% Interpolate 2D
from utils.data_utils import interpDroppedData

csv_paths = [os.path.join(pickle_folder, cam+'.csv') for cam in cams]
timestamp_paths = [os.path.join(os_path, session_folder, cam+'_logfile.txt') for cam in cams]
paths_to_save = [os.path.join(pickle_folder, cam+'_interp.csv') for cam in cams]

interpDroppedData(postfix, joints, csv_paths, timestamp_paths, paths_to_save)

#%% Extract frames
import os
from utils.vis_utils import extract_frames

img_format = 'png'
vidfolder = '/media/minyoungpark/T7 Touch/pop_1217/'
output_folder = '/media/minyoungpark/T7 Touch/for_raquel/'
vidnames = ['cam_0.avi']
folders = [vid.split('.')[0] for vid in vidnames]
vidpaths = [os.path.join(vidfolder, vidname) for vidname in vidnames]

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
paths_to_save = [os.path.join(output_folder, folder) for folder in folders]
for vidpath, path_to_save in zip(vidpaths, paths_to_save):
    extract_frames(vidpath, times, every_n_frames, path_to_save, img_format)
    

    
#%%
