# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:12:29 2021

@author: robin
"""
#%% Import
import re
import os
import pandas as pd
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
# cam_0DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_140000_full.pickle
# cams = ['cam_0_trimmed2', 'cam_1_trimmed2', 'cam_2_trimmed2', 'cam_3_trimmed2']
cams = ['cam_0_trimmed', 'cam_1_trimmed', 'cam_2_trimmed', 'cam_3_trimmed']
# cams = ['cam_1']
# cams = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
# postfix = 'DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_190000_full.pickle'
# postfix = 'DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_140000_full.pickle'
postfix = 'DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_170000_full.pickle'
# postfix = 'DLC_resnet50_Pop_freeReach_0317_mergedApr19shuffle1_200000_full.pickle'
# cams = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
# pickle_folder =  '/media/minyoungpark/New Volume/pop_0903/iteration-13'
# iteration = 17
iteration = 19
session_folder = 'pop_0811_1'
calib_folder = 'calib_0811'
# session_folder = 'pop_1103'
# calib_folder = 'calib_1103'
# session_folder = 'pop_0827'
# calib_folder = 'calib_0827'
# session_folder = 'pop_0903'
# session_folder = 'pop_0610'
# session_folder = 'pop_1028'
# calib_folder = 'calib_1028'
# session_folder = 'pop_1217'
# calib_folder = 'calib_1217'

# os_path = '/media/minyoungpark/T7 Touch/'
os_path = 'F:/'
# os_path = '/Volumes/T7 Touch/'
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

#%% Data augmentation
import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from matplotlib import pyplot as plt

path = 'C:/Users/robin/Dropbox/Research/Miller/Codes/cam_calib/img03315.png'
image = np.expand_dims(cv2.cvtColor(cv2.imread(path), 
                                    cv2.COLOR_BGR2RGB), 
                       axis=0)[:, 700:1100, 1080:1480, :]
motion_blur = iaa.MotionBlur(k=15, angle=[-90, 90])(images=image)
color = iaa.MultiplyHueAndSaturation((0.7, 1.3), per_channel=True)(images=image)
rotation = iaa.Affine(rotate=(-180, 180))(images=image)

plt.imshow(np.squeeze(image))
plt.axis('off')
plt.savefig('original.png', bbox_inches='tight')

#%% Compute 2-D error - 1
path = os.path.join(pickle_folder, 'dist.csv')
df = pd.read_csv(path, index_col=[0,1,2], header=[0])
df = df.T

pickles = [file for file in os.listdir(pickle_folder) if file.endswith('.pickle') and \
           bool(re.search('meta', file))]

pickle_path = os.path.join(pickle_folder, pickles[1])
metadata = pd.read_pickle(pickle_path)

train_indices = metadata['data']['trainIndices']
test_indices = metadata['data']['testIndices']

train_error = np.nanmean(df.iloc[train_indices])
test_error = np.nanmean(df.iloc[test_indices])
train_error_std = np.nanstd(df.iloc[train_indices])
test_error_std = np.nanstd(df.iloc[test_indices])

print(f'Train error: {train_error:.2f} ± {train_error_std:.2f} (pixels)')
print(f'Test error: {test_error:.2f} ± {test_error_std:.2f} (pixels)')

#%% Compute 2-D error - 2
df.columns = df.columns.set_levels(joints, level=2)
errs_mean = np.zeros((2, len(joints)))
errs_std = np.zeros((2, len(joints)))


for j, joint in enumerate(joints):
    errs_mean[0, j] = np.nanmean(df['Min']['right_hand'][joint].iloc[train_indices])
    errs_std[0, j] = np.nanstd(df['Min']['right_hand'][joint].iloc[train_indices])
    errs_mean[1, j] = np.nanmean(df['Min']['right_hand'][joint].iloc[test_indices])
    errs_std[1, j] = np.nanstd(df['Min']['right_hand'][joint].iloc[test_indices])


# errs = np.empty((2, len(joints)), dtype=np.char)
errs_train = ['']*21
errs_test = ['']*21

for i in range(len(joints)):
    errs_train[i] = "{:1.2f} ± {:1.2f}".format(errs_mean[0, i], errs_std[0, i])
    errs_test[i] = "{:1.2f} ± {:1.2f}".format(errs_mean[1, i], errs_std[1, i])
        # cell_text.append(['%1.1f' % (x) for x in analysis])
    
#%% Compute 2-D error - Violin plot
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
train_errors = np.array(df.iloc[train_indices])
test_errors = np.array(df.iloc[test_indices])
train_errors_finite = []
test_errors_finite = []
for j in range(21):
    train_errors_finite.append(train_errors[np.isfinite(train_errors[:, j]), j])
    test_errors_finite.append(test_errors[np.isfinite(test_errors[:, j]), j])

fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(111)

def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    
labels = []
positions = np.arange(1, 22)

# ax.violinplot(train_errors_finite, showmeans=True, showextrema=False)
add_label(plt.violinplot(train_errors_finite, 
                         positions, showmeans=True, 
                         showextrema=False), "Train")    
# ax.violinplot(test_errors_finite, showmeans=True, showextrema=False)
add_label(plt.violinplot(test_errors_finite, 
                         positions, showmeans=True, 
                         showextrema=False), "Test")    

joints_label = ['Wrist', 'CMC of thumb', 'MCP of thumb', 
                'MCP 1', 'MCP 2', 'MCP 3', 'MCP 4',
          'IP of thumb', 'PIP 1', 'PIP 2', 'PIP 3', 'PIP 4', 
          'DIP 1', 'DIP 2', 'DIP 3', 'DIP 4',
          'Tip of thumb', 'Tip 1', 'Tip 2', 'Tip 3', 'Tip 4']

plt.xticks(np.arange(1,22), joints_label, rotation=45, fontsize=12)
# plt.set_xticklabels()
plt.ylim([0, 35])
plt.ylabel('Mean Euclidean Error (pixels)', fontsize=14)
plt.legend(*zip(*labels), loc=1, fontsize=12)
plt.tight_layout()

#%%
df_speed = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_speed.csv'))
df_cut1 = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_out1.csv'))

fig = plt.figure()
connection = ['PIP2', 'Dip2']
poi1 = connection[0]
poi2 = connection[1]
poi1_coord = np.stack([df_speed[poi1+'_x'],
                       df_speed[poi1+'_y'], 
                       df_speed[poi1+'_z']], axis=1)
poi2_coord = np.stack([df_speed[poi2+'_x'], 
                       df_speed[poi2+'_y'], 
                       df_speed[poi2+'_z']], axis=1)
dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
dist_med = np.median(dist[np.isfinite(dist)])
ax = fig.add_subplot(111)
ax.hist(dist, bins, density=True)
# ax.set_title(poi1+' to '+poi2 + '\n' +
#              f'Mean: {dist_mean:.1f} Std: {dist_std:.1f} (mm) NAN prop: {nan_prop:.3f}')
ax.axvline(x=dist_med*0.6, ymin=0, ymax=1, color='tab:red', linewidth=1.5)
ax.axvline(x=dist_med*1.4, ymin=0, ymax=1, color='tab:red', linewidth=1.5)
ax.set_xlabel('Connection length (mm)', fontsize=14)
ax.set_ylabel('Proportion', fontsize=14)
ax.set_xlim([0, 40])
ax.set_ylim([0, 0.3])

poi1 = connection[0]
poi2 = connection[1]
poi1_coord = np.stack([df_cut1[poi1+'_x'],
                       df_cut1[poi1+'_y'], 
                       df_cut1[poi1+'_z']], axis=1)
poi2_coord = np.stack([df_cut1[poi2+'_x'], 
                       df_cut1[poi2+'_y'], 
                       df_cut1[poi2+'_z']], axis=1)
dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
dist_med = np.median(dist[np.isfinite(dist)])
dist_std = np.std(dist[np.isfinite(dist)])
ax.axvline(x=dist_med-2*dist_std, ymin=0, ymax=1, color='tab:orange', linewidth=1.5)
ax.axvline(x=dist_med+2*dist_std, ymin=0, ymax=1, color='tab:orange', linewidth=1.5)
fig.tight_layout()



#%% 2D interpolation
import pandas as pd
import more_itertools as mit
df_origin = pd.read_csv(os.path.join(pickle_folder, 'cam_0.csv'), 
                        header=[2,3], index_col=0)
# df_origin = pd.read_csv(os.path.join(pickle_folder, 
#                                      'backup_0113', 'output_3d_data_lpf_cut.csv'))
# 
joints = ['PIP2']

xyzs_origin = []
for joint in joints:
    x = df_origin[joint, 'x']
    y = df_origin[joint, 'y']
    z = np.zeros((len(x),))
    # x = df_origin[joint+'_x']
    # y = df_origin[joint+'_y']
    # z = df_origin[joint+'_z']
    err = df_origin[joint, 'likelihood']
    xyz = np.stack([x,y,z], axis=1)
    xyz[err < 0.3] = np.nan
    xyzs_origin.append(xyz)
xyzs_origin = np.array(xyzs_origin)
xyzs_origin = np.transpose(xyzs_origin, (1,0,2))
xyzs_origin = xyzs_origin.reshape((len(xyzs_origin), -1))

tds = np.array(timestamp_diffs)
tt = tds.reshape(-1,)-1
drops = np.array([len(np.argwhere(tt == i))/len(tt) for i in range(5)])
drops = drops[drops > 1e-12][1:]
xlabel1 = np.arange(1,len(drops)+1)
xlabel2 = np.arange(0,len(drops)+2)
fig, ax = plt.subplots(1, 1)
ax.bar(xlabel1, drops, width = 0.25)
ax.set_xticks(xlabel2)
ax.set_xticklabels(['']+[str(i) for i in xlabel1]+[''])
# ax.set_xticklabels([str(i) for i in xlabel2l])
ax.set_ylim([0, np.max(drops)*1.05])
# ax.set_yticks([0, 1, 2])
# ax.set_yticklabels([0, 0.5, 1])
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.set_xlabel('Number of frames dropped consecutively', fontsize=12)
ax.set_ylabel('Proportion of frames', fontsize=12)
ax.set_title('Proportion of frames dropped consecutively', fontsize=14)
# ax.legend(rows, fontsize=11, loc='upper left')
fig.subplots_adjust(left=0.12, bottom=0.12)
tds = np.sum(tds, axis=0)
tds[tds != 4] = np.nan
tds = np.insert(tds, 0, np.nan)
xyzs_origin[np.isnan(tds)] = np.nan

where_finite = [i for i in range(len(xyzs_origin)) if np.isfinite(xyzs_origin[i,0])]
# where_finite = [i for i in range(len(tds)) if np.isfinite(tds[i])]
finite_groups = [list(group) for group in mit.consecutive_groups(where_finite)]
longest = np.argmax([len(group) for group in finite_groups])

#%%
import re
from numpy.linalg import norm
from scipy.interpolate import splprep, LSQBivariateSpline, CubicSpline
import more_itertools as mit
from numpy import array as arr

from numpy.linalg import norm

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
    
def interp_3d_test(data, win, ext, method):
    window = win + ext*2
    offsets = []
    for i in range(len(data)):
        if i+window >= len(data):
            continue
        
        xyz = data[i:i+window,:]
        xyz_drop = xyz.copy()
        xyz_drop[ext:ext+win,:] = np.nan
    
        x = xyz_drop[:,0]
        # print(xy_drop[i+ext:i+ext+win,:])
        y = xyz_drop[:,1]
        z = xyz_drop[:,2]
        
        nans, x_fun = nan_helper(x)
        _, y_fun = nan_helper(y)
        _, z_fun = nan_helper(y)
        if nans[0] or nans[-1]:
            continue
        
        if method == 'linear':
            x[nans] = np.interp(x_fun(nans), x_fun(~nans), x[~nans])
            y[nans] = np.interp(y_fun(nans), y_fun(~nans), y[~nans])
            z[nans] = np.interp(z_fun(nans), z_fun(~nans), z[~nans])
            
        elif method == 'spline':
            frames = np.arange(len(nans))
            cs = CubicSpline(frames[~nans], [x[~nans], y[~nans], z[~nans]], axis=1)
            interp = cs(frames[nans])
            x[nans] = interp[0,:]
            y[nans] = interp[1,:]
            z[nans] = interp[2,:]
        
        
        xyz_dist = np.stack([x,y,z], axis=1) - xyz
        # print(np.stack([x,y], axis=1))
        # print(xy)
        dist = norm(xyz_dist[nans], axis=1)
        # print(dist)
        # print(np.mean(dist))
        offsets.append(np.mean(dist))
    
    offsets = arr(offsets)
    offsets = offsets[np.isfinite(offsets)]
    return offsets

def dropping_data(data, win, ext):
    data_drop = data.copy()
    window = win + ext*2
    for i in range(0, len(data)-window, window):
        data_drop[i+ext:i+ext+win,:] = np.nan
    return data_drop

def interp_3d(data, origin, win, ext, method):
    data_interp = data.copy()
    window = win + ext*2
    offsets = []
    for i in range(0, len(data), window):
        if i+window >= len(data):
            continue
        
        xyz = origin[i:i+window,:]
        xyz_drop = data_interp[i:i+window,:].copy()
    
        x = xyz_drop[:,0]
        y = xyz_drop[:,1]
        z = xyz_drop[:,2]
        
        nans, x_fun = nan_helper(x)
        _, y_fun = nan_helper(y)
        _, z_fun = nan_helper(z)
        
        if nans[0] or nans[-1]:
            continue
        
        if method == 'linear':
            x[nans] = np.interp(x_fun(nans), x_fun(~nans), x[~nans])
            y[nans] = np.interp(y_fun(nans), y_fun(~nans), y[~nans])
            z[nans] = np.interp(z_fun(nans), z_fun(~nans), z[~nans])
            
        elif method == 'spline':
            frames = np.arange(len(nans))
            cs = CubicSpline(frames[~nans], [x[~nans], y[~nans], z[~nans]],
                               axis=1, bc_type='not-a-knot')
                              # axis=1, bc_type='clamped')
            interp = cs(frames[nans])
            x[nans] = interp[0,:]
            y[nans] = interp[1,:]
            z[nans] = interp[2,:]
        
        data_interp[i:i+window,:] = np.stack([x,y,z], axis=1)
        xyz_dist = np.stack([x,y,z], axis=1) - xyz
        dist = norm(xyz_dist[nans], axis=1)

    return data_interp
#%%
from numpy import array as arr

off_means = []
off_stds = []
# for joint in joints:
joint = joints[0]
# win = 4
# exts = np.arange(1, 6, dtype=int)
xyz = xyzs_origin[finite_groups[longest]]
# for ext in exts:
#     method = 'linear'
#     offsets = interp_3d_test(xyz, win, ext, method)
#     off_mean = np.mean(offsets)
#     off_std = np.std(offsets)
#     off_means.append(off_mean)
#     off_stds.append(off_std)
#     print(joint + '    ' + f'Mean: {off_mean:.2f} Std: {off_std:.2f} (mm)')
    
# wins = np.arange(1, 5, dtype=int)
wins = np.arange(1, len(drops)+1, dtype=int)
# wins = np.arange(1, 3, dtype=int)
ext = 2
methods = ['linear', 'spline']
for method in methods:
    for win in wins:
        # method = 'linear'
        offsets = interp_3d_test(xyz, win, ext, method)
        off_mean = np.mean(offsets)
        off_std = np.std(offsets)
        off_means.append(off_mean)
        off_stds.append(off_std)
        print(joint + '    ' + f'Mean: {off_mean:.2f} Std: {off_std:.2f} (mm)')

off_means = np.array(off_means).reshape(len(methods), len(wins))
off_stds = np.array(off_stds).reshape(len(methods), len(wins))

rows = ('Linear', 'Cubic Spline')
# columns = ('Win 1', 'Win 2', 'Win 3')
columns = ('Win 1', 'Win 2')

analysis = []
cell_text = []
y_ticks = np.arange(len(wins))
tick_offsets = np.arange(len(rows))*0.15
fig = plt.figure()
for i in range(2):
    analysis = off_means[i,:]
    err = off_stds[i,:]
    
    ax = fig.add_subplot(111)
    ax.bar(y_ticks+tick_offsets[i], analysis, 0.1, yerr=err)
    cell_text.append(['%1.1f' % (x) for x in analysis])
    
# table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom')
# table.auto_set_font_size(False)
# table.set_fontsize(11)
# ax.set_xticks(np.arange(0,4,dtype=int)+0.075)
ax.set_xticks(np.arange(0,len(drops),dtype=int)+0.075)
ax.set_xticklabels([str(i) for i in range(1,len(drops)+1)])
# ax.set_xticklabels([str(i) for i in range(1,5)])
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.set_xlabel('Number of frames dropped consecutively', fontsize=12)
ax.set_ylabel('Mean Pixel Error (pixel)', fontsize=12)
# ax.set_ylabel('Mean Euclidean Error (mm)', fontsize=12)
ax.set_ylim([0, np.max(off_means + off_stds)+0.5])
ax.set_title('Interpolation errors', fontsize=14)
ax.legend(rows, fontsize=11, loc='upper left')
fig.subplots_adjust(left=0.12, bottom=0.12)

#%% Plot 2D interpolation test
xyz_drop = dropping_data(xyz, 1, 2)
linear = interp_3d(xyz_drop, xyz, 1, 2, 'linear')[:30]
spline = interp_3d(xyz_drop, xyz, 1, 2, 'spline')[:30]
xyz_drop = xyz_drop[:30]
t = np.arange(len(xyz_drop))
# t = longest_seq

# axis = 1
axis_dict = {0: 'x',
             1: 'y',
             2: 'z'}

for axis in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xx = xyz_drop[:, axis]
    start_indices = []
    end_indices = []
    
    nans_consec = [i for i in range(len(xx)) if np.isnan(xx[i])]
    nans_groups = [list(group) for group in mit.consecutive_groups(nans_consec)]
    
    for nans_group in nans_groups:
        # else:
        end_indices.append(np.max([nans_group[-1]+1, 1]))
        start_indices.append(np.min([nans_group[0], len(xx)-1]))
    end_indices.append(len(xx))
    
    ax.plot(t, xyz[:30, axis], alpha=0.8)
    # ax.plot(t, xyz[:, axis], alpha=0.8)
    ax.plot(t, linear[:, axis], alpha=0.8)
    ax.plot(t, spline[:, axis], alpha=0.8)
    # ax.fill_between(nans_consec, xy[nans_consec, xory]-5, xy[nans_consec, xory]+5)
    for start_idx, end_idx in zip(start_indices, end_indices):
        fill_range = np.arange((start_idx-1), (end_idx+1), 1)
        print(fill_range)
        ax.fill_between(fill_range, 
                        # np.min([np.min(xyz[:, axis]), 
                        np.min([np.min(xyz[:30, axis]), 
                                np.min(linear[:, axis]),
                                np.min(spline[:, axis])]), 
                        # np.max([np.max(xyz[:, axis]), 
                        np.max([np.max(xyz[:30, axis]), 
                                np.max(linear[:, axis]),
                                np.max(spline[:, axis])]), 
                        color='k', alpha=0.1)
    #     ax.axvline(x=start_idx, ymin=0.2, ymax=0.5)
    # for end_idx in end_indices[:-1]:
    #     ax.axvline(x=end_idx, ymin=0.2, ymax=0.5)
    # ax.set_xticklabels([str[for ])
    ax.legend(['Original', 'Linear Interpolation', 'Cubic spline'], fontsize=12)
    ax.set_title('PIP 2 ' + axis_dict[axis] + '-trajectory', fontsize=14)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Coordinate (pixel)', fontsize=12)
    
#%% Plot trajectory
import cv2
from matplotlib import pyplot as plt
df = pd.read_csv(os.path.join(pickle_folder, 'cam_1.csv'), header=[2,3], index_col=0)
vidpath = os.path.join(os_path, session_folder, 'cam_1.avi')
period = 5
start = 2700
# start = 3470
# frame_count = start
joints = ['Wrist', 'MCP2', 'PIP2', 'Dip2', 'Tip2']
time = np.arange(start, start+period)
colorclass = plt.cm.ScalarMappable(cmap='jet')
C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
colors = C[:, :3]
cap = cv2.VideoCapture(vidpath) 
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_num/fps
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
ret, frame = cap.read()
cv2.imwrite(os.path.join(os_path, session_folder, 'traj_raw'+str(start)+'.png'), frame)
img = plt.imread(os.path.join(os_path, session_folder, 'traj_raw'+str(start)+'.png'))
plt.imshow(img)

# for t in time:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, t)
#     ret, frame = cap.read()
#     cv2.imwrite(os.path.join(os_path, session_folder, 'traj_raw'+str(t)+'.png'), frame)

# for t in time:
#     img = plt.imread(os.path.join(os_path, session_folder, 'traj_raw'+str(t)+'.png'))
#     plt.imshow(img, alpha=0.5)
    
for j, joint in enumerate(joints):
# joint = 'PIP2'
    x = np.array(df[joint, 'x'][time])
    y = np.array(df[joint, 'y'][time])
    err = np.array(df[joint, 'likelihood'][time])
    x[err < 0.2] = np.nan
    y[err < 0.2] = np.nan
    # for t in range(len(time)):
    #     plt.scatter(x[t], y[t], c=colors[j])
    plt.scatter(x, y, c=colors[j])

# joints_label = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
#           'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'DIP1', 'DIP2', 'DIP3', 'DIP4',
#           'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']

joints_label = ['Wrist', 'MCP2', 'PIP2', 'DIP2', 'Tip2']

# plt.ylim([300, 700])
plt.xlabel('x (pixel)', fontsize=14)
plt.ylabel('y (pixel)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(joints_label, loc='upper center', bbox_to_anchor=(0.48, 1.15),
          ncol=5, fontsize=12)

#%% Heatmap
import cv2
from matplotlib import pyplot as plt
df = pd.read_csv(os.path.join(pickle_folder, 'cam_0.csv'), header=[2,3], index_col=0)
vidpath = os.path.join(os_path, session_folder, 'cam_0.avi')
# period = 15
start = 3480
# frame_count = start
# joints = ['Wrist', 'MCP2', 'PIP2', 'Dip2', 'Tip2']
time = np.arange(start, start+period)
colorclass = plt.cm.ScalarMappable(cmap='jet')
C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
colors = C[:, :3]
cap = cv2.VideoCapture(vidpath) 
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_num/fps
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
ret, frame = cap.read()
cv2.imwrite(os.path.join(os_path, session_folder, 'heatmap'+str(start)+'.png'), frame)
img = plt.imread(os.path.join(os_path, session_folder, 'heatmap'+str(start)+'.png'))
# plt.imshow(img)

heatmap_size = 20
heatmap = np.zeros((1536, 2048))
x, y = np.meshgrid(np.arange(-heatmap_size,heatmap_size), 
                  np.arange(-heatmap_size,heatmap_size))
d = np.sqrt(x*x+y*y)

for j, joint in enumerate(joints):
    x = int(df[joint, 'y'][start])
    y = int(df[joint, 'x'][start])
    err = np.array(df[joint, 'likelihood'][start])
    sigma, mu = 5.0, err
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    heatmap[-heatmap_size+x:heatmap_size+x, 
            -heatmap_size+y:heatmap_size+y] += g 
    

heatmap = heatmap.reshape((1536, 2048, 1))
heatmap = np.array(heatmap * 255, dtype = np.uint8)
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

masked_image = colored_heatmap*0.001 + img*0.999
plt.imshow(masked_image)
plt.axis('off')
plt.tight_layout()

#%% Stick figure
import cv2
from matplotlib import pyplot as plt
df = pd.read_csv(os.path.join(pickle_folder, 'cam_0.csv'), header=[2,3], index_col=0)
vidpath = os.path.join(os_path, session_folder, 'cam_0.avi')
# period = 15
start = 3480
# frame_count = start

joints = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']


time = np.arange(start, start+period)
colorclass = plt.cm.ScalarMappable(cmap='jet')
C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
colors = C[:, :3]
cap = cv2.VideoCapture(vidpath) 
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_num/fps
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
ret, frame = cap.read()
cv2.imwrite(os.path.join(os_path, session_folder, 'stick'+str(start)+'.png'), frame)
img = plt.imread(os.path.join(os_path, session_folder, 'stick'+str(start)+'.png'))
plt.imshow(img)
for j, joint in enumerate(joints):
    x = np.array(df[joint, 'x'][start])
    y = np.array(df[joint, 'y'][start])
    err = np.array(df[joint, 'likelihood'][start])
    x[err < 0.2] = np.nan
    y[err < 0.2] = np.nan
    # for t in range(len(time)):
    #     plt.scatter(x[t], y[t], c=colors[j])
    plt.scatter(x, y, c=colors[j])

joints_label = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'DIP1', 'DIP2', 'DIP3', 'DIP4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']
plt.legend(joints_label, loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=5, fontsize=12)

for finger in fingers:
    for connection in finger:
        poi1 = connection[0]
        poi2 = connection[1]
        x = [df[poi1, 'x'][start], df[poi2, 'x'][start]]
        y = [df[poi1, 'y'][start], df[poi2, 'y'][start]]
        plt.plot(x, y, c='w', alpha=0.5)
        


# plt.ylim([300, 700])
plt.xlabel('x (pixel)', fontsize=14)
plt.ylabel('y (pixel)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axis('off')
# plt.tight_layout()

#%% 3-D error - 3-D reconstruct
import os
import re
import numpy as np
import pandas as pd
from tqdm import trange
from numpy import array as arr
from utils.utils import load_config
from utils.calibration_utils import get_video_path, load_calib_new, load_intrinsics, \
    load_extrinsics
from triangulation.triangulate import undistort_points, triangulate_ransac, \
    triangulate_simple, reprojection_error_und
    

iteration = 18
joints = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']
folder_path = os.path.join(os_path, 'pop_1217/', 'iteration-' + str(iteration))
cams = ['cam_0_trimmed', 'cam_1_trimmed', 'cam_2_trimmed', 'cam_3_trimmed']

# images_folder = os.path.join(folder_path, cams[0])
# image_indices = [int(re.findall(r'\d+', file)[0]) \
    # for file in os.listdir(images_folder) if file.endswith('.png')]

read_one = pd.read_csv(os.path.join(folder_path, cams[0] + '.csv'),
                       header=[2,3], index_col=0)

all_points_raw = []
all_scores = []
for cam in cams:
    data = pd.read_csv(os.path.join(folder_path, cam + '.csv'),
                        header=[2,3], index_col=0)
                        # 'df_with_missing')
    length = len(data)
    index = np.arange(length)
    coords = np.zeros((length, len(joints), 2))
    scores = np.zeros((length, len(joints)))
    for bp_idx, joint in enumerate(joints):
        bp_coords = arr(data[joint])
        coords[index, bp_idx, :] = bp_coords[:, :2]
        scores[index, bp_idx] = np.isfinite(bp_coords[:, 0])
    
#     # (30, 3, 21, 2)
#     # (frames, cams, joints, 2)
    all_points_raw.append(coords)
#     # (30, 4, 21)
    all_scores.append(scores)
    

config = load_config('config_pop_1217.toml')
path, videos, vid_indices = get_video_path(config)
bp_interested = joints
reconstruction_threshold = config['triangulation']['reconstruction_threshold']

output_path = folder_path

intrinsics = load_intrinsics(path, vid_indices)
extrinsics = load_extrinsics(path)
# intrinsics, extrinsics = load_calib_new(config)

cam_mats = []
cam_mats_dist = []
vid_indices = vid_indices
for vid_idxs in vid_indices:
    mat = arr(extrinsics[vid_idxs])
    left = arr(intrinsics[vid_idxs]['camera_mat'])
    cam_mats.append(mat)
    cam_mats_dist.append(left)

cam_mats = arr(cam_mats)
cam_mats_dist = arr(cam_mats_dist)

# Default 3D
all_points_raw = np.stack(all_points_raw, axis=1)
all_scores = np.stack(all_scores, axis=1)

all_points_und = undistort_points(all_points_raw, vid_indices, intrinsics)
all_points_und[all_scores < reconstruction_threshold] = np.nan


n_frames, n_cams, n_joints, _ = all_points_und.shape
points_2d = np.transpose(all_points_und, (1, 0, 2, 3))

points_shaped = points_2d.reshape(n_cams, n_frames*n_joints, 2)

min_cams = 2
# out (491400, 3)
# picked_vals (4, 491400, 1)
out, picked_vals, _, errors = triangulate_ransac(points_shaped, cam_mats, 
                                                 cam_mats_dist, min_cams)

all_points_3d = np.reshape(out, (n_frames, n_joints, 3))

num_cams = np.reshape(picked_vals, (n_cams, n_frames, n_joints))
num_cams = np.sum(num_cams, axis=0)
num_cams = num_cams.astype(float)
num_cams[num_cams==0] = np.nan

errors = np.reshape(errors, (n_frames, n_joints))
errors[errors==0] = np.nan

if 'reference_point' in config['triangulation'] and 'axes' in config['triangulation']:
    all_points_3d_adj, recovery = correct_coordinate_frame(config, 
                                                           all_points_3d, 
                                                           bp_interested)
else:
    all_points_3d_adj = all_points_3d

dout = pd.DataFrame()
for bp_num, bp in enumerate(bp_interested):
    for ax_num, axis in enumerate(['x','y','z']):
        dout[bp + '_' + axis] = all_points_3d_adj[:, bp_num, ax_num]
    dout[bp + '_error'] = errors[:, bp_num]
    dout[bp + '_ncams'] = num_cams[:, bp_num]
    # dout[bp + '_score'] = scores_3d[:, bp_num]

dout['fnum'] = np.arange(n_frames)

dout.to_csv(os.path.join(output_path, 'output_3d_data_raw.csv'), index=False)
# length = all_points_raw.shape[0]
# shape = all_points_raw.shape

# all_points_3d = np.zeros((shape[0], shape[2], 3))
# all_points_3d.fill(np.nan)

# errors = np.zeros((shape[0], shape[2]))
# errors.fill(np.nan)

# scores_3d = np.zeros((shape[0], shape[2]))
# scores_3d.fill(np.nan)

# num_cams = np.zeros((shape[0], shape[2]))
# num_cams.fill(np.nan)

# all_points_und[all_scores < reconstruction_threshold] = np.nan
# all_points_3d_adj = all_points_3d
# for i in trange(all_points_und.shape[0], ncols=70):
#     for j in range(all_points_und.shape[2]):
#         pts = all_points_und[i, :, j, :]
#         good = ~np.isnan(pts[:, 0])
#         if np.sum(good) >= 2:
#             # TODO: make triangulation type configurable
#             # p3d = triangulate_optim(pts[good], cam_mats[good])
#             p3d = triangulate_simple(pts[good], cam_mats[good])
#             all_points_3d[i, j] = p3d[:3]
            # errors[i,j] = reprojection_error_und(p3d, 
            #                                      pts[good], 
            #                                      cam_mats[good], 
            #                                      cam_mats_dist[good])
#             num_cams[i,j] = np.sum(good)
#             scores_3d[i,j] = np.min(all_scores[i, :, j][good])

# dout = pd.DataFrame()
# for bp_num, bp in enumerate(bp_interested):
#     for ax_num, axis in enumerate(['x','y','z']):
#         dout[bp + '_' + axis] = all_points_3d_adj[:, bp_num, ax_num]
#     dout[bp + '_error'] = errors[:, bp_num]
#     dout[bp + '_ncams'] = num_cams[:, bp_num]

# dout['fnum'] = image_indices

# dout.to_csv(os.path.join(folder_path, 'output_3d_data_labeled.csv'), index=False)

#%% 3-D error load
labeled = pd.read_csv(os.path.join(os_path, 
                                    # 'pop_1217/test/output_3d_data_labeled_ransac.csv'))
                                    'pop_1217/test/output_3d_data_labeled.csv'))
infered = pd.read_csv(os.path.join(os_path,
                                   'pop_1217/test/iteration-14/output_3d_data_raw.csv'))

images_folder = os.path.join(os_path, 'for_raquel/cam_0_1217/')
image_indices = [int(re.findall(r'\d+', file)[0]) \
    for file in os.listdir(images_folder) if file.endswith('.png')]
    
dists = []
for j, joint in enumerate(joints):
    
    poi1_coord = np.stack([labeled[joint+'_x'], 
                           labeled[joint+'_y'], 
                           labeled[joint+'_z']], axis=1)
    poi2_coord = np.stack([infered[joint+'_x'][image_indices], 
                           infered[joint+'_y'][image_indices], 
                           infered[joint+'_z'][image_indices]], axis=1)
    dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
    dists.append(dist)
    
#%% 3-D error Violin plot
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

fig = plt.figure()
ax = fig.add_subplot(111)

dists_finite = []
for dist in dists:
    dists_finite.append(dist[np.isfinite(dist)])
# def add_label(violin, label):
#     color = violin["bodies"][0].get_facecolor().flatten()
#     labels.append((mpatches.Patch(color=color), label))
    
positions = np.arange(1, 22)

ax.violinplot(dists_finite, showmeans=True, showextrema=False, points=50)

joints_label = ['Wrist', 'CMC of thumb', 'MCP of thumb', 
                'MCP 1', 'MCP 2', 'MCP 3', 'MCP 4',
          'IP of thumb', 'PIP 1', 'PIP 2', 'PIP 3', 'PIP 4', 
          'DIP 1', 'DIP 2', 'DIP 3', 'DIP 4',
          'Tip of thumb', 'Tip 1', 'Tip 2', 'Tip 3', 'Tip 4']

plt.xticks(np.arange(1,22), joints_label, rotation=45, fontsize=12)
# plt.set_xticklabels()
plt.ylim([0, 100])
plt.ylabel('Mean Euclidean Error (mm)', fontsize=14)

#%% 3-D erorr - reprojection error
from scipy import stats

labeled_repro = []
infer_repro = []
labeled_mean = []
for j, joint in enumerate(joints):
    poi1_coord = np.array(labeled[joint+'_error'])
    poi2_coord = np.array(infered[joint+'_error'][image_indices])
    labeled_mean = np.concatenate((labeled_mean, poi1_coord))
    labeled_repro.append(poi1_coord[np.isfinite(poi1_coord)])
    infer_repro.append(poi2_coord[np.isfinite(poi2_coord)])
    stat, pval = stats.ttest_ind(labeled[joint+'_error'], infered[joint+'_error'][image_indices], 
                           nan_policy='omit',equal_var=False)
    if pval < 0.01:
        # print(pval)
        print(joint)
    
fig = plt.figure()
ax = fig.add_subplot(111)
    
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    
labels = []
positions = np.arange(1, 22)

# ax.violinplot(train_errors_finite, showmeans=True, showextrema=False)
add_label(plt.violinplot(labeled_repro, positions, showmeans=True,
                         showextrema=False), "Labeled")    
# ax.violinplot(test_errors_finite, showmeans=True, showextrema=False)
add_label(plt.violinplot(infer_repro, positions, showmeans=True, 
                         showextrema=False), "Inferred")    

joints_label = ['Wrist', 'CMC of thumb', 'MCP of thumb', 
                'MCP 1', 'MCP 2', 'MCP 3', 'MCP 4',
          'IP of thumb', 'PIP 1', 'PIP 2', 'PIP 3', 'PIP 4', 
          'DIP 1', 'DIP 2', 'DIP 3', 'DIP 4',
          'Tip of thumb', 'Tip 1', 'Tip 2', 'Tip 3', 'Tip 4']

plt.xticks(np.arange(1,22), joints_label, rotation=45, fontsize=12)
# plt.set_xticklabels()
plt.ylabel('Mean Reprojection Error (pixels)', fontsize=14)
plt.legend(*zip(*labels), loc='upper center', fontsize=12)

#%% MVB image
import cv2 
df_train = pd.read_hdf(os.path.join(os_path,'/Pop_freeReach_0317_merged-Min-2020-04-19/'\
                       'labeled-data/cam_1_0811/CollectedData_Min.h5'),
                       index_col=0, key='df_with_missing')
df_train = df_train['Min', 'right_hand']
df_pre = pd.read_csv(os.path.join(os_path, '/pop_0811_1/iteration-13/cam_1.csv'), 
                     index_col=0, header=[2,3])
df_post = pd.read_csv(os.path.join(os_path, 'pop_0811_1/iteration-18/cam_1.csv'), 
                      index_col=0, header=[2,3])
dfs = [df_train, df_pre, df_post]
img_idx = 1389
img_path = 'labeled-data/cam_1_0811/img001389.png'

vidpath = os.path.join(os_path, session_folder, 'cam_1.avi')
colorclass = plt.cm.ScalarMappable(cmap='jet')
C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
colors = C[:, :3]
cap = cv2.VideoCapture(vidpath) 
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_num/fps
cap.set(cv2.CAP_PROP_POS_FRAMES, img_idx)
ret, frame = cap.read()
cv2.imwrite(os.path.join(os_path, session_folder, 'cam_1_1389.png'), frame)


for i, df in enumerate(dfs):
    img = plt.imread(os.path.join(os_path, session_folder, 'cam_1_1389.png'))
    plt.imshow(img[900:1120, 1100:1450, :])
    for j, joint in enumerate(joints):
        if i is 0:
            x = np.array(df[joint, 'x'][img_path]) - 1100
            y = np.array(df[joint, 'y'][img_path]) - 900
            plt.scatter(x, y, c=colors[j], s=60)
        else:
            x = np.array(df[joint, 'x'][img_idx]) - 1100
            y = np.array(df[joint, 'y'][img_idx]) - 900
            if x > 0 and y > 0:
                plt.scatter(x, y, c=colors[j], s=60)
    plt.axis('off')
    plt.savefig(os.path.join(os_path, session_folder, 'cam_1_1389_'+str(i)+'.png'), 
                bbox_inches='tight', dpi=600)
    plt.close()
# [1100, 900]
# [1450, 1120]

#%% Post-processing connection length
df_raw = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_raw.csv'))
# df_outlier = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_out2.csv'))
# df_interp = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_interp.csv'))
df_filt = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_lpf_linear.csv'))


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
dfs = [df_raw, df_outlier, df_interp, df_filt]
titles = ['Pre-processing', 'Post-outlier-removal', 'Post-interpolation', 'Post-LPF']

fig = plt.figure(figsize=[8,6])
connection_lengths = []
bins = np.arange(0,100,2)
stds = []
nans = []

connection = ["PIP1", "Dip1"]
for i, (df, title) in enumerate(zip(dfs, titles)):
    poi1 = connection[0]
    poi2 = connection[1]
    poi1_coord = np.stack([df[poi1+'_x'], 
                           df[poi1+'_y'], 
                           df[poi1+'_z']], axis=1)
    poi2_coord = np.stack([df[poi2+'_x'], 
                           df[poi2+'_y'], 
                           df[poi2+'_z']], axis=1)
    dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
    connection_lengths.append(dist)
    dist_med = np.median(dist[np.isfinite(dist)])
    
    dist_std = np.std(dist[np.isfinite(dist)])
    stds.append(dist_std)
    
    nan_prop = np.sum(np.isnan(dist))/len(dist)
    nans.append(nan_prop)
    
    # if i >= 2:
    #     ax = fig.add_subplot(2,3,i+2)
    # else:
    ax = fig.add_subplot(2,2,i+1)
    ax.hist(dist, bins, density=True)
    ax.set_title(title + '\n' + 
                  # poi1+' to '+'DIP2' + '\n' +
                 # poi1+' to '+poi2 + '\n' +
                 f'Std: {dist_std:.1f} (mm)\nProp. missing: {nan_prop:.3f}',
                 fontsize=14)
    ax.axvline(x=dist_med-1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
    ax.axvline(x=dist_med+1.0*dist_std, ymin=0, ymax=1, color='r', linewidth=0.5)
    ax.axvline(x=dist_meds_real[connection[0]+'-'+connection[1]][-1], 
               ymin=0, ymax=1, color='tab:orange', linewidth=2)
    ax.set_xlabel('Connection length (mm)', fontsize=12)
    ax.tick_params(axis='x', which='major', labelsize=10)
    if i == 1 or i == 3:
        ax.set_ylabel('Proportion of frames', fontsize=12)
    if i == 0 or i == 2:
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', which='both', labelsize=11)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 0.3])

left = 0.01  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.88     # the top of the subplots of the figure
wspace = 0.35  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.99 # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

#%% Post-processing 2
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.1)  # adjust space between axes
ax1.plot(stds, c='tab:blue', linestyle='-', marker='o')
ax2.plot(nans, c='tab:red', linestyle='-', marker='o')

ax1.set_ylim(1.4, 3.)  # outliers only
ax2.set_ylim(0, .21)  # most of the data

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.set_ylabel('Standard deviation (mm)', fontsize=13)
ax2.set_ylabel('Proportion of\nmissing points', fontsize=13)

ax2.set_xticks([0, 1, 2, 3])
ax2.set_xticklabels(titles, rotation=30)

ax1.set_yticks([1.5, 2.0, 2.5, 3.0])
ax1.set_yticklabels(['1.5', '2.0', '2.5', '3.0'])
ax2.set_yticks([0, 0.05, 0.1, 0.15])
ax2.set_yticklabels(['0', '0.05', '0.1', '0.15'])

ax1.tick_params(axis='y', which='both', labelsize=11)
ax2.tick_params(axis='both', which='both', labelsize=11)

left = 0.25  # the left side of the subplots of the figure
right = 0.95   # the right side of the subplots of the figure
bottom = 0.20  # the bottom of the subplots of the figure
top = 0.95     # the top of the subplots of the figure
wspace = 0.35  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.1 # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)

#%% 3-D error for Post-processing - 1
import os
import re
from matplotlib import pyplot as plt

# os_path = 'F:/'
pickle_folder =  os.path.join(os_path, session_folder, 'iteration-'+str(iteration))
iterations = ['iteration-14', 'iteration-15', 'iteration-17', 'iteration-18']
# datas = ['output_3d_data_raw.csv', 'output_3d_data_out2.csv', 'output_3d_data_interp.csv',
#  'output_3d_data_lpf.csv']
datas = ['output_3d_data_raw.csv', 'output_3d_data_lpf_linear.csv', 
         'output_3d_data_lpf_spline.csv']
# pickle_folders = [os.path.join(os_path, session_folder, iteration) for iteration in iterations]

images_folder = os.path.join(os_path, 'for_raquel/cam_0_1217/')
image_indices = [int(re.findall(r'\d+', file)[0]) \
    for file in os.listdir(images_folder) if file.endswith('.png')]
    
joints = ['CMC_thumb', 'MCP_thumb', 'IP_thumb', 'Tip_thumb',
          'MCP1', 'PIP1', 'Dip1', 'Tip1',
          'MCP2', 'PIP2', 'Dip2', 'Tip2', 
          'MCP3', 'PIP3', 'Dip3', 'Tip3', 
          'MCP4', 'PIP4', 'Dip4', 'Tip4', 
          'Wrist']

colorclass = plt.cm.ScalarMappable(cmap='jet')
C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
colors = C[:, :3]
labeled = pd.read_csv(os.path.join(os_path, 
                                   'pop_1217/test/output_3d_data_labeled_ransac.csv'))

dists_iters = []

    
nans = np.zeros((len(datas), 21))
for i, data in enumerate(datas):
    infered = pd.read_csv(os.path.join(pickle_folder, data))
    dists = []
    for j, joint in enumerate(joints):
        
        poi1_coord = np.stack([labeled[joint+'_x'], 
                               labeled[joint+'_y'], 
                               labeled[joint+'_z']], axis=1)
        poi2_coord = np.stack([infered[joint+'_x'][image_indices], 
                               infered[joint+'_y'][image_indices], 
                               infered[joint+'_z'][image_indices]], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        dists.append(dist)
    dists = np.array(dists).reshape(-1,)
    dists_iters.append(dists)
dists_iters = np.array(dists_iters)
dists_mean = []
dists_std = []
for dists in dists_iters:
    dists_mean.append(np.nanmean(dists))
    dists_std.append(round(np.nanstd(dists)/np.sqrt(np.sum(np.isfinite(dists))),1))
#%% 3-D error for Post-processing - 2

fig = plt.figure()
ax = fig.add_subplot(111)
# positions = np.arange(1,5)*4
positions = np.arange(0, 2)*2 + 3
# for i, dists in enumerate(dists_iters):
#     dists_finite = dists[np.isfinite(dists)]
#     ax.boxplot(dists_finite, positions=[i])
# for mean, std in zip(dists_mean, dists_std):
#     dists_finite = dists[np.isfinite(dists)]
#     ax.boxplot(dists_finite, positions=[i])
ax.bar(positions, dists_mean[:2], yerr=dists_std[:2], width=0.5)
for i in range(len(datas[:2])):
    ax.text(positions[i]+0.5, dists_mean[i]+0.5, 
            f'{dists_mean[i]:.1f} ± {dists_std[i]:.1f}',
            fontsize=12)

ax.set_xticks(positions)
# ax.set_ylim([0, 18])
# ax.set_xlim([1, 20])
ax.set_ylim([0, 10])
ax.set_xlim([2, 6.5])
# ax.set_xticklabels(['Pre-processing', 'Post-outlier-removal', 
#                     'Post-interpolation', 'Post-LPF'], fontsize=12, rotation=30)
ax.set_xticklabels(['Pre-processing', 'Post-processing'], fontsize=12)
# ax.set_xticklabels(['Before', 'Fixing', 'Extra labels', 
#   'Fixing & extra labels', 'Label'], fontsize=12)
ax.set_ylabel('Mean Euclidean Error (mm)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()

#%% 3-D error for Post-processing - 3
fig = plt.figure(figsize=[5.5,5])
ax = fig.add_subplot(111)
ind = np.random.choice(range(len(dists)), 500, replace=False)
x = dists_iters[0][ind]
y = dists_iters[1][ind]
# for i, dists in enumerate(dists_iters[:2]):
    # ax.scatter(np.ones((len(dists),))*positions[i], dists)
ax.scatter(x, y, alpha=0.5)
ax.plot([0,30], [0,30], 'k-')
    
# positions = np.arange(1,5)*4
# for i in range(len(datas)):
#     ax.text(positions[i]+0.5, dists_mean[i]+0.5, 
#             f'{dists_mean[i]:.1f} ± {dists_std[i]:.1f}',
#             fontsize=12)

ax.set_ylim([0, 15])
ax.set_xlim([0, 15])
# ax.set_ylim([0, 10])
# ax.set_xlim([1, 14])
# ax.set_xticklabels(['Pre-processing', 'Post-outlier-removal', 
#                     'Post-interpolation', 'Post-LPF'], fontsize=12, rotation=30)
# ax.set_xticklabels(['Pre-processing', 'Post-processing (linear)', 
#                     'Post-processing (cubic)'], fontsize=12, rotation=30)
# ax.set_xticklabels(['Before', 'Fixing', 'Extra labels', 
#   'Fixing & extra labels', 'Label'], fontsize=12)
ax.set_xlabel('Pre-processing\nMean Euclidean Error (mm)', fontsize=14)
ax.set_ylabel('Post-processing\nMean Euclidean Error (mm)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()


#%% 3-D error for Post-processing - 
from scipy import stats

a = dists_iters[0, :]
b = dists_iters[1, :]
c = dists_iters[2, :]

stat, pvalue = stats.ttest_ind(a,b,nan_policy='omit',equal_var=False)
# stat, pvalue = stats.ttest_ind(a,c,nan_policy='omit',equal_var=False)

#%% Histogram of outliers
import more_itertools as mit
from scipy.signal import welch, periodogram
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt, freqz, freqs, lfilter

def butter_lowpass_filter(data):    
    n = len(data)   # total number of samples
    fs = 30         # sample rate, Hz
    T = n/fs        # Sample Period
    cutoff = 8      # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 4       # sin wave can be approx represented as quadratic

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
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

t = np.arange(500, 1100)
speed_thr = [0.6, 0.75, 0.9, 1.2]

df = pd.read_csv(os.path.join(pickle_folder, 'backup_full', 'output_3d_data_trimmed.csv'))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

joint = 'PIP2'
# for joint in joints:
x = np.array(df[joint+'_x'])
y = np.array(df[joint+'_y'])
z = np.array(df[joint+'_z'])
xyz = np.stack([x,y,z], axis=1)
speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)/(1/30)/1000

speed_sub = speed[t].copy()
f, Pxx = welch(speed_sub, 30, nperseg=64)
ax1.semilogy(f, np.sqrt(Pxx))

ax1.axvline(x=8, ymin=0, ymax=1, color='r', linewidth=2, alpha=0.8)
ax1.set_xlabel('frequency [Hz]', fontsize=14)
ax1.set_ylabel('Linear spectrum [V RMS]', fontsize=14)
ax1.set_title('Power spectral density - PIP2', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
fig1.tight_layout()


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

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(t/30, speed_sub)
ax2.set_title(joint + ' Speed', fontsize=16)
ax2.set_xlabel('Time (sec)', fontsize=14)
ax2.set_ylabel('Speed (m/s)', fontsize=14)
ax2.hlines(speed_thr[1], t[0]/30, t[-1]/30, 'r', linewidth=2, alpha=0.8)
ax2.tick_params(axis='both', which='major', labelsize=12)
fig2.tight_layout()

#%% 3D dropped histogram
import pandas as pd
import more_itertools as mit
df_origin = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_out2.csv'))

xyzs_origin = []
drops = np.zeros((3,))
for joint in joints:
    x = df_origin[joint+'_x']
    
    nans_consec = [i for i in range(len(x)) if np.isnan(x[i])]
    nans_groups = [list(group) for group in mit.consecutive_groups(nans_consec)]
    
    for group in nans_groups:
        i = len(group)
        if i < 3:
            drops[i-1] += i
        else:
            drops[2] += i
        # if i < 6:
        #     drops[i-1] += i
drops /= (len(df_origin)*len(joints))
xlabel1 = np.arange(0, len(drops))
xlabel2 = ['1', '2', 'More than 2']
fig = plt.figure()
ax = fig.add_subplot(111)
# fig = plt.figure(figsize=[10,4])
# ax = fig.add_subplot(121)
ax.bar(xlabel1, drops, width = 0.25)
ax.set_xticks(xlabel1)
ax.set_xticklabels(xlabel2)
# ax.set_xticklabels([str(i) for i in xlabel2l])
ax.set_ylim([0, np.max(drops)*1.05])
# ax.set_yticks([0, 1, 2])
# ax.set_yticklabels([0, 0.5, 1])
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlabel('Number of frames missing consecutively', fontsize=14)
ax.set_ylabel('Proportion of frames', fontsize=14)
# ax.set_title('Proportion of frames dropped consecutively', fontsize=16)
# ax.legend(rows, fontsize=11, loc='upper left')
fig.tight_layout()

#%% 3D interpolation comparison
from numpy import array as arr

# off_means = []
# off_stds = []
# for joint in joints:
joint = joints[0]
x = df_origin[joint+'_x']
y = df_origin[joint+'_y']
z = df_origin[joint+'_z']
xyz = np.stack([x,y,z], axis=1)

where_finite = [i for i in range(len(x)) if np.isfinite(x[i])]
finite_groups = [list(group) for group in mit.consecutive_groups(where_finite)]
longest = np.argmax([len(group) for group in finite_groups])

# xyz = xyz[finite_groups[longest]]

wins = np.arange(1, 6, dtype=int)
ext = 2
off_means = [[] for i in range(2) for j in range(len(wins))]
off_stds = [[] for i in range(2) for j in range(len(wins))]

methods = ['linear', 'spline']
for group in finite_groups:
    if len(group) > 30:
        for i, method in enumerate(methods):
            for j, win in enumerate(wins):
                offsets = interp_3d_test(xyz[group], win, ext, method)
                # off_mean = np.mean(offsets)
                # off_std = np.std(offsets)
                off_means[len(wins)*i+j].append(offsets)
                off_stds[len(wins)*i+j].append(offsets)
            # print(joint + '    ' + f'Mean: {off_mean:.2f} Std: {off_std:.2f} (mm)')

off_means = [np.mean(np.concatenate(arr)) for arr in off_means]
off_stds = [np.std(np.concatenate(arr)) for arr in off_stds]

off_means = np.array(off_means).reshape(len(methods), len(wins))
off_stds = np.array(off_stds).reshape(len(methods), len(wins))

rows = ('Linear', 'Cubic Spline')
# columns = ('Win 1', 'Win 2', 'Win 3')
columns = ('Win 1', 'Win 2')

analysis = []
cell_text = []
y_ticks = np.arange(len(wins))
tick_offsets = np.arange(len(rows))*0.15
for i in range(2):
    analysis = off_means[i,:]
    err = off_stds[i,:]
    
    ax = fig.add_subplot(122)
    ax.bar(y_ticks+tick_offsets[i], analysis, 0.1, yerr=err)
    cell_text.append(['%1.1f' % (x) for x in analysis])
    
ax.set_xticks(wins-1)
ax.set_xticklabels([str(i) for i in wins])
# ax.set_xticklabels([str(i) for i in range(1,5)])
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.set_xlabel('Number of frames missing consecutively', fontsize=13)
ax.set_ylabel('Mean Euclidean Error (mm)', fontsize=13)
# ax.set_ylabel('Mean Euclidean Error (mm)', fontsize=12)
ax.set_ylim([0, np.max(off_means + off_stds)+0.5])
# ax.set_title('Interpolation errors', fontsize=16)
ax.legend(rows, fontsize=11, loc='upper left')
fig.tight_layout()

#%% Plot 3D interpolation test
xyz_drop = dropping_data(xyz, 4, 2)
linear = interp_3d(xyz_drop, xyz, 4, 2, 'linear')[:64]
spline = interp_3d(xyz_drop, xyz, 4, 2, 'spline')[:64]
xyz_drop = xyz_drop[:64]
t = np.arange(len(xyz_drop))
# t = longest_seq

# axis = 1
axis_dict = {0: 'x',
             1: 'y',
             2: 'z'}

for axis in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xx = xyz_drop[:, axis]
    start_indices = []
    end_indices = []
    
    nans_consec = [i for i in range(len(xx)) if np.isnan(xx[i])]
    nans_groups = [list(group) for group in mit.consecutive_groups(nans_consec)]
    
    for nans_group in nans_groups:
        # else:
        end_indices.append(np.max([nans_group[-1]+1, 1]))
        start_indices.append(np.min([nans_group[0], len(xx)-1]))
    end_indices.append(len(xx))
    
    ax.plot(t, xyz[:64, axis], alpha=0.8)
    # ax.plot(t, xyz[:, axis], alpha=0.8)
    ax.plot(t, linear[:, axis], alpha=0.8)
    ax.plot(t, spline[:, axis], alpha=0.8)
    # ax.fill_between(nans_consec, xy[nans_consec, xory]-5, xy[nans_consec, xory]+5)
    for start_idx, end_idx in zip(start_indices, end_indices):
        fill_range = np.arange((start_idx-1), (end_idx+1), 1)
        print(fill_range)
        ax.fill_between(fill_range, 
                        # np.min([np.min(xyz[:, axis]), 
                        np.min([np.min(xyz[:64, axis]), 
                                np.min(linear[:, axis]),
                                np.min(spline[:, axis])]), 
                        # np.max([np.max(xyz[:, axis]), 
                        np.max([np.max(xyz[:64:, axis]), 
                                np.max(linear[:, axis]),
                                np.max(spline[:, axis])]), 
                        color='k', alpha=0.1)
    #     ax.axvline(x=start_idx, ymin=0.2, ymax=0.5)
    # for end_idx in end_indices[:-1]:
    #     ax.axvline(x=end_idx, ymin=0.2, ymax=0.5)
    # ax.set_xticklabels([str[for ])
    ax.legend(['Original', 'Linear Interpolation', 'Cubic spline'], fontsize=12)
    ax.set_title('PIP 2 ' + axis_dict[axis] + '-trajectory', fontsize=16)
    ax.set_xlabel('Frame', fontsize=14)
    ax.set_ylabel('Coordinate (mm)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()

#%% PSD of trajectory
import more_itertools as mit
from scipy.signal import welch, periodogram
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt, freqz, freqs, lfilter

def butter_lowpass_filter(data):    
    n = len(data)   # total number of samples
    fs = 30         # sample rate, Hz
    T = n/fs        # Sample Period
    cutoff = 8      # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 4       # sin wave can be approx represented as quadratic

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
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
# start = 540
# interval = 420

start = 500
interval = 548
# t = np.arange(500, 1048)
t = np.arange(start, start+interval)
speed_thr = [0.6, 0.75, 0.9, 1.2]

df = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_interp.csv'))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for joint in joints:
    x = np.array(df[joint+'_x'])
    x_sub = x[t].copy()
    for i in range(3):
        f, Pxx = welch(x_sub, 30, nperseg=64)
        ax1.semilogy(f, np.sqrt(Pxx))
# speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)/(1/30)/1000


# f, Pxx = welch(xyz_sub, 30, nperseg=64)
# ax1.semilogy(f, np.sqrt(Pxx))

ax1.axvline(x=8, ymin=0, ymax=1, color='r', linewidth=2, alpha=0.8)
ax1.set_xlabel('frequency [Hz]', fontsize=14)
ax1.set_ylabel('Linear spectrum [V RMS]', fontsize=14)
ax1.set_title('Power spectral density of x-axis of all interest points', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
# fig1.tight_layout()

joint = 'PIP2'
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
x = np.array(df[joint+'_x'])
y = np.array(df[joint+'_y'])
z = np.array(df[joint+'_z'])
xyz = np.stack([x,y,z], axis=1)/(1/30)/1000
xyz_sub = xyz[t].copy()
for i in range(3):
    f, Pxx = welch(xyz_sub[:, i], 30, nperseg=64)
    ax2.semilogy(f, np.sqrt(Pxx), alpha=0.8)
    
df_filt = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_lpf_cut.csv'))    
x = np.array(df_filt[joint+'_x'])
y = np.array(df_filt[joint+'_y'])
z = np.array(df_filt[joint+'_z'])
xyz_filt = np.stack([x,y,z], axis=1)/(1/30)/1000

speed_sub = np.linalg.norm(np.diff(xyz_filt[t], axis=0), axis=1)
f, Pxx = welch(speed_sub, 30, nperseg=64)
ax2.semilogy(f, np.sqrt(Pxx), alpha=0.8)


# f, Pxx = welch(xyz_sub, 30, nperseg=64)
# ax1.semilogy(f, np.sqrt(Pxx))

ax2.axvline(x=8, ymin=0, ymax=1, color='r', linewidth=2, alpha=0.8)
ax2.set_xlim([0, 15])
ax2.set_xlabel('frequency [Hz]', fontsize=14)
ax2.set_ylabel('PSD [V**2/Hz]', fontsize=14)
ax2.set_title('Power spectral density - PIP 2', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(['X', 'Y', 'Z', 'Speed filtered'], fontsize=12)
fig2.tight_layout()

#%% Plot 3D filter trajectory
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
    y = filtfilt(b, a, data)
    output = {'y': y,
              'b': b,
              'a': a}
    return output


def filt_3d(filt_type, nan_win, xyz):
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
                # x = xyz[expand_indices, 0].copy()
                # y = xyz[expand_indices, 1].copy()
                # z = xyz[expand_indices, 2].copy()
                
                # nans, x_fun = nan_helper(x)
                # _, y_fun = nan_helper(y)
                # _, z_fun = nan_helper(z)
                
                # x[nans]= np.interp(x_fun(nans), x_fun(~nans), x[~nans])
                # y[nans]= np.interp(y_fun(nans), y_fun(~nans), y[~nans])
                # z[nans]= np.interp(z_fun(nans), z_fun(~nans), z[~nans])
                
                # xyz_filt[nans_group, 0] = x[1:-1]
                # xyz_filt[nans_group, 1] = y[1:-1]
                # xyz_filt[nans_group, 2] = z[1:-1]
                
                frames = np.arange(len(expand_indices))
                xyz_trim = xyz[expand_indices, :].copy()
                
                x = xyz[expand_indices, 0].copy()
                y = xyz[expand_indices, 1].copy()
                z = xyz[expand_indices, 2].copy()
                nans, _ = nan_helper(x)
                
                cs = CubicSpline(frames[~nans], [x[~nans], y[~nans], z[~nans]], axis=1)
                xyz_trim[nans] = cs(frames[nans]).T
                xyz_filt[nans_group] = xyz_trim[2:-2]
        
    # xyz = smooth.smooth(xyz, best_tol, best_sigR, keepOriginal=True)
    
    if filt_type == 'interp':
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
    
coords = xyz_sub.copy()
xyz_filt = filt_3d('lpf', 4, coords)

t = np.arange(len(xyz_filt))
axis_dict = {0: 'x',
             1: 'y',
             2: 'z'}

for axis in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t/30, xyz_sub[:, axis], alpha=0.8)
    ax.plot(t/30, xyz_filt[:, axis], alpha=0.8)
    ax.legend(['Pre-LPF', 'Post-LPF'], fontsize=12)
    ax.set_title('PIP 2 ' + axis_dict[axis] + '-trajectory', fontsize=16)
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Coordinate (mm)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    
#%% NAN prop of connection lengths changes for MVB
import os
import re
from matplotlib import pyplot as plt

# os_path = 'F:/'
os_path = '/Volumes/T7 Touch/'
pickle_folder =  os.path.join(os_path, session_folder, 'iteration-'+str(iteration))
iterations = ['iteration-14', 'iteration-15', 'iteration-17', 'iteration-18']
pickle_folders = [os.path.join(os_path, session_folder, iteration) for iteration in iterations]


joints = ['CMC_thumb', 'MCP_thumb', 'IP_thumb', 'Tip_thumb',
          'MCP1', 'PIP1', 'Dip1', 'Tip1',
          'MCP2', 'PIP2', 'Dip2', 'Tip2', 
          'MCP3', 'PIP3', 'Dip3', 'Tip3', 
          'MCP4', 'PIP4', 'Dip4', 'Tip4', 
          'Wrist']

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

fingers = [
    ['CMC_thumb', 'MCP_thumb', 'IP_thumb', 'Tip_thumb'],
    ["MCP1", "PIP1", "Dip1", "Tip1"],
    ["MCP2", "PIP2", "Dip2", "Tip2"],
    ["MCP3", "PIP3", "Dip3", "Tip3"],
    ["MCP4", "PIP4", "Dip4", "Tip4"],
    ["Wrist"]]


colors = ['tab:purple', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'k']

nans = np.zeros((len(pickle_folders), 21))
for i, pickle_folder in enumerate(pickle_folders):
    df_infer = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_lpf.csv'))
    joint_nan = {joint: [] for joint in joints}
    for k, joint in enumerate(joints):
        infer_x = df_infer[joint+'_x']
        nans[i, k] = np.sum(np.isnan(infer_x))/len(df_infer)

linestyles = ['o-', 'o--', 'o-.', 'o:']
fig = plt.figure()
ax = fig.add_subplot(111)
for i, finger in enumerate(fingers):
    for j, joint in enumerate(finger):
        ax.plot(nans[:, 4*i+j], linestyles[j], color=colors[i])
ax.set_xticks([0, 1, 2, 3])
ax.set_ylim([0, 0.5])
ax.set_xticklabels(['Pre-MVB', 'Iteration-1', 'Iteration-2', 'Iteration-3'], fontsize=12)
# ax.set_xticklabels(['Before', 'Fixing', 'Extra labels', 
#   'Fixing & extra labels', 'Label'], fontsize=12)
ax.set_ylabel('Proportion of missing points', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(joints, loc='upper center', bbox_to_anchor=(1.05, 1.1),
          ncol=1, fontsize=12, handlelength=3)

#%%
# for i in range(4):
#     f = np.arange(i, 21, 4)
#     n = np.mean(nans[:, f], axis=1)
#     print(n[0]-n[-1])
for i in range(5):
    f = np.arange(i*4, i*4+4, 1)
    n = np.mean(nans[:, f], axis=1)
    print(n[0]-n[-1])
    # print(f)
#%% Stds of MVB
stds = np.zeros((len(pickle_folders), 20))
# for j, connection in enumerate(scheme)
#     poi1 = connection[0]
#     poi2 = connection[1]
#     poi1_coord = np.stack([df[poi1+'_x'], df[poi1+'_y'], df[poi1+'_z']], axis=1)
#     poi2_coord = np.stack([df[poi2+'_x'], df[poi2+'_y'], df[poi2+'_z']], axis=1)
#     dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
#     dist_std = np.std(dist[np.isfinite(dist)])
        
for i, pickle_folder in enumerate(pickle_folders):
    df_infer = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_lpf.csv'))
    
    for k, connection in enumerate(scheme):
        poi1 = connection[0]
        poi2 = connection[1]
        poi1_coord = np.stack([df_infer[poi1+'_x'], 
                               df_infer[poi1+'_y'], 
                               df_infer[poi1+'_z']], axis=1)
        poi2_coord = np.stack([df_infer[poi2+'_x'], 
                               df_infer[poi2+'_y'], 
                               df_infer[poi2+'_z']], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        dist_std = np.std(dist[np.isfinite(dist)])
        stds[i, k] = dist_std

conn = [connection[0]+'-'+connection[1] for connection in scheme]
fig = plt.figure()
ax = fig.add_subplot(111)
for i, color in enumerate(colors[:-1]):
    ax.plot(stds[:, i], 'o-', color=color)
ax.set_xticks(np.arange(len(pickle_folders)))
ax.set_ylim([0, 10])
ax.set_xticklabels(['Pre-MVB', 'Iteration-1', 'Iteration-2', 'Iteration-3'], fontsize=12)
ax.set_ylabel('Standard deviation (mm)', fontsize=14)
ax.legend(conn, loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
# fig.tight_layout()

#%% 3-D error for MVB - 1
import os
import re
from matplotlib import pyplot as plt

os_path = 'F:/'
# os_path = '/Volumes/T7 Touch/'
pickle_folder =  os.path.join(os_path, session_folder, 'test', 'iteration-'+str(iteration))
iterations = ['iteration-14', 'iteration-15', 'iteration-17', 'iteration-18']
pickle_folders = [os.path.join(os_path, session_folder, iteration) for iteration in iterations]

images_folder = os.path.join(os_path, 'for_raquel/cam_0_1217/')
image_indices = [int(re.findall(r'\d+', file)[0]) \
    for file in os.listdir(images_folder) if file.endswith('.png')]
    
joints = ['CMC_thumb', 'MCP_thumb', 'IP_thumb', 'Tip_thumb',
          'MCP1', 'PIP1', 'Dip1', 'Tip1',
          'MCP2', 'PIP2', 'Dip2', 'Tip2', 
          'MCP3', 'PIP3', 'Dip3', 'Tip3', 
          'MCP4', 'PIP4', 'Dip4', 'Tip4', 
          'Wrist']

colorclass = plt.cm.ScalarMappable(cmap='jet')
C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
colors = C[:, :3]
labeled = pd.read_csv(os.path.join(os_path, 'pop_1217/test/output_3d_data_labeled_ransac.csv'))
infered = pd.read_csv(os.path.join(os_path, 'pop_1217/test/iteration-14/output_3d_data_raw.csv'))

dists_iters = []

    
nans = np.zeros((len(pickle_folders), 21))
for i, pickle_folder in enumerate(pickle_folders):
    infered = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_lpf.csv'))
    dists = []
    for j, joint in enumerate(joints):
        
        poi1_coord = np.stack([labeled[joint+'_x'], 
                               labeled[joint+'_y'], 
                               labeled[joint+'_z']], axis=1)
        poi2_coord = np.stack([infered[joint+'_x'][image_indices], 
                               infered[joint+'_y'][image_indices], 
                               infered[joint+'_z'][image_indices]], axis=1)
        dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
        dists.append(dist)
    dists = np.array(dists).reshape(-1,)
    dists_iters.append(dists)
dists_iters = np.array(dists_iters)
dists_mean = []
dists_std = []
for dists in dists_iters:
    dists_mean.append(np.nanmean(dists))
    dists_std.append(round(np.nanstd(dists)/np.sqrt(np.sum(np.isfinite(dists))),1))
    
#%% 3-D error for MVB - 2

fig = plt.figure()
ax = fig.add_subplot(111)
positions = np.arange(1,5)*4
# for i, dists in enumerate(dists_iters):
#     dists_finite = dists[np.isfinite(dists)]
#     ax.boxplot(dists_finite, positions=[i])
# for mean, std in zip(dists_mean, dists_std):
#     dists_finite = dists[np.isfinite(dists)]
#     ax.boxplot(dists_finite, positions=[i])
ax.bar(positions, dists_mean, yerr=dists_std)
for i in range(len(pickle_folders)):
    ax.text(positions[i], dists_mean[i]+1, 
            f'{dists_mean[i]:.1f} ± {dists_std[i]:.1f}',
            fontsize=12)

ax.set_xticks(positions)
ax.set_ylim([0, 10])
ax.set_xlim([1, 20])
ax.set_xticklabels(['Pre-MVB', 'Iteration-1', 'Iteration-2', 'Iteration-3'], fontsize=12)
# ax.set_xticklabels(['Before', 'Fixing', 'Extra labels', 
#   'Fixing & extra labels', 'Label'], fontsize=12)
ax.set_ylabel('Mean Euclidean Error (mm)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()

#%% 3-D error for MVB - 3
from scipy import stats

a = dists_iters[0, :]
b = dists_iters[1, :]
c = dists_iters[2, :]
d = dists_iters[3, :]

stat, pvalue = stats.ttest_ind(a,d,nan_policy='omit',equal_var=False)
# stat, pvalue = stats.ttest_ind(a,c,nan_policy='omit',equal_var=False)

#%%
ppca_pos = PPCA()
ppca_pos.fit(position, d=63, tol=1e-6)
pos_pcs = ppca_pos.transform()

pos_inv = ppca_pos.inverse_transform()
pos_inv = pos_inv*ppca_pos.stds + ppca_pos.means
pos_inv *= 1000
#%%
iterables = [[postfix.split('.')[0]], ['right_hand'], joints, ['x', 
                                                               'y', 
                                                               'likelihood']]
header = pd.MultiIndex.from_product(iterables, names=['scorer', 
                                                      'individuals', 
                                                      'bodyparts', 
                                                      'coords'])
L = len(df)
index = np.arange(L)
# df_ppca = pd.DataFrame(np.zeros((L, len(joints)*3)), index=index, columns=header)

df_ppca = df.copy()
for j, joint in enumerate(joints):
    # df_ppca[joint+'_x'][where_finite] = recover_ppca[:, 3*j]
    # df_ppca[joint+'_y'][where_finite] = recover_ppca[:, 3*j+1]
    # df_ppca[joint+'_z'][where_finite] = recover_ppca[:, 3*j+2]
    
    df_ppca[joint+'_x'] = pos_inv[:, 3*j]
    df_ppca[joint+'_y'] = pos_inv[:, 3*j+1]
    df_ppca[joint+'_z'] = pos_inv[:, 3*j+2]


# df_interp[df_interp==0] = np.nan
df_ppca.to_csv(os.path.join(pickle_folder, 'output_3d_data_ppca.csv'), mode='w')
#%% Reprojection with full 3-D data
import os
import cv2
import numpy as np
import pandas as pd
from numpy import array as arr
from matplotlib import pyplot as plt
from utils.calibration_utils import *
from triangulation.triangulate import *
from calibration.extrinsic import *

vid_indices = ['1']
intrinsics = load_intrinsics(calib_path, vid_indices)
extrinsics = load_extrinsics(calib_path)
df = pd.read_csv(os.path.join(pickle_folder,'output_3d_data_lpf_cut.csv'))
vidpath = os.path.join(os_path, session_folder, 'cam_1.avi')
paths_to_save = os.path.join(pickle_folder, 'cam_1_lpf')
data_3d = df.copy()
labeled_path_to_save = paths_to_save

image_indices = np.arange(570, 870)
L = len(image_indices)
joints = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']

iterables = [['Min'], ['right_hand'], joints, ['x', 'y']]
header = pd.MultiIndex.from_product(iterables, names=['scorer', 
                                                      'individuals', 
                                                      'bodyparts', 
                                                      'coords'])

data_2ds = []

for vid_idx, path_to_save in zip(vid_indices, paths_to_save):
    cameraMatrix = np.matrix(intrinsics[vid_idx]['camera_mat'])
    distCoeffs = np.array(intrinsics[vid_idx]['dist_coeff'])
    Rt = np.matrix(extrinsics[vid_idx])
    rvec, tvec = get_rtvec(Rt)
    
    out = os.path.basename(path_to_save)
    index = [os.path.join('labeled-data', out, 'img' + str(frame_count).zfill(6) + '.png') 
              for frame_count in image_indices]
    
    df_2d = pd.DataFrame(np.zeros((L, len(joints)*2)), index=index, columns=header)
    
    for i, bp in enumerate(joints):
        x = data_3d[bp+'_x'][image_indices].copy()
        y = data_3d[bp+'_y'][image_indices].copy()
        z = data_3d[bp+'_z'][image_indices].copy()
        objectPoints = np.vstack([x,y,z]).T
        coord_2d = np.squeeze(cv2.projectPoints(objectPoints, 
                                                rvec, tvec, 
                                                cameraMatrix, 
                                                distCoeffs)[0], axis=1)
        where_negative = np.argwhere(coord_2d[:,0] < 0)
        where_out1 = np.argwhere(coord_2d[:,0] >= 2048)
        where_out2 = np.argwhere(coord_2d[:,1] >= 1536)
        coord_2d[where_negative, :] = np.nan
        coord_2d[where_out1, :] = np.nan
        coord_2d[where_out2, :] = np.nan
        
        df_2d.iloc[:, 2*i:2*i+2] = coord_2d
    
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    df_2d.to_hdf(os.path.join(path_to_save,'CollectedData_Min.h5'), 
                 key='df_with_missing', mode='w')
    data_2ds.append(df_2d)
    
labeled_path_to_save = os.path.join(pickle_folder, 'cam_1_lpf')
vidpath = os.path.join(os_path, session_folder, 'cam_1.avi')
for labeled_path_to_save, data_2d, vidpath in zip(labeled_paths_to_save, 
                                                  data_2ds, vidpaths):
    if not os.path.exists(labeled_path_to_save):
        os.mkdir(labeled_path_to_save)
            
    cap = cv2.VideoCapture(vidpath) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_num/fps
    count = len(image_indices)
    
    colorclass = plt.cm.ScalarMappable(cmap='jet')
    C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
    colors = C[:, :3]
    
    with tqdm(total=count) as pbar:
        for f, frame_count in enumerate(image_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            plt.figure()
            # plt.imshow(frame)
            xs = []
            ys = []
            for joint in joints:
                xs.append(data_2d['Min', 'right_hand', joint, 'x'].iloc[f])
                ys.append(data_2d['Min', 'right_hand', joint, 'y'].iloc[f])
            
            xs = np.array(xs)
            ys = np.array(ys)
            
            x0 = np.median(xs[np.isfinite(xs)])
            y0 = np.median(ys[np.isfinite(ys)])
            
            bbox = [x0-250, x0+250, max(y0-250, 0), y0+250]
            bbox = [int(b) for b in bbox]
            # frame = cv2.transpose()
            frame_cropped = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]
            
            plt.imshow(frame_cropped)
            
            for i, (color, joint) in enumerate(zip(colors, joints)):
                x = data_2d['Min', 'right_hand', joint, 'x'].iloc[f]
                y = data_2d['Min', 'right_hand', joint, 'y'].iloc[f]
                plt.scatter(x-bbox[0], y-bbox[2], s=15, color=color, marker='o')

            plt.savefig(os.path.join(labeled_path_to_save, 'img' + 
                                     str(frame_count).zfill(6) + '.jpg'),
                        bbox_inches='tight', pad_inches=0)
            
            plt.close()
            pbar.update(1)
for j, joint in enumerate(joints):
    


#%%
from utils.analysis_utils import compute_pseudo_R2, mcfadden_rsquare, mcfadden_adjusted_rsquare

pulses = np.squeeze(nev['binned_sync_pulses'])
where_pulse = np.argwhere(pulses == True)[:len(position)]
# where_pulse = np.argwhere(pulses == True)[:len(position)]
# spikes = np.squeeze(nev['binned_spikes'][:, where_pulse])

pos_pcs = position
speed_pcs = speed
spikes = np.squeeze(nev['smoothed_binned_spikes'][:, where_pulse])
spikes = spikes.T
spikes = spikes[1:, :]

kins = [pos_pcs[1:, :], speed_pcs]
kins_first = []
times = []
eig_vals = []
for i, (kin, title) in enumerate(zip(kins, datas_title)):
    
    if title == 'Position filtered':
        best_r2 = 0
        best_channel = 0
        num_pcs = 10
        r2s = []
        
        for c in range(spikes.shape[1]):
            channel = c
            clf = Ridge(alpha=1.0)
            clf.fit(kin[:, :5], spikes[:, channel])
            yhat = clf.predict(kin[:, :5])
            r2 = clf.score(kin[:, :5], spikes[:, channel])
            # r2 = compute_pseudo_R2(spikes[:, channel], 
            #                             yhat,
            #                             np.mean(spikes[:, channel]))
        
            if r2 > best_r2:
                best_r2 = r2
                best_channel = c
            r2s.append(r2)
    
        r2s_order = np.argsort(r2s)
    
    time = np.arange(start, start+interval)
    kinematics_first = kin[:, 0]
    kins_first.append(kinematics_first)
    
#%%
fig = plt.figure()
for i, (kin_first, title) in enumerate(zip(kins_first, datas_title)):
    ax = fig.add_subplot(len(kins)+1, 1, i+1)
    if title == 'Speed filtered' and np.mean(kin_first) < 0:
        kin_first = -kin_first
    ax.plot(time/30, kin_first[time])
    ax.set_title(title, fontsize=14)
    # ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_xlim([time[0]/30, time[-1]/30])
    ax.set_xticklabels([])
    # ax.axis('off')
    
    

# for i in range(spikes_filt.shape[1]):
#     spikes_filt[:, i] = filt_3d('lpf', 0, spikes[:, i])

# ax1 = fig.add_subplot(len(datas)+2, 1, len(datas)+1)
# for j in r2s_order[-10:]:
#     # ax.plot(time/30, spikes[time, j])
#     ax1.plot(time/30, spikes_filt[time, j])
# ax1.set_title('Filtered binned spikes', fontsize=14)
# ax1.set_ylabel('Number of spikes', fontsize=12)
# ax1.set_xlabel('Time (sec)', fontsize=12)
# ax1.set_ylim([0, 10])

ax2 = fig.add_subplot(len(datas)+1, 1, len(datas)+1)
for j in r2s_order[-5:]:
    # ax.plot(time/30, spikes[time, j])
    ax2.plot(time/30, spikes[time, j])
ax2.set_title('Smoothed binned spikes', fontsize=14)
ax2.set_ylabel('Number of spikes', fontsize=12)
ax2.set_xlabel('Time (sec)', fontsize=12)
ax2.set_ylim([0, 10])
ax2.set_xlim([time[0]/30, time[-1]/30])
#%%
left = 0.09  # the left side of the subplots of the figure
right = 0.99   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.94     # the top of the subplots of the figure
wspace = 0.12  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.5 # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig.subplots_adjust(left, bottom, right, top, wspace, hspace)
# fig.tight_layout()

#%%
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import statsmodels.api as sm
from matplotlib import pyplot as plt
from utils.analysis_utils import compute_pseudo_R2, mcfadden_rsquare, \
    mcfadden_adjusted_rsquare

best_r2 = 0
best_channel = 0
num_pcs = 10
r2s = []
ppca = PPCA()
ppca.fit(speed_filt, tol=1e-4)
print(ppca.C.shape)
if ppca.C.shape[0] > 20:
    num_pcs = 10
else:
    num_pcs = 5
# velocity 500 lag (500 +- millisecond)
components_ppca = ppca.transform()
kinematics = components_ppca[:, :num_pcs]
#%%
import statsmodels.api as sm
for j, (data, title) in enumerate(zip(datas, datas_title)):
    for i in range(spikes.shape[1]):
        channel = i
        
        # clf = Ridge(alpha=1.0)
        # clf.fit(data, spikes[:, channel])
        # yhat = clf.predict(data)
        # r2 = clf.score(data, spikes[:, channel])
        
        # initializes a GLM from Poisson model family
        sm.add_constant(kinematics)
        GLM_model = sm.GLM(spikes[:, channel], 
                            kinematics,  
                            family=sm.families.Poisson(),
                            alpha=0.9)
        myFit = GLM_model.fit()
        yhat = myFit.predict(kinematics)
        
        # r2 = r2_score(spikes[:, channel], yhat)
        r2 = 1 - myFit.deviance/myFit.null_deviance
        # r2 = compute_pseudo_R2(spikes.T[:, channel], 
        #                                     yhat,
        #                                     np.mean(spikes.T[:, channel]))
        # r2 = mcfadden_adjusted_rsquare(myFit.params, 
        #                                components_ppca[:, :num_pcs], 
        #                                spikes.T[:, channel])
        
        if r2 > best_r2:
            best_r2 = r2
            best_channel = i
        r2s.append(r2)
        
    print(best_r2)
    print(best_channel)
    
    plt.hist(r2s, np.arange(0, 1.05, 0.05))
    plt.xlabel('R2')
    plt.ylabel('Number of neurons')

print(len(np.argwhere(np.array(r2s) > 0.2)))

#%% R2 plot
for data in kins:
# data = kins[2]
# data = kins[3]
    pulses = np.squeeze(nev['binned_sync_pulses'])
    where_pulse = np.argwhere(pulses == True)[:len(data)]
    spikes_smooth = np.squeeze(nev['smoothed_binned_spikes'][:, where_pulse])
    spikes_smooth = spikes_smooth.T
    
    r2s = []
    for i in range(spikes.shape[1]):
        channel = i
        
        kinematics = data[:, :10]
        # kinematics = data[:, :5]
        
        # clf = Ridge(alpha=1.0)
        # clf.fit(kinematics, spikes_smooth[:, channel])
        # yhat = clf.predict(kinematics)
        # r2 = clf.score(kinematics, spikes_smooth[:, channel])
        
        # initializes a GLM from Poisson model family
        sm.add_constant(kinematics)
        link_g = link_g = sm.genmod.families.links.log.inverse()
        GLM_model = sm.GLM(spikes_smooth[:, channel], 
                            kinematics,  
                            # family=sm.families.Poisson(),
                            # family=sm.families.Poisson(link=sm.families.links.log),
                            family=sm.families.Gaussian(link_g()),
                            alpha=0.9)
        myFit = GLM_model.fit()
        yhat = myFit.predict(kinematics)
        
        # r2 = r2_score(spikes[:, channel], yhat)
        r2 = 1 - myFit.deviance/myFit.null_deviance
        r2s.append(r2)
           
    plt.hist(r2s, np.arange(0, 1.05, 0.05), alpha=0.6)
plt.legend(['Position', 'Speed'], fontsize=12)
plt.ylim([0, 50])
plt.xlabel('R2', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.ylabel('Number of channels', fontsize=14)
plt.title('Comparision of R2 of encoding models', fontsize=16)
# plt.title('R2 of encoding model with speed', fontsize=16)
plt.tight_layout()

#%% Encoding model analysis
result = pd.read_csv(os.path.join(pickle_folder, 
                                  'reaching_experiments/EncodingResults/simplified.csv'),
                     usecols=[3,4,5,6,7])

# channels = np.arange(1,97)
pr2s = [[] for i in range(96)]
# pr2s = np.zeros((96,))

for i in range(96):
    where = result['signalID_1'] == i+1
    pr2s[i].append(result['glm_posNvel_model_eval'][where])

pr2s = np.squeeze(np.array(pr2s))
pr2s_mean = np.array([np.mean(pr2s[i][np.isfinite(pr2s[i])]) for i in range(96)])

plt.axvline(x=0.2, ymin=0, ymax=1, color='tab:red', linewidth=1.5)
plt.hist(pr2s_mean, np.arange(0, 1.05, 0.05))
plt.xlabel('Pseudo R2', fontsize=14)
# plt.xlabel('Position & Speed pR2', fontsize=14)
plt.ylabel('Number of neurons', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
# plt.title('Pseudo-R2 of Encoding model\nwith Position & Speed', fontsize=16)
plt.tight_layout()

print(sum(pr2s_mean > 0.2))
#%% Python PPCA - 1
import os
import smooth
from ppca_old import PPCA
import pandas as pd
import more_itertools as mit
from matplotlib import pyplot as plt
from scipy import stats, signal
from scipy.io import savemat, loadmat
# from sklearn.linear_model import Ridge
from scipy.interpolate import CubicSpline
from scipy.signal import butter,filtfilt, freqz, freqs, welch, periodogram, savgol_filter


def butter_lowpass_filter(data):    
    n = len(data)  # total number of samples
    fs = 30       # sample rate, Hz
    T = n/fs         # Sample Period
    cutoff = 7      # desired cutoff frequency of the filter
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 4       # sin wave can be approx represented as quadratic

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    output = {'y': y,
              'b': b,
              'a': a}
    return output


def filt_3d(filt_type, nan_win, x):
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
    
    nans_consec = [i for i in range(len(x)) if np.isnan(x[i])]
    nans_groups = [list(group) for group in mit.consecutive_groups(nans_consec)]
    
    x_filt = x.copy()
    
    for nans_group in nans_groups:
        if len(nans_group) <= nan_win:
            if (nans_group[0]-2 > 0) and (nans_group[-1]+2 < len(x)):
                expand_indices = [nans_group[0]-2] + [nans_group[0]-1] + \
                nans_group + [nans_group[-1]+1] + [nans_group[-1]+2]
                
                frames = np.arange(len(expand_indices))
                x_trim = x[expand_indices].copy()
                
                x = x[expand_indices].copy()
                nans, _ = nan_helper(x)
                
                cs = CubicSpline(frames[~nans], x[~nans], axis=1)
                x_trim[nans] = cs(frames[nans]).T
                x_filt[nans_group] = x_trim[2:-2]
    
    if filt_type == 'interp':
        return x_filt
    else:
        where_finite = np.arange(len(x_filt))
        where_finite = where_finite[np.isfinite(x_filt)]
        finite_groups = [list(group) for group in mit.consecutive_groups(where_finite)]
        
        for group in finite_groups:
            x = x_filt[group].copy()
            if filt_type == 'savgol' and len(x) > 11:
                sg_x = savgol_filter(x, 7, 5)
                x[group] = sg_x.T
            elif filt_type == 'lpf' and len(x) > 15:
                lpf_x = butter_lowpass_filter(x)
                x_filt[group] = lpf_x['y']               
        return x_filt

def smooth_binned_spikes(binned_spikes, kernel_type, bin_size, kernel_SD, sqrt = 0):
    """
    Binned spikes are stored in a list, sqrt specifies 
    whether to perform square root transform
    """
    smoothed = []
    if sqrt == 1:
       for (i, each) in enumerate(binned_spikes):
           binned_spikes[i] = np.sqrt(each)
    kernel_hl = np.ceil( 3 * kernel_SD / bin_size )
    normalDistribution = stats.norm(0, kernel_SD)
    x = np.arange(-kernel_hl*bin_size, (kernel_hl+1)*bin_size, bin_size)
    kernel = normalDistribution.pdf(x)
    if kernel_type == 'gaussian':
        pass
    elif kernel_type == 'half_gaussian':
       for i in range(0, int(kernel_hl)):
            kernel[i] = 0
    n_sample = np.size(binned_spikes[0])
    nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + 
                                                    int(kernel_hl)] 
    for each in binned_spikes:
        temp1 = np.convolve(kernel,each)
        temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)]/nm
        smoothed.append(temp2)
    print('The binned spikes have been smoothed.')
    return smoothed


# ja_data = loadmat(os.path.join(pickle_folder, 'angle.mat'))
# ja = ja_data['jangles']
# angles = np.vstack([ja[0:4], ja[5:9], ja[10:14], ja[15:19], ja[20:24]])
# angles_diff = np.abs(np.diff(angles.T, axis=0))
# angles_filt = angles_diff.copy()
# for i in range(angles_filt.shape[1]):
#     angles_filt[:, i] = filt_3d('lpf', 4, angles_diff[:, i])
# angles_smooth = smooth_binned_spikes(angles_diff.T, 'gaussian', 1/30, 0.05)
# angles_smooth = np.array(angles_smooth).T

joints = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']

df = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_lpf_cut.csv'))

position = []
velocity = []
speed = []

for joint in joints:
    x = df[joint+'_x']
    y = df[joint+'_y']
    z = df[joint+'_z']
    xyz = np.stack([x,y,z], axis=1)/1000
    position.append(xyz)
    
    xyz = np.diff(xyz, axis=0)/(1/30)
    velocity.append(xyz)
    
    xyz = np.linalg.norm(xyz, axis=1)
    speed.append(xyz)
    
position = np.array(position)
velocity = np.array(velocity)
speed = np.array(speed)

position = np.transpose(position, (1,0,2))
position = position.reshape((len(position), -1))

velocity = np.transpose(velocity, (1,0,2))
velocity = velocity.reshape((len(velocity), -1))

speed = np.transpose(speed, (1,0))
speed_filt = speed.copy()
for i in range(speed_filt.shape[1]):
    speed_filt[:, i] = filt_3d('lpf', 4, speed[:, i])
speed_rect = np.abs(np.transpose(speed, (1,0)))
speed_smooth = smooth_binned_spikes(speed.T, 'gaussian', 1/30, 0.05)
speed_smooth = np.array(speed_smooth).T


nev_file = 'Pop_20200811_FR_003.mat'
session_folder = '20200811'
nev = loadmat(os.path.join(os_path, session_folder, nev_file.split('.')[0]+
                           '_simplified.mat'))

# datas = [position, velocity, speed]
# datas = [position, speed, speed_filt, angles_diff, angles_filt]
datas = [position, speed_filt]
# datas = [position, speed_filt, angles_filt]
# datas = [position, speed, speed_filt, speed_smooth, angles_diff, angles_smooth]
# datas = [speed]
# datas = [position]

# datas_title = ['Position', 'Speed', 'Speed filtered', 'Joint angular velocity', 
#                'Joint angular velocity filtered']

# datas_title = ['Position filtered']
# datas_title = ['Speed']
datas_title = ['Position filtered', 'Speed filtered']
# datas_title = ['Position filtered', 'Speed filtered', 'Joint angular speed smoothed']
# datas_title = ['Position', 'Velocity', 'Speed']
# datas_title = ['Position', 'Speed', 'Speed filtered', 'Speed smoothed', 
#                'Joint angular velocity', 'Joint angular velocity smoothed']

start = 540
interval = 420
# start = 540
# interval = 420

# start = 3160
# interval = 600

#%% Python PPCA - 2
ppca_pos = PPCA()
ppca_pos.fit(position, tol=1e-12, d=5)
pos_pcs = ppca_pos.transform()

ppca_vel = PPCA()
ppca_vel.fit(velocity, tol=1e-12, d=7)
vel_pcs = ppca_vel.transform()

ppca_speed = PPCA()
ppca_speed.fit(speed_filt, tol=1e-12, d=7)
speed_pcs = ppca_speed.transform()


pulses = np.squeeze(nev['binned_sync_pulses'])

where_pulse = np.argwhere(pulses == True)[:len(position)]
# spikes = np.squeeze(nev['binned_spikes'][:, where_pulse])
spikes = np.squeeze(nev['smoothed_binned_spikes'][:, where_pulse])
spikes = spikes.T

savemat(os.path.join(pickle_folder, 'simplified.mat'), 
                      mdict={'smoothed_spikes': spikes,
                              'pos': pos_pcs,
                              'speed': speed_pcs,
                              'velocity': vel_pcs})

#%% PPCA Recover

recover_pos = ppca_pos.inverse_transform()
recover_pos = recover_pos*ppca_pos.stds+ppca_pos.means
recover_vel = ppca_vel.inverse_transform()
recover_vel = recover_vel*ppca_vel.stds+ppca_vel.means

start = 4506
interval = 600
new_t = np.arange(start, start+interval)
ttt = new_t/30

fig = plt.figure(figsize=[9,6])
ax1 = fig.add_subplot(2,1,1)
ax1.plot(ttt,position[new_t, 27])
ax1.plot(ttt,recover_pos[new_t, 27])
# ax1.set_xlabel('Time (sec)', fontsize=12)
ax1.set_ylabel('Position (m)', fontsize=12)
ax1.legend(['Pre-PPCA', 'Post-PPCA'])
ax1.set_xlim([ttt[0], ttt[-1]])
ax1.set_title('Position(x) of PIP1')
ax1.set_xticklabels([])

ax2 = fig.add_subplot(2,1,2)
ax2.plot(ttt,velocity[new_t, 27])
ax2.plot(ttt,recover_vel[new_t, 27])
ax2.set_xlabel('Time (sec)', fontsize=12)
ax2.set_ylabel('Velocity (m/s)', fontsize=12)
ax2.legend(['Pre-PPCA', 'Post-PPCA'])
ax2.set_xlim([ttt[0], ttt[-1]])
ax2.set_title('Velocity(x) of PIP1')


#%% Draw cumsum eigvals
from scipy.io import savemat, loadmat

# pos_eig = data['pos']['eig_vals'][0][0][0]
# speed_eig = data['speed']['eig_vals'][0][0][0]
pos_eig = ppca_pos.eig_vals
vel_eig = ppca_vel.eig_vals

plt.plot(np.cumsum(pos_eig)/np.sum(pos_eig), 
         color='tab:blue',  marker='o', linestyle='-')
# plt.plot(np.cumsum(speed_eig)/np.sum(speed_eig), 
#          color='tab:orange',  marker='o', linestyle='-')
plt.plot(np.cumsum(vel_eig)/np.sum(vel_eig), 
         color='tab:orange',  marker='o', linestyle='-')
# plt.xlim(0, len(ppca_pos.eig_vals))
plt.axhline(y=0.97, c='tab:red', linestyle=':')
plt.axhline(y=0.86, c='tab:red', linestyle=':')
plt.xlim(0, 63)
plt.text(4 - 0.9, 
         np.sum(pos_eig[:5])/np.sum(pos_eig) + 0.03, 
         '5', fontsize=14, color='tab:blue')
# plt.text(6 + 0.3, 
         # np.sum(speed_eig[:7])/np.sum(speed_eig) - 0.03, 
         # '7', fontsize=14, color='tab:orange')
plt.text(7 + 0.3, 
         np.sum(vel_eig[:7])/np.sum(vel_eig) - 0.05, 
         '7', fontsize=14, color='tab:orange')
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.86, 0.9, 0.97, 1.0])
plt.xlabel('Number of components', fontsize=14)
plt.ylabel('Total variance explained', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
# plt.legend(['Position', 'Speed'], loc='center right', fontsize=12)
plt.legend(['Position', 'Velocity'], loc='center right', fontsize=12)
# plt.title('Total variance explained\nfor Position and Speed', fontsize=16)
plt.tight_layout()

#%% PPCA - 1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


joints = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']

# df = pd.read_csv(os.path.join(pickle_folder, 'backup_1218','output_3d_data_lpf.csv'))
df = pd.read_csv(os.path.join(pickle_folder, 'output_3d_data_out2.csv'))

xyzs = []
for joint in joints:
    x = df[joint+'_x']
    y = df[joint+'_y']
    z = df[joint+'_z']
    xyz = np.stack([x,y,z], axis=1)
    xyzs.append(xyz)
xyzs = np.array(xyzs)
xyzs = np.transpose(xyzs, (1,0,2))
xyzs = xyzs.reshape((len(xyzs), -1))
where_finite = ~np.isnan(xyzs).any(axis=1)

#%% PPCA - 2
from scipy.io import savemat, loadmat
from sklearn.linear_model import Ridge
from utils.analysis_utils import compute_pseudo_R2, mcfadden_rsquare, mcfadden_adjusted_rsquare

# pos_pcs = loadmat(os.path.join(pickle_folder, 'position.mat'))['score']
# speed_pcs = loadmat(os.path.join(pickle_folder, 'speed.mat'))['score']

# start = 540
interval = 420
start = 4506
interval = 600
data = loadmat(os.path.join(pickle_folder, 'simplified.mat'))
datas_title = ['Position', 'Velocity']
# datas_title = ['Position', 'Speed']



spikes = data['smoothed_spikes'][1:, :]
posPcs = data['pos'][1:, :]
# posSpeed = data['speed']
posVel = data['velocity']

# kins = np.concatenate((posPcs[:, :5], posVel[:, :33]), axis=1)
kins = [posPcs[:, :5], posVel[:, :33]]
kins_first = []
times = []
eig_vals = []
for i, (kin, title) in enumerate(zip(kins, datas_title)):
    
    if title == 'Position':
        best_r2 = 0
        best_channel = 0
        num_pcs = 10
        r2s = []
        
        for c in range(spikes.shape[1]):
            channel = c
            clf = Ridge(alpha=1.0)
            clf.fit(kin, spikes[:, channel])
            yhat = clf.predict(kin)
            r2 = clf.score(kin, spikes[:, channel])
            if r2 > best_r2:
                best_r2 = r2
                best_channel = c
            r2s.append(r2)
    
        r2s_order = np.argsort(r2s)
    
    time = np.arange(start, start+interval)
    kinematics_first = kin[:, 2]
    kins_first.append(kinematics_first)

# best_r2 = 0
# best_channel = 0
# num_pcs = 10
# r2s = []

# for c in range(spikes.shape[1]):
#     channel = c
#     clf = Ridge(alpha=1.0)
#     clf.fit(kins, spikes[:, channel])
#     yhat = clf.predict(kins)
#     r2 = clf.score(kins, spikes[:, channel])
#     if r2 > best_r2:
#         best_r2 = r2
#         best_channel = c
#     r2s.append(r2)

# r2s_order = np.argsort(r2s)

fig = plt.figure()
for i, (kin_first, title) in enumerate(zip(kins_first, datas_title)):
    ax = fig.add_subplot(len(kins_first)+1, 1, i+1)
    if title == 'Velocity' and np.mean(kin_first) < 0:
        kin_first = -kin_first
    ax.plot(time/30, kin_first[time])
    ax.set_title('First PC of '+title, fontsize=14)
    # ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_xlim([time[0]/30, time[-1]/30])
    ax.set_xticklabels([])
    # ax.axis('off')
    
    

# for i in range(spikes_filt.shape[1]):
#     spikes_filt[:, i] = filt_3d('lpf', 0, spikes[:, i])

# ax1 = fig.add_subplot(len(datas)+2, 1, len(datas)+1)
# for j in r2s_order[-10:]:
#     # ax.plot(time/30, spikes[time, j])
#     ax1.plot(time/30, spikes_filt[time, j])
# ax1.set_title('Filtered binned spikes', fontsize=14)
# ax1.set_ylabel('Number of spikes', fontsize=12)
# ax1.set_xlabel('Time (sec)', fontsize=12)
# ax1.set_ylim([0, 10])

ax2 = fig.add_subplot(3, 1, 3)
for j in r2s_order[-5:]:
    ax2.plot(time/30, spikes[time, j])
ax2.set_title('Smoothed binned spikes', fontsize=14)
ax2.set_ylabel('Number of spikes', fontsize=12)
ax2.set_xlabel('Time (sec)', fontsize=12)
ax2.set_ylim([0, 10])
ax2.set_xlim([time[0]/30, time[-1]/30])
plt.tight_layout()

#%%
aaxis = 3
a = velocity[time,aaxis]
b = posVel[time, aaxis]

plt.plot(stats.zscore(a, nan_policy='omit'))
plt.plot(stats.zscore(b, nan_policy='omit'))


#%% PPCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ppca_old import PPCA

which = position.copy()
ppca = PPCA()
d = 5
ppca.fit(which, d=d, tol=1e-10)
print(ppca.C.shape)
recover_ppca = ppca.inverse_transform()
recover = recover_ppca*ppca.stds+ppca.means
b = ppca.transform()
# a= np.diff(recover, axis=0)
a = recover

jjoint = 27
x = which[:, jjoint]
where_finite = [i for i in range(len(x)) if np.isfinite(x[i])]
finite_groups = [list(group) for group in mit.consecutive_groups(where_finite)]
longest = np.argmax([len(group) for group in finite_groups])
# longest=33
#%%
new_t = np.array(finite_groups[longest])

# interval = 420
start = 4506
interval = 600
new_t = np.arange(start, start+interval)
fs = 30
colors = ['tab:blue', 'tab:orange', 'tab:red']
linestyles = ['-', '-.']
for i in range(3):
    f, Pxx = welch(which[new_t, jjoint+i], fs, 'flattop',
                          nperseg=50, 
                    average='mean'
                    )
    # f, Pxx = welch(which[new_t, jjoint+i], fs, nperseg=64)
    plt.semilogy(f, np.sqrt(Pxx), linestyles[0], color=colors[i])

for i in range(3):
    f, Pxx = welch(a[new_t, jjoint+i], fs, 'flattop', 
                          nperseg=50,
                    average='mean'
                    )
    # f, Pxx = welch(a[new_t, jjoint+i], fs, nperseg=64)
    plt.semilogy(f, np.sqrt(Pxx), linestyles[1], color=colors[i])

# for i in range(3):
#     f, Pxx = welch(b[new_t, i], fs, nperseg=64)
#     plt.semilogy(f, np.sqrt(Pxx))
plt.legend(['post-PPCA-x', 'post-PPCA-y','post-PPCA-z',
            'pre-PPCA-x','pre-PPCA-y','pre-PPCA-z'
            ])
# plt.legend(['pre-PPCA-x','pre-PPCA-y','pre-PPCA-z',
#             'post-PPCA-x', 'post-PPCA-y','post-PPCA-z'
#             ])
# plt.legend(['pre-PPCA-x','pre-PPCA-y','pre-PPCA-z'])
# plt.legend(['pre-PPCA-x','post-PPCA-x'])
plt.xlim([0, 15])
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Linear spectrum (V RMS)', fontsize=12)
plt.title('Power spectral density of PIP1', fontsize=14)

#%%