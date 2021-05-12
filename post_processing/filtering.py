#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 07:02:53 2020

@author: minyoungpark
"""
import os
import smooth
import numpy as np
import pandas as pd
import more_itertools as mit
# from post_processing.outlier_removal import remove_finger_connection
from scipy.signal import butter, filtfilt, savgol_filter

from scipy.interpolate import CubicSpline

def remove_finger_connection(df_origin, finger, lowers, uppers):
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
    # z, p, k = butter(order, normal_cutoff, btype='low', analog=False)
    
    y = filtfilt(b, a, data)
    output = {'y': y,
              'b': b,
              'a': a}
    return output


def filt_3d(interp_type, filt_type, nan_win, xyz):
    def nan_helper(y):
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
            elif filt_type == 'lpf' and len(y) > 16:
                lpf_x = butter_lowpass_filter(x)
                lpf_y = butter_lowpass_filter(y)
                lpf_z = butter_lowpass_filter(z)
                xyz_filt[group, 0] = lpf_x['y']
                xyz_filt[group, 1] = lpf_y['y']
                xyz_filt[group, 2] = lpf_z['y']                
        return xyz_filt


def filter_3d(config, joints, fingers, interp_type, filt_type):
    path = config['triangulation']['reconstruction_output_path']
    df = pd.read_csv(os.path.join(path,'output_3d_data_out2.csv'))
    df_filt = df.copy()
    
    for joint in joints:
        x = df[joint+'_x'].copy()
        y = df[joint+'_y'].copy()
        z = df[joint+'_z'].copy()
        coords = np.stack([x,y,z]).T
        xyz_filt = filt_3d('linear', filt_type, 2, coords)
        df_filt[joint+'_x'] = xyz_filt[:, 0]
        df_filt[joint+'_y'] = xyz_filt[:, 1]
        df_filt[joint+'_z'] = xyz_filt[:, 2]

    dist_meds = []
    for i, finger in enumerate(fingers):
        finger_meds = []
        for j, connection in enumerate(finger):
            poi1 = connection[0]
            poi2 = connection[1]
            poi1_coord = np.stack([df_filt[poi1+'_x'],
                                   df_filt[poi1+'_y'], 
                                   df_filt[poi1+'_z']], axis=1)
            poi2_coord = np.stack([df_filt[poi2+'_x'], 
                                   df_filt[poi2+'_y'], 
                                   df_filt[poi2+'_z']], axis=1)
            dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
            dist_med = np.median(dist[np.isfinite(dist)])
            finger_meds.append(dist_med)
            
        dist_meds.append(finger_meds)
        
    df_filt_cut = df_filt.copy()
    
    # if filt_type is not 'interp':
    for i, (finger, meds) in enumerate(zip(fingers, dist_meds)):
        df_filt_cut = remove_finger_connection(df_filt_cut, finger, 
                                                np.array(meds)*0.6, np.array(meds)*1.4)
        
    df_filt_cut.to_csv(os.path.join(path, 'output_3d_data_'+filt_type+'.csv'), 
                     index=False)
