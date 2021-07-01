#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import more_itertools as mit


def outlier_speed_removal(config, joints, speed_thr):
    path = config['triangulation']['reconstruction_output_path']
    df_origin = pd.read_csv(os.path.join(path,'output_3d_data_raw.csv'))
    
    df_speed = df_origin.copy()
    for i, joint in enumerate(joints):
        joint_coord = np.stack([df_origin[joint+'_x'], 
                                df_origin[joint+'_y'], 
                                df_origin[joint+'_z']], axis=1)
        joint_speed = np.linalg.norm(np.diff(joint_coord, axis=0), axis=1)/(1/30)/1000
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
        df_speed[joint+'_x'][above_thr] = np.nan
        df_speed[joint+'_y'][above_thr] = np.nan
        df_speed[joint+'_z'][above_thr] = np.nan
        
        joint_coord = np.stack([df_speed[joint+'_x'], 
                                df_speed[joint+'_y'], 
                                df_speed[joint+'_z']], axis=1)
        
    df_speed.to_csv(os.path.join(path, 'output_3d_data_speed.csv'), index=False)
    
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

def outlier_connection_removal(config, fingers):
    path = config['triangulation']['reconstruction_output_path']
    df = pd.read_csv(os.path.join(path, 'output_3d_data_speed.csv'))
    
    dist_meds = []
    for i, finger in enumerate(fingers):
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
            dist_med = np.median(dist[np.isfinite(dist)])
            finger_meds.append(dist_med)
            
        dist_meds.append(finger_meds)
            
    dist_meds2 = []
    dist_stds = []
    for i, (finger, uppers) in enumerate(zip(fingers, dist_meds)):
        df = remove_finger_connection(df, finger, np.array(uppers)*0.6, 
                                      np.array(uppers)*1.4)
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
            dist_med = np.median(dist[np.isfinite(dist)])
            finger_meds.append(dist_med)
            dist_std = np.std(dist[np.isfinite(dist)])
            finger_stds.append(dist_std)
            
        dist_meds2.append(finger_meds)
        dist_stds.append(finger_stds)
    
    df.to_csv(os.path.join(path, 'output_3d_data_out1.csv'), index=False)
    
    for i, (finger, med, std) in enumerate(zip(fingers, dist_meds2, dist_stds)):
        df = remove_finger_connection(df, finger, np.array(med)-2.0*np.array(std), 
                                      np.array(med)+2.0*np.array(std))
        
    df.to_csv(os.path.join(path, 'output_3d_data_out2.csv'), index=False)
