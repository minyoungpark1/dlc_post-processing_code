# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 02:20:07 2021

@author: robin
"""
import os
import pandas as pd
import numpy as np
from tqdm import trange


def convertPickleToCSV(postfix, joints, pickle_paths, paths_to_save):
    for pickle_path, path_to_save in zip(pickle_paths, paths_to_save):
        if not os.path.exists(pickle_path):
            print('Pickle file does not exist.')
            return
        
        if not os.path.exists(os.path.dirname(path_to_save)):
            print(os.path.dirname(path_to_save) + ' does not exsit.')
            folder_input = input('Do you want to create this path (folder)? (y/n) ')
            if folder_input is 'y':
                os.mkdir(os.path.dirname(path_to_save))
            elif folder_input is 'n':
                return
            else:
                print('Wrong input.')
                return
            
    iterables = [[postfix.split('.')[0]], ['right_hand'], joints, ['x', 'y', 'likelihood']]
    header = pd.MultiIndex.from_product(iterables, names=['scorer', 'individuals', 'bodyparts', 'coords'])
    
    for pickle_path, path_to_save in zip(pickle_paths, paths_to_save):
        data = pd.read_pickle(pickle_path)
        L = len(data) - 1
        index = np.arange(L)
        df = pd.DataFrame(np.zeros((L, len(joints)*3)), index=index, columns=header)
        for i in trange(L):
            framenum = 'frame' + str(i).zfill(int(np.ceil(np.log10(L))))
            for k, joint in enumerate(joints):
                coord = data[framenum]['coordinates'][0][k]
                conf = data[framenum]['confidence'][k]
                if len(coord) > 1:
                    max_conf_idx = np.argmax(conf)
                    # all_points_raw[i, j, k, :] = coord[max_conf_idx]
                    # df.iloc[i, 3*k:3*k+2] = coord[max_conf_idx]
                    # df.iloc[i, 3*k+2] = conf[max_conf_idx]
                    df[postfix.split('.')[0], 'right_hand', joint, 'x'][i] = coord[max_conf_idx][0]
                    df[postfix.split('.')[0], 'right_hand', joint, 'y'][i] = coord[max_conf_idx][1]
                    df[postfix.split('.')[0], 'right_hand', joint, 'likelihood'][i] = conf[max_conf_idx][0]
                    
                elif len(coord) > 0:
                    # df.iloc[i ,3*k:3*k+2] = coord
                    # df.iloc[i, 3*k+2] = conf
                    df[postfix.split('.')[0], 'right_hand', joint, 'x'][i] = coord[0][0]
                    df[postfix.split('.')[0], 'right_hand', joint, 'y'][i] = coord[0][1]
                    df[postfix.split('.')[0], 'right_hand', joint, 'likelihood'][i] = conf[0]
        
        df.to_csv(path_to_save, mode='w')
        df.to_hdf(path_to_save.split('.')[0]+postfix.split('.')[0][:-5]+'.h5', key='df_with_missing', mode='w')



def interp_drop(data, timestamp):
    tstamp = []
    data_interp = np.zeros((int(np.sum(timestamp))+1,))
    for i, t in enumerate(timestamp):
        if t > 1:
            xp = np.array([i, i+t])
            fp = data[i:i+2]
            interp = np.interp(np.arange(i+1,i+t), xp, fp)
            start = int(np.cumsum(timestamp[:i])[-1])
            end = int(np.cumsum(timestamp[:i+1])[-1])
            data_interp[start+1:end] = interp
            data_interp[end] = data[i]
            tstamp = np.append(tstamp, np.arange(start+1, end+1))
        else:
            end = int(np.cumsum(timestamp[:i+1])[-1])
            data_interp[end] = data[i]
            tstamp = np.append(tstamp, end)
    return data_interp, tstamp


def interpDroppedData(postfix, joints, csv_paths, timestamp_paths, paths_to_save):
    for path_to_save in paths_to_save:
        if not os.path.exists(os.path.dirname(path_to_save)):
            print(os.path.dirname(path_to_save) + ' does not exsit.')
            folder_input = input('Do you want to create this path (folder)? (y/n) ')
            if folder_input is 'y':
                os.mkdir(os.path.dirname(path_to_save))
            elif folder_input is 'n':
                return
            else:
                print('Wrong input.')
                return
        
    min_len = 0
    iterables = [[postfix.split('.')[0]], ['right_hand'], joints, ['x', 'y', 'likelihood']]
    header = pd.MultiIndex.from_product(iterables, names=['scorer', 'individuals', 'bodyparts', 'coords'])
    Ls = []
    timestamps = []
    timestamp_diffs = []
    dfs = []
    
    for csv_path, timestamp_path in zip(csv_paths, timestamp_paths):
        df = pd.read_csv(csv_path, header=[2,3], index_col=0)
        where_off = []
        timestamp = np.loadtxt(timestamp_path)
        timestamp_diff = np.array(np.diff(timestamp))/(1/30)
        td_copy = timestamp_diff.copy()
        timestamp_diff = np.round(timestamp_diff)
        
        if timestamp_diff[0] > 4:
            df = df.drop(0)
            timestamp_diff = timestamp_diff[1:]
            timestamp = timestamp[1:]
            
        timestamp -= timestamp[0]
        timestamps.append(timestamp)    
        timestamp_diff = np.insert(timestamp_diff, 0, 0)
        timestamp_diffs.append(timestamp_diff)
        dfs.append(df)
        
    td = np.array(timestamp_diffs)
    
    print(np.array(timestamps)[:,-1]*30)
    print(np.sum(td, axis=1))
    if len(set(np.sum(td, axis=1))) > 1:
        td_mod = td.copy()
        for i in range(1, td.shape[1]-1):
            td_sum1 = np.sum(td[:, i-1:i+1], axis=1)
            td_sum2 = np.sum(td[:, i:i+2], axis=1)
            td_sums = np.sum(td[:, i-1:i+2], axis=1)
            if len(set(td_sum1)) > 1 and len(set(td_sum2)) > 1 and \
                len(set(td_sums)) > 1 and len(set(np.sum(td[:, :i+3], axis=1))) > 1:
                print('Timestamp mismatch happened at: ' + str(i))
                mode = np.median(td_sums)
                where = np.argwhere(td_sums != mode)
                modify = (mode - td_sums[where])
                td[where, i] = td[where, i] + modify
                
        td = np.round(td)
        timestamp_diffs = [td[i, :] for i in range(td.shape[0])]
    
    print(np.sum(timestamp_diffs, axis=1))
    
    for path_to_save, timestamp_diff, df in zip(paths_to_save, timestamp_diffs, dfs):
        L = int(np.sum(timestamp_diff)+1)
        index = np.arange(L)
        df_interp = pd.DataFrame(np.zeros((L, len(joints)*3)), index=index, columns=header)
        
        for i, joint in enumerate(joints):
            x = np.array(df.iloc[:, 3*i])
            y = np.array(df.iloc[:, 3*i+1])
            err = np.array(df.iloc[:, 3*i+2])
            x_interp, tstamp = interp_drop(x, timestamp_diff)
            y_interp, _ = interp_drop(y, timestamp_diff)
            err_interp, _ = interp_drop(err, timestamp_diff)
            
            df_interp[postfix.split('.')[0], 'right_hand', joint, 'x'] = x_interp
            df_interp[postfix.split('.')[0], 'right_hand', joint, 'y'] = y_interp
            df_interp[postfix.split('.')[0], 'right_hand', joint, 'likelihood'] = err_interp
        
        df_interp[df_interp==0] = np.nan
        df_interp.to_csv(path_to_save, mode='w')
        
def interpDroppedData2(postfix, joints, csv_paths, timestamp_paths, paths_to_save):
    for path_to_save in paths_to_save:
        if not os.path.exists(os.path.dirname(path_to_save)):
            print(os.path.dirname(path_to_save) + ' does not exsit.')
            folder_input = input('Do you want to create this path (folder)? (y/n) ')
            if folder_input is 'y':
                os.mkdir(os.path.dirname(path_to_save))
            elif folder_input is 'n':
                return
            else:
                print('Wrong input.')
                return
        
    iterables = [[postfix.split('.')[0]], ['right_hand'], joints, ['x', 'y', 'likelihood']]
    header = pd.MultiIndex.from_product(iterables, names=['scorer', 'individuals', 'bodyparts', 'coords'])
    
    timestamps = []
    max_timestamps = []
    dfs = []
    
    for csv_path, timestamp_path in zip(csv_paths, timestamp_paths):
        df = pd.read_csv(csv_path, header=[2,3], index_col=0)
        timestamp = np.loadtxt(timestamp_path)
        timestamp_diff = np.round(timestamp[1] - timestamp[0])
        
        if timestamp_diff > 4:
            df = df.drop(0)
            timestamp = timestamp[1:]
            
        timestamp -= timestamp[0]
        timestamps.append(timestamp*30)
        max_timestamps.append(timestamp[-1]*30)
        dfs.append(df)
    
    max_timestamp = max(max_timestamps)
    print(np.array(max_timestamps))
    
    for path_to_save, timestamp, df in zip(paths_to_save, timestamps, dfs):
        L = int(max_timestamp + 1)
        index = np.arange(L)
        df_interp = pd.DataFrame(np.zeros((L, len(joints)*3)), index=index, columns=header)
        t = np.arange(0, L)
        for i, joint in enumerate(joints):
            x = np.array(df.iloc[:, 3*i])
            y = np.array(df.iloc[:, 3*i+1])
            err = np.array(df.iloc[:, 3*i+2])
            x_interp = np.interp(t, timestamp, x)
            y_interp = np.interp(t, timestamp, y)
            err_interp = np.interp(t, timestamp, err)
            
            df_interp[postfix.split('.')[0], 'right_hand', joint, 'x'] = x_interp
            df_interp[postfix.split('.')[0], 'right_hand', joint, 'y'] = y_interp
            df_interp[postfix.split('.')[0], 'right_hand', joint, 'likelihood'] = err_interp
        
        df_interp[df_interp==0] = np.nan
        df_interp.to_csv(path_to_save, mode='w')
