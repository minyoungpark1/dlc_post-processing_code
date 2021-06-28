#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:07:13 2019

@author: minyoungpark
"""

import os
import cv2
import itertools

import numpy as np
import pandas as pd
from tqdm import trange
from numpy import array as arr
from collections import defaultdict
from scipy import optimize

from utils.triangulation_utils import load_2d_data, load_labeled_2d_data

from utils.calibration_utils import get_video_path, load_intrinsics, load_extrinsics, load_calib_new


def expand_matrix(mtx):
    z = np.zeros((4,4))
    z[0:3,0:3] = mtx[0:3,0:3]
    z[3,3] = 1
    return z


def reproject_points(p3d, points2d, camera_mats):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    return proj


def reprojection_error(p3d, points2d, camera_mats):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    errors = np.linalg.norm(proj - points2d, axis=1)
    return np.mean(errors)


def distort_points_cams(points, camera_mats):
    out = []
    for i in range(len(points)):
        point = np.append(points[i], 1)
        mat = camera_mats[i]
        new = mat.dot(point)[:2]
        out.append(new)
    return np.array(out)


def reprojection_error_und(p3d, points2d, camera_mats, camera_mats_dist):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    proj_d = distort_points_cams(proj, camera_mats_dist)
    points2d_d = distort_points_cams(points2d, camera_mats_dist)
    errors = np.linalg.norm(proj_d - points2d_d, axis=1)
    return np.mean(errors)


def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams*2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i*2):(i*2+1)] = x*mat[2]-mat[0]
        A[(i*2+1):(i*2+2)] = y*mat[2]-mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d / p3d[3]
    return p3d


def triangulate_points(the_points, cam_mats):
    p3ds = []
    errors = []
    for ptnum in range(the_points.shape[0]):
        points = the_points[ptnum]
        good = ~np.isnan(points[:, 0])
        p3d = triangulate_simple(points[good], cam_mats[good])
        err = reprojection_error(p3d, points[good], cam_mats[good])
        p3ds.append(p3d)
        errors.append(err)
    p3ds = np.array(p3ds)
    errors = np.array(errors)
    return p3ds, errors


def triangulate_undistorted_points(pts, cam_mats, cam_mats_dist):
    one_point = False
    
    if len(pts.shape) == 2:
        pts = pts.reshape(-1, 1, 2)
        one_point = True

    p3ds = []
    errors = []
    for ptnum in range(pts.shape[1]):
        points = pts[:, ptnum, :]
        # print(points)
        good = ~np.isnan(points[:, 0])
        p3d = triangulate_simple(points[good], cam_mats[good])
        err = reprojection_error_und(p3d, points[good], cam_mats[good], cam_mats_dist[good])
        p3ds.append(p3d)
        errors.append(err)
    p3ds = np.array(p3ds)
    errors = np.array(errors)
    
    if one_point:
        p3ds = p3ds[0]
        errors = errors[0]
    return p3ds, errors

# points_shaped = points_2d.reshape(n_cams, n_frames*n_joints, 1, 2)


def triangulate_ransac(points, cam_mats, cam_mats_dist, 
                         undistort=True, min_cams=2, threshold=0.5):
    """Given an CxNxPx2 array, this returns an Nx3 array of points
    by triangulating all possible points and picking the ones with
    best reprojection error
    where:
    C: number of cameras
    N: number of points
    P: number of possible options per point
    """
    
    min_cams = min_cams
    n_cams, n_points, _ = points.shape
    points = points.reshape(n_cams, n_points, 1, 2)    
    n_cams, n_points, n_possible, _ = points.shape
    cam_nums, point_nums, possible_nums = np.where(
        ~np.isnan(points[:, :, :, 0]))

    all_iters = defaultdict(dict)
    
    for cam_num, point_num, possible_num in zip(cam_nums, point_nums,
                                                possible_nums):
        if cam_num not in all_iters[point_num]:
            all_iters[point_num][cam_num] = []
        all_iters[point_num][cam_num].append((cam_num, possible_num))

    for point_num in all_iters.keys():
        for cam_num in all_iters[point_num].keys():
            all_iters[point_num][cam_num].append(None)

    out = np.full((n_points, 3), np.nan, dtype='float64')
    picked_vals = np.zeros((n_cams, n_points, n_possible), dtype='bool')
    errors = np.zeros(n_points, dtype='float64')
    points_2d = np.full((n_cams, n_points, 2), np.nan, dtype='float64')

    iterator = trange(n_points, ncols=70)
    
    for point_ix in iterator:
        best_point = None
        best_error = 200

        n_cams_max = len(all_iters[point_ix])
        if n_cams_max < min_cams:
            continue
        
        for picked in itertools.product(*all_iters[point_ix].values()):
            picked = [p for p in picked if p is not None]
            if len(picked) < min_cams:
                continue
            
            cnums = [p[0] for p in picked]
            xnums = [p[1] for p in picked]

            pts = points[cnums, point_ix, xnums]
            # print(pts)
            sub_cam_mats = np.array([cam_mats[idx].copy() for idx in cnums])
            sub_cam_mats_dist = np.array([cam_mats_dist[idx].copy() for idx in cnums])
            p3d, err = triangulate_undistorted_points(pts, sub_cam_mats, sub_cam_mats_dist)
            
            # cc = subset_cameras(cnums)

            # p3d = cc.triangulate(pts, undistort=undistort)
            # err = cc.reprojection_error(p3d, pts, mean=True)
            # print(err)
            
            if best_point is not None:
                if len(cnums) > len(best_point['picked']) and err < 10:
                    best_point = {
                        'error': err,
                        'point': p3d[:3],
                        'points': pts,
                        'picked': picked,
                        'joint_ix': point_ix
                    }
                    best_error = err
                else:
                    if err < best_error:
                        best_point = {
                            'error': err,
                            'point': p3d[:3],
                            'points': pts,
                            'picked': picked,
                            'joint_ix': point_ix
                        }
                        best_error = err
            else: 
                if err < 10:
                    best_point = {
                                'error': err,
                                'point': p3d[:3],
                                'points': pts,
                                'picked': picked,
                                'joint_ix': point_ix
                            }
                    best_error = err
        # print(picked)                
        
        if best_point is not None:
            out[point_ix] = best_point['point']
            picked = best_point['picked']
            cnums = [p[0] for p in picked]
            xnums = [p[1] for p in picked]
            picked_vals[cnums, point_ix, xnums] = True
            errors[point_ix] = best_point['error']
            points_2d[cnums, point_ix] = best_point['points']

    return out, picked_vals, points_2d, errors


def triangulate_finger_ransac(points, cam_mats, cam_mats_dist, 
                         undistort=True, min_cams=3, threshold=0.5):
    """Given an CxNxPx2 array, this returns an Nx3 array of points
    by triangulating all possible points and picking the ones with
    best reprojection error
    where:
    C: number of cameras
    N: number of points
    P: number of possible options per point
    """
    
    min_cams = min_cams
    n_cams, n_points, _ = points.shape
    
    points = points.reshape(n_cams, n_points, 1, 2)
    
    n_cams, n_points, n_possible, _ = points.shape

    cam_nums, point_nums, possible_nums = np.where(
        ~np.isnan(points[:, :, :, 0]))

    all_iters = defaultdict(dict)

    for cam_num, point_num, possible_num in zip(cam_nums, point_nums,
                                                possible_nums):
        if cam_num not in all_iters[point_num]:
            all_iters[point_num][cam_num] = []
        all_iters[point_num][cam_num].append((cam_num, possible_num))

    for point_num in all_iters.keys():
        for cam_num in all_iters[point_num].keys():
            all_iters[point_num][cam_num].append(None)

    out = np.full((n_points, 3), np.nan, dtype='float64')
    picked_vals = np.zeros((n_cams, n_points, n_possible), dtype='bool')
    errors = np.zeros(n_points, dtype='float64')
    points_2d = np.full((n_cams, n_points, 2), np.nan, dtype='float64')

    iterator = trange(n_points, ncols=70)
    print(min_cams)
    for point_ix in iterator:
        best_point = None
        best_error = 200

        n_cams_max = len(all_iters[point_ix])
        if n_cams_max < min_cams:
            continue
        
        for picked in itertools.product(*all_iters[point_ix].values()):
            picked = [p for p in picked if p is not None]
            if len(picked) < min_cams:
                continue
            
            # print(picked)
            
            cnums = [p[0] for p in picked]
            xnums = [p[1] for p in picked]

            pts = points[cnums, point_ix, xnums]
            # print(pts)
            sub_cam_mats = np.array([cam_mats[idx].copy() for idx in cnums])
            sub_cam_mats_dist = np.array([cam_mats_dist[idx].copy() for idx in cnums])
            p3d, err = triangulate_undistorted_points(pts, sub_cam_mats, sub_cam_mats_dist)
            
            # cc = subset_cameras(cnums)

            # p3d = cc.triangulate(pts, undistort=undistort)
            # err = cc.reprojection_error(p3d, pts, mean=True)
            # print(err)
            if len(cnums) == 4 and err < 10:
                best_point = {
                    'error': err,
                    'point': p3d[:3],
                    'points': pts,
                    'picked': picked,
                    'joint_ix': point_ix
                }
                best_error = err
                continue
            else:
                if err < best_error:
                    best_point = {
                        'error': err,
                        'point': p3d[:3],
                        'points': pts,
                        'picked': picked,
                        'joint_ix': point_ix
                    }
                    best_error = err
                    if best_error < threshold:
                        break
        # print(picked)                
        if best_point is not None:
            out[point_ix] = best_point['point']
            picked = best_point['picked']
            cnums = [p[0] for p in picked]
            xnums = [p[1] for p in picked]
            picked_vals[cnums, point_ix, xnums] = True
            errors[point_ix] = best_point['error']
            points_2d[cnums, point_ix] = best_point['points']

    return out, picked_vals, points_2d, errors


def optim_error_fun(points, camera_mats):
    def fun(x):
        p3d = np.array([x[0], x[1], x[2], 1])
        proj = np.dot(camera_mats, p3d)
        resid = points - proj[:, :2] / proj[:, 2, None]
        return resid.flatten()
        # return np.linalg.norm(resid, axis=1)
    return fun


def triangulate_optim(points, camera_mats, max_error=20):
    try:
        p3d = triangulate_simple(points, camera_mats)
        # error = reprojection_error(p3d, points, camera_mats)
    except np.linalg.linalg.LinAlgError:
        return np.array([0,0,0,0])

    fun = optim_error_fun(points, camera_mats)
    try:
        res = optimize.least_squares(fun, p3d[:3], loss='huber', f_scale=1e-3)
        x = res.x
        p3d = np.array([x[0], x[1], x[2], 1])
    except ValueError:
        pass

    return p3d


def proj(u, v):
    """Project u onto v"""
    return u * np.dot(v,u) / np.dot(u,u)


def ortho(u, v):
    """Orthagonalize u with respect to v"""
    return u - proj(v, u)


def get_median(all_points_3d, ix):
    pts = all_points_3d[:, ix]
    pts = pts[~np.isnan(pts[:, 0])]
    return np.median(pts, axis=0)


def correct_coordinate_frame(config, all_points_3d, bodyparts):
    """Given a config and a set of points and bodypart names, this function will rotate the coordinate frame to match the one in config"""
    bp_interested = config['labeling']['bodyparts_interested']
    bp_index = dict(zip(bp_interested, range(len(bp_interested))))
    axes_mapping = dict(zip('xyz', range(3)))

    ref_point = config['triangulation']['reference_point']
    axes_spec = config['triangulation']['axes']
    a_dirx, a_l, a_r = axes_spec[0]
    b_dirx, b_l, b_r = axes_spec[1]

    a_dir = axes_mapping[a_dirx]
    b_dir = axes_mapping[b_dirx]

    ## find the missing direction
    done = np.zeros(3, dtype='bool')
    done[a_dir] = True
    done[b_dir] = True
    c_dir = np.where(~done)[0][0]

    a_lv = get_median(all_points_3d, bp_index[a_l])
    a_rv = get_median(all_points_3d, bp_index[a_r])
    b_lv = get_median(all_points_3d, bp_index[b_l])
    b_rv = get_median(all_points_3d, bp_index[b_r])

    a_diff = a_rv - a_lv
    b_diff = ortho(b_rv - b_lv, a_diff)

    M = np.zeros((3,3))
    M[a_dir] = a_diff
    M[b_dir] = b_diff
    if (a_dir==0 and b_dir==1) or (a_dir==1 and b_dir==2) or (a_dir==2 and b_dir==0):
        M[c_dir] = np.cross(a_diff, b_diff)
    else:
        M[c_dir] = np.cross(b_diff, a_diff)

    M /= np.linalg.norm(M, axis=1)[:,None]

    center = get_median(all_points_3d, bp_index[ref_point])

    # all_points_3d_adj = np.dot(all_points_3d - center, M.T)
    all_points_3d_adj = (all_points_3d - center).dot(M.T)
    center_new = get_median(all_points_3d_adj, bp_index[ref_point])
    all_points_3d_adj = all_points_3d_adj - center_new

    recovery = {'center_new': center_new,
                'registration_mat': M,
                'center': center}
    
    return all_points_3d_adj, recovery


def undistort_points(all_points_raw, cam_names, intrinsics):
    all_points_und = np.zeros(all_points_raw.shape)

    for ix_cam, cam_name in enumerate(cam_names):
        calib = intrinsics[cam_name]
        points = all_points_raw[:, ix_cam].reshape(-1, 1, 2)
        points_new = cv2.undistortPoints(
            points, arr(calib['camera_mat']), arr(calib['dist_coeff']))
        all_points_und[:, ix_cam] = points_new.reshape(
            all_points_raw[:, ix_cam].shape)

    return all_points_und


def reconstruct_3d(config, **kwargs):
    path, videos, vid_indices = get_video_path(config)
    bp_interested = config['labeling']['bodyparts_interested']
    reconstruction_threshold = config['triangulation']['reconstruction_threshold']
    if config['triangulation'].get('reconstruction_output_path') is None:
        output_path = kwargs.get('output_path', '')
    else:
        output_path = config['triangulation']['reconstruction_output_path']

    try:
        intrinsics = load_intrinsics(path, vid_indices)
    except:
        print("Intrinsic calibration output does not exist.")
        return

    try:
        extrinsics = load_extrinsics(path)
    except:
        print("Extrinsic calibration output does not exist.")
        return
    
    # intrinsics, extrinsics = load_calib_new(config)
    
    cam_mats = []
    cam_mats_dist = []

    for vid_idxs in vid_indices:
        mat = arr(extrinsics[vid_idxs])
        left = arr(intrinsics[vid_idxs]['camera_mat'])
        cam_mats.append(mat)
        cam_mats_dist.append(left)

    cam_mats = arr(cam_mats)
    cam_mats_dist = arr(cam_mats_dist)

    out = load_2d_data(config, vid_indices, bp_interested)
    
    all_points_raw = out['points']
    all_scores = out['scores']

    all_points_und = undistort_points(all_points_raw, vid_indices, intrinsics)

    length = all_points_raw.shape[0]
    shape = all_points_raw.shape

    all_points_3d = np.zeros((shape[0], shape[2], 3))
    all_points_3d.fill(np.nan)

    errors = np.zeros((shape[0], shape[2]))
    errors.fill(np.nan)

    scores_3d = np.zeros((shape[0], shape[2]))
    scores_3d.fill(np.nan)

    num_cams = np.zeros((shape[0], shape[2]))
    num_cams.fill(np.nan)

    all_points_und[all_scores < reconstruction_threshold] = np.nan

    for i in trange(all_points_und.shape[0], ncols=70):
        for j in range(all_points_und.shape[2]):
            pts = all_points_und[i, :, j, :]
            good = ~np.isnan(pts[:, 0])
            if np.sum(good) >= 2:
                # TODO: make triangulation type configurable
                # p3d = triangulate_optim(pts[good], cam_mats[good])
                p3d = triangulate_simple(pts[good], cam_mats[good])
                all_points_3d[i, j] = p3d[:3]
                errors[i,j] = reprojection_error_und(p3d, pts[good], cam_mats[good], cam_mats_dist[good])
                num_cams[i,j] = np.sum(good)
                scores_3d[i,j] = np.min(all_scores[i, :, j][good])

    if 'reference_point' in config['triangulation'] and 'axes' in config['triangulation']:
        all_points_3d_adj, recovery = correct_coordinate_frame(config, all_points_3d, bp_interested)
    else:
        all_points_3d_adj = all_points_3d

    dout = pd.DataFrame()
    for bp_num, bp in enumerate(bp_interested):
        for ax_num, axis in enumerate(['x','y','z']):
            dout[bp + '_' + axis] = all_points_3d_adj[:, bp_num, ax_num]
        dout[bp + '_error'] = errors[:, bp_num]
        dout[bp + '_ncams'] = num_cams[:, bp_num]
        dout[bp + '_score'] = scores_3d[:, bp_num]

    dout['fnum'] = np.arange(length)

    dout.to_csv(os.path.join(output_path, 'output_3d_data.csv'), index=False)
    
    if 'reference_point' in config['triangulation'] and 'axes' in config['triangulation']:
        return recovery
    else:
        return None


<<<<<<< HEAD
def reconstruct_3d_ransac(config, min_cams=2, model_type='dlc', **kwargs):
=======
def reconstruct_3d_ransac(config, min_cams=2, **kwargs):
>>>>>>> f86fa74cc15f1d374d7857ccff3d57f3ee6ba515
    path, videos, vid_indices = get_video_path(config)
    bp_interested = config['labeling']['bodyparts_interested']
    reconstruction_threshold = config['triangulation']['reconstruction_threshold']
    if config['triangulation'].get('reconstruction_output_path') is None:
        output_path = kwargs.get('output_path', '')
    else:
        output_path = config['triangulation']['reconstruction_output_path']
    
    try:
        intrinsics = load_intrinsics(path, vid_indices)
    except:
        print("Intrinsic calibration output does not exist.")
        return

    try:
        extrinsics = load_extrinsics(path)
    except:
        print("Extrinsic calibration output does not exist.")
        return
    
    # intrinsics, extrinsics = load_calib_new(config)

    cam_mats = []
    cam_mats_dist = []

    for vid_idxs in vid_indices:
        mat = arr(extrinsics[vid_idxs])
        left = arr(intrinsics[vid_idxs]['camera_mat'])
        cam_mats.append(mat)
        cam_mats_dist.append(left)

    cam_mats = arr(cam_mats)
    cam_mats_dist = arr(cam_mats_dist)

    # (fnum, n, j, 2)
<<<<<<< HEAD
    out = load_2d_data(config, vid_indices, bp_interested, model_type)
=======
    out = load_2d_data(config, vid_indices, bp_interested)
>>>>>>> f86fa74cc15f1d374d7857ccff3d57f3ee6ba515
    
    
    all_points_raw = out['points']
    all_scores = out['scores']

    all_points_und = undistort_points(all_points_raw, vid_indices, intrinsics)
    all_points_und[all_scores < reconstruction_threshold] = np.nan
    
    n_frames, n_cams, n_joints, _ = all_points_und.shape
    points_2d = np.transpose(all_points_und, (1, 0, 2, 3))
    
    points_shaped = points_2d.reshape(n_cams, n_frames*n_joints, 2)
    
    min_cams = min_cams
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
        all_points_3d_adj, recovery = correct_coordinate_frame(config, all_points_3d, bp_interested)
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
    
    if 'reference_point' in config['triangulation'] and 'axes' in config['triangulation']:
        return recovery
    else:
        return None


def reconstruct_3d_finger(config, min_cams=3, **kwargs):
    path, videos, vid_indices = get_video_path(config)
    bp_interested = config['labeling']['bodyparts_interested']
    reconstruction_threshold = config['triangulation']['reconstruction_threshold']
    if config['triangulation'].get('reconstruction_output_path') is None:
        output_path = kwargs.get('output_path', '')
    else:
        output_path = config['triangulation']['reconstruction_output_path']
    
    # try:
    #     intrinsics = load_intrinsics(path, vid_indices)
    # except:
    #     print("Intrinsic calibration output does not exist.")
    #     return

    # try:
    #     extrinsics = load_extrinsics(path)
    # except:
    #     print("Extrinsic calibration output does not exist.")
    #     return
    
    intrinsics, extrinsics = load_calib_new(config)

    cam_mats = []
    cam_mats_dist = []

    for vid_idxs in vid_indices:
        mat = arr(extrinsics[vid_idxs])
        left = arr(intrinsics[vid_idxs]['camera_mat'])
        cam_mats.append(mat)
        cam_mats_dist.append(left)

    cam_mats = arr(cam_mats)
    cam_mats_dist = arr(cam_mats_dist)

    out = load_2d_data(config, vid_indices, bp_interested)
    
    all_points_raw = out['points']
    all_scores = out['scores']

    all_points_und = undistort_points(all_points_raw, vid_indices, intrinsics)
    all_points_und[all_scores < reconstruction_threshold] = np.nan
    
    n_frames, n_cams, n_joints, _ = all_points_und.shape
    points_2d = np.transpose(all_points_und, (1, 0, 2, 3))
    
    points_shaped = points_2d.reshape(n_cams, n_frames*n_joints, 2)
    
    # Finger 1, 2, 3, 4 
    connection_indices = [[3,8,12,17],
                          [4,9,13,18],
                          [5,10,14,19],
                          [6,11,15,20]]
    a = all_points_raw[:, :, connection_indices[0], :]
    min_cams = min_cams
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
        all_points_3d_adj, recovery = correct_coordinate_frame(config, all_points_3d, bp_interested)
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

    dout.to_csv(os.path.join(output_path, 'output_3d_data.csv'), index=False)
    
    return recovery



def validate_3d(config, **kwargs):
    path, videos, vid_indices = get_video_path(config)
    bp_interested = config['labeling']['bodyparts_interested']
    reconstruction_threshold = config['triangulation']['reconstruction_threshold']
    if config['triangulation'].get('reconstruction_output_path') is None:
        output_path = kwargs.get('output_path', '')
    else:
        output_path = config['triangulation']['reconstruction_output_path']

    try:
        intrinsics = load_intrinsics(path, vid_indices)
    except:
        print("Intrinsic calibration output does not exist.")
        return

    try:
        extrinsics = load_extrinsics(path)
    except:
        print("Extrinsic calibration output does not exist.")
        return

    cam_mats = []
    cam_mats_dist = []

    for vid_idxs in vid_indices:
        mat = arr(extrinsics[vid_idxs])
        left = arr(intrinsics[vid_idxs]['camera_mat'])
        cam_mats.append(mat)
        cam_mats_dist.append(left)

    cam_mats = arr(cam_mats)
    cam_mats_dist = arr(cam_mats_dist)

    out = load_labeled_2d_data(config, vid_indices, bp_interested)
    
    all_points_raw = out['points']

    all_points_und = undistort_points(all_points_raw, vid_indices, intrinsics)

    length = all_points_raw.shape[0]
    shape = all_points_raw.shape

    all_points_3d = np.zeros((shape[0], shape[2], 3))
    all_points_3d.fill(np.nan)

    errors = np.zeros((shape[0], shape[2]))
    errors.fill(np.nan)

    scores_3d = np.zeros((shape[0], shape[2]))
    scores_3d.fill(np.nan)

    num_cams = np.zeros((shape[0], shape[2]))
    num_cams.fill(np.nan)

    for i in trange(all_points_und.shape[0], ncols=70):
        for j in range(all_points_und.shape[2]):
            pts = all_points_und[i, :, j, :]
            good = ~np.isnan(pts[:, 0])
            if np.sum(good) >= 2:
                # TODO: make triangulation type configurable
                # p3d = triangulate_optim(pts[good], cam_mats[good])
                p3d = triangulate_simple(pts[good], cam_mats[good])
                all_points_3d[i, j] = p3d[:3]
                errors[i,j] = reprojection_error_und(p3d, pts[good], cam_mats[good], cam_mats_dist[good])
                num_cams[i,j] = np.sum(good)

    if 'reference_point' in config['triangulation'] and 'axes' in config['triangulation']:
        all_points_3d_adj = correct_coordinate_frame(config, all_points_3d, bp_interested)
    else:
        all_points_3d_adj = all_points_3d

    dout = pd.DataFrame()
    for bp_num, bp in enumerate(bp_interested):
        for ax_num, axis in enumerate(['x','y','z']):
            dout[bp + '_' + axis] = all_points_3d_adj[:, bp_num, ax_num]
        dout[bp + '_error'] = errors[:, bp_num]
        dout[bp + '_ncams'] = num_cams[:, bp_num]

    dout['fnum'] = np.arange(length)

    dout.to_csv(os.path.join(output_path, 'validate_3d_data.csv'), index=False)
