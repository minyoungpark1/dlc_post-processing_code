#%% Multi-view Bootstrapping
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.calibration_utils import *
from utils.vis_utils import extract_specific_frames
from triangulation.triangulate import *
from calibration.extrinsic import *


class MVB():
    def __init__(self, config, data_path):
        self.config = config
        self.vid_indices = []
        self.data_path = data_path
        self.intrisics = {}
        self.extrinsics = {}
        self.good_reconstruction_indices = []
        self.new_indices = []
        self.img_format = None
        self.scorer = None
        self.individuals = None
        
        
        
    def _load_camera_matrices(self):
        calib_path = self.config['calibration']['calib_video_path']
        self.intrinsics = load_intrinsics(calib_path, self.vid_indices)
        self.extrinsics = load_extrinsics(calib_path)
        
        
    def _load_timestamps(self, video_paths):
        timestamp_paths = [video_path.split('.')[0] + '_logfile.txt' for video_path in video_paths]
            
        max_timestamps = []
        timestamps = []
        for timestamp_path in timestamp_paths:
            try:
                timestamp = np.loadtxt(timestamp_path)
            except FileNotFoundError:
                print('Timestamp log files must be located in a same folder where videos are located.')
                break
            timestamp_diff = np.round(timestamp[1] - timestamp[0])

            if timestamp_diff > 4:
                timestamp = timestamp[1:]

            timestamp -= timestamp[0]
            timestamps.append(timestamp*30)
        
        return timestamps
        
        
    # This method recovers indices before the interpolation
    def _recover_indices(self, video_paths, indices):
        timestamps = self._load_timestamps(video_paths)
        new_indices = []
        for timestamp in timestamps:
            timestamp_round = np.round(timestamp)
            new_idx = np.searchsorted(timestamp_round, indices)
            new_indices.append(new_idx)
        return new_indices
        
        
    def _generate_bootstrapped_images(self, video_paths, save_paths):
#         for video_path, save_path in zip(video_paths, save_paths):
#             extract_specific_frames(video_path, self.good_reconstruction_indices, save_path, self.img_format)
        for video_path, save_path, new_idx in zip(video_paths, save_paths, self.new_indices):
            extract_specific_frames(video_path, new_idx, save_path, self.img_format)
            
            
    def _generate_bootstrapped_h5(self, save_paths):
        data_3d = pd.read_csv(self.data_path)
        self._load_camera_matrices()
        joints = self.config['labeling']['bodyparts_interested']
        L = len(self.good_reconstruction_indices)

        iterables = [[self.scorer], [self.individuals], joints, ['x', 'y']]
        header = pd.MultiIndex.from_product(iterables, names=['scorer', 'individuals', 'bodyparts', 'coords'])

        data_2ds = []
        
        for save_path, vid_idx in zip(save_paths, self.vid_indices):
            cameraMatrix = np.matrix(self.intrinsics[vid_idx]['camera_mat'])
            distCoeffs = np.array(self.intrinsics[vid_idx]['dist_coeff'])
            Rt = np.matrix(self.extrinsics[vid_idx])
            rvec, tvec = get_rtvec(Rt)

            out = os.path.basename(save_path)
            index = [os.path.join('labeled-data', out, 'img' + str(frame_count).zfill(6) + '.' + self.img_format)
                      for frame_count in self.good_reconstruction_indices]

            df_2d = pd.DataFrame(np.zeros((L, len(joints)*2)), index=index, columns=header)

            for i, bp in enumerate(tqdm(joints)):
#                 x = data_3d[bp+'_x'][new_idx].copy()
#                 y = data_3d[bp+'_y'][new_idx].copy()
#                 z = data_3d[bp+'_z'][new_idx].copy()
                x = data_3d[bp+'_x'][self.good_reconstruction_indices].copy()
                y = data_3d[bp+'_y'][self.good_reconstruction_indices].copy()
                z = data_3d[bp+'_z'][self.good_reconstruction_indices].copy()
                # for idx in new_idx:
                #     if idx not in where_good_joint[bp]:
                #         x[idx] = np.nan
                #         y[idx] = np.nan
                #         z[idx] = np.nan
                objectPoints = np.vstack([x,y,z]).T
                coord_2d = np.squeeze(cv2.projectPoints(objectPoints, rvec, tvec, 
                                                        cameraMatrix, distCoeffs)[0], axis=1)
                where_negative = np.argwhere(coord_2d[:,0] < 0)
                where_out1 = np.argwhere(coord_2d[:,0] >= 2048)
                where_out2 = np.argwhere(coord_2d[:,1] >= 1536)
                coord_2d[where_negative, :] = np.nan
                coord_2d[where_out1, :] = np.nan
                coord_2d[where_out2, :] = np.nan

                df_2d.iloc[:, 2*i:2*i+2] = coord_2d

            if not os.path.exists(save_path):
                os.mkdir(path_to_save)
            df_2d.to_hdf(os.path.join(save_path, 'CollectedData_' + self.scorer + '.h5'),
                         key='df_with_missing', mode='w')
            data_2ds.append(df_2d)
        
        
    def bootstrap(self, video_paths, save_paths, vid_indices, img_format, scorer, individuals):
        indices = self.good_reconstruction_indices
        
        self.vid_indices = vid_indices
        self.img_format = img_format
        self.scorer = scorer
        self.individuals = individuals
        
        if len(indices) == 0:
            print('Zero frame was selected')
            return
        
        self.new_indices = self._recover_indices(video_paths, indices)
        self._generate_bootstrapped_images(video_paths, save_paths)
        self._generate_bootstrapped_h5(save_paths)

        
    def select_good_frames(self, criterion='measured', cutoff=3, N=1000):
        joints = self.config['labeling']['bodyparts_interested']
        scheme = self.config['labeling']['scheme']
        dist_measured = self.config['labeling']['dist_measured']

        df = pd.read_csv(self.data_path)
        
        connection_dict = {}
        arg_accurate = {}
        dist_meds = {}
        dist_means = {}
        dist_stds = {}
        dist_outliers = {}
        for i, connection in enumerate(scheme):
            poi1 = connection[0]
            poi2 = connection[1]
            poi1_coord = np.stack([df[poi1+'_x'], df[poi1+'_y'], df[poi1+'_z']], axis=1)
            poi2_coord = np.stack([df[poi2+'_x'], df[poi2+'_y'], df[poi2+'_z']], axis=1)
            dist = np.linalg.norm(poi1_coord - poi2_coord, axis=1)
            connection_dict[poi1+'-'+poi2] = dist
            dist_med = np.median(dist[np.isfinite(dist)])
            dist_std = np.std(dist[np.isfinite(dist)])
            dist_mean = np.mean(dist[np.isfinite(dist)])
            dist_meds[poi1+'-'+poi2] = dist_med
            dist_stds[poi1+'-'+poi2] = dist_std
            dist_means[poi1+'-'+poi2] = dist_mean
            
            if criterion is 'measured':
                threshold = dist_measured[i]
            elif criterion is 'median':
                threshold = dist_med
            dist[np.isnan(dist)] = -100
            
            arg_accurate[poi1+'-'+poi2] = np.argwhere((dist > (threshold*0.85)) & 
                                                      (dist < (threshold*1.15)))

        good_reconstruction_indices = []
        good_number = []
        where_good = {}
        for i in range(len(df)):
            idx_threshold = 0
            where_good[i] = []
            for connection in scheme:
                poi1 = connection[0]
                poi2 = connection[1]
                if i in arg_accurate[poi1+'-'+poi2]:
                    idx_threshold += 1
                    where_good[i].append(poi1)
                    where_good[i].append(poi2)

            good_number.append(idx_threshold)
            if idx_threshold >= len(scheme)-cutoff:
                good_reconstruction_indices.append(i)

            where_good[i] = set(where_good[i])

        print(f'Mean of standard deviation of inferred connection lengths is: {np.mean(list(dist_stds.values())):.2f} mm')
        print(f'Length of good reconstructions indices: {len(good_reconstruction_indices):d}')
    
        where_good_joint = {}
        where_joint = {}
        chose_good_joint = {}

        for joint in joints:
            where_good_joint[joint] = []
            where_joint[joint] = []
            chose_good_joint[joint] = []

        for i in range(len(df)):
            for joint in joints:
                if joint in where_good[i]:
                    where_joint[joint].append(i)

        for idx in good_reconstruction_indices:
            for joint in joints:
                if joint in where_good[idx]:
                    where_good_joint[joint].append(idx)
                    
        if len(good_reconstruction_indices) > N:
            chosen_indices = np.random.choice(good_reconstruction_indices, N)
            print(f'Randomly selecting {N:d} frames')
            for idx in chosen_indices:
                for joint in joints:
                    if joint in where_good[idx]:
                        chose_good_joint[joint].append(idx)
            self.good_reconstruction_indices = chosen_indices
        else:
            self.good_reconstruction_indices = good_reconstruction_indices

            
    def check_mvb_images(self, video_paths, image_save_paths, data_save_paths):
        joints = self.config['labeling']['bodyparts_interested']
        data_2ds = [pd.read_hdf(os.path.join(save_path, 'CollectedData_' + self.scorer + '.h5'), 
                                'df_with_missing', header=[2,3], index_col=0) for save_path in data_save_paths]
        for image_save_path, data_2d, vidpath, new_idx in zip(image_save_paths, data_2ds, video_paths, self.new_indices):
            if not os.path.exists(image_save_path):
                os.mkdir(image_save_path)

            cap = cv2.VideoCapture(vidpath) 
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_num/fps
#             count = len(self.good_reconstruction_indices)
            count = len(new_idx)

            colorclass = plt.cm.ScalarMappable(cmap='jet')
            C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
            colors = C[:, :3]

            for f, frame_count in enumerate(tqdm(new_idx)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                plt.figure()
                # plt.imshow(frame)
                xs = []
                ys = []
                for joint in joints:
                    xs.append(data_2d[self.scorer, self.individuals, joint, 'x'].iloc[f])
                    ys.append(data_2d[self.scorer, self.individuals, joint, 'y'].iloc[f])

                xs = np.array(xs)
                ys = np.array(ys)

                x0 = np.median(xs[np.isfinite(xs)])
                y0 = np.median(ys[np.isfinite(ys)])

                bbox = [max(x0-50, 0), min(x0+50, frame.shape[1]), max(y0-50, 0), min(y0+50, frame.shape[1])]
                bbox = [int(b) for b in bbox]
                # frame = cv2.transpose()
                frame_cropped = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]

                plt.imshow(frame_cropped)

                for i, (color, joint) in enumerate(zip(colors, joints)):
                    x = data_2d[self.scorer, self.individuals, joint, 'x'].iloc[f]
                    y = data_2d[self.scorer, self.individuals, joint, 'y'].iloc[f]
                    plt.scatter(x-bbox[0], y-bbox[2], s=15, color=color, marker='o')

                plt.savefig(os.path.join(image_save_path, 'img' + str(frame_count).zfill(6) + '.' + self.img_format),
                            bbox_inches='tight', pad_inches=0)

                plt.close()