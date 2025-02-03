import math
import numpy as np
import cv2
import re
import logging
import torch
from tqdm import tqdm
import sys
import os
import torch
import albumentations as albu

from torch.utils.data import Dataset
from dataloader.utils import camera_parameters, qv_mult, flip_pose, h36m_cameras_intrinsic_params

from utils.pose import heatmap_generator


subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
Mono_3d_file= True #need to choose which one I want to use.
KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
load_imgs = True
AllCameras = False
CameraView = True 

standardize_3d = True
standardize_2d = False
Normalize = False

number_of_frames = 243

class H36_dataset(Dataset):
    def __init__(self, subjectp=subjects , transform=None, target_transform=None, is_train = True, split_rate=None):
        
        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        
        self.split_rate = split_rate

        self.dataset2d, self.dataset3d, self.video_and_frame_paths = self.read_data(subjects= subjectp)
        
        if self.split_rate:
            self.dataset2d = self.dataset2d[::split_rate]
            self.dataset3d = self.dataset3d[::split_rate]
            self.video_and_frame_paths = self.video_and_frame_paths[::split_rate]

    def __len__(self):
        return len(self.dataset3d)

    def __getitem__(self, idx):

        if load_imgs:
            frame = cv2.imread(self.video_and_frame_paths[idx][0])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_2d = self.dataset2d[idx].reshape(-1 ,2)
        

        #testing cropping using the gt 2d as bounding box: 
        bc = [min(keypoints_2d[:,0]), max(keypoints_2d[:,1])] 
        br = [max(keypoints_2d[:,0]), max(keypoints_2d[:,1])] 
        pc = [min(keypoints_2d[:,0]), min(keypoints_2d[:,1])] 
        pr = [max(keypoints_2d[:,0]), min(keypoints_2d[:,1])] 
        l_m = int (max(br[0] - bc[0], br[1] - pr[1])  * 1. *1000 )
        frame = frame[ min(0,int(keypoints_2d[0,1]*1000-(l_m/2))) : max(1000,int(keypoints_2d[0,1]*1000+(l_m/2))) ,  min(0,int(keypoints_2d[0,0]*1000- (l_m/2))) : max(1000,int(keypoints_2d[0,0]*1000+ (l_m/2))) , :]
        # breakpoint()
        #resising the image for Resnet
        frame = cv2.resize(frame, (256, 256))
        # frame = cv2.resize(frame, (1024, 1024))
        frame = frame/256.0

        subject, action, frame_num = self.find_parameters_from_path(self.video_and_frame_paths[idx][0])

        return keypoints_2d, frame, subject, action, frame_num

    def find_parameters_from_path(self, path):
        pattern = r'videos/([^/]+)/outputVideos/([^/]+)\.mp4/(\d{4})\.jpg'

        match = re.search(pattern, path)

        if Mono_3d_file :
            if re:
                s = match.group(1)  # subject
                a = match.group(2)  # Discussion.23657
                frame_num = int(match.group(3))  # frame_number

        return s,a,frame_num

    def xyz_to_uvw(self, kp): #here
        return np.array([-kp[1], -kp[2], kp[0]])

    def _keypoint_to_heatmap_3D(self, keypoint, sigma=0.5): #sigma=1.75
        """
        Read the function name duh
        Args:
            keypoints (np.float32): Keypoints in 3D ranges from -1 to 1
            sigma (float, optional):Size of the Gaussian. Defaults to 1.75.
        """
        
        assert np.min(keypoint) >= -1
        assert np.max(keypoint) <= 1

        # Create an empty volumetric heatmap
        im = np.zeros((64, 64, 64), dtype=np.float32)

        # Scale keypoints from -1 to 1
        keypoint = 31.5 * (1 + keypoint)
        keypoint_int = np.rint(keypoint).astype(int)

        # Size of 3D Gaussian window.
        size = int(math.ceil(6 * sigma))
        # Ensuring that size remains an odd number
        if not size % 2:
            size += 1

        # Generate gaussian, with window=size and variance=sigma
        u = np.arange(keypoint_int[0] - (size // 2), keypoint_int[0] + (size // 2) + 1)
        v = np.arange(keypoint_int[1] - (size // 2), keypoint_int[1] + (size // 2) + 1)
        w = np.arange(keypoint_int[2] - (size // 2), keypoint_int[2] + (size // 2) + 1)
        
        uu, vv, ww = np.meshgrid(u, v, w, indexing='ij', sparse=True)
        z = np.exp(-((uu - keypoint[0]) ** 2 + (vv - keypoint[1]) ** 2 + (ww - keypoint[2]) ** 2) / (2 * (sigma ** 2)))

        # Identify indices in im that will define the crop area
        top_u = max(0, keypoint_int[0] - (size//2))
        top_v = max(0, keypoint_int[1] - (size//2))
        top_w = max(0, keypoint_int[2] - (size//2))

        bottom_u = min(64, keypoint_int[0] + (size//2) + 1)
        bottom_v = min(64, keypoint_int[1] + (size//2) + 1)
        bottom_w = min(64, keypoint_int[2] + (size//2) + 1)

        im[top_u:bottom_u, top_v:bottom_v, top_w:bottom_w] = \
            z[top_u - (keypoint_int[0] - (size//2)): top_u - (keypoint_int[0] - (size//2)) + (bottom_u - top_u),
            top_v - (keypoint_int[1] - (size//2)): top_v - (keypoint_int[1] - (size//2)) + (bottom_v - top_v),
            top_w - (keypoint_int[2] - (size//2)): top_w - (keypoint_int[2] - (size//2)) + (bottom_w - top_w)]

        return im

    def keypoints_to_heatmap_3D(self, keypoints):
        hm = []
        for i in range(keypoints.shape[0]):
            kp = self.xyz_to_uvw(keypoints[i])
            # kp = keypoints[i]
            hm.append(self._keypoint_to_heatmap_3D(kp))
        return np.stack(hm, axis=0)

    @staticmethod
    def read_data(subjects = subjects):
        
        cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]

        if Mono_3d_file:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m_mono.npz', allow_pickle=True)
        else:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m.npz', allow_pickle=True)
        data_file_2d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_2d_h36m.npz', allow_pickle=True) #2d is considered as the groundtruth

        data_file_3d = data_file_3d['positions_3d'+"_mono"*Mono_3d_file].item()
        data_file_2d = data_file_2d['positions_2d'].item()

        data_directory = '/mnt/vita/scratch/datasets/Human3.6'
        n_frame = 0 
        for s in subjects:
            for a in data_file_3d[s].keys():#For every action (even if many action for same action, a will be the different videos)
                n_frame += len(data_file_3d[s][a])  
            
        n_frame = n_frame + 3*int(not Mono_3d_file)*n_frame #would be 4*n_frame if Mono_3d_file is False

        #Chaque frame a sa GT 3d et sa gt 2D
        all_in_one_dataset_3d = np.zeros((n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
        all_in_one_dataset_2d = np.zeros((n_frame if AllCameras else n_frame, 17 ,2),  dtype=np.float32)
        video_and_frame_paths = []
        i = 0
        for s in subjects:
            for a in data_file_3d[s].keys():
                for frame_num in range(len(data_file_3d[s][a])):

                    pose_3d = data_file_3d[s][a][frame_num]  
                    pose_3d = pose_3d[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                    
                    if Mono_3d_file :
                        all_in_one_dataset_3d[i] = pose_3d
                        tmp2 = data_file_2d[s][a][frame_num]
                        all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                        
                        if load_imgs:
                            video_and_frame_paths.append( [data_directory+"/videos/"+s+"/outputVideos/"+a+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])

                        i = i + 1 
                        
                    else :
                        for c in range(1+3*int(AllCameras)) :
                            tmp = pose_3d.copy()

                            if CameraView:
                                for j in range(len(tmp)): 
                                    tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][c]['translation']),1000)
                                    tmp[j] = qv_mult(np.array(camera_parameters[s][c]['orientation']),tmp[j])
                                        
                            all_in_one_dataset_3d[i] = tmp

                            tmp2 = data_file_2d[s][a+cam_ids[c]][frame_num]    
                            all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                            if load_imgs:
                                video_and_frame_paths.append( [data_directory+"/videos/"+s+"/outputVideos/"+a+cam_ids[c]+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])

                            i = i + 1 

        
        return all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths

class Human36_dataset_3D(Dataset):
    def __init__(self, subjectp=subjects , transform=None, target_transform=None, is_train = True, split_rate=None, model_2D = None, num_frame=243, num_keypoints=17):

        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        
        self.split_rate = split_rate
        self.number_of_frames = num_frame
        self.num_keypoints = num_keypoints
        #File to take the 2D predicted keypoints from
        directory = '/mnt/vita/scratch/datasets/Human3.6/npy/'
        ground_truth = False
        self.probabilistic = False
        if model_2D == 'ViTPose':
            logging.info('Using ViTPose Prediction')
            if is_train:
                keypoints_2d_path = directory + 'keypoints_2D_ViTPose_trainset.npy'
            else:
                keypoints_2d_path = directory + 'keypoints_2D_ViTPose_testset.npy'
    
        elif model_2D == 'Hourglass':
            if is_train:
                keypoints_2d_path = directory + 'keypoints_2D_Hourglass_trainset.npy'
            else:
                keypoints_2d_path = directory + 'keypoints_2D_Hourglass_testset.npy'
        elif model_2D == 'GroundThruth':
            ground_truth = True
            keypoints_2d_path = None
        elif model_2D == 'Probabilistic':
            self.probabilistic = True
            if is_train:#Only GroundTruth for now generated with ViTPose model
                keypoints_2d_path = directory + 'keypoints_2D_GroundThruth_trainset_prob_pose.npy'
            else:
                keypoints_2d_path = directory + 'keypoints_2D_GroundThruth_testset_prob_pose.npy'
        else:
            raise ValueError(f"Unsupported model name '{model_2D}'. Please choose either 'ViTPose' or 'Hourglass' or 'GroundThruth.")
        
        dataset2d, dataset3d, mean, covariance, num_frame_in_sequence, giant_dic = self.read_data(subjects= subjectp, file_keypoints_2D = keypoints_2d_path, ground_truth = ground_truth, probabilistic = self.probabilistic, num_keypoints = num_keypoints)
        self.dataset2d = []
        self.dataset3d = []
        self.dataset2d_mean = []
        self.dataset2d_covariance = []
        if self.probabilistic:
            list_of_indexes = []
            for s in giant_dic.keys():
                count_a = 0
                for a in giant_dic[s].keys():
                    count_a= count_a+1
                    if (len(giant_dic[s][a]['num_frame_in_sequence'])!=len(giant_dic[s][a]['covariance'])) or (len(giant_dic[s][a]['mean']) !=len(giant_dic[s][a]['3d'])) or (len(giant_dic[s][a]['covariance']) !=len(giant_dic[s][a]['3d'])):
                        print("Action",a,"Subject",s, "has an inconsistency, it was ignored")
                        break
                    list_of_index = self.pick_sequence_from_frame_probabilistic(np.array(giant_dic[s][a]['num_frame_in_sequence']),num_frame,num_keypoints)
                    list_of_indexes.extend(list_of_index)

                self.list_of_index = list_of_indexes
                self.giant_dic = giant_dic
                local_vars = locals()
                total_size = sum(sys.getsizeof(var) for var in local_vars.values())
                print(f"Total memory used: {total_size / (1024 ** 3):.2f} GB")
                print("Number of sequences:", len(self.list_of_index))

        else:
            self.dataset2d, self.dataset3d = self.pick_sequence_from_frame(dataset2d, dataset3d, num_frame_in_sequence, mean, covariance, num_frame,num_keypoints)
        if self.split_rate:
            if not self.probabilistic:
                self.dataset2d = self.dataset2d[::split_rate]
                self.dataset3d = self.dataset3d[::split_rate]

    def __len__(self):
        if self.probabilistic:
            return len(self.list_of_index)
        else:
            return len(self.dataset3d)

    def __getitem__(self, idx):
        if self.probabilistic:
            subject, action, index = self.list_of_index[idx]
            ground_truth_3d = []
            keypoints_2d = []
            for idx in range(index-self.number_of_frames//2,index+self.number_of_frames//2+1):
                samples = torch.stack([torch.distributions.MultivariateNormal(self.giant_dic[subject][action]['mean'][idx][i,:],self.giant_dic[subject][action]['covariance'][idx][i,:]).sample() for i in range(self.giant_dic[subject][action]['mean'][idx].shape[0])])
            #    #samples = np.array([np.random.multivariate_normal(self.giant_dic[subject][action]['mean'][idx][i,:], self.giant_dic[subject][action]['covariance'][idx][i,:], size=1).squeeze() for i in range(self.giant_dic[subject][action]['mean'][idx].shape[0])])
                keypoints_2d.append(samples)
            ground_truth_3d.append(self.giant_dic[subject][action]['3d'][index])
            keypoints_2d = torch.stack(keypoints_2d)
            ground_truth_3d =torch.stack(ground_truth_3d)
            keypoints_2d = keypoints_2d.view(-1, self.num_keypoints, 2) / 64
            ground_truth_3d = ground_truth_3d.view(-1, 3)
            # keypoints_2d = keypoints_2d.reshape(-1,self.num_keypoints,2)/64
            # ground_truth_3d = ground_truth_3d.reshape(-1,3)
        else:
            keypoints_2d = self.dataset2d[idx].reshape(-1,self.num_keypoints,2)
            ground_truth_3d = self.dataset3d[idx].reshape(-1,3)

        return ground_truth_3d, keypoints_2d

    @staticmethod
    def read_data(subjects, file_keypoints_2D, ground_truth, probabilistic, num_keypoints):
        
        cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]

        if Mono_3d_file:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m_mono.npz', allow_pickle=True)
        else:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m.npz', allow_pickle=True)
        data_file_3d = data_file_3d['positions_3d'+"_mono"*Mono_3d_file].item()

        if ground_truth:
            data_file_2d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_2d_h36m.npz', allow_pickle=True) #2d is considered as the groundtruth
            data_file_2d = data_file_2d['positions_2d'].item()

        data_directory = '/mnt/vita/scratch/datasets/Human3.6'
        n_frame = 0 
        for s in subjects:
            for a in data_file_3d[s].keys():#For every action (even if many action for same action, a will be the different videos)
                n_frame += len(data_file_3d[s][a])  
            
        n_frame = n_frame + 3*int(not Mono_3d_file)*n_frame #would be 4*n_frame if Mono_3d_file is False

        if not ground_truth:
            keypoints2D = np.load(file_keypoints_2D, allow_pickle=True)
            keypoints2D = keypoints2D[()]



        #Chaque frame a sa GT 3d et sa gt 2D
        all_in_one_dataset_3d = np.zeros((n_frame if AllCameras else n_frame, num_keypoints ,3),  dtype=np.float32)

        all_in_one_dataset_2d_mean = None
        all_in_one_dataset_2d_covariance = None

        if probabilistic:
            all_in_one_dataset_2d_mean = np.zeros((n_frame if AllCameras else n_frame, num_keypoints*2),  dtype=np.float32)
            all_in_one_dataset_2d_covariance = np.zeros((n_frame if AllCameras else n_frame, num_keypoints*2,num_keypoints*2),  dtype=np.float32)
            all_in_one_dataset_2d_keypoints = None
        else:#GroundTruth or Normal
            all_in_one_dataset_2d_keypoints = np.zeros((n_frame if AllCameras else n_frame, num_keypoints ,2),  dtype=np.float32)

        num_frame_in_sequence = []
        i = 0
        giant_dic={}
        for s in subjects:
            giant_dic[s]={}
            for a in data_file_3d[s].keys():
                giant_dic[s][a] = {}
                giant_dic[s][a]['num_frame_in_sequence'] = []
                giant_dic[s][a]['mean'] = []
                giant_dic[s][a]['covariance'] = []
                giant_dic[s][a]['2d'] = []
                giant_dic[s][a]['3d'] = []
                for frame_num in range(len(data_file_3d[s][a])):
                    if not ground_truth:
                        if frame_num==len(keypoints2D[s][a]):
                            break
                    pose_3d = data_file_3d[s][a][frame_num]  
                    pose_3d = pose_3d[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                    
                    num_frame_in_sequence.append([frame_num+1,len(data_file_3d[s][a])])
                    if probabilistic:
                        giant_dic[s][a]['num_frame_in_sequence'].append([frame_num+1,len(data_file_3d[s][a]),s,a])
                    if Mono_3d_file :
                        all_in_one_dataset_3d[i] = pose_3d
                        if ground_truth:
                            tmp2 = data_file_2d[s][a][frame_num]
                            all_in_one_dataset_2d_keypoints[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                        elif probabilistic:
                            all_in_one_dataset_2d_mean[i] = keypoints2D[s][a][frame_num]['mean']
                            all_in_one_dataset_2d_covariance[i] = keypoints2D[s][a][frame_num]['covariance']
                            giant_dic[s][a]['mean'].append(torch.tensor(keypoints2D[s][a][frame_num]['mean'],dtype =torch.float32))
                            giant_dic[s][a]['covariance'].append(torch.tensor(keypoints2D[s][a][frame_num]['covariance'],dtype =torch.float32))
                            giant_dic[s][a]['3d'].append(torch.tensor(pose_3d,dtype =torch.float32))
                        else:
                            all_in_one_dataset_2d_keypoints[i] = keypoints2D[s][a][frame_num] #Cause we saved with-1


                        i = i + 1 
                        
                    else :
                        for c in range(1+3*int(AllCameras)) :
                            tmp = pose_3d.copy()

                            if CameraView:
                                for j in range(len(tmp)): 
                                    tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][c]['translation']),1000)
                                    tmp[j] = qv_mult(np.array(camera_parameters[s][c]['orientation']),tmp[j])
                            if ground_truth:#depends if we want the ground thruth keypoints
                                tmp2 = data_file_2d[s][a+cam_ids[c]][frame_num]    
                                all_in_one_dataset_2d_keypoints[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                            elif probabilistic:
                                all_in_one_dataset_2d_mean[i] = keypoints2D[s][a][frame_num]['mean']
                                all_in_one_dataset_2d_covariance[i] = keypoints2D[s][a][frame_num]['covariance']
                            else:
                                all_in_one_dataset_2d_keypoints[i] = keypoints2D[s][a+cam_ids[c]][frame_num]
                            all_in_one_dataset_3d[i] = tmp

                            i = i + 1 

        local_vars = locals()
        total_size = sum(sys.getsizeof(var) for var in local_vars.values())
        print(f"Total memory used: {total_size} bytes")
        return all_in_one_dataset_2d_keypoints, all_in_one_dataset_3d, all_in_one_dataset_2d_mean, all_in_one_dataset_2d_covariance, num_frame_in_sequence, giant_dic
    

    @staticmethod
    def pick_sequence_from_frame(keypoints_2D, ground_truth3d, num_frame_in_sequence, mean, covariance, number_of_frames,num_keypoints):
        #num_frame_in_sequence has len N and is of structure [number of frame, total_frame_in_sequence]
        final_keypoints_2D = []
        final_ground_truth3D = []
        for i in range(ground_truth3d.shape[0]):
            num_frame, total_num_frames_in_sequence = num_frame_in_sequence[i]
            if (num_frame>=number_of_frames//2+1) and (num_frame<=total_num_frames_in_sequence-(number_of_frames//2)):#Ensures there are enough frames after/before:
                range_index = range(i-number_of_frames//2,i+number_of_frames//2+1)
                sequence_2D = np.zeros((1,number_of_frames,num_keypoints,2))

                var = 0
                for j,idx in enumerate(range_index):#idx is in the same index world as i
                    if num_frame_in_sequence[idx][0]!=num_frame-(number_of_frames//2)+j:#Verify each frame that it is the correct one (the 2 prediction did not shuffle the set)
                        var = 1
                        break
                    sequence_2D[0,j,:,:] = keypoints_2D[idx,:,:]
                if var == 0:
                    final_keypoints_2D.append(sequence_2D)
                    final_ground_truth3D.append(ground_truth3d[i,:,:])
        final_keypoints_2D = np.concatenate(final_keypoints_2D, axis=0) if final_keypoints_2D else np.empty((0, number_of_frames, num_keypoints, 2))
        final_ground_truth3D = np.array(final_ground_truth3D) if final_ground_truth3D else np.empty((0, num_keypoints, 3))
        assert final_ground_truth3D.shape[0] != 0, "Error in the data loading"

        return final_keypoints_2D, final_ground_truth3D

    @staticmethod
    def pick_sequence_from_frame_probabilistic(num_frame_in_sequence, number_of_frames=243,num_keypoint=17):
        list_index = []
        old_index =  -100
        for i, (index, max_index, subject, action) in enumerate(num_frame_in_sequence):
            if (int(index)>=number_of_frames//2+1) and (int(index)<=int(max_index)-(number_of_frames//2)):
                if int(index)>=old_index+number_of_frames/2:
                    list_index.append([subject,action, i])
                    old_index = int(index)
        return list_index


class Human36_dataset_action(Dataset):
    def __init__(self, subjectp=subjects , transform=None, target_transform=None, is_train = True, split_rate=None, model_2D = None, num_frame=243, num_keypoints=17):


        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        self.actions = ["Directions", "Purchases", "Smoking", "Phoning", "Discussion", "Eating", "Greeting", "Photo", "SittingDown","Walking", "Waiting","WalkDog","WalkTogether","Sitting", "Posing" ]
        self.split_rate = split_rate
        self.number_of_frames = num_frame
        self.num_keypoints = num_keypoints
        if self.num_keypoints == 17:
            self.KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        elif self.num_keypoints == 20:
            self.KeyPoints_from3d = [0,1,2,3,6,7,8,11,12,13,14,15,16,17,18,19,24,25,26,27]#Added 11, 16, 24
        else:
            raise ValueError(f"Unsupported num of frames '{self.num_keypoints}'. Please choose either '17' or '20'.")
        #File to take the 2D predicted keypoints from
        directory = '/mnt/vita/scratch/datasets/Human3.6/npy/'
        ground_truth = False
        if model_2D == 'ViTPose':
            logging.info('Using ViTPose Prediction')
            if is_train:
                keypoints_2d_path = directory + 'keypoints_2D_ViTPose_trainset.npy'
            else:
                keypoints_2d_path = directory + 'keypoints_2D_ViTPose_testset.npy'
    
        elif model_2D == 'Hourglass':
            if is_train:
                keypoints_2d_path = directory + 'keypoints_2D_Hourglass_trainset.npy'
            else:
                keypoints_2d_path = directory + 'keypoints_2D_Hourglass_testset.npy'
        elif model_2D == 'GroundThruth':
            ground_truth = True
            keypoints_2d_path = None
        else:
            raise ValueError(f"Unsupported model name '{model_2D}'. Please choose either 'ViTPose' or 'Hourglass' or 'GroundThruth.")
        
        dataset2d, dataset3d, num_frame_in_sequence, action_list, view_list, subject_list = self.read_data(subjects= subjectp, file_keypoints_2D = keypoints_2d_path, ground_truth = ground_truth, KeyPoints_from3d = self.KeyPoints_from3d)
        self.dataset2d, self.dataset3d, self.action_list, self.frame_numbers, self.view_list, self.subject_list = self.pick_sequence_from_frame(dataset2d, dataset3d, num_frame_in_sequence, num_frame, action_list, view_list, subject_list, KeyPoints_from3d = self.KeyPoints_from3d)
        if self.split_rate:
            self.dataset2d = self.dataset2d[::split_rate]
            self.dataset3d = self.dataset3d[::split_rate]
            self.action_list = self.action_list[::split_rate]
            self.frame_numbers = self.frame_numbers[::split_rate]


    def __len__(self):
        return len(self.dataset3d)

    def __getitem__(self, idx):
        keypoints_2d = self.dataset2d[idx].reshape(-1,self.number_of_frames,self.num_keypoints,1)
        ground_truth_3d = self.dataset3d[idx].reshape(-1,3)
        action = self.action_list[idx]
        idx_number = self.frame_numbers[idx]
        view_point = self.view_list[idx]
        subject = self.subject_list[idx]

        return ground_truth_3d, keypoints_2d, action, idx_number, view_point, subject

    @staticmethod
    def read_data(subjects, file_keypoints_2D, ground_truth, KeyPoints_from3d):
        cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        actions = ["Directions", "Purchases", "Smoking", "Phoning", "Discussion", "Eating", "Greeting", "Photo", "SittingDown","Walking", "Waiting","WalkDog","WalkTogether","Sitting", "Posing" ]
        subjects_to_compare = ["S1", "S5", "S6", "S7", "S8", "S9","S11"]
        if Mono_3d_file:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m_mono.npz', allow_pickle=True)
        else:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m.npz', allow_pickle=True)
        data_file_2d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_2d_h36m.npz', allow_pickle=True) #2d is considered as the groundtruth

        data_file_3d = data_file_3d['positions_3d'+"_mono"*Mono_3d_file].item()
        data_file_2d = data_file_2d['positions_2d'].item()

        data_directory = '/mnt/vita/scratch/datasets/Human3.6'
        n_frame = 0 
        for s in subjects:
            for a in data_file_3d[s].keys():#For every action (even if many action for same action, a will be the different videos)
                n_frame += len(data_file_3d[s][a])  
            
        n_frame = n_frame + 3*int(not Mono_3d_file)*n_frame #would be 4*n_frame if Mono_3d_file is False

        if not ground_truth:
            keypoints2D = np.load(file_keypoints_2D, allow_pickle=True)
            keypoints2D = keypoints2D[()]


        #Chaque frame a sa GT 3d et sa gt 2D
        all_in_one_dataset_3d = np.zeros((n_frame if AllCameras else n_frame, len(KeyPoints_from3d) ,3),  dtype=np.float32)
        all_in_one_dataset_2d_keypoints = np.zeros((n_frame if AllCameras else n_frame, len(KeyPoints_from3d) ,2),  dtype=np.float32)
        action_list = np.zeros((n_frame if AllCameras else n_frame,1), dtype = np.int)
        view_list = np.zeros((n_frame if AllCameras else n_frame,1), dtype = np.int)
        subject_list = np.zeros((n_frame if AllCameras else n_frame,1), dtype = np.int)
        num_frame_in_sequence = []
        i = 0
        for s in subjects:
            for a in data_file_3d[s].keys():
                for frame_num in range(len(data_file_3d[s][a])):

                    pose_3d = data_file_3d[s][a][frame_num]  
                    pose_3d = pose_3d[KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                    
                    num_frame_in_sequence.append([frame_num+1,len(data_file_3d[s][a])])
                    if Mono_3d_file :
                        if ground_truth:
                            tmp2 = data_file_2d[s][a][frame_num]
                            all_in_one_dataset_2d_keypoints[i] = tmp2[KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                        else:
                            all_in_one_dataset_2d_keypoints[i] = keypoints2D[s][a][frame_num] #Cause we saved with-1
                        action_list[i] = next((idx for idx, action in enumerate(actions) if action in a), -1)
                        view_list[i] = next((idx for idx, view_point in enumerate(cam_ids) if view_point in a), -1)
                        subject_list[i] = next((idx for idx, subject in enumerate(subjects_to_compare) if subject in s), -1)

                        i = i + 1

                        
                    else :
                        for c in range(1+3*int(AllCameras)) :
                            tmp = pose_3d.copy()

                            if CameraView:
                                for j in range(len(tmp)): 
                                    tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][c]['translation']),1000)
                                    tmp[j] = qv_mult(np.array(camera_parameters[s][c]['orientation']),tmp[j])
                                        
                            if ground_truth:#depends if we want the ground thruth keypoints
                                tmp2 = data_file_2d[s][a+cam_ids[c]][frame_num]    
                                all_in_one_dataset_2d_keypoints[i] = tmp2[KeyPoints_from3d,:] #only keeping the 16 or 17 keypoints we want
                            else:
                                all_in_one_dataset_2d_keypoints[i] = keypoints2D[s][a+cam_ids[c]][frame_num]
                            action_list[i,0] = next((idx for idx, action in enumerate(actions) if action in a), -1)
                            view_list[i,0] = next((idx for idx, view_point in enumerate(cam_ids) if view_point in a), -1)
                            subject_list[i] = next((idx for idx, subject in enumerate(subjects_to_compare) if subject in s), -1)

                            i = i + 1 

        
        return all_in_one_dataset_2d_keypoints, all_in_one_dataset_3d, num_frame_in_sequence, action_list, view_list, subject_list
    

    @staticmethod
    def pick_sequence_from_frame(keypoints_2D, ground_truth3d, num_frame_in_sequence, number_of_frames, action_list, view_list, subject_list, KeyPoints_from3d):
        #num_frame_in_sequence has len N and is of structure [number of frame, total_frame_in_sequence]
        final_keypoints_2D = []
        final_ground_truth3D = []
        final_action_list = []
        final_view_list = []
        final_subject_list = []
        final_frame_numbers = []

        for i in range(keypoints_2D.shape[0]):
            #num_frame begins at 1
            num_frame, total_num_frames_in_sequence = num_frame_in_sequence[i]
            if (num_frame>=number_of_frames//2+1) and (num_frame<=total_num_frames_in_sequence-(number_of_frames//2)):#Ensures there are enough frames after/before:
                if (number_of_frames//2):
                    range_index = range(i-number_of_frames//2,i+number_of_frames//2)
                else:
                    range_index = range(i-number_of_frames//2,i+number_of_frames//2+1)
                sequence_2D = np.zeros((1,number_of_frames,len(KeyPoints_from3d),2,1))
                frame_numbers = np.zeros((1,number_of_frames))
                var = 0
                for j,idx in enumerate(range_index):#idx is in the same index world as i
                    if num_frame_in_sequence[idx][0]!=num_frame-(number_of_frames//2)+j:#Verify each frame that it is the correct one (the 2 prediction did not shuffle the set)
                        var = 1
                        break
                    sequence_2D[0,j,:,:,0] = keypoints_2D[idx,:,:]
                    frame_numbers[0,j] = j
                #Start from [B, F, J, D, M]
                #To B, D, F, J, M.
                sequence_2D = np.transpose(sequence_2D, (0, 3, 1, 2, 4))
                if var == 0:
                    final_keypoints_2D.append(sequence_2D)
                    final_ground_truth3D.append(ground_truth3d[i,:,:])
                    final_action_list.append(action_list[i,0])
                    final_view_list.append(view_list[i,0])
                    final_subject_list.append(subject_list[i,0])
                    final_frame_numbers.append(frame_numbers)

        final_keypoints_2D = np.concatenate(final_keypoints_2D, axis=0) if final_keypoints_2D else np.empty((0, number_of_frames, len(KeyPoints_from3d), 2,1))
        final_ground_truth3D = np.array(final_ground_truth3D) if final_ground_truth3D else np.empty((0, len(KeyPoints_from3d), 3))
        final_action_list = np.array(final_action_list) if final_action_list else np.empty((0,1))
        final_view_list = np.array(final_view_list) if final_view_list else np.empty((0,1))
        final_subject_list = np.array(final_subject_list) if final_subject_list else np.empty((0,1))
        final_frame_numbers = np.array(final_frame_numbers) if final_frame_numbers else np.empty((0,1))
        assert final_keypoints_2D.shape[0] != 0, "Error in the data loading"

        return final_keypoints_2D, final_ground_truth3D, final_action_list, final_frame_numbers, final_view_list, final_subject_list
    

class H36_dataset_probabilistic_pose(Dataset):
    def __init__(self, subjectp=subjects , conf =None,transform=None, target_transform=None, is_train = True, split_rate=None):
        
        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        self.occlusion = True
        self.hm_shape = (64, 64)
        self.hm_peak = 30
        self.split_rate = split_rate
        self.augmentation_flag = False
        self.flip_prob = 0.5
        self.conf = conf

        self.dataset2d, self.dataset3d, self.video_and_frame_paths = self.read_data(subjects= subjectp)
        
        if self.split_rate:
            self.dataset2d = self.dataset2d[::split_rate]
            self.dataset3d = self.dataset3d[::split_rate]
            self.video_and_frame_paths = self.video_and_frame_paths[::split_rate]
        
        # Deciding augmentation techniques
        self.shift_scale_rotate = self.augmentation(
            [albu.ShiftScaleRotate(
                p=1, shift_limit=0.2, scale_limit=0.25, rotate_limit=45, interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT, value=0)
            ])

        self.flip_prob = 0.5
        self.horizontal_flip = self.augmentation([albu.HorizontalFlip(p=1)])

    def __len__(self):
        return len(self.dataset3d)

    def __getitem__(self, idx):

        if load_imgs:
            frame = cv2.imread(self.video_and_frame_paths[idx][0])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_2d = self.dataset2d[idx].reshape(-1 ,2)
        

        #Bounding Box: 
        bc = [min(keypoints_2d[:,0]), max(keypoints_2d[:,1])] 
        br = [max(keypoints_2d[:,0]), max(keypoints_2d[:,1])] 
        pc = [min(keypoints_2d[:,0]), min(keypoints_2d[:,1])] 
        pr = [max(keypoints_2d[:,0]), min(keypoints_2d[:,1])] 
        l_m = int (max(br[0] - bc[0], br[1] - pr[1])  * 1. *1000 )
        frame = frame[ min(0,int(keypoints_2d[0,1]*1000-(l_m/2))) : max(1000,int(keypoints_2d[0,1]*1000+(l_m/2))) ,  min(0,int(keypoints_2d[0,0]*1000- (l_m/2))) : max(1000,int(keypoints_2d[0,0]*1000+ (l_m/2))) , :]
        # breakpoint()
        #resising the image for Resnet
        frame = cv2.resize(frame, (256, 256))
        # frame = cv2.resize(frame, (1024, 1024))
        frame = frame/256.0

        #Pairs to switch for left right: 
        #sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
        # [1,4], [2,5], [3,6], [11,14], [12,15],[13,16]
        # Augmentation

        ones_column = np.ones((keypoints_2d.shape[0], 1))
        gt = np.expand_dims(np.hstack((keypoints_2d*frame.shape[0], ones_column)),axis=0)
        if self.augmentation_flag:

            # Horizontal flip can't be done using albu's probability
            if torch.rand(1) < self.flip_prob:
                # Augment image and keypoints
                augmented = self.horizontal_flip(image=frame, keypoints=gt.reshape(-1, 3)[:, :2])
                frame = augmented['image']
                gt[:, :, :2] = np.stack(augmented['keypoints'], axis=0).reshape(-1, self.conf.experiment_settings['num_hm'], 2)

                # Flip ground truth to match horizontal flip
                gt[:, [1,4], :] = gt[:, [4,1], :]
                gt[:, [2,5], :] = gt[:, [5,2], :]
                gt[:, [3,6], :] = gt[:, [6,3], :]
                gt[:, [11,14], :] = gt[:, [14,11], :]
                gt[:, [12,15], :] = gt[:, [15,12], :]
                gt[:, [13,16], :] = gt[:, [16,13], :]

            # Ensure shift scale rotate augmentation retains all joints
            tries = 5
            augment_ok = False
            image_, gt_ = None, None

            while tries > 0:
                tries -= 1
                augmented = self.shift_scale_rotate(image=frame, keypoints=gt.reshape(-1, 3)[:, :2])
                image_ = augmented['image']
                gt_ = np.stack(augmented['keypoints'], axis=0).reshape(
                    -1, self.conf.experiment_settings['num_hm'], 2)

                # I don't remember why I set the threshold to -+5 but I don't want to break it
                if (np.all(gt_[0]) > -5) and (np.all(gt_[0]) < 261):  # 0 index single person
                    augment_ok = True
                    break

            if augment_ok:
                frame = image_
                gt[:, :, :2] = gt_


        heatmaps, _ = heatmap_generator(joints=np.copy(gt), occlusion=self.occlusion, hm_shape=self.hm_shape, img_shape=frame.shape)
        heatmaps = self.hm_peak * heatmaps

        keypoints_2d = gt
        keypoints_2d[:,:,:2]/=frame.shape[0]

        subject, action, frame_num = self.find_parameters_from_path(self.video_and_frame_paths[idx][0])

        return keypoints_2d, frame,heatmaps, subject, action, frame_num

    def find_parameters_from_path(self, path):
        pattern = r'videos/([^/]+)/outputVideos/([^/]+)\.mp4/(\d{4})\.jpg'

        match = re.search(pattern, path)

        if Mono_3d_file :
            if re:
                s = match.group(1)  # subject
                a = match.group(2)  # Discussion.23657
                frame_num = int(match.group(3))  # frame_number

        return s,a,frame_num

    def xyz_to_uvw(self, kp): #here
        return np.array([-kp[1], -kp[2], kp[0]])

    def set_augmentation(self, augment: bool) -> None:
        """
        Set augmentation flag
        """
        if augment: self.augmentation_flag = True
        else: self.augmentation_flag = False

    def _keypoint_to_heatmap_3D(self, keypoint, sigma=0.5): #sigma=1.75
        """
        Read the function name duh
        Args:
            keypoints (np.float32): Keypoints in 3D ranges from -1 to 1
            sigma (float, optional):Size of the Gaussian. Defaults to 1.75.
        """
        
        assert np.min(keypoint) >= -1
        assert np.max(keypoint) <= 1

        # Create an empty volumetric heatmap
        im = np.zeros((64, 64, 64), dtype=np.float32)

        # Scale keypoints from -1 to 1
        keypoint = 31.5 * (1 + keypoint)
        keypoint_int = np.rint(keypoint).astype(int)

        # Size of 3D Gaussian window.
        size = int(math.ceil(6 * sigma))
        # Ensuring that size remains an odd number
        if not size % 2:
            size += 1

        # Generate gaussian, with window=size and variance=sigma
        u = np.arange(keypoint_int[0] - (size // 2), keypoint_int[0] + (size // 2) + 1)
        v = np.arange(keypoint_int[1] - (size // 2), keypoint_int[1] + (size // 2) + 1)
        w = np.arange(keypoint_int[2] - (size // 2), keypoint_int[2] + (size // 2) + 1)
        
        uu, vv, ww = np.meshgrid(u, v, w, indexing='ij', sparse=True)
        z = np.exp(-((uu - keypoint[0]) ** 2 + (vv - keypoint[1]) ** 2 + (ww - keypoint[2]) ** 2) / (2 * (sigma ** 2)))

        # Identify indices in im that will define the crop area
        top_u = max(0, keypoint_int[0] - (size//2))
        top_v = max(0, keypoint_int[1] - (size//2))
        top_w = max(0, keypoint_int[2] - (size//2))

        bottom_u = min(64, keypoint_int[0] + (size//2) + 1)
        bottom_v = min(64, keypoint_int[1] + (size//2) + 1)
        bottom_w = min(64, keypoint_int[2] + (size//2) + 1)

        im[top_u:bottom_u, top_v:bottom_v, top_w:bottom_w] = \
            z[top_u - (keypoint_int[0] - (size//2)): top_u - (keypoint_int[0] - (size//2)) + (bottom_u - top_u),
            top_v - (keypoint_int[1] - (size//2)): top_v - (keypoint_int[1] - (size//2)) + (bottom_v - top_v),
            top_w - (keypoint_int[2] - (size//2)): top_w - (keypoint_int[2] - (size//2)) + (bottom_w - top_w)]

        return im

    def keypoints_to_heatmap_3D(self, keypoints):
        hm = []
        for i in range(keypoints.shape[0]):
            kp = self.xyz_to_uvw(keypoints[i])
            # kp = keypoints[i]
            hm.append(self._keypoint_to_heatmap_3D(kp))
        return np.stack(hm, axis=0)

    def augmentation(self, transform: list) -> albu.Compose:
        """
        Albumentation objects for augmentation in getitem
        """
        return albu.Compose(
            transform, p=1, keypoint_params=albu.KeypointParams(format='yx', remove_invisible=False))

    @staticmethod
    def read_data(subjects = subjects):
        
        cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]

        if Mono_3d_file:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m_mono.npz', allow_pickle=True)
        else:
            data_file_3d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_3d_h36m.npz', allow_pickle=True)
        data_file_2d = np.load('/mnt/vita/scratch/datasets/Human3.6/npz/data_2d_h36m.npz', allow_pickle=True) #2d is considered as the groundtruth

        data_file_3d = data_file_3d['positions_3d'+"_mono"*Mono_3d_file].item()
        data_file_2d = data_file_2d['positions_2d'].item()

        data_directory = '/mnt/vita/scratch/datasets/Human3.6'
        n_frame = 0 
        for s in subjects:
            for a in data_file_3d[s].keys():#For every action (even if many action for same action, a will be the different videos)
                n_frame += len(data_file_3d[s][a])  
            
        n_frame = n_frame + 3*int(not Mono_3d_file)*n_frame #would be 4*n_frame if Mono_3d_file is False

        #Chaque frame a sa GT 3d et sa gt 2D
        all_in_one_dataset_3d = np.zeros((n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
        all_in_one_dataset_2d = np.zeros((n_frame if AllCameras else n_frame, 17 ,2),  dtype=np.float32)
        video_and_frame_paths = []
        i = 0
        for s in subjects:
            for a in data_file_3d[s].keys():
                for frame_num in range(len(data_file_3d[s][a])):

                    pose_3d = data_file_3d[s][a][frame_num]  
                    pose_3d = pose_3d[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                    
                    if Mono_3d_file :
                        all_in_one_dataset_3d[i] = pose_3d
                        tmp2 = data_file_2d[s][a][frame_num]
                        all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                        
                        if load_imgs:
                            video_and_frame_paths.append( [data_directory+"/videos/"+s+"/outputVideos/"+a+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])

                        i = i + 1 
                        
                    else :
                        for c in range(1+3*int(AllCameras)) :
                            tmp = pose_3d.copy()

                            if CameraView:
                                for j in range(len(tmp)): 
                                    tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][c]['translation']),1000)
                                    tmp[j] = qv_mult(np.array(camera_parameters[s][c]['orientation']),tmp[j])
                                        
                            all_in_one_dataset_3d[i] = tmp

                            tmp2 = data_file_2d[s][a+cam_ids[c]][frame_num]    
                            all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                            if load_imgs:
                                video_and_frame_paths.append( [data_directory+"/videos/"+s+"/outputVideos/"+a+cam_ids[c]+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])

                            i = i + 1 

        
        return all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths
