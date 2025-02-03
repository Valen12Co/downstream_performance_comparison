

import math
import numpy as np
from torch.utils.data import Dataset
from dataloader.utils import camera_parameters, qv_mult, flip_pose, h36m_cameras_intrinsic_params
import cv2
# import albumentations as A

systm = "vita17"  #izar,vita17,laptop
act = "" #"Walking"
load_imgs = True
from_videos = False

zero_centre = True
standardize_3d = False
standardize_2d = False
Normalize = False

sample = False
Samples = np.random.randint(0,74872 if act=="Walk" else 389938, 200) #389938+135836=525774
AllCameras = False
CameraView = True 
Mono_3d_file= True
if AllCameras:
    CameraView = True
MaxNormConctraint = False 


num_cameras = 1
input_dimension = num_cameras*2
output_dimension = 3

num_of_joints = 17 #data = np.insert(data, 0 , values= [0,0,0], axis=0 )

dataset_direcotories = {"izar":"/work/vita/datasets/h3.6", #/home/rhossein/venvs/codes/VideoPose3D/data/
                "vita17":"/data/rh-data/h3.6", 
                "laptop": "/Users/rh/test_dir/h3.6/VideoPose3D/data"}  #vita17 used to be: /home/rh/h3.6/dataset/npz/",

data_directory =  dataset_direcotories[systm]
path_positions_2d_VD3d = data_directory + "/npz/data_2d_h36m.npz" #"data_2d_h36m_gt.npz" 
path_positions_3d_VD3d =data_directory + "/npz/data_3d_h36m.npz"
path_positions_3d_VD3d_mono =data_directory + "/npz/data_3d_h36m_mono.npz"


subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
# KeyPoints_from3d = list(range(32))
KeyPoints_from3d_to_delete = [4,5,9,10,11,16,20,21,22,23,24,28,29,30,31]


class H36_dataset(Dataset):
    def __init__(self, subjectp=subjects , action=act, transform=None, target_transform=None, is_train = True, split_rate=None):
        
        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        
        self.split_rate = split_rate

        self.dataset2d, self.dataset3d, self.video_and_frame_paths = self.read_data(subjects= subjectp,action=action,is_train = is_train)
        
        if self.split_rate:
            self.dataset2d = self.dataset2d[::split_rate]
            self.dataset3d = self.dataset3d[::split_rate]
            self.video_and_frame_paths = self.video_and_frame_paths[::split_rate]
        
        self.dataset2d = self.process_data(self.dataset2d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_2d, z_c = False)
        self.dataset3d = self.process_data(self.dataset3d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_3d, z_c = True)

        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
                
      
    def __len__(self):
        return len(self.dataset3d) #number of all the frames 

    def __getitem__(self, idx):

        if load_imgs:
            if from_videos:     
                cap = cv2.VideoCapture(self.video_and_frame_paths[idx][0])
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_and_frame_paths[idx][1]) 
                res, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else :
                frame = cv2.imread(self.video_and_frame_paths[idx][0])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                   
        keypoints_2d = self.dataset2d[idx].reshape(-1 ,2)
        keypoints_3d = self.dataset3d[idx]
        
        if standardize_3d and (not Normalize): 
            heatmap_3d = np.array([])
        else:    
            heatmap_3d = self.keypoints_to_heatmap_3D(keypoints_3d)
   
        #augmentation
        # if self.is_train:
        #     if np.random.sample() < 1 and zero_centre: #translate
        #         pass
        #         # shift_scale = np.random.sample()-0.5
        #         # self.dataset2d[idx] = self.dataset2d[idx] + shift_scale*((0.02))
        #         # # frame[int(shift_scale*1000): ,int(shift_scale*1000): ] = cv2.
                
        #         # breakpoint()
                
        #     if np.random.sample() < 0.1 : #rotate
        #         pass
                
        #     if np.random.sample() < 0.1 : #scale
        #         pass 
        
        #     if np.random.sample() < 0.5  :  #flip
        #         frame = cv2.flip(frame,1)
        #         self.dataset2d[idx] = flip_pose(self.dataset2d[idx])
        #         self.dataset3d[idx] = flip_pose(self.dataset3d[idx])
                
        
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
        
        
        for k in range(4):
            if self.cam_ids[k] in self.video_and_frame_paths[idx][0]:
                r_cam_id = k
                
        # cam_inf = np.array([ [r_cam_id,r_cam_id ], h36m_cameras_intrinsic_params[r_cam_id]['center'],h36m_cameras_intrinsic_params[r_cam_id]['focal_length'] ])
            
        return keypoints_2d, keypoints_3d, frame, np.array(h36m_cameras_intrinsic_params[r_cam_id]['focal_length']), heatmap_3d

    
    def xyz_to_uvw(self, kp): #here
        return np.array([-kp[1], -kp[2], kp[0]])
        # return np.array([kp[2], kp[1], kp[0]])
    
    
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
        
    
    def process_data(self, dataset , sample=sample, is_train = True, standardize = False, z_c = zero_centre) :

        n_frames, n_joints, dim = dataset.shape

        if z_c:
            for i in range(n_frames):
                dataset[i,1:] = dataset[i,1:] - dataset[i,0]


        if is_train :
            data_sum = np.sum(dataset, axis=0)
            data_mean = np.divide(data_sum, n_frames)

            diff_sq2_sum =np.zeros((n_joints,dim))
            for i in range(n_frames):
                diff_sq2_sum += np.power( dataset[i]-data_mean ,2)
            data_std = np.divide(diff_sq2_sum, n_frames)
            data_std = np.sqrt(data_std)

            if dim == 2:
                with open("./logs/run_time_utils/mean_train_2d.npy","wb") as f:
                    np.save(f, data_mean)
                with open("./logs/run_time_utils/std_train_2d.npy","wb") as f:
                    np.save(f, data_std)  

            elif dim == 3:
                with open("./logs/run_time_utils/mean_train_3d.npy","wb") as f:
                    np.save(f, data_mean)  
                with open("./logs/run_time_utils/std_train_3d.npy","wb") as f:
                    np.save(f, data_std)  
                    
                with open("./logs/run_time_utils/max_train_3d.npy","wb") as f:
                    temp_max =  np.max(dataset, axis=0)
                    temp_max = np.ones((num_of_joints,3))
                    np.save(f, temp_max)        
                with open("./logs/run_time_utils/min_train_3d.npy","wb") as f:
                    temp_min = np.min(dataset, axis=0)
                    temp_min = np.ones((num_of_joints,3)) * -1
                    np.save(f, temp_min ) 

        if dim == 2:
            with open("./logs/run_time_utils/mean_train_2d.npy","rb") as f:
                mean_train_2d = np.load(f)
            with open("./logs/run_time_utils/std_train_2d.npy","rb") as f:
                std_train_2d = np.load(f)  
        elif dim == 3:
            with open("./logs/run_time_utils/mean_train_3d.npy","rb") as f:
                mean_train_3d =np.load(f)  
            with open("./logs/run_time_utils/std_train_3d.npy","rb") as f:
                std_train_3d = np.load(f)  
                
            with open("./logs/run_time_utils/max_train_3d.npy","rb") as f:
                max_train_3d =np.load(f)  
            with open("./logs/run_time_utils/min_train_3d.npy","rb") as f:
                min_train_3d = np.load(f)  

        if standardize :
            if dim == 2 :
                for i in range(n_frames):
                    if Normalize:
                        # max_dataset, min_dataset = np.max(dataset, axis=0), np.min(dataset, axis=0)
                        # print(max_dataset, min_dataset)
                        # dataset[i] = np.divide(2*dataset[i], (max_dataset-min_dataset))
                        # dataset[i] = dataset[i] - np.divide(min_dataset, (max_dataset-min_dataset))
                        dataset[i] = 2*dataset[i] -1 

                    else:
                        dataset[i] = np.divide(dataset[i] - mean_train_2d, std_train_2d)
            elif dim == 3:
                for i in range(n_frames):
                    if Normalize:
                        # max_dataset, min_dataset = np.max(dataset, axis=0), np.min(dataset, axis=0)
                        # dataset[i] = np.divide(dataset[i]- min_train_3d, (max_train_3d-min_train_3d)) # map to 0 and 1

                        dataset[i] = np.divide(dataset[i]- min_train_3d, (max_train_3d-min_train_3d)) # map to 0 and 1
                        dataset[i] -= 0.5  # map to 0 and 1
                        # pass
                    else:
                        dataset[i] = np.divide(dataset[i] - mean_train_3d, std_train_3d)


        if num_of_joints == 16: #Should through an error if num of joints is 16 but zero centre is false    
            dataset = dataset[:, 1:, :].copy()
        elif z_c :
            dataset [:,:1,:] *= 0


        if dim == 2 and sample :
            dataset = dataset.reshape((int(n_frames/4),4, num_of_joints,2))

        dataset = dataset[Samples] if sample else dataset

        if dim == 2 and sample :
            dataset = dataset.reshape(-1, num_of_joints,2)  

        return dataset
    
    @staticmethod
    def read_data(subjects = subjects, action = "", is_train=True):
        
        cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]

        if Mono_3d_file:
            data_file_3d = np.load(path_positions_3d_VD3d_mono, allow_pickle=True)
        else:
            data_file_3d = np.load(path_positions_3d_VD3d, allow_pickle=True)
        data_file_2d = np.load(path_positions_2d_VD3d, allow_pickle=True)

        data_file_3d = data_file_3d['positions_3d'+"_mono"*Mono_3d_file].item()
        data_file_2d = data_file_2d['positions_2d'].item()

        n_frame = 0 
        for s in subjects:
            for a in data_file_3d[s].keys():
                if (action in a ) :
                    n_frame += len(data_file_3d[s][a])  
            
        n_frame = n_frame + 3*int(not Mono_3d_file)*n_frame #would be 4*n_frame if Mono_3d_file is False

        all_in_one_dataset_3d = np.zeros((n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
        all_in_one_dataset_2d = np.zeros((n_frame if AllCameras else n_frame, 17 ,2),  dtype=np.float32)
        video_and_frame_paths = []
        i = 0
        for s in subjects:
            for a in data_file_3d[s].keys():
                if action in a :
                    print(s,a,len(data_file_3d[s][a]))
                    for frame_num in range(len(data_file_3d[s][a])):

                        pose_3d = data_file_3d[s][a][frame_num]  
                        pose_3d = pose_3d[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                        
                        if Mono_3d_file :
                            all_in_one_dataset_3d[i] = pose_3d
                            tmp2 = data_file_2d[s][a][frame_num]
                            all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want
                            
                            if load_imgs:
                                if from_videos:
                                    video_and_frame_paths.append( [data_directory+"/videos/"+s+"/Videos/"+a+".mp4",frame_num])
                                else:
                                    if systm == "laptop":
                                        video_and_frame_paths.append( ["/Users/rh/test_dir/h3.6/dataset/S1_frames/"+a+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])
                                    else:
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
                                    if from_videos:
                                        video_and_frame_paths.append( [data_directory+"/videos/"+s+"/Videos/"+a+cam_ids[c]+".mp4",frame_num])
                                    else:
                                        if systm == "laptop":
                                            video_and_frame_paths.append( ["/Users/rh/test_dir/h3.6/dataset/S1_frames/"+a+cam_ids[c]+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])
                                        else:
                                            video_and_frame_paths.append( [data_directory+"/videos/"+s+"/outputVideos/"+a+cam_ids[c]+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])

                                i = i + 1 

        
        return all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths


if __name__ == "__main__" :
    
    training_set = H36_dataset(subjectp=subjects[0:5], is_train = True, action="Posing", split_rate=81) 
    keypoints_2d, keypoints_3d, frame, intnsc, hm3d = training_set.__getitem__(0)
    
    frame = cv2.resize(frame, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)
    import matplotlib.pyplot as plt
    plt.figure()    
    plt.imshow(frame)
    plt.savefig("filename.png")
    plt.close()
    
    breakpoint()
    
    all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths = H36_dataset.read_data(subjects=subjects[0:5])
    for i in range(all_in_one_dataset_3d.shape[0]):
        all_in_one_dataset_3d[i,1:] = all_in_one_dataset_3d[i,1:] - all_in_one_dataset_3d[i,0]
        all_in_one_dataset_3d[i,0] *= 0 
    print(all_in_one_dataset_3d.max(axis=0))
    print(all_in_one_dataset_3d.min(axis=0))
    breakpoint()
    
    #___
    
    # import matplotlib.pyplot as plt
 
    # all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths = H36_dataset.read_data(subjects=subjects[0:5])
    
    # for i in range(all_in_one_dataset_3d.shape[0]):
    #     all_in_one_dataset_3d[i,1:] = all_in_one_dataset_3d[i,1:] - all_in_one_dataset_3d[i,0]
    #     all_in_one_dataset_3d[i,0] *= 0 
    
    # for dim in range(3):
    #     fig, ax = plt.subplots()
    #     for joint in range(1,17):
    #         ax.hist(all_in_one_dataset_3d[:, joint, dim], bins=50, alpha=0.5, label=f"Joint {joint+1}")
    #     ax.set_title(f"Histogram of All Joints in {['x', 'y', 'z'][dim]} Dimension")
    #     ax.set_xlabel("Value")
    #     ax.set_ylabel("Frequency")
    #     ax.legend()
    #     fig.savefig(f"all_joints_{['x', 'y', 'z'][dim]}.png")
    #     plt.show()
    
    # all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths = H36_dataset.read_data(subjects=subjects[5:7])
    # for i in range(all_in_one_dataset_3d.shape[0]):
    #     all_in_one_dataset_3d[i,1:] = all_in_one_dataset_3d[i,1:] - all_in_one_dataset_3d[i,0]
    #     all_in_one_dataset_3d[i,0] *= 0 
        
    # for dim in range(3):
    #     fig2, ax = plt.subplots()
    #     for joint in range(1,17):
    #         ax.hist(all_in_one_dataset_3d[:, joint, dim], bins=50, alpha=0.5, label=f"Joint {joint+1}")
    #     ax.set_title(f"Histogram of All Joints in {['x', 'y', 'z'][dim]} Dimension")
    #     ax.set_xlabel("Value")
    #     ax.set_ylabel("Frequency")
    #     ax.legend()
    #     fig2.savefig(f"test_all_joints_{['x', 'y', 'z'][dim]}.png")
    #     plt.show()
        
   