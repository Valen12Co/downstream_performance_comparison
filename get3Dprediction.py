import os
import copy
import logging
import numpy as np
from tqdm import tqdm
from typing import Union

from matplotlib import pyplot as plt#for visualize
import cv2
import random

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ParseConfig
from utils.pose import count_parameters, soft_argmax
from utils.eval import mpjpe_p1, mpjpe_p2, compute_ap
from utils.visualisation import compute_length, visualize_2d, compute_length_2D
from utils.utils_graphmlp import parse_args, AccumLoss, flip_data


from dataloader.dataloader import Human36_dataset_3D

from models.GraphMLP.graphmlp import Model as GraphMLP


training_methods = ['TIC']#do one method preferrably for the 2D keypoints saving, constant is here in case someone wants to add other methods
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Evaluate_3D_prediction(object):
    def __init__(self, sampler: Human36_dataset_3D, model, conf: ParseConfig,training_pkg: dict,args) -> None:
        """
        Train and compare various covariance methods.
        :param sampler: Instance of H36_dataset
        :param models: Contains Hourglass or ViTPose model
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        :param isTrain: If it is the train or test set
        """
        self.sampler = sampler
        self.conf = conf
        self.args = args
        self.training_pkg = training_pkg
        self.batch_size = conf.experiment_settings['batch_size']
        self.torch_dataloader = DataLoader(self.sampler, batch_size=self.batch_size, shuffle=False, num_workers=1, drop_last=False)

        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        for method in training_methods:
            self.training_pkg[method]['networks'] = (copy.deepcopy(model).cuda())
            if self.conf.model_2D_prediction == 'ViTPose':
                model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/ViTPose1/model_2_19.pth'
                self.training_pkg[method]['networks'].load_state_dict(torch.load(model_path))
            elif self.conf.model_2D_prediction == 'Hourglass':
                model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/Hourglass1/model_1_13.pth'
                self.training_pkg[method]['networks'].load_state_dict(torch.load(model_path))
            elif self.conf.model_2D_prediction == 'GroundThruth':
                model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/pretrained/243/model_4379.pth'
                #model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/GroundTruth/model_9_7.pth'
                #model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/Noise/model_14_8.pth'
                #model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/Noise4/model_5_9.pth'

                pre_dict = torch.load(model_path)
                model_dict = self.training_pkg[method]['networks'].state_dict()
                state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                self.training_pkg[method]['networks'].load_state_dict(model_dict)
            elif self.conf.model_2D_prediction == 'Probabilistic':
                #model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/ProbPose2/model_26_16.pth'
                model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/ProbPoseFinal/model_26_16.pth'
                self.training_pkg[method]['networks'].load_state_dict(torch.load(model_path))
            else:
                raise ValueError(f"Unsupported model name '{self.conf.model_2D_prediction}'. Please choose either 'ViTPose' or 'Hourglass'.")

    def run_model(self) -> dict:
        print("Model running to obtain 2D keypoints: Batch Size - {} total number of batches = {}".format(self.batch_size, len(self.torch_dataloader)),  flush = True)

        method = training_methods[0]
        net = self.training_pkg[method]['networks']
        net.eval()

        error_1 = []
        error_2 = []

        with torch.no_grad():
            lengths_gt = np.empty((0,16))
            lengths_pred = np.empty((0,16))
            lengths_2D = np.empty((0,16))
            for i, (gt_3d, keypoints_2D) in tqdm(enumerate(self.torch_dataloader), ascii=True):
                gt_3d = gt_3d.cuda()
                keypoints_2D = keypoints_2D.cuda()
                keypoints_2D = keypoints_2D.float()
                flipped_keypoints_2D = flip_data(keypoints_2D)
                predicted_3D = net(keypoints_2D)
                flipped_predicted_3D = net(flipped_keypoints_2D)

                output_3D = (predicted_3D + flip_data(flipped_predicted_3D))/2

                out_target = gt_3d.clone()
                out_target[:, 1:, :] = out_target[:, 1:, :] - out_target[:, self.args.root_joint, :].unsqueeze(1)
                out_target[:, self.args.root_joint, :] = 0
                output_3D[:,:, self.args.root_joint, :] = 0
                #out_target = out_target[:, self.args.pad].unsqueeze(1)
                out_target = out_target.unsqueeze(1)

                error_1.append(mpjpe_p1(output_3D,out_target).detach().cpu().numpy())
                error_2.append(mpjpe_p2(output_3D,out_target))

                if i ==0:
                    print("FOR IMAGE THE ERROR P1 is",mpjpe_p1(output_3D[0,:],out_target[0,:]).detach().cpu().numpy(),"P2 is", mpjpe_p2(output_3D,out_target))

                """
                length_2D = compute_length_2D(keypoints_2D)
                length_gt = compute_length(out_target)
                length_pred = compute_length(output_3D)
                lengths_2D = np.concatenate((lengths_2D,length_2D),axis=0)
                lengths_gt = np.concatenate((lengths_gt,length_gt),axis=0)
                lengths_pred = np.concatenate((lengths_pred,length_pred),axis=0)
                """

            error_1 = np.mean(np.concatenate(error_1, axis=0),axis=0)
            error_2 = np.mean(np.concatenate(error_2, axis=0),axis=0)
            print("\nIn evaluation: MJPE P1", error_1, "MJPE P2", error_2)

            """
            lengths_2D = np.mean(lengths_2D,axis=0)
            lengths_gt = np.mean(lengths_gt,axis = 0) 
            lengths_pred = np.mean(lengths_pred,axis=0)
            print("\nLength joints",lengths_gt,lengths_pred)
            print("\nLength 2D", lengths_2D)
            """
        return self.training_pkg



def init_3D_prediction_models(conf: ParseConfig, args) -> tuple:
    """
    Initializes and returns Hourglass and AuxNet models
    """

    print('Initializing Auxiliary Network')
    

    if conf.model_3D_prediction == 'GraphMLP':
        print('Initializing GraphMLP Network')
        pose_net = GraphMLP(args).cuda()
        print('Number of parameters (GraphMLP) for{}: {}\n'.format(conf.model_2D_prediction, count_parameters(pose_net)))
    
    else:
        raise ValueError(f"Unsupported model name '{conf.model_3D_prediction}'. Please choose 'GraphMLP'.")

    print('Successful: Model transferred to GPUs.\n')


    if torch.cuda.is_available():
        pose_net = torch.nn.DataParallel(pose_net)
        
    return pose_net



def main() -> None:
    """
    Control flow for the code
    """

    # 1. Load configuration file ----------------------------------------------------------------------------
    print('Loading configurations.\n')

    conf  = ParseConfig()
    args = parse_args()

    training_pkg = dict()

    num_hm = conf.experiment_settings['num_hm']
    epochs = conf.experiment_settings['epochs']
    trials = conf.trials

    
    training_pkg = dict()
    for method in training_methods:
        training_pkg[method] = dict()
        training_pkg[method]['tac'] = torch.zeros((trials, num_hm), dtype=torch.float32)#, device='cuda')
        training_pkg[method]['ll'] = torch.zeros(trials, dtype=torch.float32)#, device='cuda')
        training_pkg[method]['loss'] = torch.zeros((trials, epochs))#, device='cuda')
    training_pkg['training_methods'] = training_methods 
    
    
    # 3. Defining DataLoader --------------------------------------------------------------------------------
    print('Defining DataLoader.\n',  flush = True)
    train_dataset = Human36_dataset_3D(subjectp = ['S1', 'S5', 'S6', 'S7', 'S8'], is_train=True, model_2D = conf.model_2D_dataset,num_frame=243, num_keypoints=17) #subjectp = ['S1', 'S5', 'S6', 'S7', 'S8']
    test_dataset = Human36_dataset_3D(subjectp = ['S9', 'S11'], is_train=False, model_2D = conf.model_2D_dataset, num_frame=243, num_keypoints=17) #subjectp = ['S9', 'S11']
    print('Ending DataLoader definition.\n', flush = True)


    #4 Do one loop of prediction 2D pose
    pose_model = init_3D_prediction_models(conf=conf, args= args)

    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Defining the network -------------------------------------------------------------------------

        # 4.b: Train the covariance approximation model
        train_obj_2D_prediction = Evaluate_3D_prediction(train_dataset, pose_model, conf, training_pkg,args=args)
        training_pkg_2D_prediction = train_obj_2D_prediction.run_model()

        test_obj_2D_prediction = Evaluate_3D_prediction(test_dataset, pose_model, conf, training_pkg, args=args)
        training_pkg_2D_prediction = test_obj_2D_prediction.run_model()



main()