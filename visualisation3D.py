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
from utils.visualisation import visualize_3D_all
from utils.utils_graphmlp import parse_args, AccumLoss, flip_data


from dataloader.dataloader import Human36_dataset_3D

from models.GraphMLP.graphmlp import Model as GraphMLP


#training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']
training_methods = ['MSE']#do one method preferrably for the 2D keypoints saving
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Visualise_3D_prediction(object):
    def __init__(self, sampler_vitpose: Human36_dataset_3D,sampler_hourglass: Human36_dataset_3D, model, conf: ParseConfig,training_pkg: dict,args) -> None:
        """
        Train and compare various covariance methods.
        :param sampler: Instance of H36_dataset
        :param models: Contains Hourglass or ViTPose model
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        :param isTrain: If it is the train or test set
        """
        self.sampler_vitpose = sampler_vitpose
        self.sampler_hourglass = sampler_hourglass
        self.conf = conf
        self.args = args
        self.training_pkg = training_pkg
        self.batch_size = conf.experiment_settings['batch_size']
        self.torch_dataloader_vitpose = DataLoader(self.sampler_vitpose, batch_size=self.batch_size, shuffle=False, num_workers=1, drop_last=False)
        self.torch_dataloader_hourglass = DataLoader(self.sampler_hourglass, batch_size=self.batch_size, shuffle=False, num_workers=1, drop_last=False)

        #self.ind_to_jnt = self.sampler.ind_to_jnt
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        for method in training_methods:
            self.training_pkg[method]['networks_vitpose'] = (copy.deepcopy(model).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/ViTPose1/model_2_19.pth'
            self.training_pkg[method]['networks_vitpose'].load_state_dict(torch.load(model_path))
            self.training_pkg[method]['networks_hourglass'] = (copy.deepcopy(model).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/Hourglass1/model_1_13.pth'
            self.training_pkg[method]['networks_hourglass'].load_state_dict(torch.load(model_path))

    def run_model(self) -> dict:
        print("Model running to obtain 2D keypoints: Batch Size - {} total number of batches = {}".format(self.batch_size, len(self.torch_dataloader_hourglass)))

        method = training_methods[0]
        net_vitpose = self.training_pkg[method]['networks_vitpose']
        net_vitpose.eval()

        net_hourglass = self.training_pkg[method]['networks_hourglass']
        net_hourglass.eval()

        with torch.no_grad():
            for i, (gt_3d, keypoints_2D) in tqdm(enumerate(self.torch_dataloader_vitpose), ascii=True):
                gt_3d = gt_3d.cuda()
                out_target = gt_3d.clone()
                out_target[:, 1:, :] = out_target[:, 1:, :] - out_target[:, self.args.root_joint, :].unsqueeze(1)
                out_target[:, self.args.root_joint, :] = 0
                out_target = out_target.unsqueeze(1)
                #out_target = out_target[:, self.args.pad].unsqueeze(1)

                keypoints_2D = keypoints_2D.cuda()
                keypoints_2D = keypoints_2D.float()
                flipped_keypoints_2D = flip_data(keypoints_2D)

                predicted_3D = net_vitpose(keypoints_2D)
                flipped_predicted_3D = net_vitpose(flipped_keypoints_2D)
                output_3D_vitpose_vitpose = (predicted_3D + flip_data(flipped_predicted_3D))/2
                output_3D_vitpose_vitpose[:,:, self.args.root_joint, :] = 0

                predicted_3D = net_hourglass(keypoints_2D)
                flipped_predicted_3D = net_hourglass(flipped_keypoints_2D)
                output_3D_vitpose_hourglass = (predicted_3D + flip_data(flipped_predicted_3D))/2
                output_3D_vitpose_hourglass[:,:, self.args.root_joint, :] = 0

                if i ==0:
                    output_3D_vitpose_vitpose = output_3D_vitpose_vitpose[0,:]
                    output_3D_vitpose_hourglass = output_3D_vitpose_hourglass[0,:]
                    break
            
            for i, (gt_3d, keypoints_2D) in tqdm(enumerate(self.torch_dataloader_hourglass), ascii=True):
                gt_3d = gt_3d.cuda()
                out_target = gt_3d.clone()
                out_target[:, 1:, :] = out_target[:, 1:, :] - out_target[:, self.args.root_joint, :].unsqueeze(1)
                out_target[:, self.args.root_joint, :] = 0
                out_target = out_target.unsqueeze(1)
                #out_target = out_target[:, self.args.pad].unsqueeze(1)

                keypoints_2D = keypoints_2D.cuda()
                keypoints_2D = keypoints_2D.float()
                flipped_keypoints_2D = flip_data(keypoints_2D)

                predicted_3D = net_vitpose(keypoints_2D)
                flipped_predicted_3D = net_vitpose(flipped_keypoints_2D)
                output_3D_hourglass_vitpose = (predicted_3D + flip_data(flipped_predicted_3D))/2
                output_3D_hourglass_vitpose[:,:, self.args.root_joint, :] = 0

                predicted_3D = net_hourglass(keypoints_2D)
                flipped_predicted_3D = net_hourglass(flipped_keypoints_2D)
                output_3D_hourglass_hourglass = (predicted_3D + flip_data(flipped_predicted_3D))/2
                output_3D_hourglass_hourglass[:,:, self.args.root_joint, :] = 0

                if i ==0:
                    output_3D_hourglass_vitpose = output_3D_hourglass_vitpose[0,:]
                    output_3D_hourglass_hourglass = output_3D_hourglass_hourglass[0,:]
                    gt = out_target[0,:]
                    break

            #output_dataset_modelused
            e_v_v = mpjpe_p1(output_3D_vitpose_vitpose,gt).detach().cpu().numpy()
            e_v_h = mpjpe_p1(output_3D_vitpose_hourglass,gt).detach().cpu().numpy()
            e_h_v = mpjpe_p1(output_3D_hourglass_vitpose,gt).detach().cpu().numpy()
            e_h_h = mpjpe_p1(output_3D_hourglass_hourglass,gt).detach().cpu().numpy()
            print("ERROR E_v_v",e_v_v,"E_V_H",e_v_h,"E_H_V",e_h_v,"E_H_H",e_h_h)
            print(gt.shape,output_3D_vitpose_vitpose.shape, output_3D_hourglass_vitpose.shape, output_3D_vitpose_hourglass.shape, output_3D_hourglass_hourglass.shape)
            visualize_3D_all(gt[0,:],output_3D_vitpose_vitpose[0,:],output_3D_hourglass_vitpose[0,:],'model_vitpose')
            visualize_3D_all(gt[0,:],output_3D_vitpose_hourglass[0,:],output_3D_hourglass_hourglass[0,:],'model_hourglass')

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
    print('Defining DataLoader.\n')
    vitpose_dataset = Human36_dataset_3D(subjectp = ['S9'], is_train=False, model_2D = 'ViTPose', num_frame=243, num_keypoints=17) #subjectp = ['S9', 'S11']
    hourglass_dataset = Human36_dataset_3D(subjectp = ['S9'], is_train=False, model_2D = 'Hourglass', num_frame=243, num_keypoints=17) #subjectp = ['S9', 'S11']
    print('Ending DataLoader definition.\n')


    #4 Do one loop of prediction 2D pose
    pose_model = init_3D_prediction_models(conf=conf, args= args)

    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Defining the network -------------------------------------------------------------------------

        # 4.b: Train the covariance approximation model
        train_obj_2D_prediction = Visualise_3D_prediction(vitpose_dataset,hourglass_dataset, pose_model, conf, training_pkg,args=args)
        training_pkg_2D_prediction = train_obj_2D_prediction.run_model()



main()