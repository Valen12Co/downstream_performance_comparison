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


from dataloader.dataloader import H36_dataset

from models.GraphMLP.graphmlp import Model as GraphMLP
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
from models.vit_pose import vitpose_config
from models.vit_pose.ViTPose import ViTPose


#training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']
training_methods = ['MSE']#do one method preferrably for the 2D keypoints saving
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Visualise_2D3D_prediction(object):
    def __init__(self, sampler: H36_dataset,pose_model, vitpose_model, hourglass_model, conf: ParseConfig,training_pkg: dict,args) -> None:
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

        self.action_reference = 'Phoning.58860488'
        #self.ind_to_jnt = self.sampler.ind_to_jnt
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        for method in training_methods:

            #2D models
            self.training_pkg[method]['networks_vitpose_2D'] = (copy.deepcopy(vitpose_model).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/InThis/'+'ViTPose'+'2_1/'+'ViTPose'+'.pth'
            self.training_pkg[method]['networks_vitpose_2D'].load_state_dict(torch.load(model_path))
            self.training_pkg[method]['networks_hourglass_2D'] = (copy.deepcopy(hourglass_model).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/InThis/'+'Hourglass'+'2_1/'+'Hourglass'+'.pth'
            self.training_pkg[method]['networks_hourglass_2D'].load_state_dict(torch.load(model_path))

            #3D models can easily switch bewteen models
            #GT 3D
            self.training_pkg[method]['networks_gt_3D'] = (copy.deepcopy(pose_model).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/ViTPose1/model_2_19.pth'
            self.training_pkg[method]['networks_gt_3D'].load_state_dict(torch.load(model_path))
            #ViTPose
            self.training_pkg[method]['model_1'] = (copy.deepcopy(pose_model).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/ViTPose1/model_2_19.pth'
            self.training_pkg[method]['model_1'].load_state_dict(torch.load(model_path))
            #GraphMLP
            self.training_pkg[method]['model_2'] = (copy.deepcopy(pose_model).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/checkpoint/Hourglass1/model_1_13.pth'
            self.training_pkg[method]['model_2'].load_state_dict(torch.load(model_path))

    def run_model(self) -> dict:
        print("Model running to obtain 2D keypoints: Batch Size - {} total number of batches = {}".format(self.batch_size, len(self.torch_dataloader)))

        method = training_methods[0]
        ground_truth_model = self.training_pkg[method]['networks_gt_3D']
        ground_truth_model.eval()

        model_1 = self.training_pkg[method]['model_1']
        model_1.eval()

        model_2 = self.training_pkg[method]['model_2']
        model_2.eval()

        net_vitpose = self.training_pkg[method]['networks_vitpose_2D']
        net_vitpose.eval()

        net_hourglass = self.training_pkg[method]['networks_hourglass_2D']
        net_hourglass.eval()

        coordinates = []
        images = []
        with torch.no_grad():
            
            #for i, (gt_2d, image, subject, action, frame_num) in tqdm(enumerate(self.torch_dataloader), ascii=True):

            #We get the image and the 2D coordinates of the different predictor
            # Following this we have: coordinates = [[gt_2d, pred_uv_vitpose, pred_uv_hourglass],...]
            #and we also have images = [[image, frame_num, pre_length(of the sequence of the image of tha batch)]
            for i, (gt_2d, image, subject, action, frame_num) in enumerate(self.torch_dataloader):
                if i >= 200:
                    break
                indices = [j for j, a in enumerate(action) if a == self.action_reference]

                if len(indices) == 0:
                    continue
                
                gt_2d = gt_2d.to('cuda')
                image = image.float()

                outputs, pose_features = net_vitpose(image)# images.cuda() done internally within the model
                pred_uv_vitpose = soft_argmax(outputs[:, -1]).view(outputs.shape[0], self.num_hm,2)
                outputs, pose_features = net_hourglass(image)# images.cuda() done internally within the model
                pred_uv_hourglass = soft_argmax(outputs[:, -1]).view(outputs.shape[0], self.num_hm,2)

                for idx in indices:
                    coordinates.append([gt_2d[idx,:], pred_uv_vitpose[idx,:], pred_uv_hourglass[idx,:]])
                    pre_length = len(images)
                    images.append([image[idx,:], frame_num[idx].item(),pre_length])

            print(len(coordinates), len(images))

            sequence_keypoints, images = self.get_frame(images, coordinates)

            print("Debuging:", len(sequence_keypoints))
            print("Debuging:", len(sequence_keypoints[0]))
            print("to add in get frame", sequence_keypoints[0][0].size(),sequence_keypoints[0][1].size(),sequence_keypoints[0][2].size(),sequence_keypoints[0][3])

            #Need a way to keep track of the 3D GT.

            #3 models so simulate the three models each time for. Always begin with gt then how we want
            # Each model predicts on the 2D prediction always returns table [gt_pred,vit_pred,hour_pred, gt_idx] of size [[17,2],[17,2],[17,2],1]
            model_gt_pred = []
            for i, (gt_2d, vitpose_2d, hourglass_2d, gt_idx) in enumerate(sequence_keypoints):
                gt_pred = ground_truth_model(gt_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                vit_pred = ground_truth_model(vitpose_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                hour_pred = ground_truth_model(hourglass_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                print("pred gt",i, gt_pred.size(), vit_pred.size(), hour_pred.size(),gt_idx)
                model_gt_pred.append([gt_pred.cpu().numpy(), vit_pred.cpu().numpy(), hour_pred.cpu().numpy(), gt_idx])
                #break
            print(model_gt_pred[0][0].shape)
            model_1_pred = []
            for i, (gt_2d, vitpose_2d, hourglass_2d, gt_idx) in enumerate(sequence_keypoints):
                gt_pred = model_1(gt_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                vit_pred = model_1(vitpose_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                hour_pred = model_1(hourglass_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                model_1_pred.append([gt_pred.cpu().numpy(), vit_pred.cpu().numpy(), hour_pred.cpu().numpy(),gt_idx])
                #break

            model_2_pred = []
            for i, (gt_2d, vitpose_2d, hourglass_2d, gt_idx) in enumerate(sequence_keypoints):
                gt_pred = model_2(gt_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                vit_pred = model_2(vitpose_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                hour_pred = model_2(hourglass_2d.unsqueeze(0)).squeeze(0).squeeze(0)
                model_2_pred.append([gt_pred.cpu().numpy(), vit_pred.cpu().numpy(), hour_pred.cpu().numpy(),gt_idx])
                #break
                #if i ==0:
                #    output_3D_vitpose_vitpose = output_3D_vitpose_vitpose[0,:]
                #    output_3D_vitpose_hourglass = output_3D_vitpose_hourglass[0,:]
                #    break
            print(len(model_gt_pred), len(model_1_pred), len(model_2_pred))
            print(len(images))
            """
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
            """

            """
            #output_dataset_modelused
            e_v_v = mpjpe_p1(output_3D_vitpose_vitpose,gt).detach().cpu().numpy()
            e_v_h = mpjpe_p1(output_3D_vitpose_hourglass,gt).detach().cpu().numpy()
            e_h_v = mpjpe_p1(output_3D_hourglass_vitpose,gt).detach().cpu().numpy()
            e_h_h = mpjpe_p1(output_3D_hourglass_hourglass,gt).detach().cpu().numpy()
            print("ERROR E_v_v",e_v_v,"E_V_H",e_v_h,"E_H_V",e_h_v,"E_H_H",e_h_h)
            print(gt.shape,output_3D_vitpose_vitpose.shape, output_3D_hourglass_vitpose.shape, output_3D_vitpose_hourglass.shape, output_3D_hourglass_hourglass.shape)
            visualize_3D_all(gt[0,:],output_3D_vitpose_vitpose[0,:],output_3D_hourglass_vitpose[0,:],'model_vitpose')
            visualize_3D_all(gt[0,:],output_3D_vitpose_hourglass[0,:],output_3D_hourglass_hourglass[0,:],'model_hourglass')
            """
        return self.training_pkg
    
    def get_frame(self, images, coordinates, num_frames = 243):
        """
        We create sequences of frame for the 2D-3D lifting pose estimator.

        coordinates: [ground_truth, vitpose, hourglass]
        images: [image, frame_number, idx]

        returns: sequence_keypoints: [gt, vitpose, hourglass, frame_idx (the predicted one)]) each line contains torch tensor of size: [[243, 17, 2],torch.Size([243, 17, 2]),torch.Size([243, 17, 2],1]
                 images: images sorted based on the number of the frame

        """
        images.sort(key=lambda x: x[1]) #Works cause we only search for one action so one sequence of frame. 
        sorted_coordinates = [coordinates[img[2]] for img in images]

        #Important to keep in mind that frame_num is +1 compared to the idx
        frames_to_keep = []
        frame_idx = []
        for i in range(num_frames//2,len(images)-num_frames//2):
            frames_to_keep.append(range(i-num_frames//2,i+num_frames//2+1))
            frame_idx.append(i)
        
        sequence_keypoints= []
        for i in range(len(frames_to_keep)):
            gt = torch.zeros(243,17,2)
            vitpose = torch.zeros(243,17,2)
            hourglass = torch.zeros(243,17,2)


            for j, idx in enumerate(frames_to_keep[i]):
                #print(idx, frames_to_keep[i])
                gt[j,:,:] = sorted_coordinates[idx][0]
                vitpose[j,:,:] = sorted_coordinates[idx][1]
                hourglass[j,:,:] = sorted_coordinates[idx][2]
            
            sequence_keypoints.append([gt, vitpose, hourglass, frame_idx[i]]) #add if possible the 3D GT with it.

        return sequence_keypoints, images



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

def init_2D_prediction_models(conf, name) -> tuple:
    """
    Initializes and returns Hourglass and AuxNet models
    """

    print('Initializing Auxiliary Network')
    

    if name == 'ViTPose':
        logging.info('Initializing ViTPose Network')
        pose_net = ViTPose(vitpose_config.model)#.cuda()
        print('Number of parameters (ViTPose): {}\n'.format(count_parameters(pose_net)))
    
    elif name == 'Hourglass':
        logging.info('Initializing Hourglass Network')
        pose_net = Hourglass(arch=conf.architecture['hourglass'])
        print('Number of parameters (Hourglass): {}\n'.format(count_parameters(pose_net)))
    
    else:
        raise ValueError(f"Unsupported model name '{conf.model_2D_prediction}'. Please choose either 'ViTPose' or 'Hourglass'.")

    logging.info('Successful: Model transferred to GPUs.\n')


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
    dataset = H36_dataset(subjectp = ['S9'], is_train=False) #subjectp = ['S1', 'S5', 'S6', 'S7', 'S8']
    print('Ending DataLoader definition.\n')


    vitpose_model = init_2D_prediction_models(conf, name = 'ViTPose')
    hourglass_model = init_2D_prediction_models(conf, name='Hourglass')
    pose_model = init_3D_prediction_models(conf=conf, args= args)

    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Defining the network -------------------------------------------------------------------------

        # 4.b: Train the covariance approximation model
        train_obj_2D_prediction = Visualise_2D3D_prediction(dataset, pose_model, vitpose_model, hourglass_model,conf, training_pkg,args=args)
        training_pkg_2D_prediction = train_obj_2D_prediction.run_model()



main()