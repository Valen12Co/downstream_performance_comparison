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
from utils.eval import mpjpe, compute_ap, mpjpe_p1
#from utils.visualisation import compute_length_2d#, visualize_2d

from dataloader.dataloader import H36_dataset

from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
from models.vit_pose import vitpose_config
from models.vit_pose.ViTPose import ViTPose

#training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']
training_methods = ['MSE']#do one method preferrably for the 2D keypoints saving
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Train_2D_prediction(object):
    def __init__(self, sampler: H36_dataset, model, conf: ParseConfig,training_pkg: dict, trial: int,isTrain:int) -> None:
        """
        Train and compare various covariance methods.
        :param sampler: Instance of H36_dataset
        :param models: Contains Hourglass or ViTPose model
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        :param isTrain: If it is the train or test set
        """
        self.conf = conf
        self.sampler = sampler
        self.training_pkg = training_pkg
        self.trial = trial
        self.isTrain = isTrain

        self.batch_size = conf.experiment_settings['batch_size']
        self.epoch = conf.experiment_settings['epochs']
        self.learning_rate = conf.experiment_settings['lr']
        #we want all the data to get all the 2D prediction
        self.torch_dataloader = DataLoader(self.sampler, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=False)

        #self.ind_to_jnt = self.sampler.ind_to_jnt
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        for method in training_methods:
            self.training_pkg[method]['networks'] = (copy.deepcopy(model).cuda())
    
            all_param = []
            all_param += list(self.training_pkg[method]['networks'].parameters())
            self.training_pkg[method]['optimizer'] = torch.optim.Adam(all_param, lr=self.learning_rate)
            self.training_pkg[method]['scheduler'] = ReduceLROnPlateau(self.training_pkg[method]['optimizer'],factor=0.25, patience=10, cooldown=0, min_lr=1e-6, verbose=True)
            
    def run_model(self) -> dict:
        print("Model running to obtain 2D keypoints: Batch Size - {} total number of batches = {}".format(self.batch_size, len(self.torch_dataloader)))

        #self.sampler.set_augmentation(augment=True)

        # Evaluation loop
        for e in range(self.epoch):
            print('Training for epoch: {}'.format(e+1))
            for i, (gt_2d,image,subject,action,frame_num) in tqdm(enumerate(self.torch_dataloader), ascii=True):
                batch_size = gt_2d.shape[0]
                gt_2d = gt_2d.to('cuda')
                image = image.float()
                for method in training_methods:
                    net = self.training_pkg[method]['networks']
                    optimizer = self.training_pkg[method]['optimizer']

                    net.train()
                    outputs, pose_features = net(image)# images.cuda() done internally within the model
                    pred_uv = soft_argmax(outputs[:, -1]).view(outputs.shape[0], self.num_hm,2)
                    loss = mpjpe(pred_uv,gt_2d)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            print('The last loss: {}'.format(loss.mean()))

        return self.training_pkg

class Evaluate_2D_prediction(object):
    def __init__(self, sampler: H36_dataset, model, conf: ParseConfig,training_pkg: dict, trial: int,isTrain:int) -> None:
        """
        Train and compare various covariance methods.
        :param sampler: Instance of H36_dataset
        :param models: Contains Hourglass or ViTPose model
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        :param isTrain: If it is the train or test set
        """
        self.conf = conf
        self.sampler = sampler
        self.training_pkg = training_pkg
        self.trial = trial
        self.isTrain = isTrain

        self.batch_size = conf.experiment_settings['batch_size']
        #we want all the data to get all the 2D prediction
        self.torch_dataloader = DataLoader(self.sampler, batch_size=self.batch_size, shuffle=False, num_workers=1, drop_last=False)

        #self.ind_to_jnt = self.sampler.ind_to_jnt
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps


    def run_model(self) -> dict:
        print("Model running to obtain 2D keypoints: Batch Size - {} total number of batches = {}".format(self.batch_size, len(self.torch_dataloader)))

        #self.sampler.set_augmentation(augment=True)

        # Evaluation loop
        method_results = {}
        with torch.no_grad():
            for i, (gt_2d,image,subject,action,frame_num) in tqdm(enumerate(self.torch_dataloader), ascii=True):
                batch_size = gt_2d.shape[0]
                gt_2d = gt_2d.to('cuda')
                image = image.float()
                for method in training_methods:
                    net = self.training_pkg[method]['networks']

                    net.eval()
                    outputs, pose_features = net(image)# images.cuda() done internally within the model
                    pred_uv = soft_argmax(outputs[:, -1]).view(outputs.shape[0], self.num_hm,2)
                    p1 = mpjpe_p1(pred_uv,gt_2d)
                    ap = compute_ap(pred_uv,gt_2d,0.5)
                    p1 = p1.detach().cpu().numpy()
                    ap = ap.detach().cpu().numpy()
                    if method not in method_results:
                        method_results[method] = {"AP": [], "P1": []}
                    method_results[method]["AP"].append(ap)
                    method_results[method]["P1"].append(p1)


            for method in training_methods:
                method_results[method]["AP"] = np.mean(np.concatenate(method_results[method]["AP"], axis=0),axis=0)
                method_results[method]["P1"] = np.mean(np.concatenate(method_results[method]["P1"], axis=0),axis=0)
                print(f"For {method} the values are AP = {method_results[method]['AP']} and P1 = {method_results[method]['P1']}")


def init_2D_prediction_models(conf: ParseConfig) -> tuple:
    """
    Initializes and returns Hourglass and AuxNet models
    """

    logging.info('Initializing Auxiliary Network')
    

    if conf.model_2D_prediction == 'ViTPose':
        logging.info('Initializing ViTPose Network')
        pose_net = ViTPose(vitpose_config.model)#.cuda()
        print('Number of parameters (ViTPose): {}\n'.format(count_parameters(pose_net)))
    
    elif conf.model_2D_prediction == 'Hourglass':
        logging.info('Initializing Hourglass Network')
        pose_net = Hourglass(arch=conf.architecture['hourglass'])
        print('Number of parameters (Hourglass): {}\n'.format(count_parameters(pose_net)))
    
    else:
        raise ValueError(f"Unsupported model name '{conf.model_2D_prediction}'. Please choose either 'ViTPose' or 'Hourglass'.")

    logging.info('Successful: Model transferred to GPUs.\n')


    if torch.cuda.is_available():
        pose_net = torch.nn.DataParallel(pose_net)
        
    return pose_net

def save_model(conf, model, model_name): #Juste do a small print to
    print(conf.save_path)
    os.makedirs(conf.save_path, exist_ok=True)


    previous_name = '%s/%s.pth' % (conf.save_path, model_name)
    print(previous_name)
    torch.save(model.state_dict(), previous_name)
    
    return previous_name


def main() -> None:
    """
    Control flow for the code
    """

    # 1. Load configuration file ----------------------------------------------------------------------------
    logging.info('Loading configurations.\n')

    conf  = ParseConfig()


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
    logging.info('Defining DataLoader.\n')
    train_dataset = H36_dataset(subjectp = ['S1', 'S5', 'S6', 'S7', 'S8'], is_train=True) #subjectp = ['S1', 'S5', 'S6', 'S7', 'S8']
    test_dataset = H36_dataset(subjectp = ['S9','S11'], is_train=False)

    #4 Do one loop of prediction 2D pose
    pose_model = init_2D_prediction_models(conf=conf)

    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Defining the network -------------------------------------------------------------------------

        # 4.b: Train the covariance approximation model
        train_obj_2D_prediction = Train_2D_prediction(train_dataset, pose_model, conf, training_pkg, trial, isTrain=True)
        training_pkg_2D_prediction = train_obj_2D_prediction.run_model()

        test_obj_2D_prediction = Evaluate_2D_prediction(test_dataset, pose_model, conf, training_pkg_2D_prediction, trial, isTrain=False)
        test_obj_2D_prediction.run_model()

        save_model(conf,training_pkg_2D_prediction[training_methods[0]]['networks'],conf.model_2D_prediction)



main()