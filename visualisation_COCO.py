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
from utils.visualisation import visualize_on_COCO

from dataloader.dataloader import H36_dataset

from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
from models.vit_pose import vitpose_config
from models.vit_pose.ViTPose import ViTPose

#training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']
training_methods = ['MSE']#do one method preferrably for the 2D keypoints saving
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Visualize_2D_prediction(object):
    def __init__(self, sampler: H36_dataset, model_vitpose, model_hourglass, conf: ParseConfig,training_pkg: dict, trial: int,isTrain:int) -> None:
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

        #self.ind_to_jnt = self.sampler.ind_to_jnt
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        for method in training_methods:
            self.training_pkg[method]['networks_vitpose'] = (copy.deepcopy(model_vitpose).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/InThis/'+'ViTPose'+'2_1/'+'ViTPose'+'.pth'
            self.training_pkg[method]['networks_vitpose'].load_state_dict(torch.load(model_path))

            self.training_pkg[method]['networks_hourglass'] = (copy.deepcopy(model_hourglass).cuda())
            model_path = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/InThis/'+'Hourglass'+'2_1/'+'Hourglass'+'.pth'
            self.training_pkg[method]['networks_hourglass'].load_state_dict(torch.load(model_path))
    
    def run_model(self) -> dict:
        print("Started running model.")
        #self.sampler.set_augmentation(augment=True)

        # Evaluation loop
        image_folder_path_test = '/mnt/vita/scratch/datasets/MS_COCO_rwx/images/test2017'
        with torch.no_grad():
            for i, filename in tqdm(enumerate(os.listdir(image_folder_path_test)), ascii=True):
                if i == 0 or i == 6 or i==7 or i ==8 or i ==9 or i == 17 or i == 18 or i ==19:
                    continue
                if i == 30:
                    break
                if filename.endswith(('.jpg')): #Just to make sure

                    image_path = os.path.join(image_folder_path_test, filename)
                    image = cv2.imread(image_path)
                    assert image.shape[2] == 3, "Image loaded with wrong dimensions"

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (256, 256))
                    image = image/256.0
                    image = torch.from_numpy(image).float().unsqueeze(0)

                    for method in training_methods:
                        net_vitpose = self.training_pkg[method]['networks_vitpose']
                        net_hourglass = self.training_pkg[method]['networks_hourglass']

                        net_vitpose.eval()
                        net_hourglass.eval()

                        outputs, pose_features = net_vitpose(image)# images.cuda() done internally within the model
                        pred_uv_vitpose = soft_argmax(outputs[:, -1]).view(outputs.shape[0], self.num_hm,2)
                        outputs, pose_features = net_hourglass(image)# images.cuda() done internally within the model
                        pred_uv_hourglass = soft_argmax(outputs[:, -1]).view(outputs.shape[0], self.num_hm,2)

                    if i < 30:
                        #def visualize_on_image(ground_truth_keypoints,vitpose_keypoints, hourglass_keypoints, image, name):
                        visualize_on_COCO(pred_uv_vitpose[0,:], pred_uv_hourglass[0,:], image[0,:], str(i))

        return self.training_pkg



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


    num_hm = conf.experiment_settings['num_hm']
    trials = conf.trials

    
    training_pkg = dict()
    for method in training_methods:
        training_pkg[method] = dict()
    training_pkg['training_methods'] = training_methods 
    
    
    # 3. Defining DataLoader --------------------------------------------------------------------------------
    logging.info('Defining DataLoader.\n')
    dataset = H36_dataset(subjectp = ['S9'], is_train=False) #subjectp = ['S1', 'S5', 'S6', 'S7', 'S8']

    #4 Do one loop of prediction 2D pose
    vitpose_model = init_2D_prediction_models(conf, name = 'ViTPose')
    hourglass_model = init_2D_prediction_models(conf, name='Hourglass')

    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Defining the network -------------------------------------------------------------------------

        # 4.b: Train the covariance approximation model
        train_obj_2D_prediction = Visualize_2D_prediction(dataset, vitpose_model,hourglass_model, conf, training_pkg, trial, isTrain=True)
        training_pkg_2D_prediction = train_obj_2D_prediction.run_model()



main()