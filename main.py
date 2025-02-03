##############################################################
##### USE THIS FIL ONLY AFTER RUNNING get2Dprediction.py #####
#####           OR BY HAVING THE NPY FILES               #####
##############################################################


import os
import copy
import logging
import yaml
import time
import csv
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

from utils.pose import count_parameters
from utils.utils_graphmlp import parse_args, AccumLoss, flip_data
from utils.utils_skateformer import get_parser
from utils.eval import mpjpe_p1, mpjpe, mpjpe_p2, compute_X_sub, compute_X_view
from utils.visualisation import visualize_3d, compute_length

from models.GraphMLP.graphmlp import Model as GraphMLP
from models.SkateFormer.SkateFormer import SkateFormer
from dataloader.dataloader import Human36_dataset_action, Human36_dataset_3D

from timm.scheduler.cosine_lr import CosineLRScheduler

subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Train(object):
    def __init__(self, sampler: Human36_dataset_action, model,args, conf: ParseConfig,training_pkg: dict, trial: int) -> None:
        """
        Train and compare various covariance methods.
        :param sampler: Instance of Human36_dataset_action
        :param models: Contains GraphMLP
        :param args: contains arguments for model
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        """
        self.conf = conf
        self.sampler = sampler
        self.training_pkg = training_pkg
        self.trial = trial
        self.args = args

        # Experiment Settings
        self.batch_size = conf.experiment_settings['batch_size']
        self.epoch = conf.experiment_settings['epochs']
        self.learning_rate = conf.experiment_settings['lr']
        
        #self.loss_fn = torch.nn.MSELoss()  # MSE
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        #self.joint_names = self.sampler.ind_to_jnt
        self.model_save_path = conf.save_path

        self.torch_dataloader = DataLoader(self.sampler, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)

        self.training_pkg['networks'] = (copy.deepcopy(model).cuda())

        all_param = []
        all_param += list(self.training_pkg['networks'].parameters())
        if conf.model_3D_prediction == 'GraphMLP':
            self.training_pkg['optimizer'] = torch.optim.Adam(all_param, lr=args.lr, amsgrad=True)
        elif conf.model_3D_prediction == 'SkateFormer':
            self.training_pkg['optimizer'] = torch.optim.Adam(all_param,lr=self.args.base_lr, weight_decay=self.args.weight_decay)
            self.loss = torch.nn.CrossEntropyLoss().cuda()
            num_step = int(self.epoch*len(self.torch_dataloader))
            warmup_steps = int(self.args.warm_up_epoch*len(self.torch_dataloader))
            self.lr_scheduler = CosineLRScheduler(self.training_pkg['optimizer'],
                t_initial=(num_step - warmup_steps) if self.args.warmup_prefix else num_step,
                lr_min=self.args.min_lr,
                warmup_lr_init=self.args.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=self.args.warmup_prefix,
            )
            self.global_step = 0
        self.epoch = 0 #Now used to store information of the epoch we are at

        #self.training_pkg['scheduler'] = ReduceLROnPlateau(self.training_pkg['optimizer'],factor=0.25, patience=10, cooldown=0, min_lr=1e-6, verbose=True)


    def train_model(self, training_pkg) -> dict:
        if self.conf.model_3D_prediction == 'GraphMLP':
            training_pkg, loss = self.train_model_lifting(training_pkg)
        elif self.conf.model_3D_prediction == 'SkateFormer':
            training_pkg, loss = self.train_model_action_recognition(training_pkg)
        return training_pkg, loss

    def train_model_action_recognition(self, training_pkg) -> dict:
        """
        Training loop
        """
        print("Action-Recognition: training: Epochs - {}\tBatch Size - {}\t Number of batches- {}".format(self.epoch, self.batch_size, len(self.torch_dataloader)))

        #actions = ["Directions", "Purchases", "Smoking", "Phoning", "Discussion", "Eating", "Greeting", "Photo", "SittingDown","Walking", "Waiting","WalkDog","WalkTogether","Sitting", "Posing" ]
        #self.sampler.set_augmentation(augment=True)

        self.training_pkg =training_pkg
        # Training loop
        loss_value = []
        acc_value = []
        net = self.training_pkg['networks']
        optimizer = self.training_pkg['optimizer']
        net.train()
        data_for_x_sub = torch.zeros((self.batch_size*len(self.torch_dataloader),4))

        for i, (gt_3d, keypoints_2D, label, index_frame, view_point, subject) in tqdm(enumerate(self.torch_dataloader), ascii=True):
            self.lr_scheduler.step_update(self.global_step)
            self.global_step +=1
            #if i == 0:
            #    print(label, view_point, subject)
            gt_3d = gt_3d.cuda()
            keypoints_2D = keypoints_2D.cuda()
            keypoints_2D = keypoints_2D.float()
            index_frame = index_frame.float()
            index_frame = index_frame.squeeze(1)
            output = net(keypoints_2D, index_frame)#index_t has been put at False just to run through and see if we have any error's further on
            label = label.to(output.device)
            loss = self.loss(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value.append(loss.data.item())

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())

            data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 0] = label
            data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 1] = predict_label
            data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 2] = subject
            data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 3] = view_point

        array = data_for_x_sub.numpy()
        name = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/csv_files/train_256_'+str(self.epoch)+'_epoch.csv'
        with open(name, 'w', newline='') as csvfile:#delete import csv at the end
            writer = csv.writer(csvfile)
            writer.writerows(array)
        

        return self.training_pkg, np.mean(loss_value)

    def train_model_lifting(self, training_pkg) -> dict:
        """
        Training loop
        """
        print("Lifting 2D-3D: training: Epochs - {}\tBatch Size - {}\t Len of Dataloader{}".format(self.epoch, self.batch_size, len(self.torch_dataloader)))

        #self.sampler.set_augmentation(augment=True)

        self.training_pkg =training_pkg
        # Training loop
        loss_all = {'loss': AccumLoss()}
        net = self.training_pkg['networks']
        optimizer = self.training_pkg['optimizer']
        net.train()

        for i, (gt_3d, keypoints_2D) in tqdm(enumerate(self.torch_dataloader), ascii=True):
            gt_3d = gt_3d.cuda()
            keypoints_2D = keypoints_2D.cuda()
            keypoints_2D = keypoints_2D.float()
            """
            mean = 0.0
            std = 0.02 #0.01
            gaussian_noise = torch.normal(mean, std, size=keypoints_2D.shape, device=keypoints_2D.device)
            keypoints_2D = keypoints_2D + gaussian_noise
            """
            predicted_3D = net(keypoints_2D)

            out_target = gt_3d.clone()
            #print(out_target.shape)
            out_target[:, 1:, :] = out_target[:, 1:, :] - out_target[:, self.args.root_joint, :].unsqueeze(1)
            out_target[:, self.args.root_joint, :] = 0
            #out_target = out_target[:, self.args.pad].unsqueeze(1)
            #print(out_target.shape)
            out_target = out_target.unsqueeze(1)
            #print(out_target.shape, predicted_3D.shape)
            loss = mpjpe(predicted_3D,out_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            N = keypoints_2D.shape[0]
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)


            # if i < 10:
            #     with torch.no_grad():
            #         visualize_3d(out_target[0,0,:],predicted_3D[0,0,:],str(i)+'3dv')
        return self.training_pkg, loss_all['loss'].avg

class Evaluate(object):
    def __init__(self, sampler: Human36_dataset_action, conf: ParseConfig, args) -> None:
        """
        Compare the TAC error or likelihood for various covariance methods.
        :param sampler: Instance of HumanPoseDataLoader, samples from MPII + LSP
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        """

        self.sampler = sampler
        self.conf = conf
        self.args = args
        self.batch_size = conf.experiment_settings['batch_size']
        self.torch_dataloader = DataLoader(self.sampler, batch_size=self.batch_size, shuffle=False, num_workers=1, drop_last=True)
        if conf.model_3D_prediction == 'SkateFormer':
            self.loss = torch.nn.CrossEntropyLoss().cuda()

    def evaluation(self, training_pkg, e) -> dict:
        if self.conf.model_3D_prediction == 'GraphMLP':
            err1, err2 = self.evaluation_lifting(training_pkg)
        elif self.conf.model_3D_prediction == 'SkateFormer':
            err1, err2 = self.evaluation_action_recognition(training_pkg, e)
        return err1, err2

    def evaluation_action_recognition(self, training_pkg, epoch):
        cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        actions = ["Directions", "Purchases", "Smoking", "Phoning", "Discussion", "Eating", "Greeting", "Photo", "SittingDown","Walking", "Waiting","WalkDog","WalkTogether","Sitting", "Posing" ]
        subjects_to_compare = ["S1", "S5", "S6", "S7", "S8", "S9","S11"]

        net = training_pkg['networks']
        net.eval()

        loss_value = []
        acc_value = []
        print("Evalation \tBatch Size - {}".format( self.batch_size))
        with torch.no_grad():
            data_for_x_sub = torch.zeros((self.batch_size*len(self.torch_dataloader),4))
            for i, (gt_3d, keypoints_2D, label, index_frame,  view_point, subject) in tqdm(enumerate(self.torch_dataloader), ascii=True):
                gt_3d = gt_3d.cuda()
                keypoints_2D = keypoints_2D.cuda()
                keypoints_2D = keypoints_2D.float()
                index_frame = index_frame.float()
                index_frame = index_frame.squeeze(1)

                output = net(keypoints_2D, index_frame)
                label = label.to(output.device)
                loss = self.loss(output, label)

                loss_value.append(loss.data.item())

                value, predict_label = torch.max(output.data, 1)
                acc = torch.mean((predict_label == label.data).float())
                acc_value.append(acc.data.item())

                data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 0] = label
                data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 1] = predict_label
                data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 2] = subject
                data_for_x_sub[i*self.batch_size:(i+1)*self.batch_size, 3] = view_point

                
        array = data_for_x_sub.numpy()
        name = '/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/csv_files/test_256_'+str(epoch)+'_epoch.csv'
        with open(name, 'w', newline='') as csvfile:#delete import csv at the end
            writer = csv.writer(csvfile)
            writer.writerows(array)
        x_sub = compute_X_sub(data_for_x_sub[:,0], data_for_x_sub[:,1],data_for_x_sub[:,2], possible_subject = np.array([5,6]))#5,6 cause, S9 and S11 are in positions 5 and 6 in the subject list
        x_view = compute_X_view(data_for_x_sub[:,0], data_for_x_sub[:,1],data_for_x_sub[:,3], possible_view = np.array([0,1,2,3]))#0,1,2,3 cause we have the four viewpoints
        del data_for_x_sub
        print("RESULT", x_sub, x_view)


        return x_sub, x_view

    def evaluation_lifting(self, training_pkg):
        
        net = training_pkg['networks']
        net.eval()

        error_1 = []
        error_2 = []
        print("evaluation")
        with torch.no_grad():
            for i,(gt_3d, keypoints_2D) in tqdm(enumerate(self.torch_dataloader), ascii=True):
                gt_3d = gt_3d.cuda()
                keypoints_2D = keypoints_2D.cuda()
                keypoints_2D = keypoints_2D.float()
                flipped_keypoints_2D = flip_data(keypoints_2D)
                predicted_3D = net(keypoints_2D)
                flipped_predicted_3D = net(flipped_keypoints_2D)

                output_3D = (predicted_3D + flip_data(flipped_predicted_3D))/2

                out_target = gt_3d.clone()
                #print(out_target.shape, output_3D.shape)
                out_target[:, 1:, :] = out_target[:, 1:, :] - out_target[:, self.args.root_joint, :].unsqueeze(1)
                out_target[:, self.args.root_joint, :] = 0
                output_3D[:,:, self.args.root_joint, :] = 0
                #print(out_target.shape, output_3D.shape)

                #out_target = out_target[:, self.args.pad].unsqueeze(1)
                out_target = out_target.unsqueeze(1)

                error_1.append(mpjpe_p1(output_3D,out_target).detach().cpu().numpy())
                error_2.append(mpjpe_p2(output_3D,out_target))
                #if i == 10:
                #    break
            error_1 = np.mean(np.concatenate(error_1, axis=0),axis=0)
            error_2 = np.mean(np.concatenate(error_2, axis=0),axis=0)
            print("In evaluation: MJPE P1", error_1, "MJPE P2", error_2)

        return error_1,error_2


def init_3D_prediction_models(conf: ParseConfig, args) -> tuple:
    """
    Initializes and returns Hourglass and AuxNet models
    """

    logging.info('Initializing Auxiliary Network')
    

    if conf.model_3D_prediction == 'GraphMLP':
        print('Initializing GraphMLP Network')
        pose_net = GraphMLP(args).cuda()
        print('Number of parameters (GraphMLP) for{}: {}\n'.format(conf.model_2D_prediction, count_parameters(pose_net)))
    elif conf.model_3D_prediction == 'SkateFormer':
        print('Initializing SkateFormer Network')
        pose_net = SkateFormer(in_channels=2,depths=(2, 2, 2, 2),channels=(96, 192, 192, 192),embed_dim=96,num_frames=256,**args.model_args).cuda()
        print('Number of parameters (SkateFormer) for{}: {}\n'.format(conf.model_2D_prediction, count_parameters(pose_net)))
    else:
        raise ValueError(f"Unsupported model name '{conf.model_3D_prediction}'. Please choose either 'GraphMLP'.")

    logging.info('Successful: Model transferred to GPUs.\n')


    if torch.cuda.is_available():
        pose_net = torch.nn.DataParallel(pose_net)
        
    return pose_net

def save_model(args, epoch, mpjpe, model, model_name): #Juste do a small print to
    os.makedirs(args.checkpoint, exist_ok=True)

    if os.path.exists(args.previous_name):
        os.remove(args.previous_name)

    previous_name = '%s/%s_%d_%d.pth' % (args.checkpoint, model_name, epoch, mpjpe * 100)
    torch.save(model.state_dict(), previous_name)
    
    return previous_name

def main() -> None:

    print('Loading configurations.\n')

    conf  = ParseConfig()

    if conf.model_3D_prediction == 'GraphMLP':
        args = parse_args()
    elif conf.model_3D_prediction == 'SkateFormer':
        parser = get_parser()
        p = parser.parse_args()
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                    assert (k in key)
            parser.set_defaults(**default_arg)

        args = parser.parse_args()
        args.previous_best = -np.inf
    else:
        raise ValueError(f"Unsupported model name '{conf.model_3D_prediction}'. Please choose either 'GraphMLP'.")
    
    print('Loading configurations done.\n')

    training_pkg = dict()
    num_hm = conf.experiment_settings['num_hm']
    epochs = conf.experiment_settings['epochs']
    trials = conf.trials
        

    if conf.model_3D_prediction == 'GraphMLP':
        args.nepoch = epochs
        lr = args.lr
        train_dataset = Human36_dataset_3D(subjectp = ['S1', 'S5', 'S6', 'S7', 'S8'], is_train=True, model_2D = conf.model_2D_prediction,num_frame=243, num_keypoints=17) #subjectp = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_dataset = Human36_dataset_3D(subjectp = ['S9','S11'], is_train=False, model_2D = conf.model_2D_prediction, num_frame=243, num_keypoints=17) #subjectp = ['S9', 'S11']


    elif conf.model_3D_prediction == 'SkateFormer':
        args.num_epoch =epochs
        lr=args.base_lr
        train_dataset = Human36_dataset_action(subjectp = ['S1', 'S5', 'S6', 'S7', 'S8'], is_train=True, model_2D = conf.model_2D_prediction,num_frame=256, num_keypoints=20) #subjectp = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_dataset = Human36_dataset_action(subjectp = ['S9', 'S11'], is_train=False, model_2D = conf.model_2D_prediction, num_frame=256, num_keypoints=20) #subjectp = ['S9', 'S11']
        logtime = time.strftime('%m%d_%H%M_%S_')
        args.checkpoint = 'checkpoint/'+ 'Skateformer' + logtime
        args.previous_name = ''
        os.makedirs(args.checkpoint, exist_ok=True)
    start_epoch = 0
    loss_epochs = []
    mpjpes = []
    model = init_3D_prediction_models(conf,args)
    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):
        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))
        train_obj_2D_prediction = Train(train_dataset, model, args, conf, training_pkg, trial)
        test_obj_2D_prediction = Evaluate(test_dataset, conf, args)
        training_pkg = train_obj_2D_prediction.training_pkg
        for e in range(epochs):
            training_pkg_2D_prediction, loss = train_obj_2D_prediction.train_model(training_pkg)
            loss_epochs.append(loss)
            p1, p2 = test_obj_2D_prediction.evaluation(training_pkg_2D_prediction, e)
            mpjpes.append(p1)
            optimizer = train_obj_2D_prediction.training_pkg['optimizer']
            
            if conf.model_3D_prediction == 'GraphMLP':
                if p1 < args.previous_best:#and args.train
                    best_epoch = e
                    args.previous_name = save_model(args, e, p1, training_pkg_2D_prediction['networks'], 'model')
                    args.previous_best = p1
            elif conf.model_3D_prediction == 'SkateFormer':
                if p1 > args.previous_best:#and args.train
                    best_epoch = e
                    args.previous_name = save_model(args, e, p1, training_pkg_2D_prediction['networks'], 'model')
                    args.previous_best = p1
            ###IF ARGS.train
            if conf.model_3D_prediction == 'GraphMLP':
                if e % args.lr_decay_epoch == 0:
                    lr *= args.lr_decay_large
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.lr_decay_large
                else:
                    lr *= args.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.lr_decay

            if e > start_epoch:###AND ARGS.TRAIN
                plt.figure()
                epoch_x = np.arange(start_epoch+1, len(loss_epochs)+1)
                plt.plot(epoch_x, loss_epochs[start_epoch:], '.-', color='C0')
                plt.plot(epoch_x, mpjpes[start_epoch:], '.-', color='C1')
                plt.legend(['Loss-Train', '1st Metric test'])
                plt.ylabel('Loss/Score')
                plt.xlabel('Epoch')
                plt.xlim((start_epoch+1, len(loss_epochs)+1))
                plt.savefig(os.path.join(args.checkpoint, 'loss.png'))
                plt.close()

            train_obj_2D_prediction.epoch +=1
            train_obj_2D_prediction.training_pkg['optimizer'] = optimizer
            training_pkg = train_obj_2D_prediction.training_pkg

main()