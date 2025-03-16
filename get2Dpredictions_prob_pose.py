# System imports
import os
import copy
import logging
from tqdm import tqdm
from typing import Union

import numpy as np
from matplotlib import pyplot as plt#for visualize
import cv2
import random

# Science-y imports
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# File imports
from config import ParseConfig
from dataloader.dataloader import H36_dataset_probabilistic_pose

from models.joint_prediction.JointPrediction import JointPrediction_ViTPose, JointPrediction_StackedHourglass
from models.auxiliary.AuxiliaryNet import AuxNet_HG, AuxNet_ViTPose
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
from models.vit_pose import vitpose_config
from models.vit_pose.ViTPose import ViTPose

from utils.pose import fast_argmax, soft_argmax
from utils.pose import heatmap_loss, count_parameters

from utils.tic import get_positive_definite_matrix, get_tic_covariance
from utils.tic import calculate_tac, calculate_ll

from utils.loss import mse_loss, nll_loss, diagonal_loss
from utils.loss import beta_nll_loss, faithful_loss
from utils.loss import tic_loss

# Global declarations
logging.getLogger().setLevel(logging.INFO)
os.chdir(os.path.dirname(os.path.realpath(__file__)))


training_method = ['TIC']

class Evaluate(object):
    def __init__(self, sampler: H36_dataset_probabilistic_pose, conf: ParseConfig,
                 training_pkg: dict, trial: int, is_test_set:bool) -> None:
        """
        Compare the TAC error or likelihood for various covariance methods.
        :param sampler: Instance of H36_dataset_probabilistic_pose
        :param conf: Stores the configuration for the experiment
        :param training_pkg: Dictionary which will hold models, optimizers, schedulers etc.
        :param trial: Which trial is ongoing
        """

        self.sampler = sampler
        self.conf = conf
        self.training_pkg = training_pkg
        self.trial = trial
        self.is_test_set = is_test_set
        
        self.batch_size = conf.experiment_settings['batch_size']
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps

        self.torch_dataloader = DataLoader(
            self.sampler, batch_size=self.batch_size, shuffle=False, num_workers=1, drop_last=True)


    def calculate_metric(self, metric: str) -> None:
        """
        Calculate and store TAC or LL for all methods
        """
        print("Covariance Estimation: {} Evaluation".format(metric.upper()))

        self.sampler.set_augmentation(augment=False)

        with torch.no_grad():
            keypoints_2D = {}
            for i, (gt, images, heatmaps, subject, action, frame_num) in tqdm(enumerate(self.torch_dataloader), ascii=True):
                for method in training_method:
                    images = images.float()
                    net = self.training_pkg[method]['networks'][0]
                    aux_net = self.training_pkg[method]['networks'][1]

                    net.eval()
                    aux_net.eval()

                    outputs, pose_features = net(images)

                    # At 64 x 64 level
                    pred_uv = soft_argmax(outputs[:, -1]).view(
                        outputs.shape[0], self.num_hm * 2)
                    gt_uv = fast_argmax(heatmaps.to(pred_uv.device)).view(
                        outputs.shape[0], self.num_hm * 2)

                    matrix = self._aux_net_inference(pose_features, aux_net)
                    covariance = self._get_covariance(method, matrix, net, pose_features, images)

                    covariance = covariance*images.shape[1]/heatmaps.shape[2]
                    pred_uv= pred_uv*images.shape[1]/heatmaps.shape[2]

                    for idx in range(self.batch_size):
                        
                        mean = pred_uv[idx,:].detach().cpu().numpy()
                        covariance_matrix = covariance[idx,:].detach().cpu().numpy()
                        if subject[idx] not in keypoints_2D:
                            keypoints_2D[subject[idx]] = {}
                        if action[idx] not in keypoints_2D[subject[idx]]:
                            keypoints_2D[subject[idx]][action[idx]] = {}
                        keypoints_2D[subject[idx]][action[idx]][frame_num[idx].item() - 1] = {}
                        keypoints_2D[subject[idx]][action[idx]][frame_num[idx].item() - 1]['mean'] = mean
                        keypoints_2D[subject[idx]][action[idx]][frame_num[idx].item() - 1]['covariance']= covariance_matrix
           
            keypoints_2D_path = "/mnt/vita/scratch/datasets/Human3.6/npy"
            if self.is_test_set == False: #(is_test_set = 0 if train set)
                keypoints_2D_filename = "keypoints_2D_"+self.conf.model_2D_prediction+'_trainset_prob_pose2.npy'
            else:
                keypoints_2D_filename = "keypoints_2D_"+self.conf.model_2D_prediction+'_testset_prob_pose2.npy'
            np.save(file=os.path.join(keypoints_2D_path, keypoints_2D_filename),arr=keypoints_2D, allow_pickle=True)


    def _get_covariance(self, name: str, matrix: torch.Tensor, pose_net: Union[Hourglass, ViTPose],
                        pose_encodings: dict, imgs: torch.Tensor) -> torch.Tensor:
        
        out_dim = 2 * self.num_hm
        if name == 'MSE':
            return torch.eye(out_dim).expand(matrix.shape[0], out_dim, out_dim).cuda()

        # Various covariance implentations ------------------------------------------------------------
        elif name in ['NLL', 'Faithful']:
            precision_hat = get_positive_definite_matrix(matrix, out_dim)
            return torch.linalg.inv(precision_hat)
        
        elif name in ['Diagonal', 'Beta-NLL']:
            var_hat = matrix[:, :out_dim] ** 2
            return torch.diag_embed(var_hat)

        elif name in ['TIC']:
            psd_matrix = get_positive_definite_matrix(matrix, out_dim)
            covariance_hat = get_tic_covariance(
                pose_net, pose_encodings, matrix, psd_matrix, self.conf.use_hessian, self.conf.model_prob_pose, imgs)

            return covariance_hat

        else:
            raise NotImplementedError

    def _aux_net_inference(self, pose_features: dict,
                           aux_net: Union[AuxNet_HG, AuxNet_ViTPose]) -> torch.Tensor:
        """
        Obtaining the flattened matrix from the aux net inference module
        """
        if self.conf.model_prob_pose == 'Hourglass':
            with torch.no_grad():
                depth = len(self.conf.architecture['aux_net']['spatial_dim'])
                encodings = torch.cat(
                    [pose_features['feature_{}'.format(i)].reshape(
                        self.batch_size, pose_features['feature_{}'.format(i)].shape[1], -1) \
                        for i in range(depth, 0, -1)],
                    dim=2)
        else:
            encodings = pose_features

        aux_out = aux_net(encodings)
        return aux_out


def visualize_heatmaps(data_obj):
    '''
    Small code snippet to visualize heatmaps
    :return:
    '''
    random_integers = [random.randint(0, data_obj.model_input_dataset['name'].shape[0]) for _ in range(3)]

    for i in random_integers:
        image, hm = data_obj.__getitem__(i)

        plt.subplot(4, 5, 1)
        plt.imshow(image.numpy())
        plt.axis('off')
        plt.show()
        
        
        for j in range(hm.shape[0]):
            
            plt.subplot(4, 5, j+1)
            plt.imshow(image.numpy())
            plt.subplot(4, 5, j+1)
            plt.imshow(cv2.resize(hm[j].numpy(), dsize=(256, 256), interpolation=cv2.INTER_CUBIC), alpha=.5)
            plt.title('{}'.format(data_obj.coco_idx_to_jnt[j]), fontdict = {'fontsize' : 6})
            plt.axis('off')

        plt.show()
        plt.close()

def main() -> None:
    """
    Control flow for the code
    """

    # 1. Load configuration file ----------------------------------------------------------------------------
    logging.info('Loading configurations.\n')

    conf  = ParseConfig()


    num_hm = conf.architecture['aux_net']['num_hm']
    trials = conf.trials

    training_pkg = dict()
    for method in training_method:
        training_pkg[method] = dict()
        training_pkg[method]['tac'] = torch.zeros((trials, num_hm,2), dtype=torch.float32, device='cuda')
        training_pkg[method]['ll'] = torch.zeros(trials, dtype=torch.float32, device='cuda')
    training_pkg['training_method'] = training_method 
    


    # 3. Defining DataLoader --------------------------------------------------------------------------------
    logging.info('Defining DataLoader.\n')
    train_dataset = H36_dataset_probabilistic_pose(subjectp = ['S1', 'S5', 'S6', 'S7', 'S8'],conf = conf,  is_train=True)
    test_dataset = H36_dataset_probabilistic_pose(subjectp = ['S9','S11'], conf = conf, is_train=False)

    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Open the saved model -------------------------------------------------------------------------
        file_path = os.path.join("/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/InThis/ViTPoseDiv102", "training_pkg_0.pt")#ViTPoseDiv10Epoch100
        training_pkg = torch.load(file_path)


        with torch.no_grad():
            eval_obj = Evaluate(
                sampler=train_dataset, conf=conf, training_pkg=training_pkg, trial=trial, is_test_set = False)
            eval_obj.calculate_metric(metric='ll')
            eval_obj = Evaluate(
                sampler=test_dataset, conf=conf, training_pkg=training_pkg, trial=trial, is_test_set = True)
            eval_obj.calculate_metric(metric='ll')



main()