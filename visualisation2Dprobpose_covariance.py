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
from utils.pose import count_parameters, soft_argmax, fast_argmax
from utils.eval import mpjpe_p1, mpjpe_p2, compute_ap
from utils.visualisation import visualize_on_image_prob_pose
from utils.tic import get_positive_definite_matrix, get_tic_covariance

from dataloader.dataloader import H36_dataset_probabilistic_pose
from matplotlib.patches import Circle


from models.joint_prediction.JointPrediction import JointPrediction_ViTPose, JointPrediction_StackedHourglass
from models.auxiliary.AuxiliaryNet import AuxNet_HG, AuxNet_ViTPose
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
from models.vit_pose import vitpose_config
from models.vit_pose.ViTPose import ViTPose

#training_methods = ['MSE', 'Diagonal', 'NLL', 'Beta-NLL', 'Faithful', 'TIC']
training_methods = ['TIC']#do one method preferrably for the 2D keypoints saving
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Visualize_2D_prediction(object):
    def __init__(self, sampler: H36_dataset_probabilistic_pose, conf: ParseConfig,training_pkg: dict, trial: int,isTrain:int) -> None:
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
        #self.sampler.set_augmentation(augment=True)
        with torch.no_grad():
            for i, (gt_2d, image, heatmaps, subject, action, frame_num) in tqdm(enumerate(self.torch_dataloader), ascii=True):
                if i == 10:
                    break

                gt_2d = gt_2d.to('cuda')
                image = image.float()
                for method in training_methods:
                    images = image.float()
                    images = images[0,:]


                    gt_2d = gt_2d[0,:]
                    print(images.shape)
                    print(gt_2d.shape)

                    for j in range(17):
                        images = images.unsqueeze(0)


                        net = self.training_pkg[method]['networks'][0]
                        aux_net = self.training_pkg[method]['networks'][1]

                        net.eval()
                        aux_net.eval()
                        outputs, pose_features = net(images)

                        # At 64 x 64 level
                        pred_uv = soft_argmax(outputs[:, -1]).view(
                            outputs.shape[0], self.num_hm * 2)

                        matrix = self._aux_net_inference(pose_features, aux_net).unsqueeze(0)
                        covariance = self._get_covariance(method, matrix, net, pose_features, images)

                        uv_to_xy = lambda uv: (uv[1], uv[0])
                        point = np.array([np.random.multivariate_normal(pred_uv[0,:].detach().cpu(), covariance[0,:,:].detach().cpu(), size=1).squeeze()]).reshape(-1,17,2)*image.shape[1]/heatmaps.shape[2]
                        pred_uv = pred_uv.reshape(-1,17,2)*image.shape[1]/heatmaps.shape[2]


                        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                        ax[0].imshow(image[0])
                        #ax.imshow(heatmaps.sum(dim=1, keepdim=True)[0,0,:].unsqueeze(2))

                        for person in range(gt_2d.shape[0]):
                            if person ==1:
                                break
                            for joint in range(gt_2d.shape[1]):
                                gt_xy = uv_to_xy(gt_2d[person][joint])
                                pred_xy = uv_to_xy(pred_uv[person][joint])
                                point_xy = uv_to_xy(point[person][joint])
                                if gt_2d[person][joint][2] >= 0 and gt_2d[person][joint][0] > 0 and gt_2d[person][joint][1] > 0:
                                    ax[0].add_patch(Circle(gt_xy, radius=2.5, color='green', fill=True))
                                    ax[0].add_patch(Circle(pred_xy, radius=2.5, color='red', fill=True))
                                    ax[0].add_patch(Circle(point_xy, radius=2.5, color='blue', fill=True))
                                    ax[0].text(point_xy[0], point_xy[1], str(joint), fontsize=5, color='white',ha='center', va='center',bbox=dict(facecolor='black', alpha=0, edgecolor='none'))
                        ax[0].set_title("Image with Keypoints")
                        cov_matrix = covariance.squeeze(0).cpu().numpy()  # Convert tensor to numpy array
                        heatmap = ax[1].imshow(cov_matrix, cmap='hot', interpolation='nearest', vmin=-0.5, vmax=0.5)
                        ax[1].set_title("Covariance Heatmap")
                        fig.colorbar(heatmap, ax=ax[1])

                        plt.show()
                        output_dir = "/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisationprobpose/viz_covariance2"
                        os.makedirs(output_dir, exist_ok=True)
                        fig.savefig(os.path.join(output_dir, f'{j}.jpg'), dpi=350)

                        gt_2d = gt_2d.squeeze(0)

                        images = images[0,:]
                        x_min, x_max = max(0,gt_2d[j,0].item()-10), min(images.shape[1],gt_2d[j,0].item()+10)
                        y_min, y_max = max(0,gt_2d[j,1].item()-10), min(images.shape[0],gt_2d[j,1].item()+10)

                        images[int(x_min):int(x_max),int(y_min):int(y_max)]=0
                        gt_2d = gt_2d.unsqueeze(0)

                break


                mean = pred_uv/image.shape[1]
                mean = mean.detach().cpu().numpy()
                covariance = covariance.detach().cpu().numpy()
                print(image.shape)
                print(mean.shape)
                print(covariance.shape)
                breakpoint()
                for idx in range(gt_2d.size(0)):
                    points = []
                    for j in range(5):
                        #points.append(np.array([np.random.multivariate_normal(mean[0,:], covariance[0,:], size=1).squeeze()]).reshape(-1,2))
                        points.append(mean[0,:].reshape(-1,2))
                        if idx ==0:
                            print("\nGT", gt_2d[0,:])
                            print("\nMean",mean[0,:])
                            print("\nCovariance", covariance[0,:])
                            print("\nSampled point", points[j])
                    if idx<10:
                        visualize_on_image_prob_pose(gt_2d[idx,:].squeeze(0)[:,:2],points,images[idx,:].squeeze(0),str(i)+str(idx))

        return self.training_pkg

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

    def get_only_dimensions_of_interest(self,vector:torch.Tensor,list_visible_joints:torch.Tensor):
        visible_joints = (list_visible_joints[:] >= 0.5).int()
        visible_indices = visible_joints.nonzero(as_tuple=True)[1]  # Indices of True values
        cov_indices = torch.cat([visible_indices * 2, visible_indices * 2 + 1]).sort()[0]
        if vector.dim() == 3:
            vector = vector[0][cov_indices][:,cov_indices]
        elif vector.dim() == 2:
            vector = vector[0][cov_indices]
        else:
            raise NotImplementedError
        return vector

            
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

    def _number_visible_joints_inference(self, pose_features: dict, jnt_net: Union[JointPrediction_ViTPose], name: str, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Computes the visible joints in the image. Should return 0 or 1 if not visible or visible/occluded
        """
        out_dim = self.num_hm

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

        jnt_out = jnt_net(encodings)
        return jnt_out


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
    
    elif name == 'Probabilistic':
        logging.info('Initializing Probabilistic Pose Network')
        #file_path = os.path.join("/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/InThis/ViTPoseDiv10Epoch100", "training_pkg_0.pt")
        file_path = os.path.join("/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/InThis/ViTPoseDiv102", "training_pkg_0.pt")

        pose_net = torch.load(file_path)
    else:
        raise ValueError(f"Unsupported model name '{name}'. Please choose either 'ViTPose' or 'Hourglass'.")

    logging.info('Successful: Model transferred to GPUs.\n')
    
        
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

    
    
    # 3. Defining DataLoader --------------------------------------------------------------------------------
    logging.info('Defining DataLoader.\n')
    dataset = H36_dataset_probabilistic_pose(subjectp = ['S11','S9'], conf = conf, is_train=False) #subjectp = ['S1', 'S5', 'S6', 'S7', 'S8']

    #4 Do one loop of prediction 2D pose
    vitpose_model = init_2D_prediction_models(conf, name = 'ViTPose')
    hourglass_model = init_2D_prediction_models(conf, name='Hourglass')
    training_pkg = init_2D_prediction_models(conf, name='Probabilistic')

    # 4. Run the training loop ------------------------------------------------------------------------------
    for trial in range(trials):

        print('\n\n\n\n######## Trial: {}/{} ########\n\n\n\n'.format(trial + 1, trials))

        # 4.a: Defining the network -------------------------------------------------------------------------

        # 4.b: Train the covariance approximation model
        train_obj_2D_prediction = Visualize_2D_prediction(dataset, conf, training_pkg, trial, isTrain=True)
        training_pkg_2D_prediction = train_obj_2D_prediction.run_model()



main()