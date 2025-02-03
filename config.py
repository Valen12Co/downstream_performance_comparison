import os
import yaml
import logging
from pathlib import Path


class ParseConfig(object):
    """
    Loads and returns the configuration specified in configuration.yml
    """
    def __init__(self) -> None:

        # 1. Load the configuration file ------------------------------------------------------------------------------
        try:
            f = open('configuration.yml', 'r')
            conf_yml = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        except FileNotFoundError:
            logging.warning('Could not find configuration.yml')
            exit()


        # 2. Initializing ParseConfig object --------------------------------------------------------------------------
        self.trials = conf_yml['trials']
        self.model_2D_prediction = conf_yml['model_2D_prediction']
        self.model_2D_dataset = conf_yml['model_2D_dataset']
        self.model_3D_prediction = conf_yml['model_3D_prediction']
        self.model_prob_pose = conf_yml['model_prob_pose']
        self.dataset = conf_yml['dataset']
        self.experiment_settings = conf_yml['experiment_settings']
        self.architecture = conf_yml['architecture']
        self.use_hessian = conf_yml['use_hessian']
        self.load_images = conf_yml['load_images']


        # 3. Extra initializations based on configuration chosen ------------------------------------------------------
        # Number of convolutional channels for AuxNet
        self.architecture['aux_net']['channels'] = [self.architecture['hourglass']['channels']] * 7
        self.architecture['aux_net']['spatial_dim'] = [64, 32, 16, 8, 4, 2, 1]
        self.architecture['joint_prediction']['channels'] = [self.architecture['hourglass']['channels']] * 7
        self.architecture['joint_prediction']['spatial_dim'] = [64, 32, 16, 8, 4, 2, 1]

        # Number of heatmaps (or joints)
        self.experiment_settings['num_hm'] = 17
        self.architecture['hourglass']['num_hm'] = 17
        self.architecture['aux_net']['num_hm'] = 17

        # Number of output nodes for the aux_network
        self.architecture['aux_net']['fc'].append(((2 * self.architecture['aux_net']['num_hm']) ** 2) + 2)

        # 4. Create directory for model save path ----------------------------------------------------------------------
        self.experiment_name = conf_yml['experiment_name']
        i = 1
        model_save_path = os.path.join(conf_yml['save_path'], self.experiment_name + '_' + str(i))
        while os.path.exists(model_save_path):
            i += 1
            model_save_path = os.path.join(conf_yml['save_path'], self.experiment_name + '_' + str(i))

        logging.info('Saving the model at: ' + model_save_path)

        os.makedirs(model_save_path, exist_ok=True)

        self.save_path = model_save_path