# downstream_performance_comparison
This repo contains the framework mentioned in the `Effect of probabilistic pose estimator on downstream tasks` semester project conducted during Fall 2024. For the main results please look directly at the report. The final report can be found in the main tree under `report-Perret.pdf`.
During this project, we evaluated the effect of a probabilistic pose estimator on the downstream tasks. We based our framework on the Human3.6M dataset for 2D-3D Human Pose lifting. We also tried action recognition.

## Code Structure
There are three main components in this repo: dataloader, models and main files. Many files are present in the main repo, they were used to either train, get the predictions, or visualized the results. The `visualisationGIF.py` is a file to fully visualize the different models of the key points on the images in a small GIF, however, it never was finished.

The structure of the code is as follows:
```
├── dataloader
│   ├── dataloader.py
│   ├── utils.py
├── models
│   ├── GraphMLP
│   ├── SkateFormer
│   ├── auxiliary
│   ├── joint_prediction
│   ├── stacked_hourglass
│   ├── vit_pose
├── utils
├── torchlight
├── config
main.py
train2D.py
train2D_prob_pose.py
config.py
configuration.yml
```
## Data Folder
In the datasets/Human3.6 repo on the RCP cluster of the lab, one can find the different files used for the dataloader. `Videos` includes the frames of the various videos, `npz` files contain the 2D and 3D keypoints, and `npy` repo contains the npy files containing the predicted 2D key points from the model we used. Those npy files will be used for the key points lifting. The npy file used can be found in `dataloader/dataloader.py` and contains all the classes used for the Human3.6M data-loading: 2D keypoint recognition, 2D-3D lifting, and action recognition. During the run of the various scripts, you may generate data or use already existing data, make sure to update the location of those files in the dataloader file.
## Run code
To obtain results there are four steps to follow.
1) `train2D.py` to train a 2D keypoint model, if you want to use the probabilistic pose model use `train2D_prob_pose.py`. You can change if you want to predict with the VitPose, StackedHourglass in the `model_2D_prediction` parameter in `configuration.yml`. For the probabilistic pose, you should precise under `model_prob_pose` which base model you want to use. You should be careful to have the folder to save the experiment in the save_path variable.
2) Run `get2Dpredictions.py`, or `get2Dpredictions_prob_pose.py` if you use the probabilistic pose estimator. For the first file, there are two lines to uncomment under model_path and load_state_dict. Put the link to the trained model here. For the probabilistic pose estimator change the `file_path = os.path.join("xxx/downstream_performance_comparison/InThis/TestProbPose", "training_pkg_0.pt")` line. Again change accordingly the `model_2D_prediction` variable to have it similar to point 1.
3) Put the name of the saved npy file in the dataloader file. It should be saved under data_file_2d in read_data in the Human36_dataset_3D class. Once this is done you can run `main.py`. Change accordingly to parts 1) and 2) `model_2D_prediction` and `model_prob_pose`. You can switch between 2D-3D lifting and action recognition with `model_3D_prediction`. However during our tests, we saw that the Human3.6M dataset is not adapted for action recognition, the result may be as low as 10% on the X-sub metric. The run model may be saved under `checkpoint`, a folder that might have been created automatically. you may also use GroundTruth if you want to select the 2D ground truth keypoints provided by the Human3.6M dataset for training.
4) Change the model file name in the script either under the respective model_path in the __init__ function in the Evaluate_3D_prediction class. Run `get3Dprediction.py`. Like the previous part be careful of the parameters you select. This time `model_2D_prediction` corresponds to the train set, (so the train set that was used for the model you trained previously and `model_2D_dataset` specifies the keypoints you want to use for the evaluation set. This code only works for 2D-3D lifting.

Note that the file locations you changed, should be changed in the visualisation files.
## Configuration.yml file clarification
`configuration.yml` contains fours main variables that are important:
* model_2D_prediction: for the 2D prediction: which model you want to use, for the 3D prediction the training set used
* model_2D_dataset: only for 3D prediction: the evaluation set
* model_3D_prediction: the model you want to use based on the 2D keypoints: you will choose between action recognition ('SkateFormer') and 2D-3D lifting (GraphMlp).
* model_prob_pose: for the 2D prediction: the base model for the probabilistic pose estimator

The possible choices are written directly in the file, choose the one you want.
## Main Files
The main files are the following, if you want the main instruction on how to run the experiment please read the above instructions.
* `get2Dpredictions.py`: generate 2D keypoints based on images
* `get2Dpredictions_prob_pose.py`: generate 2D keypoints based on images with the probabilistic pose estimator
* `get3Dprediction.py`: evaluate the trained 2D-3D lifting model
* `main.py`: train either a 2D-3D lifting model or action recognition model on 2D keypoints
* `train2D.py`: train a 2D Human keypoints estimator model
* `train2D_prob_pose.py`: train a 2D Human keypoints estimator model with the probabilistic pose estimator
* `visualisation2D.py`: visualize 2D keypoints prediction
* `visualisation2Dprobpose.py`: visualize 2D keypoints prediction with the probabilistic pose estimator
*  `visualisation3D.py`: visualize 3D keypoints prediction
* `visualisationGIF.py`: visualize as a GIF the keypoints prediction, however this file is not finished.
## References
This code is based on two main repos. Certain functions where taken from other repos but cited at the beginning of the function used for. The four main ones:
- `https://github.com/vita-epfl/TIC-TAC/` for the probabilistic pose estimator [1]
- `https://github.com/Vegetebird/GraphMLP/` for the 2D-3D lifting model
- `https://github.com/KAIST-VICLab/SkateFormer/` for the action recognition model
- `https://github.com/RHnejad/`, the user provided usefull tools for visualisation and 2D-3D lifting
Finally, an other part of the project can be found on `https://github.com/Valen12Co/probabilistic_pose/`. The probabilistic pose estimator changes where made there first.

[1] SHUKLA, Megh, SALZMANN, Mathieu, et ALAHI, Alexandre. TIC-TAC: A Framework for Improved Covariance Estimation in Deep Heteroscedastic Regression. In: Proceedings of the 41st International Conference on Machine Learning (ICML) 2024. 2024.
