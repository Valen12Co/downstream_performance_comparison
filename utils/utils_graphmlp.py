import argparse
import os
import math
import time
import logging
import copy

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument('--d_hid', default=1024, type=int)
    parser.add_argument('--token_dim', default=256, type=int)
    parser.add_argument('--dataset', type=str, default='h36m')
    parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb', type=str)
    parser.add_argument('--data_augmentation', type=int, default=1)
    parser.add_argument('--reverse_augmentation', type=bool, default=False)
    parser.add_argument('--test_augmentation', type=bool, default=True)
    parser.add_argument('--crop_uv', type=int, default=0)
    parser.add_argument('--root_path', type=str, default='dataset/')
    parser.add_argument('--actions', default='*', type=str)
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--subset', default=1, type=float)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nepoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_refine', type=float, default=1e-5)
    parser.add_argument('--lr_decay_large', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--pad', type=int, default=0) 
    parser.add_argument('--refine', action='store_false')
    parser.add_argument('--refine_reload', action='store_false')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--previous_dir', type=str, default='')
    parser.add_argument('--n_joints', type=int, default=17)
    parser.add_argument('--out_joints', type=int, default=17)
    parser.add_argument('--out_all', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--previous_best', type=float, default= math.inf)
    parser.add_argument('--previous_name', type=str, default='')
    parser.add_argument('--previous_refine_name', type=str, default='')

    args = parser.parse_args()

    args.pad = (args.frames-1) // 2

    args.root_joint = 0
    if args.dataset == 'h36m':
        args.subjects_train = 'S1,S5,S6,S7,S8'
        args.subjects_test = 'S9,S11'

        args.n_joints = 17
        args.out_joints = 17

        args.joints_left = [4, 5, 6, 11, 12, 13] 
        args.joints_right = [1, 2, 3, 14, 15, 16]

    if args.train:
        logtime = time.strftime('%m%d_%H%M_%S_')

        args.checkpoint = 'checkpoint/' + logtime + '%d'%(args.frames) + '%s'%('_refine' if args.refine else '')
        os.makedirs(args.checkpoint, exist_ok=True)

        args_write = dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('_'))
        file_name = os.path.join(args.checkpoint, 'configs.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args_write.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)

    return args



class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


##From https://github.com/TaatiTeam/MotionAGFormer/blob/master/utils/data.py#L139 ####
def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data