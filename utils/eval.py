import torch
import numpy as np


def mpjpe(predicted, target):#For the general loss in the model
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))



#For mpjpe_p1, p_mpjpe, mpjpe_p2, we used https://github.com/Vegetebird/GraphMLP/blob/main/common/eval_cal.py
#https://github.com/Walter0807/MotionBERT/blob/main/lib/model/loss.py
#GraphMLP is used also for the 3D comparison of results
def mpjpe_p1(predicted, target):
    assert predicted.shape == target.shape
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)
    return dist

def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    normY = normY + 1e-8

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY 
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)

def mpjpe_p2(predicted, target):
    assert predicted.shape == target.shape
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)
    return dist


##Taken from  https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=17):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=17):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])



#Taken from https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation/blob/master/metrics.py
class PCK(object):
    """docstring for PCK"""
    def __init__(self, opts):
        super(PCK, self).__init__()
        self.opts = opts
        self.LB = -0.5 + 1e-8 if self.opts.TargetType == 'direct' else 0 + 1e-8

    def calc_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists

    def dist_acc(self, dists, thr=0.5):
         ''' Return percentage below threshold while ignoring values with a -1 '''
         dist_cal = np.not_equal(dists, -1)
         num_dist_cal = dist_cal.sum()
         if num_dist_cal > 0:
             return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
         else:
             return -1

    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

    def eval(self, pred, target, alpha=0.5):
        '''
        Calculate accuracy according to PCK,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        '''
        idx = list(range(16))
        norm = 1.0
        if True:
         h = self.opts.outputRes
         w = self.opts.outputRes
         norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        dists = self.calc_dists(pred, target, norm)

        acc = np.zeros((len(idx) + 1))
        avg_acc = 0
        cnt = 0

        for i in range(len(idx)):
         acc[i + 1] = self.dist_acc(dists[idx[i]])
         if acc[i + 1] >= 0:
             avg_acc = avg_acc + acc[i + 1]
             cnt += 1

        avg_acc = avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
         acc[0] = avg_acc
        return avg_acc,cnt

def compute_ap(groundtruth, prediction, threshold=0.5):
    assert groundtruth.shape == prediction.shape, "Groundtruth and prediction must have the same shape."
    batch_size, num_joints, _ = groundtruth.shape
    
    distances = torch.norm(groundtruth - prediction, dim=2)  # Shape: (batch_size, num_joints)
    
    bbox_min = torch.min(groundtruth, dim=1).values  # Shape: (batch_size, 2)
    bbox_max = torch.max(groundtruth, dim=1).values  # Shape: (batch_size, 2)
    bbox_sizes = torch.norm(bbox_max - bbox_min, dim=1)  # Shape: (batch_size,)
    bbox_sizes = torch.clamp(bbox_sizes, min=1e-6)  # Avoid division by zero

    normalized_distances = distances / bbox_sizes[:, None]  # Shape: (batch_size, num_joints)

    true_positives = (normalized_distances <= threshold)  # Shape: (batch_size, num_joints)

    precision_per_sample = true_positives.sum(dim=1) / num_joints  # Shape: (batch_size,)

    return precision_per_sample
    #average_precision = precision_per_sample.mean().item()
    #print(average_precision.shape)

    #return average_precision

def compute_X_sub(groundtruth, prediction, subjects, possible_subject = np.array([5,6])):
    """
    Computes the X-Sub metric. The accuracy for actions accross different subjects. 
    """

    #IN eval there are only two subjects: S9, S11
    groundtruth = groundtruth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    subjects = subjects.detach().cpu().numpy()

    acc = []
    for i in range(len(possible_subject)):
        gt = groundtruth[subjects==possible_subject[i]]
        preds = prediction[subjects==possible_subject[i]]
        acc.append(np.mean(gt==preds))
    
    acc = np.nanmean(acc)
    return acc

def compute_X_view(groundtruth, prediction, views, possible_view = np.array([0,1,2,3])):
    """
    Computes the X-view metric. The accuracy for actions accross different views. 
    """

    #In eval there are only four views
    groundtruth = groundtruth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    views = views.detach().cpu().numpy()

    acc = []
    for i in range(len(possible_view)):
        gt = groundtruth[views==possible_view[i]]
        preds = prediction[views==possible_view[i]]
        acc.append(np.mean(gt==preds))
    
    acc = np.nanmean(acc)
    return acc