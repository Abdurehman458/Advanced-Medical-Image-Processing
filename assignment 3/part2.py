import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import graphcut

"""
Used Graph Cut code from this repository
https://github.com/ishank26/svm-GraphCut 

"""

def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 


base = "./selected_ds"

def do_graphcut():
    for directory in sorted(os.listdir(base)):
        pth = os.path.join(base,directory)
        print(pth)
        graphcut.graph_cut(pth)

do_graphcut()

def cal_dice_score():
    dice_scores = 0
    num_examples = len(os.listdir(base))
    for directory in sorted(os.listdir(base)):
        pth = os.path.join(base,directory)
        seg = np.load(pth+"/seg.npy")
        seg_clip = np.clip(seg,0,1)
        graph_cut = cv2.imread(pth+"/graphcut.jpg",0)
        cv2.imshow("graph",graph_cut)
        cv2.waitKey()
        dice_score = compute_dice_coefficient(seg_clip.astype('uint8'),graph_cut)
        dice_scores+=dice_score

    print("Avg Dice Score:", 100*dice_scores/num_examples)

cal_dice_score()
