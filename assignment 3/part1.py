import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Used Region Growing code from this repository
https://github.com/zjgirl/RegionGrowing-1  
"""


def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out

def region_growing(img, seed):
    list = []
    outimg = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    processed = []
    while(len(list) > 0):
        pix = list[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] != 0:
                outimg[coord[0], coord[1]] = 255
                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)
        cv2.imshow("progress",outimg)
        cv2.waitKey(1)
    return outimg

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

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

def create_masks():
    for directory in sorted(os.listdir(base)):
        # print(directory)
        image = np.load(base+"/"+directory+"/img.npy")
        seg = np.load(base+"/"+directory+"/seg.npy")
        seg_clip = np.clip(seg,0,1)
        # label = np.load('seg.npy')
        # label = np.clip(label,0,1)
        # label = np.expand_dims(label, axis=0)
        image = image[1]

        image = convert(image, 0, 255, np.uint8)
        # cv2.imshow("Window", imgu8)
        # cv2.waitKey()
        # exit()

        M = cv2.moments(seg_clip)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cY,cX)

        ret, img = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        hori = np.concatenate((image,img,seg_clip),axis=1)
        cv2.namedWindow('Input')
        cv2.imshow('Input', hori)
        # cv2.imshow('Seg_mask',seg_clip)
        # cv2.waitKey()

        out = region_growing(img, centroid)
        # cv2.imshow('Region Growing', out)
        save = np.save(base+"/"+directory+'/rgrow', out)
        save_cv2 = cv2.imwrite(base+"/"+directory+'/rgrow.jpg',out)
        # .save(base+"/"+directory+'/rgrow', out)
        # cv2.waitKey()
        cv2.destroyAllWindows()

create_masks()

############ visualaize region growth masks########## 
dice_scores = 0
num_examples = len(os.listdir(base))
for directory in sorted(os.listdir(base)):
    # print(directory)
    orig_image = np.load(base+"/"+directory+"/img.npy")
    img = orig_image[1] 
    rgrow = np.load(base+"/"+directory+"/rgrow.npy")
    seg = np.load(base+"/"+directory+"/seg.npy")
    seg_clip = np.clip(seg,0,1)

    test = rgrow.sum()
    # dice_score = compute_dice_coefficient(np.expand_dims(seg_clip,axis=0),np.expand_dims(rgrow,axis=0))
    dice_score = compute_dice_coefficient(seg_clip.astype('uint8'),rgrow)
    dice_scores+=dice_score
    plt.subplot(1,3,1),plt.imshow(img),plt.title('Image')
    plt.subplot(1,3,2),plt.imshow(rgrow),plt.title('R_Growth')
    # plt.text(2, 350, "Dice Score: "+str(round(100*dice_score,5)))
    plt.subplot(1,3,3),plt.imshow(seg_clip),plt.title('label')
    plt.show()

print("Avg Dice Score:", 100*dice_scores/num_examples)