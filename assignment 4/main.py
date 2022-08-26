import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

pt1 =[(153,194),(56,155),(196,114)]
pt2 = [(168, 180),(82,125),(217,100)]

sause = cv2.imread('./Assignment4_data/source.png')
sause_label = cv2.imread('./Assignment4_data/source_label.png')
s_rows, s_cols, s_ch = sause.shape
for pt in pt1:
    sause = cv2.circle(sause, pt, radius=3, color=(255, 0, 0), thickness=-1)

target = cv2.imread('./Assignment4_data/target.png')
target_label = cv2.imread('./Assignment4_data/target_label.png')
t_rows, t_cols, t_ch = target.shape
for pt in pt2:
    target = cv2.circle(target, pt, radius=3, color=(255, 0, 0), thickness=-1)

pts1 = np.float32([[153, 194],[56,155],[196,114]])

pts2 = np.float32([[168, 180],[82,125],[217,100]])

####### display the dataset #######

fig, axs = plt.subplots(1, 2, constrained_layout=True) 
fig.suptitle('Source & Target Image', fontsize=16)
axs[0].imshow(sause)
axs[0].set_title('Source')
axs[1].imshow(target)
axs[1].set_title('Target')  

####### obtain affine transform #######
M = cv2.getAffineTransform(pts1, pts2)

####### transfrom source image to target using AFFINE MATRIX #######
dst = cv2.warpAffine(sause, M, (t_cols, t_rows),flags=cv2.INTER_LINEAR)
fig, axs = plt.subplots(1, 3, constrained_layout=True)  

fig.suptitle('Source Image -->> Target Image Transform', fontsize=16)
axs[0].imshow(sause)
axs[0].set_title('Input')
  
axs[1].imshow(dst)
axs[1].set_title('Transformed')  

axs[2].imshow(target)
axs[2].set_title('Target')  

####### transfrom source label to target label using AFFINE MATRIX #######
dst = cv2.warpAffine(sause_label, M, (t_cols, t_rows),flags=cv2.INTER_NEAREST) 
fig, axs = plt.subplots(1, 3, constrained_layout=True)  

fig.suptitle('Source Label -->> Target Label Transform', fontsize=16)
axs[0].imshow(sause_label)
axs[0].set_title('Input')
  
axs[1].imshow(dst)
axs[1].set_title('Transformed')  

axs[2].imshow(target_label)
axs[2].set_title('Target')  
# plt.show()

def compute_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 

dsc = compute_dice_coefficient(dst,target_label)
print("DSC SCORE: ",dsc)

####### extract boundry points #######
sause_bgr = cv2.cvtColor(sause_label, cv2.COLOR_BGR2GRAY)
target_bgr = cv2.cvtColor(target_label, cv2.COLOR_BGR2GRAY)
contours_s, hier = cv2.findContours(sause_bgr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_t, hier = cv2.findContours(target_bgr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
k = -1
for i, cnt in enumerate(contours_s):
    if (hier[0, i, 3] == -1):
        k += 1
    cv2.drawContours(sause_label, [cnt], -1, (255,0,0), 2)
k = -1
for i, cnt in enumerate(contours_t):
    if (hier[0, i, 3] == -1):
        k += 1
    cv2.drawContours(target_label, [cnt], -1, (255,0,0), 2)

fig, axs = plt.subplots(1, 2, constrained_layout=True)  
fig.suptitle('Source & Target Contour', fontsize=16)
axs[0].imshow(sause_label)
axs[0].set_title('Source Contour')
  
axs[1].imshow(target_label)
axs[1].set_title('Target Contour')  
# plt.show()

def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    # Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
    #                [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
    #                [0,                    0,                   1          ]])
    Tr = np.array([[np.cos(0), np.sin(0),  0.09],
                   [np.sin(0), np.cos(0),  -0.1],
                   [0,         0,          1]])
    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateAffine2D(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T[0])
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T[0],[0,0,1])))
    return Tr[0:2]

ang = np.linspace(-np.pi/2, np.pi/2, 320)
a = np.array([ang, np.sin(ang)])
th = np.pi/2
rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
b = np.dot(rot, a) + np.array([[0.2], [0.3]])

contours_s = contours_s[0]
contours_t = contours_t[0]
contours_s = np.transpose(contours_s,(1,2,0))
contours_t = np.transpose(contours_t,(1,2,0))
contours_s = np.squeeze(contours_s,axis=0)
contours_t = np.squeeze(contours_t,axis=0)
cs_max = contours_s.max()
ct_max = contours_t.max()
contours_s = contours_s/cs_max
contours_t = contours_t/ct_max

M2 = icp(contours_s, contours_t, [0.1,  0.33, np.pi/2.2], 30)

src = np.array([contours_s.T]).astype(np.float32)
res = cv2.transform(src, M2)
# res = res.astype(np.uint8)
plt.figure()
plt.plot(contours_t[0],contours_t[1])
plt.plot(res[0].T[0], res[0].T[1], 'r.')
plt.plot(contours_s[0], contours_s[1])

####### transfrom source image to target using ICP #######
dst = cv2.warpAffine(sause, M2, (t_cols, t_rows),flags=cv2.INTER_LINEAR)
fig, axs = plt.subplots(1, 3, constrained_layout=True)  

fig.suptitle('Source Image -->> Target Image ICP Transform', fontsize=16)
axs[0].imshow(sause)
axs[0].set_title('Input')
  
axs[1].imshow(dst)
axs[1].set_title('Transformed')  

axs[2].imshow(target)
axs[2].set_title('Target') 
# plt.show()

####### transfrom source label to target label using ICP #######
dst = cv2.warpAffine(sause_label, M2, (t_cols, t_rows),flags=cv2.INTER_NEAREST) 
fig, axs = plt.subplots(1, 3, constrained_layout=True)  

fig.suptitle('Source Label -->> Target Label ICP Transform', fontsize=16)
axs[0].imshow(sause_label)
axs[0].set_title('Input')
  
axs[1].imshow(dst)
axs[1].set_title('Transformed')  

axs[2].imshow(target_label)
axs[2].set_title('Target')  

dsc = compute_dice_coefficient(dst,target_label)
print("DSC SCORE ICP: ",dsc)

plt.show()