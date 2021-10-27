from time import time
import numpy as np
import scipy as sp
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import os
import random
from pathlib import Path
import skimage
from skimage import io
import pydicom
import cv2 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import exposure, img_as_ubyte

main_path = "AAPM"
patch_size = (5, 5)
nd_list = []
batch_size=100
max_patches = 1000
n_batches = max_patches // batch_size

imgND_path=[]

dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)

for folder in os.listdir(main_path):
    # path_LDCT = os.path.join(os.path.join(main_path,folder),"quarter_3mm")
    path_NDCT = os.path.join(os.path.join(main_path,folder),"full_3mm")
    for i,img_nd in enumerate(sorted(os.listdir(path_NDCT))):
        # if i > 1: break
        path = os.path.join(path_NDCT,img_nd)
        imgND_path.append(path)
random.seed(20)
random.shuffle(imgND_path)

transform_algorithms = [('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 2})]

def train_dict(ND_path, iterations=10):
    for it in range(iterations):
        buffer = []
        for i,path in enumerate(tqdm(ND_path)):
            # if i > 5:break
            img = pydicom.dcmread(path).pixel_array
            img = img/img.max()

            data = extract_patches_2d(img, patch_size, max_patches=max_patches) # (256036,7,7)
            data = data.reshape(data.shape[0], -1) # (256036,49)

            for i in range(n_batches):
                batch = data[i * batch_size:(i + 1) * batch_size]
                batch -= np.mean(batch, axis=0)
                batch /= np.std(batch, axis=0)
                # t0 = time()
                V = dico.partial_fit(batch) #learned dictionary (100,49)
                # dt = time() - t0
                # print('done in %.2fs.' % dt)


train_dict(imgND_path,iterations=1)
V = dico.components_ 
np.save("test3_dict",V)
# V = np.load("test3_dict.npy")


### for Visualizing DICTIONARY #####

# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(V[:100]):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle('Dictionary learned from patches\n',fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
# plt.show()
# exit()

# #############################################################################
# Extract noisy patches and reconstruct them using the dictionary
sub = pd.read_csv('part1_results.csv')
dico.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=2)
dict_result=[]

for folder in os.listdir(main_path):
    path_LDCT = os.path.join(os.path.join(main_path,folder),"quarter_3mm")
    path_NDCT = os.path.join(os.path.join(main_path,folder),"full_3mm")
    for i,(img_ld,img_nd) in enumerate(tqdm(zip(sorted(os.listdir(path_LDCT)),sorted(os.listdir(path_NDCT))))):
        assert img_ld.split(".")[3] == img_nd.split(".")[3]
        imgLD_path = os.path.join(path_LDCT,img_ld)
        imgND_path = os.path.join(path_NDCT,img_nd)

        imageLD = pydicom.dcmread(imgLD_path).pixel_array
        imageND = pydicom.dcmread(imgND_path).pixel_array
        
        imageLD = imageLD/65535
        imageND = imageND/65535
        data = extract_patches_2d(imageLD, patch_size)
        data = data.reshape(data.shape[0], -1)
        intercept = np.mean(data, axis=0)
        data -= intercept
        code = dico.transform(data) #estimate sparse solution
        patches = np.dot(code, V)
        patches += intercept
        patches = patches
        patches = patches.reshape(len(data), *patch_size)
        reconstruct = reconstruct_from_patches_2d(patches, (512,512))
        dict_diff = reconstruct - imageND
        LDND_diff = imageLD - imageND
        diff_wdict = np.sqrt(np.sum(dict_diff ** 2))
        LDND_diff = np.sqrt(np.sum(LDND_diff ** 2))

        psnrND = cv2.PSNR(imageND*65535,reconstruct*65535)
        psnrLD = cv2.PSNR(imageLD*65535,reconstruct*65535)
        NDvsLD = cv2.PSNR(imageLD*65535,imageND*65535)

        to_add = [NDvsLD,psnrND,diff_wdict]
        dict_result.append(to_add)
        
columns =['NDvsLD','Dict_PSNR','ND-dict_reconstruction']  
dict_result = pd.DataFrame(dict_result,columns=columns)
sub = sub.join(dict_result)
sub.to_csv('part3_partial_fit.csv', index=False)  