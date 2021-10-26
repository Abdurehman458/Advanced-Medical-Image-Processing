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
patch_size = (7, 7)
nd_list = []
batch_size=100
max_patches = 10000
n_batches = max_patches // batch_size

imgND_path=[]

# #############################################################################
# Extract noisy patches and reconstruct them using the dictionary
sub = pd.read_csv('part1_results.csv')

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
        imgND_max = imageND.max()
        imgLD_max = imageLD.max()
        imageLD = imageLD/imgLD_max
        imageND = imageND/imgND_max

        img =imageND
        data = extract_patches_2d(img, patch_size) # (256036,7,7)
        data = data.reshape(data.shape[0], -1) # (256036,49)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
        V = dico.fit(data).components_ 

        data = extract_patches_2d(imageLD, patch_size)
        data = data.reshape(data.shape[0], -1)
        intercept = np.mean(data, axis=0)
        data -= intercept
        dico.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=2)
        code = dico.transform(data) #estimate sparse solution
        patches = np.dot(code, V)
        patches += intercept
        patches = patches
        patches = patches.reshape(len(data), *patch_size)
        reconstruct = reconstruct_from_patches_2d(patches, (512,512))
        
        dict_diff = reconstruct - imageND
        LDND_diff = imageLD - imageND

        diff_wdict = np.sqrt(np.sum(dict_diff ** 2))
        psnrND = cv2.PSNR(imageND*imgND_max,reconstruct*imgLD_max)
        psnrLD = cv2.PSNR(imageLD*imgLD_max,reconstruct*imgLD_max)
        NDvsLD = cv2.PSNR(imageLD*imgLD_max,imageND*imgND_max)

        # print(NDvsLD,psnrND,diff_wdict)
        to_add = [NDvsLD,psnrND,diff_wdict]
        dict_result.append(to_add)

        ######### for Visualizing results ##########
        plt.subplot(221),plt.imshow(imageND),plt.title('ImageND')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222),plt.imshow(imageLD),plt.title('ImageLD-PSNR:'+str(round(NDvsLD,2)))
        plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(reconstruct),plt.title('Reconstruct-PSNR:'+str(round(psnrND,2)))
        plt.xticks([]), plt.yticks([])
        plt.show()
        
columns =['NDvsLD','Dict_PSNR','ND-dict_reconstruction']  
dict_result = pd.DataFrame(dict_result,columns=columns)
sub = sub.join(dict_result)
sub.to_csv('part3_fit.csv', index=False)  