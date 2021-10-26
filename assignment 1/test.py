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

transform_algorithms = [
    # ('Orthogonal Matching Pursuit\n1 atom', 'omp',
    #  {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 2}),
    # ('Least-angle regression\n5 atoms', 'lars',
    #  {'transform_n_nonzero_coefs': 5}),
    # ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})
    ]

def train_dict(ND_path, iterations=10):
    for it in range(iterations):
        buffer = []
        for i,path in enumerate(tqdm(ND_path)):
            # if i > 5:break
            img = pydicom.dcmread(path).pixel_array
            # uint8 = img_as_ubyte(exposure.rescale_intensity(img))
            # uint8 = uint8/255
            # uint8 = uint8[::4, ::4] + uint8[1::4, ::4] + uint8[::4, 1::4] + uint8[1::4, 1::4]
            # uint8 /= 4.0
            img = img/img.max()
            # io.imshow(imageND)
            # print('Extracting reference patches...')
            # t0 = time()
            data = extract_patches_2d(img, patch_size, max_patches=400) # (256036,7,7)
            data = data.reshape(data.shape[0], -1) # (256036,49)
            buffer.append(data)
            if i % 3 == 0 and i !=0:
                data = np.concatenate(buffer, axis=0)
                data -= np.mean(data, axis=0)
                data /= np.std(data, axis=0)
                t0 = time()
                # V = dico.fit(data).components_ 
                V = dico.partial_fit(data) #learned dictionary (100,49)
                dt = time() - t0
                # print('done in %.2fs.' % dt)
                buffer = []

train_dict(imgND_path,iterations=10)
V = dico.components_ 
np.save("dict",V)
# V = np.load("dict.npy")

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
        # if i > 1:break
        # exit()
        imgLD_path = os.path.join(path_LDCT,img_ld)
        imgND_path = os.path.join(path_NDCT,img_nd)
        # print(imgLD_path)
        # print(imgND_path)
        imageLD = pydicom.dcmread(imgLD_path).pixel_array
        imageND = pydicom.dcmread(imgND_path).pixel_array
        # imageLD = imageLD/imageLD.max()
        # imageND = imageND/imageND.max()
        
        # imageLD = img_as_ubyte(exposure.rescale_intensity(imageLD))
        # imageLD = imageLD/255
        # imageND = img_as_ubyte(exposure.rescale_intensity(imageND))
        # imageND = imageND/255
    
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
        # print("%.2f" % diff_wdict,"%.2f" % LDND_diff,NDvsLD,psnrND,psnrLD)
        to_add = [NDvsLD,psnrND,diff_wdict]
        dict_result.append(to_add)
        
columns =['NDvsLD','Dict_PSNR','ND-dict_reconstruction']  
dict_result = pd.DataFrame(dict_result,columns=columns)
sub = sub.join(dict_result)
sub.to_csv('final_results.csv', index=False)  
        
        
        # plt.figure(figsize=(10, 10))
        # plt.subplot(2, 2, 1),plt.title('Reconstructed'),plt.imshow(reconstruct),plt.xticks(()),plt.yticks(())
        # plt.subplot(2, 2, 2),plt.title('ND'),plt.imshow(imageND),plt.xticks(()),plt.yticks(())
        # plt.subplot(2, 2, 4),plt.title('LD'),plt.imshow(imageLD),plt.xticks(()),plt.yticks(())
        # plt.show()
        # exit()
        