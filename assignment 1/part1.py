import os
from pathlib import Path
import skimage
import pydicom
import cv2 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

path_LDCT = "L506_QD_3_1.CT.0003.0105.2015.12.22.20.45.42.541197.358793241.IMA"
path_NDCT = "L506_FD_3_1.CT.0001.0105.2015.12.22.20.19.39.34094.358586575.IMA"
main_path = "AAPM"

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])


data = []

for folder in tqdm(os.listdir(main_path)):
    path_LDCT = os.path.join(os.path.join(main_path,folder),"quarter_3mm")
    path_NDCT = os.path.join(os.path.join(main_path,folder),"full_3mm")
    # box_dir = os.path.join(main_path,folder,"1_box_filter")
    # gaussian_dir = os.path.join(main_path,folder,"2_gaussian_filter")
    # sharp_dir = os.path.join(main_path,folder,"3_sharpening_filter")
    # median_dir = os.path.join(main_path,folder,"4_median_filter")
    # Path(box_dir).mkdir(parents=True, exist_ok=True)
    # Path(gaussian_dir).mkdir(parents=True, exist_ok=True)
    # Path(sharp_dir).mkdir(parents=True, exist_ok=True)
    # Path(median_dir).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(boxf_dir):
    #     os.mkdir(box_dir)
    for img_ld,img_nd in zip(sorted(os.listdir(path_LDCT)),sorted(os.listdir(path_NDCT))):
        # print(img_ld)
        # print(img_nd)
        assert img_ld.split(".")[3] == img_nd.split(".")[3]
        # exit()
        imgLD_path = os.path.join(path_LDCT,img_ld)
        imgND_path = os.path.join(path_NDCT,img_nd)
        # print(imgLD_path)
        # print(imgND_path)
        
        imageLD = pydicom.dcmread(imgLD_path).pixel_array
        imageND = pydicom.dcmread(imgND_path).pixel_array
        image_box = cv2.boxFilter(imageLD, -1, (3,3))
        image_gauss = cv2.GaussianBlur(imageLD,(3,3),0)
        image_sharp = cv2.filter2D(imageLD,-1,sharpen)
        image_med = cv2.medianBlur(imageLD,3)

        orig_psnr = cv2.PSNR(imageND,imageLD)
        box_psnr = cv2.PSNR(imageND,image_box)
        gaus_psnr = cv2.PSNR(imageND,image_gauss)
        sharp_psnr = cv2.PSNR(imageND,image_sharp)
        med_psnr = cv2.PSNR(imageND,image_med)
        # print(orig_psnr,box_psnr,gaus_psnr,sharp_psnr,med_psnr)
        psnr_scores = [img_ld,orig_psnr,box_psnr,gaus_psnr,sharp_psnr,med_psnr]
        data.append(psnr_scores)
        
        ######### for Visualizing results ##########
        # plt.subplot(335),plt.imshow(imageND),plt.title('ImageND')
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(331),plt.imshow(image_box),plt.title('Box Filter-PSNR:'+str(round(box_psnr,2)))
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(333),plt.imshow(image_gauss),plt.title('Gaussian-PSNR:'+str(round(gaus_psnr,2)))
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(337),plt.imshow(image_sharp),plt.title('Sharp-PSNR:'+str(round(sharp_psnr,2)))
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(339),plt.imshow(image_med),plt.title('Median-PSNR:'+str(round(med_psnr,2)))
        # plt.xticks([]), plt.yticks([])
        # plt.show()

column_names = ["LD_Name", "ND-LD_psnr", "box_psnr","gaus_psnr","sharp_psnr","med_psnr"]
part1 = pd.DataFrame(data, columns=column_names)
part1.to_csv('part1_results.csv', index=False)  
print(part1)
# img_LDCT = pydicom.dcmread(path_LDCT).pixel_array
# img_NDCT = pydicom.dcmread(path_NDCT).pixel_array