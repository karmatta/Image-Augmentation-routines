import numpy as np
import pandas as pd
import cv2
import glob
import os
import Augmentor
import matplotlib.pyplot as plt




path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9/train_data/train/sat_imgs/"

path_label = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9/train_data/label/sat_imgs_mask/"

output_path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_augmented/"




 
def adjust_gamma(image, gamma=2.2):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
 
	
    return cv2.LUT(image, table)

#img = cv2.imread("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/30005279950000_25796256.png")
#plt.imshow(img)
#gamma_img = adjust_gamma(img)
#plt.imshow(gamma_img)  
#cv2.imwrite("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/output/Gamma_Image.png",gamma_img)






image_files =  glob.glob(path + "*.png")
image_mask_files =  glob.glob(path_label + "*.png")

for i in range(len(image_files)):
    
    print(i)
    
    img =  cv2.imread(image_files[i])
    basename = os.path.basename(image_files[i]).split(".")[0]
    uwi    =  basename.split("_")[0]
    prodcatid = basename.split("_")[1]
    
    adjusted_image = adjust_gamma(img)
    
    
    output_path_image = output_path + "GammaCorrection/Augmented_Images/"
    
    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)
     
     
    image_name = uwi + "_" + prodcatid + "_" + "GammaCorrection.png"   
        
    cv2.imwrite(output_path_image + image_name,adjusted_image)
    
    
    mask =  cv2.imread(image_mask_files[i])
    basename_mask = os.path.basename(image_mask_files[i]).split(".")[0]
    uwi_mask    =  basename_mask.split("_")[0]
    prodcatid_mask = basename_mask.split("_")[1]
    
    
    output_path_mask = output_path + "GammaCorrection/Augmented_Masks/"
    
    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)
     
     
    mask_name = uwi_mask + "_" + prodcatid_mask + "_" + "GammaCorrection.png"   
        
    cv2.imwrite(output_path_mask + image_name, mask)
    
    