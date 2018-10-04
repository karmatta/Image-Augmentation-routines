import numpy as np
import pandas as pd
import cv2
import glob
import os
import Augmentor
import matplotlib.pyplot as plt
from PIL import Image



def add_snow(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 2.5 
    snow_point=140 ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

img = cv2.imread("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/30005279950000_25796256.png")
plt.imshow(img)
snow_img = add_snow(img)
plt.imshow(snow_img)  
cv2.imwrite("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/output/Snow_Image.png",snow_img)


image_path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9/train_data/train/sat_imgs/"
label_path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9/train_data/label/sat_imgs_mask/"
output_path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_augmented/"

image_files =  glob.glob(image_path + "*.png")
image_mask_files =  glob.glob(label_path + "*.png")

for i in range(len(image_files)):
    
    print(i)
    
    img =  cv2.imread(image_files[i])
    basename = os.path.basename(image_files[i]).split(".")[0]
    uwi    =  basename.split("_")[0]
    prodcatid = basename.split("_")[1]
    
    adjusted_image = add_snow(img)
    
    
    output_path_image = output_path + "SNOW/Augmented_Images/"
    
    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)
     
     
    image_name = uwi + "_" + prodcatid + "_" + "SNOW.png"   
        
    cv2.imwrite(output_path_image + image_name,adjusted_image)
    
    
    mask =  cv2.imread(image_mask_files[i])
    basename_mask = os.path.basename(image_mask_files[i]).split(".")[0]
    uwi_mask    =  basename_mask.split("_")[0]
    prodcatid_mask = basename_mask.split("_")[1]
    
    
    output_path_mask = output_path + "SNOW/Augmented_Masks/"
    
    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)
     
     
    mask_name = uwi_mask + "_" + prodcatid_mask + "_" + "SNOW.png"   
        
    cv2.imwrite(output_path_mask + image_name, mask)
    
    

