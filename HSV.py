import numpy as np
import pandas as pd
import cv2
import glob
import os
import Augmentor
import matplotlib.pyplot as plt
from PIL import Image




path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9/train_data/train/sat_imgs/"

path_label = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9/train_data/label/sat_imgs_mask/"

output_path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_augmented/"


def HSV(image,hue_shift,saturation_scale,saturation_shift,value_scale,value_shift):
    
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv = hsv/255

    hsv[..., 0] += np.random.uniform(-hue_shift, hue_shift)
    hsv[..., 1] *= np.random.uniform(1 / (1 + saturation_scale), 1 + saturation_scale)
    hsv[..., 1] += np.random.uniform(-saturation_shift, saturation_shift)
    hsv[..., 2] *= np.random.uniform(1 / (1 + value_scale), 1 + value_scale)
    hsv[..., 2] += np.random.uniform(-value_shift, value_shift)

    hsv.clip(0, 1, hsv)
    hsv = np.uint8(np.round(hsv * 255.))

    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)




#img = cv2.imread("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/30005279950000_25796256.png")
#plt.imshow(img)
#adjusted_image = HSV(img, 0.5,0.5,0.5,0.5,0.5)
#plt.imshow(adjusted_image)  
#cv2.imwrite("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/output/HSV_Image.png",adjusted_image)



image_files =  glob.glob(path + "*.png")
image_mask_files =  glob.glob(path_label + "*.png")

for i in range(len(image_files)):
    
    print(i)
    
    img =  cv2.imread(image_files[i])
    basename = os.path.basename(image_files[i]).split(".")[0]
    uwi    =  basename.split("_")[0]
    prodcatid = basename.split("_")[1]
    
    adjusted_image = HSV(img, 0.5,0.5,0.5,0.5,0.5)
    
    
    output_path_image = output_path + "HSV/Augmented_Images/"
    
    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)
     
     
    image_name = uwi + "_" + prodcatid + "_" + "HSV.png"   
        
    cv2.imwrite(output_path_image + image_name,adjusted_image)
    
    
    mask =  cv2.imread(image_mask_files[i])
    basename_mask = os.path.basename(image_mask_files[i]).split(".")[0]
    uwi_mask    =  basename_mask.split("_")[0]
    prodcatid_mask = basename_mask.split("_")[1]
    
    
    output_path_mask = output_path + "HSV/Augmented_Masks/"
    
    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)
     
     
    mask_name = uwi_mask + "_" + prodcatid_mask + "_" + "HSV.png"   
        
    cv2.imwrite(output_path_mask + image_name, mask)
    
    