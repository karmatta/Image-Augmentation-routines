import matplotlib.pyplot as plt
import cv2 
from PIL import Image, ImageEnhance
import pandas as pd
import glob
import os
import numpy as np

"""
PATH

path:        Path of the original images
path_label:  Path of the original image_mask
output_path: Folder Level 1 for the output to be stored 

TAGS

image:       label for enhanced image folder
image_mask:  label for corresponding masks folder
enh_1:       type of enhacnement to be added as a suffix to images
enh_2:       type of enhacnement to be added as a suffix to images

"""

path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous/train_data/train/sat_imgs/"

path_label = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous/train_data/label/sat_imgs_mask/"

output_path = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous_augmented/"


image = "Augmented_Images"
image_mask = "Augmented_Masks"
enh_1 = "Brightness"
enh_2 = "Contrast"


def enhancement(image_path,C,B):
    
    """ Performs the Main Enhancement Operations
    Args:
        image_path: path of the image whose brightness/contrast is to be adjusted This is of type string. 
        C:          Contrast factor of type.float32.
        B:          Brightness factor of type.float32.
    Returns:
        contrast:   Image with contrast enhanced
        brightness: Image with brightness enhanced
    """
    ###
    
    ## The brighntess/contrast can be adjusted below 1 so that the brightness/contrast 
    ## bis reduced. For enhacnement keep the values higher than 1.
    
    ## .convert is used to change the image mode
    
    im = Image.open(image_path).convert("RGBA")
    plt.imshow(im)
    brightness = ImageEnhance.Brightness(im)
    contrast   = ImageEnhance.Contrast(im)
    FACTOR_C = C
    FACTOR_B = B
    contrast = contrast.enhance(FACTOR_C)
    brightness = brightness.enhance(FACTOR_B)
    return contrast,brightness


img = cv2.imread("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/30005279950000_25796256.png")
plt.imshow(img)
path = "/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/30005279950000_25796256.png"

contrast_img,bright_img = enhancement(path,4,2)
plt.imshow(contrast_img)
plt.imshow(bright_img)  
contrast_img.save("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/output/Contrast_Image.png")
bright_img.save("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/output/Brightness_Image.png")



files = glob.glob(path + "*.png")
files_labels =  glob.glob(path_label +  "*.png")


for i in range(len(files)):

    print(i)
    
    ###
    ## Extract base filename and get UWI, product catalog id
    
    base =  os.path.basename(files[i])
    uwi =   base.split("_")[0]
    productid = base.split("_")[1].split(".")[0]
    
    ###
    ## call the enhancement function
    
    contrast,brightness = enhancement(files[i],4,2)
    
    
    ###
    ## create folders and subsequently save the images in the folder 
    
    
    image_B = output_path + enh_1 + "/" + image + "/"
    if not os.path.exists( image_B):
        os.makedirs(image_B) 
    brightness.save(image_B + uwi + "_" + productid + "_" + enh_1 + ".png")
    
    image_C = output_path + enh_2 + "/" + image + "/"
    if not os.path.exists( image_C):
        os.makedirs(image_C) 
    contrast.save(image_C + uwi + "_" + productid + "_" + enh_2 + ".png")
    
    base_label =  os.path.basename(files_labels[i])
    uwi_label =   base_label.split("_")[0]
    productid_label = base_label.split("_")[1].split(".")[0]
    
    
    label_image = cv2.imread(files_labels[i])
    
    label_B = output_path +  enh_1 + "/" + image_mask + "/"
    label_C = output_path +  enh_2 + "/" + image_mask + "/"
    
    if not os.path.exists( label_B):
        os.makedirs(label_B) 
    cv2.imwrite(label_B + uwi_label + "_" + productid_label + "_" + enh_1 + ".png",label_image)
    
    if not os.path.exists( label_C):
        os.makedirs(label_C) 
    cv2.imwrite(label_C + uwi_label + "_" + productid_label + "_" + enh_2 + ".png",label_image)
    
    


