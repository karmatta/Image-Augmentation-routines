import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
import glob
import scipy.misc


image_folder = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous/train_data/train/sat_imgs/"
label_folder = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous/train_data/label/sat_imgs_mask/"
output_folder = "/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous_augmented/"

image_files = glob.glob(image_folder + "*.png")
label_files = glob.glob(label_folder + "*.png")




augmentation_method = "blackwhite"
aug_img_path = output_folder + augmentation_method + "/Augmented_Images/"
aug_label_path = output_folder + augmentation_method + "/Augmented_Masks/"

for i in range(len(image_files)):
    
    print(i)
    
    if not os.path.exists(aug_img_path):
        os.makedirs(aug_img_path)    

    if not os.path.exists(aug_label_path):
        os.makedirs(aug_label_path)    

    
    image = cv2.imread(image_files[i],0)
    mask = cv2.imread(label_files[i],0)
    
    basename = os.path.basename(image_files[i]).split(".")[0]
    uwi_image = basename.split("_")[0]
    prodcatid_image = basename.split("_")[1]
    
    img_name = uwi_image + "_" + prodcatid_image + "_"+ augmentation_method +".png"
    cv2.imwrite(aug_img_path + img_name,image)
    
    basename_label = os.path.basename(label_files[i]).split(".")[0]
    uwi_image_label = basename_label.split("_")[0]
    prodcatid_image_label = basename_label.split("_")[1]
    
    img_name_label = uwi_image_label + "_" + prodcatid_image_label + "_"+ augmentation_method +".png"
    cv2.imwrite(aug_label_path + img_name_label,mask)
    
    
