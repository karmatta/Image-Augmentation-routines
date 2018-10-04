# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:33:05 2018

@author: nitish
"""

import Augmentor
from shutil import *
import os
import re
import cv2 as cv


# Defining Pipeline with Images and Ground Truth Labels
SRC_IMG = ("/home/affine/Downloads/TGS_OilPad_Identify/Gamma Images 300/New Gamma Images/")
SRC_MSK = ("/home/affine/Downloads/TGS_OilPad_Identify/Gamma Images 300/New Gamma Images Masks/")
p = Augmentor.Pipeline(SRC_IMG)
p.ground_truth(SRC_MSK)

# Fuction to Split images

def split_images(augmentation_method):
    OUT = SRC_IMG + "output/"
    AUG_MSK = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Masks/"
    AUG_IMG = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Images/"

    if not os.path.exists(AUG_MSK):
        os.makedirs(AUG_MSK)
    
    if not os.path.exists(AUG_IMG):
        os.makedirs(AUG_IMG)    

    files1= [d for d in os.listdir(OUT) if 'groundtruth' in d]
    files2= [d for d in os.listdir(OUT) if 'original' in d]

    for f in files1:
        src = OUT+f
        dst = AUG_MSK+f
        move(src,dst)

    for f in files2:
        src = OUT+f
        dst = AUG_IMG+f
        move(src,dst)
    rmtree(OUT)
        
# Function to rename files in (UWI_Product ID_Augmentation Method_.png) Format
def rename_files(augmentation_method,sample):
    AUG_MSK = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Masks/"
    AUG_IMG = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Images/"
    
    file_names_masks = os.listdir(AUG_MSK)
    count = 0
    for i in file_names_masks[-sample:]:
        j = i.rsplit(".")
        k = j[0].rsplit("_")
        l = k[4] + "_" + k[5] + "_" + augmentation_method + ".png"
        if os.path.exists(AUG_MSK+l):
            count+=1
            l = k[4] + "_" + k[5] + "_" + augmentation_method + "(" + str(count) + ")"+ ".png"
        else:
            count = 0
        os.rename(AUG_MSK+i,AUG_MSK+l)
        img = cv.imread(AUG_MSK+l,0)
        ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
        thresh1.dtype='uint8'
        cv.imwrite(AUG_MSK+l,thresh1)
        
    file_names_images = os.listdir(AUG_IMG)
    count = 0
    for i in file_names_images[-sample:]:
        j = i.rsplit(".")
        k = j[0].rsplit("_")
        l = k[2] + "_" + k[3] + "_" + augmentation_method + ".png"
        if os.path.exists(AUG_IMG+l):
            count+=1
            l = k[2] + "_" + k[3] + "_" + augmentation_method + "(" + str(count) + ")"+ ".png"
        else:
            count = 0
        os.rename(AUG_IMG+i,AUG_IMG+l)
        
      
def split_images_random_erasing(augmentation_method):
    OUT = SRC_IMG +"output/" 
    AUG_MSK = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Masks/"
    AUG_IMG = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Images/"

    if not os.path.exists(AUG_MSK):
        os.makedirs(AUG_MSK)
    
    if not os.path.exists(AUG_IMG):
        os.makedirs(AUG_IMG)    

    files2= [d for d in os.listdir(OUT) if 'original' in d]

    for f in files2:
        src = OUT+f
        dst = AUG_IMG+f
        move(src,dst)    
    rmtree(OUT)
    
def rename_files_random_erasing(augmentation_method,sample):
    AUG_MSK = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Masks/"
    AUG_IMG = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/" + augmentation_method + "/Augmented_Images/"
        
    file_names_images = os.listdir(AUG_IMG)
    count = 0
    for i in file_names_images[-sample:]:
        j = i.rsplit(".")
        k = j[0].rsplit("_")
        l = k[2] + "_" + k[3] + "_" + augmentation_method + ".png"
        m = k[2] + "_" + k[3] + ".png"
        if os.path.exists(AUG_IMG+l):
            count+=1
            l = k[4] + "_" + k[5] + "_" + augmentation_method + "(" + str(count) + ")"+ ".png"
        else:
            count = 0
        os.rename(AUG_IMG+i,AUG_IMG+l)
        m = k[2] + "_" + k[3] + ".png"
        copy(SRC_MSK+m,AUG_MSK+m)
        os.rename(AUG_MSK+m,AUG_MSK+l)

# User Input and Augmentation Operations
print("1. Rotate 2. Skew tilt 3. Random Distortion 4. Random Erasing 5. Rotate without Crop 6. Flip Random 7. Shear 8.Zoom")
choice = input("Enter your choice: ")

if choice == "1":
    augmentation_method = "rotate"
    probability = input("Probability:? ")
    max_left_rotation = input("Max_Left_Rotation:? ") 
    max_right_rotation = input("Max_Right_Rotation:? ")
    sample=int(input("Sample:?"))
    exec("p."+augmentation_method+"(probability="+str(probability)+", max_left_rotation=" +str(max_left_rotation)+", max_right_rotation="+str(max_right_rotation)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
    
elif choice == "2":
    
    augmentation_method = "skew_tilt"
    probability = input("Probability:? ")
    sample=int(input("Sample:? "))
    exec("p."+augmentation_method+"(probability="+str(probability)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
elif choice == "3":
    augmentation_method = "random_distortion"
    probability = input("Probability:? ")
    grid_width = input("Grid_Width:? ")
    grid_height = input("Grid_Height:? ")
    magnitude = input("Magnitude:? ")
    sample=int(input("Sample:? "))
    exec("p."+augmentation_method+"(probability="+str(probability)+", grid_width=" +str(grid_width)+", grid_height="+str(grid_height)+",magnitude="+str(magnitude)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
elif choice == "4":
    augmentation_method = "random_erasing"
    probability = input("Probability:? ")
    rectangle_area = input("Rectangle_area:? ")
    sample=int(input("Sample:? "))
    exec("p."+augmentation_method+"(probability="+str(probability)+", rectangle_area="+str(rectangle_area)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images_random_erasing(augmentation_method)
    rename_files_random_erasing(augmentation_method,sample)
        
elif choice == "5":
    augmentation_method = "rotate_without_crop"
    probability = input("Probability:? ")
    max_left_rotation = input("Max_Left_Rotation:? ") 
    max_right_rotation = input("Max_Right_Rotation:? ")
    sample=int(input("Sample:? "))
    expand = False
    exec("p."+augmentation_method+"(probability="+str(probability)+", max_left_rotation=" +str(max_left_rotation)+", max_right_rotation="+str(max_right_rotation)+", expand=" +str(expand)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
elif choice == "6":
    augmentation_method = "flip_random"
    probability = input("Probability:? ")
    sample=int(input("Sample:? "))
    exec("p."+augmentation_method+"(probability="+str(probability)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
elif choice == "7":
    augmentation_method = "shear"
    probability = input("Probability:? ")
    max_shear_left = input("Max_Shear_Left:? ") 
    max_shear_right = input("Max_Shear_Right:? ")
    sample=int(input("Sample:? "))
    exec("p."+augmentation_method+"(probability="+str(probability)+", max_shear_left=" +str(max_shear_left)+", max_shear_right="+str(max_shear_right)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
elif choice == "8":
    augmentation_method = "zoom"
    probability = input("Probability:? ")
    min_factor = input("Min_factor:? ") 
    max_factor = input("Max_Factor:? ")
    sample=int(input("Sample:? "))
    exec("p."+augmentation_method+"(probability="+str(probability)+", min_factor=" +str(min_factor)+", max_factor="+str(max_factor)+")")
    #exec("p.sample("+str(sample)+")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
elif choice == "9":
    augmentation_method = "gaussian_distortion"
    probability = input("Probability:? ")
    grid_width = input("Grid Width:? ")
    grid_height = input("Grid Height:? ")
    magnitude = input("Magnitude:? ")
    corner = "bell"
    sample = int(input("Sample:?"))
    exec("p." + augmentation_method + "(probability=" + str(probability) + ", grid_width = " + str(grid_width) + ",grid_height = " + str(grid_height) + ",magnitude = " + str(magnitude) + ",corner = 'bell' "  + ", method = 'in'" +")" )
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method,sample)
    
elif choice == "10":
    augmentation_method = "histogram_equalisation"
    probability = input("Probability:? ")
    sample = int(input("Sample:?"))
    exec("p." + augmentation_method + "(probability=" + str(probability) + ")")
    p.process()
    split_images(augmentation_method)
    rename_files(augmentation_method, sample)
    

    
else:
    print("Ivalid Input!")
    


