import Augmentor
from shutil import *
import os
import re
import cv2 as cv
import pandas as pd
import numpy as np


"""

PATH

SRC_IMG: Path of the original images
SRC_MSK: Path of the original image mask


"""


SRC_IMG = ("/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous/train_data/train/sat_imgs/")
SRC_MSK = ("/home/affine/Downloads/TGS_OilPad_Identify/data_v9_ambiguous/train_data/label/sat_imgs_mask/")



###
## intializing augmentor pipeline with the path of the original image 
## ground truth is not be added as the images are Gamma Images hence all  have black masks

p = Augmentor.Pipeline(SRC_IMG)
p.ground_truth(SRC_MSK)


###
## crop by size mentioned in the parameters 
## random area cropping 

p.crop_by_size(probability = 1.0,width =256, height =  256, centre = False)
p.resize(probability = 1.0,width =  500, height = 500)
p.process()


def split_images(augmentation_method):
    
    """ Split images into Augmentation and Mask folder
    
    Args:
        augmentation_method: string type, type of augmentation being used
    
    """
    
    ###
    ## basic separation operation by identifying strings in the 
    ## path names 
    
    OUT = SRC_IMG + "/output/"
    AUG_MSK = "./Downloads/TGS_OilPad_Identify/data_v9_ambiguous_augmented/" + augmentation_method + "/Augmented_Masks/"
    AUG_IMG = "./Downloads/TGS_OilPad_Identify/data_v9_ambiguous_augmented/" + augmentation_method + "/Augmented_Images/"

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
    #rmtree("./Images/output")





def rename_files(augmentation_method,sample):
    """ Rename images to remove unnecessary suffixes in the name strings
    
    Args:
        augmentation_method: string type, type of augmentation being used
        sample: number of images to be used of type.int32
        
    """
    
    ### 
    ## Process to strip the file names of unnecessary strings and replace with 
    ## the standard nomenclature
    
    
    AUG_MSK = "./Downloads/TGS_OilPad_Identify/data_v9_ambiguous_augmented/" + augmentation_method + "/Augmented_Masks/"
    AUG_IMG = "./Downloads/TGS_OilPad_Identify/data_v9_ambiguous_augmented/" + augmentation_method + "/Augmented_Images/"

    file_names_masks = os.listdir(AUG_MSK)
    count = 0
    for i in file_names_masks[-sample:]:
        j = i.rsplit(".")
        k = j[0].rsplit("_")
        l = k[5] + "_" + k[6] + "_" + augmentation_method + ".png"
        if os.path.exists(AUG_MSK+l):
            count+=1
            l = k[5] + "_" + k[6] + "_" + augmentation_method + "(" + str(count) + ")"+ ".png"
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
        l = k[3] + "_" + k[4] + "_" + augmentation_method + ".png"
        if os.path.exists(AUG_IMG+l):
            count+=1
            l = k[3] + "_" + k[4] + "_" + augmentation_method + "(" + str(count) + ")"+ ".png"
        else:
            count = 0
        os.rename(AUG_IMG+i,AUG_IMG+l)


split_images("crop_resize")
rename_files("crop_resize",1900)
