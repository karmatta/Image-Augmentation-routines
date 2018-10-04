######################################
######################################

### Salt and Pepper Noise Addition ###

######################################
######################################


import pandas as pd 
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import shutil


image_folder = "/home/affine/Downloads/TGS_OilPad_Identify/Gamma Images 300/New Gamma Images/"
label_folder = "/home/affine/Downloads/TGS_OilPad_Identify/Gamma Images 300/New Gamma Images Masks/"
output_folder = "/home/affine/Downloads/TGS_OilPad_Identify/GammaImages_Augmented_v1/"



label = "s_p_noise_black_white"
mask  = "mask"

def s_p_noise_black_white(image,amount):
    
    """ Salt and Pepper Noise Addition with actual Black and White Pixels
    
    Args:
        image:   the image, numpy array, on which the salt and pepper noise is to be added
        amount:  the percentage of the image pixels to be used for S&P noise
    Returns:
        out:     augmented or enhanced image with S&P noise  
        
    """
    ###
    ## The black and white S&P is added by making all the 3 channels of have the same
    ## value of 255 or 0
    
    row,col,ch = image.shape
    s_vs_p = 0.6
    amount = amount
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
           for i in image[:,:,0].shape]
    for l in range(len(coords[0])):
        out[coords[0][l],coords[1][l],:] = 255
        
    
    
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image[:,:,0].shape]
    for l in range(len(coords[0])):
        out[coords[0][l],coords[1][l],:] = 0
    
    
    
    return out


def s_p_noise_rgb(image,amount):
    
    """ Salt and Pepper Noise Addition with actual Black and White Pixels
    
    Args:
        image:   the image, numpy array, on which the salt and pepper noise is to be added
        amount:  the percentage of the image pixels to be used for S&P noise
    Returns:
        out:     augmented or enhanced image with S&P noise  
        
    """
    
    ###
    ## Randomly select coordinates(channels included) and impute 255 or 0 
    ## to produce a colored S&P noise in the image
    
    row,col,ch = image.shape
    s_vs_p = 0.6
    amount = amount
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
           for i in image.shape]
    out[coords] = 255    
    
    
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0 
    
    
    return out




input_image_path = glob.glob(image_folder + "*.png")
label_image_path = glob.glob(label_folder + "*.png")
count = 0

#img = cv2.imread("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/30005279950000_25796256.png")
#plt.imshow(img)
#sp_img = s_p_noise_rgb(img,0.05)
#plt.imshow(sp_img)  
#cv2.imwrite("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/output/Salt_Pepper_COLOR_Image.png",sp_img)
#





###
## The loop below goes thorugh all the files in the input_image_path folder and 
## adds the noise and writes the images to folder

for i in range(len(input_image_path)):
    
    base = os.path.basename(input_image_path[i])
    uwi_name = base.split("_")[0]
    product_catalog_id = base.split("_")[1]
    product_catalog_id = product_catalog_id.split(".")[0] 
    
    img = cv2.imread(input_image_path[i])
    
    image = s_p_noise_black_white(img,0.05)
    
    output_folder_final = output_folder + label + "/Augmented_Images/" 
    
    if not os.path.exists(output_folder_final):
        os.makedirs(output_folder_final) 
        
    img_name =  uwi_name + "_" + product_catalog_id + "_"+ label
    cv2.imwrite(output_folder_final + img_name + ".png",image)
    count =  count + 1
    print(count)
    
    label_output_folder = output_folder + label + "/Augmented_Masks/"
    
    label_image = cv2.imread(label_image_path[i])
    
    if not os.path.exists(label_output_folder):
        os.makedirs(label_output_folder) 

    base_label = os.path.basename(label_image_path[i])
    uwi_name_label = base_label.split("_")[0]
    product_catalog_id_label = base_label.split("_")[1]
    product_catalog_id_label = product_catalog_id_label.split(".")[0] 
    

    img_name_label = uwi_name_label + "_" + product_catalog_id_label + "_" + label
    cv2.imwrite(label_output_folder + img_name_label  + ".png",label_image)

#shutil.copytree(label_folder,output_folder + label+ "_" +mask)

    

