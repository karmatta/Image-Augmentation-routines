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



def generate_shadow_coordinates(imshape, no_of_shadows=2):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(3,7)): ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices


def add_shadow(img,no_of_shadows=1):
    #image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image = img.copy()
    mask = np.zeros_like(image) 
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape,no_of_shadows) #3 getting list of shadow vertices
    for vertices in vertices_list: 
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    
    image[:,:,:][mask[:,:,0]==255] = image[:,:,:][mask[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered 
    #image_RGB = cv2.cvtColor(image,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image

image_files =  glob.glob(path + "*.png")
image_mask_files =  glob.glob(path_label + "*.png")

for i in range(len(image_files)):
    
    print(i)
    
    img =  cv2.imread(image_files[i])
    basename = os.path.basename(image_files[i]).split(".")[0]
    uwi    =  basename.split("_")[0]
    prodcatid = basename.split("_")[1]
    
    adjusted_image = add_shadow(img)
    
    
    output_path_image = output_path + "PolygonShadow/Augmented_Images/"
    
    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)
     
     
    image_name = uwi + "_" + prodcatid + "_" + "PolygonShadow.png"   
        
    cv2.imwrite(output_path_image + image_name,adjusted_image)
    
    
    mask =  cv2.imread(image_mask_files[i])
    basename_mask = os.path.basename(image_mask_files[i]).split(".")[0]
    uwi_mask    =  basename_mask.split("_")[0]
    prodcatid_mask = basename_mask.split("_")[1]
    
    
    output_path_mask = output_path + "PolygonShadow/Augmented_Masks/"
    
    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)
     
     
    mask_name = uwi_mask + "_" + prodcatid_mask + "_" + "PolygonShadow.png"   
        
    cv2.imwrite(output_path_mask + image_name, mask)
    


