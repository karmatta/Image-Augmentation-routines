import pandas as pd
import numpy as np
import cv2 
import matplotlib.pyplot  as plt
import glob 
import os



color_choice = [0,255]

def generate_random_lines(imshape,slant,drop_length):
    drops=[]
    for i in range(500): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops
        
def add_white_rain(image):
    #image = img
    imshape = image.shape
    slant_extreme=10
    slant= np.random.randint(-slant_extreme,slant_extreme) 
    drop_length=20
    drop_width=2
    
    ## a shade of gray
    rain_drops= generate_random_lines(imshape,slant,drop_length)
    
    for rain_drop in rain_drops:
         w = np.random.choice(color_choice)
         drop_color=(int(w),int(w),int(w))
         if slant > 0 :
             
             cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+np.random.randint(0,slant),rain_drop[1]+drop_length),drop_color,drop_width)
         elif slant == 0:
             cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+0,rain_drop[1]+drop_length),drop_color,drop_width)
             
         else:
             cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+np.random.randint(slant,0),rain_drop[1]+drop_length),drop_color,drop_width)
    #image= cv2.blur(image,(7,7)) ## rainy view are blurry
    
    #brightness_coefficient = 0.7 ## rainy days are usually shady 
    #image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    #image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    #image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image
    
def add_rain(image):
    #image = img
    imshape = image.shape
    slant_extreme=10
    slant= np.random.randint(-slant_extreme,slant_extreme) 
    drop_length=20
    drop_width=2
    
    ## a shade of gray
    rain_drops= generate_random_lines(imshape,slant,drop_length)
    
    for rain_drop in rain_drops:
         r = np.random.randint(0,255)
         g = np.random.randint(0,255)
         b = np.random.randint(0,255)
         drop_color=(r,g,b)
         if slant > 0 :
             
             cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+np.random.randint(0,slant),rain_drop[1]+drop_length),drop_color,drop_width)
         elif slant == 0:
             cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+0,rain_drop[1]+drop_length),drop_color,drop_width)
             
         else:
             cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+np.random.randint(slant,0),rain_drop[1]+drop_length),drop_color,drop_width)
    #image= cv2.blur(image,(7,7)) ## rainy view are blurry
    
    #brightness_coefficient = 0.7 ## rainy days are usually shady 
    #image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    #image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    #image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image



#img = cv2.imread("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/30005279950000_25796256.png")
#plt.imshow(img)
#rain_img = add_rain(img)
#plt.imshow(rain_img)  
#cv2.imwrite("/home/affine/Downloads/TGS_OilPad_Identify/Augmentation Methods/Image/output/ColorRain_Image.png",rain_img)



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
    
    adjusted_image = add_white_rain(img)
    
    
    output_path_image = output_path + "WHITE_RAIN/Augmented_Images/"
    
    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)
     
     
    image_name = uwi + "_" + prodcatid + "_" + "WHITE_RAIN.png"   
        
    cv2.imwrite(output_path_image + image_name,adjusted_image)
    
    
    mask =  cv2.imread(image_mask_files[i])
    basename_mask = os.path.basename(image_mask_files[i]).split(".")[0]
    uwi_mask    =  basename_mask.split("_")[0]
    prodcatid_mask = basename_mask.split("_")[1]
    
    
    output_path_mask = output_path + "WHITE_RAIN/Augmented_Masks/"
    
    if not os.path.exists(output_path_mask):
        os.makedirs(output_path_mask)
     
     
    mask_name = uwi_mask + "_" + prodcatid_mask + "_" + "WHITE_RAIN.png"   
        
    cv2.imwrite(output_path_mask + image_name, mask)
    
    

