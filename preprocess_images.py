# import required libraries
import cv2
import numpy as np
import os
from pathlib import Path
from cjm_pil_utils.core import get_img_files

#centering and cropping the image
def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

#create dataset using the source dataset

#dimension for the target images: (400,400) - All the models tested use images with
#dimensions below this values
dim = (400,400)
#path to the source dataset
source_dataset = "./images/images-datasetDouglascompleto"

#paths to the destination dataset, with the pre-processed images
#image_path_train = "./images/train"
#image_path_val = "./images/val"
#image_path_test = "./images/test"

#categories of the dataset: train, val, test
categories = ['train','val','test']
#path to the target dataset
image_target_path = "./images"

for category in categories:
    print(f"creating category {category}")
	#Obtaining the images to be pre-processed
	#source_dataset_train = Path(source_dataset+'/train')
    source_dataset_category = Path(source_dataset+'/'+category)
    
    #img_folder_paths_train = [folder_train for folder_train in source_dataset_train.iterdir() if folder_train.is_dir()]
    img_folder_paths_category = [folder_category for folder_category in source_dataset_category.iterdir() if folder_category.is_dir()]
    class_file_paths_category = [get_img_files(folder) for folder in img_folder_paths_category]

    ordered_classes_names = [os.path.basename(folder) for folder in source_dataset_category.iterdir() if folder.is_dir()]
    
    #print(img_folder_paths_train[0])
    #print(ordered_classes_names[0])

	#if the target directory does not exist, create it:
    if not os.path.exists(image_target_path+'/'+category):
        os.mkdir(image_target_path+'/'+category)

    j=0
    
    for directory in img_folder_paths_category:        
		
        if not os.path.exists(image_target_path+'/'+category+'/'+ordered_classes_names[j]):
            os.mkdir(image_target_path+'/'+category+'/'+ordered_classes_names[j])
            print(f"creating directory {ordered_classes_names[j]}")
		
        for subdir, dir, files in os.walk(directory):
            i=1        
			
            for file in files:
				#print(file)
                image = cv2.imread(os.path.join(subdir, file))
                if image is not None:
                    ccrop_img = center_crop(image,dim)
					#print(f"{image_path_train}/{ordered_classes_names[0]}/img-{str(i)}.jpg")
                    cv2.imwrite(image_target_path+'/'+category+'/'+ordered_classes_names[j]+'/img-'+str(i)+'.jpg',ccrop_img)
                i += 1
		
        j += 1



#Just for support:
#image_path = './images/train/giallo_fiorito/1.jpg'
#image = cv2.imread(image_path)
#print(image.shape)
#cv2.imshow('Image',image)
#cv2.waitKey(0)

#processed_image = center_crop(image, (400,400))
#cv2.imshow('Processed Image',processed_image)
#cv2.waitKey(0)