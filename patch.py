import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
# we can also import unpatchify

# Read images and labels
large_image_stack = tiff.imread('/mnt/c/projects/unet/external_data/image.tif')
large_mask_stack = tiff.imread('/mnt/c/projects/unet/external_data/mask.tif')

print(large_image_stack) # every image is a 3D array (every element of 3D array is matrix that represents values of one image)
print(large_mask_stack) # masks have 0 for black and 255 for white

######################################### IMAGES #########################################    

for img in range(large_image_stack.shape[0]): # shape[0] is image height
    
    large_image = large_image_stack[img]
    
    patches_img = patchify(large_image, (256, 256), step=256) # step 256 refers to exact crop
    print(patches_img.shape)
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite('/mnt/c/projects/unet/external_data/images/' + 'image_' + str(img) + '_' + str(i)+str(j) + '.tif', single_patch_img)
            
######################################### MASKS #########################################    
    
for mask in range(large_mask_stack.shape[0]): 
    
    large_mask = large_mask_stack[mask]
    
    patches_mask = patchify(large_mask, (256, 256), step=256)
    
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            
            single_patch_mask = patches_mask[i,j,:,:]
            tiff.imwrite('/mnt/c/projects/unet/external_data/masks/' + 'mask_' + str(mask) + '_' + str(i)+str(j) + '.tif', single_patch_mask)
            single_patch_mask = single_patch_mask / 255
