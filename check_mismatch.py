import os
import numpy as np

images_folder = '/mnt/c/projects/unet/patches/images'
masks_folder = '/mnt/c/projects/unet/patches/masks'

images_dir = [i for i in os.listdir(images_folder) if i.endswith(('.tif'))]
masks_dir = [i for i in os.listdir(masks_folder) if i.endswith(('.tif'))]

# Extract image and labels names
image_names = [os.path.splitext(i)[0] for i in images_dir]
mask_names = [os.path.splitext(i)[0] for i in masks_dir]

mismatched_images = []
mismatched_labels = []

#print('mismatch images:')
#print(mismatched_images)
#print('mismatch masks:')
#print(mismatched_labels)

for image_name in image_names:
    if image_name not in mask_names:
        mismatched_labels.append(image_name)
print('Labels with these names do not exist: ')
print(mismatched_labels)

for label_name in mask_names:
    if label_name not in image_names:
        mismatched_images.append(label_name)
print('Images with these names do not exist: ')
print(mismatched_images)

# Delete mismatched files
