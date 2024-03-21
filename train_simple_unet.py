from simple_unet_model import simple_unet_model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random

image_directory = '/mnt/c/projects/unet/patches2/images/'
mask_directory =  '/mnt/c/projects/unet/patches2/masks/'

# Define size to resize all images if some of them are not 256
SIZE = 256

# We capture all images in these lists
image_dataset = []
mask_dataset = []

# For each element of the list read it and convert it to pillow objec, resize it and convert it to numpy array, append it to datasets
images = os.listdir(image_directory)
for i, image_name in enumerate(images): # zasto nije ovde samo stavio images?
    if (image_name.split('.')[1] == 'jpg'): # tekst pre ('.') je na poziciji [0], a posle tog znaka je na [1]
        #print(image_directory + image_name)
        image = cv2.imread(image_directory + image_name, 0) # ucitaj slike
        if image is not None:
            image = Image.fromarray(image) # stvaramo sliku iz niza
            image = image.resize((SIZE, SIZE))
            image_dataset.append(np.array(image)) # dodaj u niz sliku u vidu niza
        else:
            print('Failed to load image:', image_name)

masks = os.listdir(mask_directory)
for i, mask_name in enumerate(masks): 
    if (mask_name.split('.')[1] == 'tif'): 
        mask = cv2.imread(mask_directory + mask_name, 0) 
        if mask is not None:
            mask = Image.fromarray(mask) 
            mask = mask.resize((SIZE, SIZE))
            mask_dataset.append(np.array(mask)) 
        else:
            print('Failed to load mask:', mask_name)
    
# Normalize images. Convert list to numpy arrays and always scale to [0,1]. Now they are 8bit images [0,255], and expand dimensions by 1 so they are ready for deep learning
# We added one dimension for color
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)

# Rescale masks to 0 to 1. 
mask_dataset = np.expand_dims((np.array(mask_dataset)),3)/255.

# ogranici sve liste na prvih 100 elemenata pre train test split
image_dataset_100 = image_dataset[:50]
mask_dataset_100 = mask_dataset[:50]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset_100, mask_dataset_100, test_size=0.10, random_state=0) #random state: shuffle the data before splitting

# Sanity check, view few images
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12,6))

plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

############################################################

IMG_HEIGHT = image_dataset.shape[1]
IMG_WEIGHT = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WEIGHT, IMG_CHANNELS)

model = get_model()

history = model.fit(X_train, y_train, # takes two arrays as training data for supervised learning
                    batch_size=1,
                    verbose=1, # verbosity mode: specify the logging level 
                    epochs=40,
                    validation_data=(X_test, y_test),
                    shuffle=False)

model.save('mitochondria_test.keras') # hdf5 - hierarchical data format 5 (store large amount of data, ali je legacy)

############################################################

# Evaluate the model

_, acc = model.evaluate(X_test, y_test)
print('Accuracy = ', (acc * 100.0), '%')

# Plot the training and validation accuracy and loss at each point
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

############################################################
# IOU

y_pred=model.predict(X_test) # racuna verovatnoce za svaki piksel
y_pred_thresholded = y_pred > 0.5 # binarizujemo vrednosti

# Pretvaramo u binarne maske
y_pred_thresholded = y_pred_thresholded.astype(np.uint8)
y_test = y_test.astype(np.uint8)

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)

############################################################
# Predict on a few images

model = get_model()
model.load_weights('/mnt/c/projects/unet/mitochondria_test.keras')

test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

test_img_other = cv2.imread('/mnt/c/projects/unet/external_data/images/image_0_00.tif', 0)
test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
test_img_other_norm = test_img_other_norm[:,:,0][:,:,None]
test_img_other_input = np.expand_dims(test_img_other_norm, 0)

# Predict and threshold for values above 0.5 probability
prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(235)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()