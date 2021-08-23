import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from unet import unet

#resize images if needed

size_x = 128
size_y = 128
#num_classes = 14
num_classes = 30#38

#training list

train_images = []
for img_path in tqdm(glob.glob("camvid/train/images/*.png"), total=701):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size_x, size_y))
    train_images.append(img)

"convert list to np array for ml processing"
train_images = np.array(train_images)

train_masks = []
for mask_path in tqdm(glob.glob("camvid/train/masks/*.png"), total=701):
    mask = cv2.imread(mask_path,0)
    mask = cv2.resize(mask, (size_x, size_y), interpolation= cv2.INTER_NEAREST)
    train_masks.append(mask)

"convert list to np array for ml processing"
train_masks = np.array(train_masks)
print(np.unique(train_masks))
print(train_masks.shape)
#apply label encoder 0,1,2,3.... used for iou keras
#works for single vector, so convert train msak array (701,256,256,3) to vector, encode, and again reshape back

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n,h,w)
print("encoded labels is ", np.unique(train_masks_encoded_original_shape))# 0 is background or  unlabelled


from keras.utils import normalize
#train_images = np.expand_dims(train_images, axis=3)#701x256x256 to 701x256x256x3
train_images = normalize(train_images, axis=1)#uint8 to float64
train_masks = np.expand_dims(train_masks_encoded_original_shape, axis=3)#701x256x256 to 701x256x256x1

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_images, train_masks_encoded_original_shape, test_size=0.3, random_state=None)
#x_train 490,256,256,3 x_test 211,256,256,3 y_train 490,256,256 y_test 211,256,256

#when doing this class changes from 38 to 30
y_train = np.expand_dims(y_train, axis=3)#490x256x256 to 490x256x256x1
y_test = np.expand_dims(y_test, axis=3)#211x256x256 to 211x256x256x1
print("num of class is ", np.unique(y_train))

#one hot encoding the masks
from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=num_classes)# 490,256,256,38
y_train_cat = train_masks_cat
#y_train_cat = train_masks_cat.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2],y_train.shape[3], num_classes )
test_masks_cat = to_categorical(y_test, num_classes=num_classes)# 490,256,256,38
y_test_cat = test_masks_cat
#y_test_cat = test_masks_cat.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], y_test.shape[3], num_classes )

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_masks_reshaped_encoded), train_masks_reshaped_encoded)
print("class weights are :", class_weights)

img_height = x_train.shape[1]
img_width = x_train.shape[2]
img_channels = x_train.shape[3]
input_shape = (img_height,img_width,img_channels)
print(input_shape)

def get_model():
    return unet(input_shape=input_shape, num_classes=num_classes)

model = get_model()
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.load_weights("unet50adam.hdf5") #Mean IOU : 0.057816256 Accuracy: 55.813437700271606 %

_, acc = model.evaluate(x_test, y_test_cat)
print("Accuracy:", (acc*100.0),"%")

y_pred = model.predict(x_test)
print(y_pred.shape)#(211, 128, 128, 38) prob 0 to 1
y_pred_argmax = np.argmax(y_pred, axis=3)
print(y_pred_argmax.shape)#(211, 128, 128)# argmax returns classs of max prob

from keras.metrics import MeanIoU
num_classes = 30#38
iou = MeanIoU(num_classes=num_classes)
iou.update_state(y_test[:,:,:,0], y_pred_argmax)
print(y_test[:,:,:,0],"-------------",y_pred_argmax)#211,128,128 same
plt.imshow(y_test[100])
plt.show()
plt.imshow(y_pred_argmax[100])
plt.show()
print("Mean IOU :", iou.result().numpy())

plt.imshow(train_images[0, :,:,0], cmap="gray")
plt.show()
plt.imshow(train_masks[0], cmap="gray")
plt.show()
plt.imshow(x_test[0, :,:,0], cmap="gray")
plt.show()
plt.imshow(y_test[0], cmap="gray")
plt.show()
plt.imshow(y_pred_argmax[100], cmap="gray")
plt.show()

f, axarr = plt.subplots(1,2)
axarr[0].imshow(y_test[50])#,cmap="gray")
axarr[1].imshow(y_pred_argmax[50])#,cmap="gray")

for i in range (5):
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow(y_test[i])  # ,cmap="gray")
    axarr[0,1].imshow(y_pred_argmax[i])  # ,cmap="gray")
    axarr[1,0].imshow(y_test[i+10])  # ,cmap="gray")
    axarr[1,1].imshow(y_pred_argmax[i+10])  # ,cmap="gray")
a=1