import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from tensorflow.keras.optimizers import *
from keras.utils.vis_utils import *
from keras.preprocessing.image import *
from keras.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import cv2

BATCH_SIZE = 32
EPOCHS = 20
path_covid = "Dataset/COVID"
path_normal = "Dataset/NORMAL"

X = []
Y = []

for image in os.listdir(path_covid):
    img_path = os.path.join(path_covid, image)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    X.append(img)
    Y.append(1)

for image in os.listdir(path_normal):
    img_path = os.path.join(path_normal, image)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    X.append(img)
    Y.append(0)

X = np.array(X)
Y = np.array(Y)
# print(X.shape)
# print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow(
    X_train,
    y = Y_train,
    batch_size = BATCH_SIZE,
    shuffle = True
)
test_generator = test_datagen.flow(
    X_test,
    y = Y_test,
    batch_size = BATCH_SIZE,
    shuffle = False
)

def Unet ():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

model_Unet = Unet()
# plot_model(model_Unet, to_file='model_Unet.png', show_shapes=True, show_layer_names=True)

output_Unet = model_Unet.fit_generator(
    train_generator,
    steps_per_epoch = 10,
    epochs = EPOCHS,
    validation_data = test_generator,
    validation_steps = 5
)

model_Unet.save("Model_Unet.h5")

train_loss_Unet = output_Unet.history['loss']
val_loss_Unet = output_Unet.history['val_loss']
epochs_Unet = range(1, EPOCHS + 1)
plt.plot(epochs_Unet, train_loss_Unet, color='green', label='Training loss - Unet model')
plt.plot(epochs_Unet, val_loss_Unet, color='red', label='Validating loss - Unet model')
plt.title('Training and validating loss of Unet model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure(figsize=(10, 8))
plt.show()

train_acc_Unet = output_Unet.history['accuracy']
val_acc_Unet = output_Unet.history['val_accuracy']
plt.plot(epochs_Unet, train_acc_Unet, color='green', label='Training accuracy - Unet model')
plt.plot(epochs_Unet, val_acc_Unet, color='red', label='Validating accuracy - Unet model')
plt.title('Training and validating accuracy of Unet model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure(figsize=(10, 8))
plt.show()

pred_Unet = model_Unet.predict(X_test)
y_pred_Unet = np.round(pred_Unet)
print("GIÁ TRỊ DỰ ĐOÁN")
print(y_pred_Unet)

conf_Unet = confusion_matrix(Y_test, y_pred_Unet)
print(conf_Unet)
sns.heatmap(conf_Unet, annot=True)
plt.figure(figsize=(10, 8))
plt.show()

print(classification_report(Y_test, y_pred_Unet))