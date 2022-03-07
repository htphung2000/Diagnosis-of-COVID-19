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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import *
from keras.applications.vgg19 import VGG19
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
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
test_datagen = ImageDataGenerator()

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

def VGG_19 ():
    model = Sequential()
    model.add(InputLayer(input_shape=(224, 224, 3)))

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2,2), padding='valid', name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(BatchNormalization(name='block4_bn1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(BatchNormalization(name='block4_bn2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(BatchNormalization(name='block4_bn3'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(BatchNormalization(name='block5_bn1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(BatchNormalization(name='block5_bn2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(BatchNormalization(name='block5_bn3'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu', name='fc3'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name='prediction'))

    return model

base_model = VGG19(weights='imagenet', include_top=False)
base_model.summary()

model_VGG19 = VGG_19()
model_VGG19.summary()

# plot_model(model_VGG19, to_file='model_VGG19.png', show_shapes=True, show_layer_names=True)

layer_count = 0
for layer in model_VGG19.layers:
    if layer.name[:6] in ['block1', 'block2', 'block3']:
        model_VGG19.layers[layer_count].set_weights = base_model.layers[layer_count].get_weights()
    layer_count += 1

adam = Adam(lr = 1e-5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=1e-7)
model_VGG19_chkpoint = ModelCheckpoint(filepath='vgg_19_model', save_best_only=True, save_weights_only=True)

model_VGG19.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam)

output_VGG19 = model_VGG19.fit_generator(
    train_generator, 
    epochs = EPOCHS, 
    steps_per_epoch = 10, 
    callbacks = [reduce_lr, model_VGG19_chkpoint], 
    validation_data = test_generator,
    validation_steps = 5,
    class_weight = {0:3, 1:1}
)

model_VGG19.save("Model_VGG19.h5")

print(model_VGG19.evaluate(test_generator))

loss = output_VGG19.history['loss']
val_loss = output_VGG19.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='green', label='Training loss')
plt.plot(epochs, val_loss, color='red', label='Validating loss')
plt.title('Training and validating loss - VGG19 model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure(figsize=(10, 8))
plt.show()

acc = output_VGG19.history['accuracy']
val_acc = output_VGG19.history['val_accuracy']
plt.plot(epochs, acc, color='green', label='Training accuracy')
plt.plot(epochs, val_acc, color='red', label='Validating accuracy')
plt.title('Training and validating accuracy - VGG19 model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure(figsize=(10, 8))
plt.show()

pred_VGG19 = model_VGG19.predict(test_generator)
print(pred_VGG19)
y_pred_VGG19 = np.round(pred_VGG19)
print(y_pred_VGG19)

cm = confusion_matrix(Y_test, y_pred_VGG19)
print("----- Confusion matrix -----")
print(cm)
sns.heatmap(cm, annot=True)
plt.figure(figsize=(10, 8))
plt.show()

print("----- Classification report -----")
print(classification_report(Y_test, y_pred_VGG19))
