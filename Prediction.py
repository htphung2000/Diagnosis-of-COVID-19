import numpy as np
from flask import Flask, json, render_template, request
import tensorflow as tf
from tensorflow.python.eager import tape
from tensorflow import keras
from keras.preprocessing.image import *
from keras.models import *
from keras.applications.xception import *
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from os.path import *
import calendar
import time


app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app.static_folder = 'static'
model_CNN = load_model('Model_CNN.h5')
model_VGG19 = load_model('Model_VGG19.h5')


def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


"""
    ROUTE
"""
 
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['image']
    name = str(calendar.timegm(time.gmtime())) + ".jpg"
    f.filename = name

    STORAGE_FOLDER = join(dirname(realpath(__file__)), "static\images")
    saved_path = join(STORAGE_FOLDER, f.filename)
    f.save(saved_path)
    
    test = get_img_array(saved_path, (224, 224))
    output = {0: "Âm tính", 1: "Dương tính"}

    # CNN
    pred_CNN = model_CNN.predict(test)
    y_pred_CNN = np.round(pred_CNN)
    result_CNN = output[y_pred_CNN[0][0]]

    # VGG19
    pred_VGG19 = model_VGG19.predict(test)
    y_pred_VGG19 = np.round(pred_VGG19)
    result_VGG19 = output[y_pred_VGG19[0][0]]

    return render_template("predict.html", photo='{}'.format("/static/images/" + f.filename), 
                            output_CNN='{}'.format(result_CNN), output_VGG19='{}'.format(result_VGG19))

 
if __name__ == '__main__':
    app.run(debug=True)