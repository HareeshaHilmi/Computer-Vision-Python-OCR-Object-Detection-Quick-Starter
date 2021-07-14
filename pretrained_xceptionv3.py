# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:56:10 2021

@author: user
"""

from keras.applications import Xception
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2

#Loading the image to predict
#E:\Practice\OCR\code\images\test5.jpg
img_path = 'E:\\Practice\\OCR\\code\\images\\test11.jpg'
img = load_img(img_path)

#resize the image to 299x299 squre shape
img = img.resize((299,299))
#convert the image to array
img_array = img_to_array(img)

#covert the image into a 4 dimentional Tensor
#convert from (height, width, channels),(batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis = 0)

#preprocess the input image array
img_array = preprocess_input(img_array)

#Load the model from internet / computer
#approximately 530 MB
pretrained_model = Xception(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

#decode the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)
predicted_name = (actual_prediction[0][0][1])

print("Predicted object is: %s" % predicted_name)
print("with accuracy %f" % (actual_prediction[0][0][2]*100))

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, predicted_name, (20,20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,0,0))
#show the image
cv2.imshow("Prediction", disp_img)