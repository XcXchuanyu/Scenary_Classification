# IMPORTING LIBRARIES
import keras
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# LOADING MODEL
model = load_model("model.h5")

# IMPORTING PREDICTION IMAGE
test_img=image.load_img("data/test/bedroom/image_0001.jpg",target_size=(150,150),color_mode='grayscale')

# IMAGE PREPROCESSING
test_img=image.img_to_array(test_img)
test_img=test_img/255
test_img=np.expand_dims(test_img,axis=0)

# PREDICTION
a=model.predict(test_img)
x = np.argmax(a)
print(x)

# FETCHING INDICES
image_gen=ImageDataGenerator(rescale=1./255)
train_image_gen=image_gen.flow_from_directory("data/train",target_size=(150,150),batch_size=16,class_mode="categorical",color_mode="grayscale")
train_image_gen.class_indices

# # OUTPUT OF RESULT
# if(x == 0):
#     print("Predicted Buildings")
# elif(x == 1):
#     print("Predicted Forest")
# elif(x == 2):
#     print("Predicted Glaciers")
# elif(x == 3):
#     print("Predicted Mountains")
# elif(x == 4):
#     print("Predicted Sea")
# elif(x == 5):
#     print("Predicted Streets")