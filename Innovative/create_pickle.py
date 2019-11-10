import keras as keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model,load_model
import numpy as np
import tensorflow as tf

def imagePreprocess(imgPath):
	img=load_img(imgPath)
	img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
	img = img.convert('RGB')  # Make sure img is color
	# To np.array. Height x Width x Channel. dtype=float32
	x = image.img_to_array(img)
	# (H, W, C)->(1, H, W, C), where the first elem is the number of img
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)  # Subtracting avg values for each pixel
	return x

def defineModel():
	base_model = MobileNet(weights='imagenet')
	model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_preds').output)
	return model1

featureModel=defineModel()
face1=imagePreprocess("testImage/Akul.18.1.jpg")
face2=imagePreprocess("testImage/Akul.18.2.jpg")

feature1 = featureModel.predict(face1)[0][0][0]  # (1, 4096) -> (4096, )
feature1=feature1 / np.linalg.norm(feature1)

feature2 = featureModel.predict(face2)[0][0][0]  # (1, 4096) -> (4096, )
feature2=feature2 / np.linalg.norm(feature2)


euc_dists = np.linalg.norm(feature1 - feature2)
cos_sim = np.dot(feature1,feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
print(euc_dists)
print(cos_sim)