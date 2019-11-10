import keras as keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model,load_model
import numpy as np
import tensorflow as tf
import glob,pickle

def imagePreprocess(imgPath):
	img=load_img(imgPath)
	img = img.resize((224, 224))
	img = img.convert('RGB')
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	# x=x.reshape()
	return x

def defineModel():
	base_model = MobileNet(weights='imagenet')
	model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_preds').output)
	return model1

def comparePickle(newImage,allFeatures):
	resultDict={}
	# newImage=np.reshape(newImage,(1,1000))
	print("THIOS IS MY SHAPEEEE",newImage.shape)
	for keys in allFeatures:
		loaded_feature=allFeatures[keys]
		# print(loaded_feature.shape)
		cos_sim = np.dot(newImage,loaded_feature) / (np.linalg.norm(newImage) * np.linalg.norm(loaded_feature))		
		resultDict[keys]=cos_sim
	return resultDict

# featureModel=defineModel()
# face1=imagePreprocess("11.13.jpg")

# feature1 = featureModel.predict(face1)[0][0][0]  # (1, 4096) -> (4096, )
# feature1=feature1 / np.linalg.norm(feature1)

# comparePickle(feature1)

# euc_dists = np.linalg.norm(feature1 - feature2)
# cos_sim = np.dot(feature1,feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
# print(euc_dists)
# print(cos_sim)