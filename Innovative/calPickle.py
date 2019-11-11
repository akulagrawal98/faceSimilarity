import os,sys
import pickle
import numpy as np
import keras as keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model,load_model
import numpy as np
import tensorflow as tf
import glob

def imagePreprocess(imgPath):
	img=load_img(imgPath)
	img = img.resize((224, 224))
	img = img.convert('RGB')
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def defineModel():
	base_model = MobileNet(weights='imagenet')
	model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_preds').output)
	return model1

def savePickle(personName,fe):
	# personName=""
	# for i in range(1,len(name)):
	# 	print(name[i])
	# 	if(i!=1):
	# 		personName+=" "
	# 		personName+=name[i]
	# 	else:
	# 		personName+=name[i]
	# print(personName)
	imgpath="testImage/"+personName+"/*"

	avg_feature=0
	# (1,1000) length
	file_count=0
	sum_feature=np.zeros((1000,))
	print(imgpath)
	for file in glob.glob(imgpath):
		print(file)
		face1=imagePreprocess(file)
		feature1=fe.extract(face1)
		# feature1 = featureModel.predict(face1)[0][0][0]  # (1, 4096) -> (4096, )
		# feature1=feature1 / np.linalg.norm(feature1)
		sum_feature=sum_feature+feature1
		file_count+=1
	print(file_count)
	print("NO OF FILE COUNT",file_count)
	avg_feature=sum_feature/file_count
	print(avg_feature.shape)
	pickle_path="testImage/features/"+personName+".pickle"
	pickle_out = open(pickle_path,"wb")
	pickle.dump(avg_feature, pickle_out)
	pickle_out.close()

if __name__ == '__main__':
	featureModel=defineModel()
	
	personName=""
	name=sys.argv
	for i in range(1,len(name)):
		print(name[i])
		if(i!=1):
			personName+=" "
			personName+=name[i]
		else:
			personName+=name[i]
	# print(personName)

	savePickle(featureModel,personName)

def saveP(name,fe):
	# featureModel=defineModel()
	savePickle(name,fe)	
