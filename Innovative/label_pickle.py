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

def generatePickle(personName):
	path=personName+"/*"
	avg_feature=0
	# (1,1000) length
	file_count=0
	sum_feature=np.zeros((1,1000))
	for file in glob.glob(path):
		face1=imagePreprocess(file)		
		feature1 = featureModel.predict(face1)[0][0][0]  # (1, 4096) -> (4096, )
		feature1=feature1 / np.linalg.norm(feature1)
		sum_feature=avg_feature+feature1
		file_count+=1
	avg_feature=sum_feature/file_count
	pickle_path=personName.split("/")[0]+"/features/"+personName.split("/")[1]+".pickle"
	pickle_out = open(pickle_path,"wb")
	pickle.dump(avg_feature, pickle_out)
	pickle_out.close()
	# print(avg_feature.shape)
def createDumps(name):
	featureModel=defineModel()
	imgPath="testImage/*"
	for nameFolder in glob.glob("testImage/*"):
		if(nameFolder!="testImage/features"):
			generatePickle(nameFolder)

# face1=imagePreprocess("testImage/Akul.18.1.jpg")
# face2=imagePreprocess("testImage/Akul.18.2.jpg")

# feature1 = featureModel.predict(face1)[0][0][0]  # (1, 4096) -> (4096, )
# feature1=feature1 / np.linalg.norm(feature1)

# feature2 = featureModel.predict(face2)[0][0][0]  # (1, 4096) -> (4096, )
# feature2=feature2 / np.linalg.norm(feature2)


# euc_dists = np.linalg.norm(feature1 - feature2)
# cos_sim = np.dot(feature1,feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
# print(euc_dists)
# print(cos_sim)