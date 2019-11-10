from keras.preprocessing import image
import keras as keras
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model,load_model
import numpy as np
import tensorflow as tf


class FeatureExtractor:
    def __init__(self):
        self.graph1 = tf.Graph()
        base_model = MobileNet(weights='imagenet')
        self.model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_preds').output)
        self.graph = tf.get_default_graph()

    def extract(self, img): #img is preprocessed image
        print("I have got into extract method")
        with self.graph.as_default():
            print("I have loaded the model")
            feature = self.model1.predict(img)[0][0][0]  # (1, 4096) -> (4096, )
            return feature / np.linalg.norm(feature)  # Normalize