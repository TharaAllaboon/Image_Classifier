# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:55:42 2020

@author: Thara
"""


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('image_file', type=str, help='image file path')
parser.add_argument('model_file', type=str, help='model file path')
parser.add_argument('--top_k', required = False, type=int, default = 5, help='Top 5 predictions')
parser.add_argument('--category_names', required = False, type=str, default = 'label_map.json', help='json file name for labels')
args = parser.parse_args()

class ImageClassifer:
    def __init__(self):
        self.IMG_SIZE = 224

    def get_params(self, image_path, model_path, top_k, category_names):
        self.image_file = image_path
        self.model_file = model_path
        self.top_k = top_k
        self.category_names = category_names

        print('image_file:', self.image_file)
        print('model_file:', self.model_file)
        print('top_k:', self.top_k)
        print('category_file:', category_names)

    def load_model(self):
        self.reloaded_keras_model = tf.keras.models.load_model(self.model_file, custom_objects={'KerasLayer': hub.KerasLayer})
        print('\n  Model Summary  \n')
        print('==========================')
        print(self.reloaded_keras_model.summary())

    def load_labels(self):
        with open(self.category_names, 'r') as f:
            self.class_names = json.load(f)

    def process_image(self, image):
        return tf.cast(tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE)), tf.float32).numpy()/255

    def predict(self):
        image_loaded = self.process_image(np.asarray(Image.open(self.image_file)))
        predicted = self.reloaded_keras_model.predict(np.expand_dims(image_loaded, axis=0))[0]
        probs, indices = tf.math.top_k(predicted, k=self.top_k)
        probs = probs.numpy().tolist()
        classes = indices.numpy().tolist()
        return probs, classes

    def run(self):
        self.load_model()
        self.load_labels()
        probs, classes = self.predict()
        labels = [self.class_names[str(idx + 1)] for idx in classes]

        print('\n')
        print('K    Probability    Flower')
        print('==========================')
        for i, d in enumerate(zip(probs,labels)):
            print(f"{i}    {round(d[0],4):.4f}         {d[1]}")
        print('\n')
        
ic = ImageClassifer()
ic.get_params(args.image_file, args.model_file, args.top_k, args.category_names)
ic.run()