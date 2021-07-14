import json
import os
import sys

import tensorflow as tf
import tensorflowjs as tfjs

from exp_input.py import load_image, load_images

# main

dirPath=lambda p: f"../../data/_data{p}"
shape=(224,224)
inputShape=tuple(shape+(3,))
n=0
modelSettings = {
    "optimizer":tf.keras.optimizers.Adam(learning_rate=0.0003),
    "loss": tf.keras.losses.binary_crossentropy,
    "metrics":['accuracy']
}

def dataset(bs, s):
    input_benign = load_images(dirPath('/train/benign'), newShape=shape, limit=n)
    input_malignant = load_images(dirPath('/train/malignant'), newShape=shape, limit=n)
    input_tensor = tf.concat([input_benign, input_malignant], axis=0)
    output_tensor = tf.one_hot(tf.cast(tf.concat([tf.zeros([input_benign.shape[0]]), tf.ones([input_malignant.shape[0]])], axis=0), 'int32'), depth=2)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor)).shuffle(s).repeat().batch(bs)

    return dataset

def model():
    model = tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(input_shape=inputShape),

        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1)),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(1,1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1)),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(1,1)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),    
        
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='sigmoid')
    ])
    model.summary()
    model.compile(
        optimizer=modelSettings["optimizer"],
        loss=modelSettings["loss"],
        metrics=modelSettings["metrics"]
    )
    return model