import json
import os
import sys

import tensorflow as tf
import tensorflowjs as tfjs

from exp_input import load_image, load_images

# functions
def train(ds):
    response = model.fit(ds, epochs=trainSettings["epochs"])
    return

def save(model, path):
    savedModel = model.save(f"{path}/py")
    savedModeljs = tfjs.converters.save_keras_model(model, f"{path}/js")
    print("saved model")
    return

# consts
startTraining=1
saveModel=True
savePath=f"{os.getcwd()}/../../models/model_2_2"
dirPath=lambda p: f"{os.getcwd()}/../../data/_data{p}"
shape=(224,224)
inputShape=tuple(shape+(3,))
n=0
trainSettings = {
    "epochs":60,
    "shuffle":True
}
modelSettings = {
    "optimizer":tf.keras.optimizers.Adam(learning_rate=0.0003),
    "loss": tf.keras.losses.binary_crossentropy,
    "metrics":['accuracy']
}

# main
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
model.compile(optimizer=modelSettings["optimizer"], loss=modelSettings["loss"], metrics=modelSettings["metrics"])

# inputs
input_benign = load_images(dirPath('/train/benign'), newShape=shape, limit=n)
input_malignant = load_images(dirPath('/train/malignant'), newShape=shape, limit=n)
input_tensor = tf.concat([input_benign, input_malignant], axis=0)
output_tensor = tf.one_hot(tf.cast(tf.concat([tf.zeros([input_benign.shape[0]]), tf.ones([input_malignant.shape[0]])], axis=0), 'int32'), depth=2)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(64)


print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")

if(startTraining):
    train(dataset)
    save(model, savePath)