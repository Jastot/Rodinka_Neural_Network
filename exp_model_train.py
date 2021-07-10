import json
import os
import sys

import tensorflow as tf
import tensorflowjs as tfjs

from exp_input import load_image, load_images

# functions
def train(i, o):
    response = model.fit(i, o, epochs=trainSettings["epochs"], batch_size=trainSettings["batch_size"])

def save(model, path):
    savedModel = model.save(f"{path}/py")
    savedModeljs = tfjs.converters.save_keras_model(model, f"{path}/js")
    print("saved model")

# consts
startTraining=True
saveModel=True
savePath=f"{os.getcwd()}/model_testpy"
dirPath=lambda p: f"{os.getcwd()}/data/_data{p}"
shape=(128,128)
inputShape=tuple(shape+(3,))
n=10
trainSettings = {
    "epochs":10,
    "batch_size":32,
    "shuffle":True
}
modelSettings = {
    "optimizer":tf.keras.optimizers.Adam(learning_rate=0.0003),
    "loss": tf.keras.losses.binary_crossentropy,
    "metrics":['accuracy']
}

# main
model = tf.keras.Sequential(layers=[
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=inputShape),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

model.summary()
model.compile(optimizer=modelSettings["optimizer"], loss=modelSettings["loss"], metrics=modelSettings["metrics"])

# inputs
input_benign = load_images(dirPath('/train/benign'), newShape=shape, limit=n)
input_malignant = load_images(dirPath('/train/malignant'), newShape=shape, limit=n)
input = tf.concat([input_benign, input_malignant], axis=0)
output = tf.one_hot(tf.cast(tf.concat([tf.zeros([n]), tf.ones([n])], axis=0), 'int32'), depth=2)

print(f"Input shape: {input.shape}")
print(f"Output shape: {output.shape}")
print(output)

if(startTraining):
    train(input, output)
    save(model, savePath)