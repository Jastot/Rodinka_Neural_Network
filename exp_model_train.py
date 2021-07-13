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
startTraining=1
saveModel=True
savePath=f"{os.getcwd()}/model_testpy2"
dirPath=lambda p: f"{os.getcwd()}/data/_data{p}"
shape=(224,224)
inputShape=tuple(shape+(3,))
n=200
trainSettings = {
    "epochs":60,
    "batch_size":32,
    "shuffle":True
}
modelSettings = {
    "optimizer":tf.keras.optimizers.Adam(learning_rate=0.000001),
    "loss": tf.keras.losses.binary_crossentropy,
    "metrics":['accuracy']
}

# main
model = tf.keras.Sequential(layers=[
    tf.keras.layers.ZeroPadding2D(input_shape=inputShape),
    tf.keras.layers.Conv2D(filters=96, kernel_size=(7,7), strides=2, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2),


    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),

    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),

    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),


    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2),

    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),

    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=48, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=192, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), activation='relu'),

    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=48, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=192, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), activation='relu'),

    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),


    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2),


    tf.keras.layers.ZeroPadding2D(padding=(1,1)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),

    tf.keras.layers.Conv2D(filters=1000, kernel_size=(1,1), strides=1, activation='relu'),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Dense(units=2, activation='sigmoid'),
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