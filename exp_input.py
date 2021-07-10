import json
import os
import sys

import tensorflow as tf
import numpy as np
from PIL import Image

def load_image(path, newShape=(32,32)):
    print(f"Loading image {os.getcwd()+'/'+path}")
    buffer = Image.open(f"{path}")
    buffer.load()
    data = np.asarray(buffer, dtype="float32")
    data = data[:,:,::-1]
    data = data/255
    data = np.expand_dims(data, axis=0)
    tensor = tf.convert_to_tensor(data)
    tensor = tf.image.resize(tensor, size=newShape, method='nearest')    
    return tensor



def load_images(path, newShape=(32,32), limit=0):
    print(f"Loading images {os.getcwd()+'/'+path}")
    unsorted = []
    sorted=[]
    tensors=[]

    for file in os.listdir(f"{path}"):
        if(file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')):
            if(limit>0):
                if(len(sorted)<limit):
                    sorted.append(file)
            else:
                sorted.append(file)
    
    for name in sorted:
        buffer = Image.open(f"{path}/{name}")
        buffer.load()
        data = np.asarray(buffer, dtype="float32")
        data = data[:,:,::-1]
        data = data/255
        data = np.expand_dims(data, axis=0)
        tensor = tf.convert_to_tensor(data)
        tensor = tf.image.resize(tensor, size=newShape, method='nearest')
        tensors.append(tensor)
    print("Loaded images")
    output = tf.concat(tensors, axis=0)
    return output