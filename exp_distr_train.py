import os
import json

import tensorflow as tf

import exp_distr_model

# main
batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
model_config = json.loads(os.environ['MD_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

epochs = model_config['epochs']
spe = model_config['spe']

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = batch_size * num_workers
multi_worker_dataset = exp_distr_model.dataset(global_batch_size)

with strategy.scope():
    multi_worker_model = exp_distr_model.model()

multi_worker_model.fit(multi_worker_dataset, epochs=epochs, steps_per_epoch=spe)