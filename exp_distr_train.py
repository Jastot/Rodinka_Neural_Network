import os
import json

import tensorflow as tf
import tensorflowjs as tfjs

import exp_distr_model

# f
def _is_chief(task_type, task_id):
    return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)

# main
model_path = './model_2_1_2'

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

task_type, task_id = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
write_model_path = write_filepath(model_path, task_type, task_id)
if(_is_chief(task_type, task_id)):
    multi_worker_model.save(write_model_path)
    tfjs.converters.save_keras_model(multi_worker_model, f"{model_path}/js")