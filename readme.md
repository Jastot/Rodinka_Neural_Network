### N20-7 Image classifier

#### To-do

- [ ] Research about convolutional layers 
- [ ] Research about basic layers:
  - [ ] Dense
  - [ ] Activation
  - [ ] Dropout
- [ ] Research about pooling layers
- [ ] Research about losses
- [ ] Research about optimizers
- [ ] Make a nice classifier model out of layers mentioned above

#### Comments on a model

##### About loss function

​	From the first sight, `tf.losses.meanSquaredError` was a choice. However, after a bit of research, we realised that `tf.losses.meanSquaredError` is mostly used for *regression* problems because of its mathematic nature. Long story short, it can't penalise the model enough when solving *classification* problems.

​	Thereby, a cross entropy function was chosen. However, there are two versions of it: `tf.losses.sigmoidCrossEntropy` and `tf.losses.softmaxCrossEntropy`.

