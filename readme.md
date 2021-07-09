## N20-7 Image classifier

### To-do

- [ ] Research about convolutional layers 
- [ ] Research about basic layers:
  - [ ] Dense
  - [ ] Activation
  - [ ] Dropout
- [ ] Research about pooling layers
- [ ] Research about losses
- [ ] Research about optimizers
- [ ] Make a nice classifier model out of layers mentioned above

### Comments on a model

#### About activation function

​	Definitely, our problem is a classification problem, meaning we will be using something like sigmoid function in our output layer. There are two sigmoid-like functions in the **tensorflow** library: `sigmoid` (genuine sigmoid) and `softmax` (sigmoid-like function). The difference is that in softmax inputs are depended, so the output probabilities will always sum to one, which is probably good for multiclass classification. We have only two classes, thereby using `sigmoid` in output layer.

​	`ReLU` actiovation function is considered a standard in DeepLearning hidden layers nowadays as it is easy to calculate , it doesn't saturate. Also, it was shown that `ReLU` layers after filters suprisingly improve image classifiers performance. `ReLU` was used in convolutional layers.

#### About loss function

​	From the first sight, `tf.losses.meanSquaredError` was a choice. However, after a bit of research, we realised that `tf.losses.meanSquaredError` is mostly used for *regression* problems because of its mathematic nature. Long story short, it can't penalise the model enough when solving *classification* problems.

​	Thereby, a cross entropy function was chosen. It minimizes -log(likelihood)​, thus penalising the model more when it outputs a very bed result. However, there are two versions of it in a tensorflow library: `tf.losses.sigmoidCrossEntropy` and `tf.losses.softmaxCrossEntropy`. The differnce is unknown, but the softmax performs better so it was used.

| Loss Function       | Loss  | Accuracy |
| ------------------- | ----- | -------- |
| softmaxCrossEntropy | 0.395 | 0.877    |
| sigmoidCrossEntropy | 0.486 | 0.746    |
|                     |       |          |

