## N20-7 Image classifier

### To-do

- [x] Research about convolutional layers 
- [x] Research about basic layers:
  - [x] Dense
  - [x] Activation
  - [x] Dropout
- [x] Research about pooling layers
- [x] Research about losses
- [ ] Research about optimizers
- [ ] Make a nice classifier model out of layers mentioned above

### Comments on a model

#### About activation function

​	Definitely, our problem is a classification problem, meaning we will be using something like sigmoid function in our output layer. There are two sigmoid-like functions in the **tensorflow** library: `sigmoid` (genuine sigmoid) and `softmax` (sigmoid-like function). The difference is that in softmax inputs are depended, so the output probabilities will always sum to one, which is probably good for multiclass classification. We have only two classes, thereby using `sigmoid` in output layer.

​	`ReLU` actiovation function is considered a standard in DeepLearning hidden layers nowadays as it is easy to calculate , it doesn't saturate, and its non-linear, which helps layers to cooperate. Also, it was shown that `ReLU` layers after filters suprisingly improve image classifiers performance. `ReLU` was used in convolutional layers. 

#### About loss function

​	A cross entropy function was chosen. It minimizes -log(likelihood)​, thus strongly penalising the model when it outputs a very bed result. However, there are many versions of in the tensorflow library. The differnce is unknown, but the `binary` one was used as we have only two classes.

#### Created models

| Model name | Loss | Accuracy | Pic res | Params |
| :--------- | :--- | :------- | ------- | ------ |
| model_2_1  | 1.02 | 0.837    | 224x224 | 555k   |



### Used to understand the topic

- https://www.quora.com/What-makes-ReLU-so-much-better-than-Linear-Activation-As-half-of-them-are-exactly-the-same
- https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
- https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
- https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
- https://towardsdatascience.com/recognizing-cats-and-dogs-with-tensorflow-105eb56da35f
- https://medium.com/@nutanbhogendrasharma/tensorflow-classify-images-of-cats-and-dogs-by-using-transfer-learning-59da26723bda
- https://miro.medium.com/max/2000/1*ooVUXW6BIcoRdsF7kzkMwQ.png
- https://neurohive.io/en/popular-networks/vgg16/
- https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#train_the_model
- https://www.tensorflow.org/guide/distributed_training#multiworkermirroredstrategy
- https://arxiv.org/pdf/1409.4842.pdf
- https://arxiv.org/abs/1602.07360