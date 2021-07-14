//require
const log = require('../js_common/logger.js').log;
const tf = require('@tensorflow/tfjs-node');
const loadImages = require('../js_common/input.js').loadImages;
const loadImage = require('../js_common/input.js').loadImage;

//functions
async function train(){
    await tf.util.shuffleCombo(input, output);
    const response = await model.fit(input, output, trainSettings);
    // model.predict(loadImage('./data/train/dog/dog.5535.jpg', newShape=shape)).print();
    // model.predict(loadImage('./data/train/cat/cat.3220.jpg', newShape=shape)).print();

    model.predict(loadImage(dirPath('/train/benign/2_0.jpg'), newShape=shape)).print();
    model.predict(loadImage(dirPath('/train/malignant/1_1.jpg'), newShape=shape)).print();

    let res = model.evaluate(input, output);
    log(3, `loss: ${res[0]}`);
    log(3, `metrics: ${res[1]}`);
}

async function save(model, path){
    const savedModel = model.save(path);
}

//consts
const startTraining=1;
const saveModel=1;

const savePath=`file://${__dirname}/model_test3`;
const dirPath=(p)=>`${__dirname}/data/_data${p}`;
const shape = [128,128];
const inputShape = shape.concat([3]);
const n = 200;
const trainSettings = {
    epochs:200,
    batchSize:32,
};
const modelSettings = {
    optimizer: tf.train.adam(0.0001),
    loss: tf.losses.softmaxCrossEntropy,
    metrics:['accuracy']
};

//inputs
const input_cat = loadImages(dirPath('/train/benign'), newShape=shape, limit=n);
const input_dog = loadImages(dirPath('/train/malignant'), newShape=shape, limit=n);
const input = tf.concat([input_cat, input_dog]);
const output = tf.oneHot(tf.cast(tf.concat([tf.zeros([n]),tf.ones([n])]), 'int32'), depth=2);

log(3, input.shape);
log(3, output.shape);
log(3,output);

//model (custom [almost])
// const model = tf.sequential({layers:[
//     tf.layers.conv2d({kernelSize: [3,3], filters:32, activation:'relu', inputShape:inputShape}),
//     tf.layers.conv2d({kernelSize: [3,3], filters:32, activation:'relu'}),

//     tf.layers.maxPool2d({poolSize: [2,2]}),
//     tf.layers.conv2d({kernelSize: [3,3], filters:64, activation:'relu'}),
//     tf.layers.conv2d({kernelSize: [3,3], filters:64, activation:'relu'}),
//     tf.layers.conv2d({kernelSize: [3,3], filters:64, activation:'relu'}),

//     tf.layers.maxPool2d({poolSize: [2,2]}),
//     tf.layers.conv2d({kernelSize: [3,3], filters:128, activation:'relu'}),
//     tf.layers.conv2d({kernelSize: [3,3], filters:128, activation:'relu'}),
//     tf.layers.conv2d({kernelSize: [3,3], filters:128, activation:'relu'}),

//     tf.layers.maxPool2d({poolSize: [2,2]}),
//     tf.layers.flatten(),
//     tf.layers.dense({units:256, activation:'relu'}),
//     tf.layers.dense({units:2, activation:'softmax'}),
// ]});
const model = tf.sequential({layers:[

    tf.layers.conv2d({kernelSize: [3,3], filters:32, activation:'relu', inputShape:inputShape}),
    tf.layers.conv2d({kernelSize: [3,3], filters:32, activation:'relu', inputShape:inputShape}),
    
    tf.layers.maxPooling2d({poolSize:[2,2]}),
    tf.layers.conv2d({kernelSize: [3,3], filters:64, activation:'relu', inputShape:inputShape}),
    tf.layers.conv2d({kernelSize: [3,3], filters:64, activation:'relu', inputShape:inputShape}),

    tf.layers.maxPooling2d({poolSize:[2,2]}),
    tf.layers.conv2d({kernelSize: [3,3], filters:128, activation:'relu', inputShape:inputShape}),
    tf.layers.conv2d({kernelSize: [3,3], filters:128, activation:'relu', inputShape:inputShape}),
    tf.layers.conv2d({kernelSize: [3,3], filters:128, activation:'relu', inputShape:inputShape}),

    tf.layers.maxPooling2d({poolSize:[2,2]}),
    tf.layers.conv2d({kernelSize: [3,3], filters:256, activation:'relu', inputShape:inputShape}),
    tf.layers.conv2d({kernelSize: [3,3], filters:256, activation:'relu', inputShape:inputShape}),
    tf.layers.conv2d({kernelSize: [3,3], filters:256, activation:'relu', inputShape:inputShape}),

    tf.layers.maxPooling2d({poolSize:[2,2]}),
    tf.layers.flatten(),
    tf.layers.dense({units:1024, activation:'relu'}),
    tf.layers.dense({units:2, activation:'softmax'}),
]});
model.summary();

model.compile(modelSettings);

startTraining?train().then(()=>{
    log(0, "Training complete");
    saveModel?save(model,savePath).then(()=>log(0,"Saved model")):{};
}):{};