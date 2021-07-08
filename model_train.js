//require
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');
const loadImages = require('./input.js').loadImages;
const loadImage = require('./input.js').loadImage;

//functions
async function train(){
    const response = await model.fit(input, output, trainSettings);
    model.predict(loadImage('./data/train/dog/dog.5535.jpg', newShape=shape)).print();
    model.predict(loadImage('./data/train/cat/cat.3220.jpg', newShape=shape)).print();
}

async function save(model, path){
    const savedModel = model.save(path);
}

//consts
const startTraining=1;
const saveModel=1;
const savePath=`file://${__dirname}/model_test`;
const shape = [32,32];
const n = 200;
const trainSettings = {
    epochs:40,
    batchSize:25,
    //shuffle: true,
};

//inputs
const input_cat = loadImages('./data/train/cat', newShape=shape, limit=n);
const input_dog = loadImages('./data/train/dog', newShape=shape, limit=n);
const input = tf.concat([input_cat, input_dog]);
const output = tf.oneHot(tf.cast(tf.concat([tf.zeros([n]),tf.ones([n])]), 'int32'), depth=2);

log(3, input.shape);
log(3, output.shape);
log(3,output);

//model (custom [almost])
const model = tf.sequential({layers:[
    tf.layers.conv2d({kernelSize: [3,3], filters:64, activation:'relu',inputShape:[32,32,3]}),
    tf.layers.conv2d({kernelSize: [3,3], filters:64, activation:'relu',inputShape:[32,32,3]}),
    tf.layers.maxPooling2d({poolSize:[2,2]}),

    tf.layers.dropout({rate:0.3}),
    tf.layers.flatten(),
    tf.layers.dense({units:10, activation:'elu'}),
    tf.layers.dropout({rate:0.3}),
    tf.layers.dense({units:2, activation:'softmax'}),
]});
model.summary();

model.compile({
    optimizer:tf.train.adam('0.001'),
    loss:tf.losses.softmaxCrossEntropy,
    metrics:['accuracy']
})

startTraining?train().then(()=>{
    log(0, "Training complete");
    saveModel?save(model,savePath).then(()=>log(0,"Saved model")):{};
}):{};