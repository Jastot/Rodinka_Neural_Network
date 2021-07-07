//require
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');

const loadImages = require('./input.js').loadImages;

//model (custom)
const model = tf.sequential({layers:[
    tf.layers.dense({units:4, activation:'relu', inputShape:[32,32,3]}),
    tf.layers.dense({units:1, activation:'relu'})
]});

model.compile({
    optimizer:tf.train.sgd('0.1'),
    loss:tf.losses.meanSquaredError
})

const input_cat = loadImages('./data/train/cat', newShape=[32,32], limit=100);
const input_dog = loadImages('./data/train/dog', newShape=[32,32], limit=100);
const input = tf.concat([input_cat, input_dog]);
const output = tf.concat([tf.ones([100]),tf.zeros([100])]).toFloat();
log(3, input.shape);
log(3, output.shape);
train().then(()=>{log(0, "Training complete")});

async function train(){
    const response = await model.fit(input, output, {
        epochs:10,
    });
    log(0,response);
}