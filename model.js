//require
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');

const loadImages = require('./input.js').loadImages;

//model (custom)
const model = tf.sequential({layers:[
    tf.layers.dense({units:4, activation:'sigmoid', inputShape:[200,]}),
    tf.layers.dense({units:1, activation:'sigmoid'})
]});

model.compile({
    optimizer:tf.train.sgd('0.1'),
    loss:tf.losses.meanSquaredError
})

const input_cat = loadImages('./data/train/cat', limit=100);
const input_dog = loadImages('./data/train/dog', limit=100);
const input = tf.concat([input_cat, input_dog]);
const output = tf.concat([tf.ones([100]),tf.zeros([100])]);

train().then(()=>{log(0, "Training complete")});

async function train(){
    for (let i = 0; i<10; i++){
        const response = await model.fit(input, output);
        log(3,response);
    }
}