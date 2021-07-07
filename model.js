//require
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');

const loadImages = require('./input.js').loadImages;
const loadImage = require('./input.js').loadImage;
//model (custom)
const model = tf.sequential({layers:[
    tf.layers.flatten({inputShape:[96,96,3]}),
    tf.layers.dense({units:4, activation:'sigmoid'}),
    tf.layers.dense({units:1, activation:'sigmoid'})
]});

model.compile({
    optimizer:tf.train.sgd('0.1'),
    loss:tf.losses.meanSquaredError
})

const input_cat = loadImages('./data/train/cat', newShape=[96,96], limit=125);
const input_dog = loadImages('./data/train/dog', newShape=[96,96], limit=125);
const input = tf.concat([input_cat, input_dog]);
const output = tf.concat([tf.ones([125]),tf.zeros([125])]).toFloat();
log(3, input.shape);
log(3, output.shape);
train().then(()=>{log(0, "Training complete")});

async function train(){
    const response = await model.fit(input, output, {
        epochs:200,
        shuffle: true,
    });
    model.predict(loadImage('./data/train/dog/dog.6535.jpg', newShape=[96,96])).print();
    model.predict(loadImage('./data/train/cat/cat.9200.jpg', newShape=[96,96])).print();
    console.log();
}