//require
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');

//model (custom)
const model = tf.sequential({layers:[
    tf.layers.dense({units:4, activation:'sigmoid', inputShape:[[2]]})
]});