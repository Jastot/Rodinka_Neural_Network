//require
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');
const loadImages = require('./input.js').loadImages;
const loadImage = require('./input.js').loadImage;

//functions
async function load(path){
    const model = await tf.loadLayersModel(path);
    return model;
}

//const
const modelPath = `file://${__dirname}/model/model.json`;
const shape = [32,32];

//trolling
(async()=>{
    const model = await load(modelPath);

    model.predict(loadImage('./data/train/dog/dog.6625.jpg', newShape=shape)).print();
    model.predict(loadImage('./data/train/cat/cat.9101.jpg', newShape=shape)).print();
})();