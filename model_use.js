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
//const modelPath = `file://${__dirname}/model/model.json`;
const modelPath = `file://${__dirname}/model_test/model.json`;
const shape = [128,128];
const inputShape = shape.concat([3]);
const ev = true;
const imgs = !false;

const modelSettings = {
    optimizer: tf.train.adam(0.001),
    loss: tf.losses.softmaxCrossEntropy,
    metrics:['accuracy']
};


//trolling
(async()=>{
    const model = await load(modelPath);
    if (imgs){
        model.predict(loadImage('./data/train/dog/dog.6130.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6131.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6132.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6136.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6138.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6139.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6142.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6143.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6145.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6147.jpg', newShape=shape)).print();
        
        model.predict(loadImage('./data/train/dog/dog.6144.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6146.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6148.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/dog/dog.6153.jpg', newShape=shape)).print();
        
        model.predict(loadImage('./data/train/cat/cat.9101.jpg', newShape=shape)).print();

        model.predict(loadImage('./data/train/cat/cat.1.jpg', newShape=shape)).print();
        model.predict(loadImage('./data/train/cat/cat.2.jpg', newShape=shape)).print();
    }
    

    if(ev){
        const data = loadImages('./data/train/test', newShape=shape);
        const ys = tf.oneHot(tf.cast(tf.concat([tf.ones([500]), tf.zeros([500])]), 'int32'), depth=2);
        model.compile(modelSettings);
        let res = model.evaluate(data, ys);
        log(3, `loss: ${res[0]}`);
        log(3, `metrics: ${res[1]}`);
    }
})();

/* dogs who are "cats": 
6130,
6131 (hard one)
6132 (blurry)
6136 (wtf)
6138 (fluffy)
6139 (two puppies)
6142 (color of a dog and asphalt is nearly the same)
6143 (fluffy dog front, back of a cat)
6145 idk
6147 puppy
*/