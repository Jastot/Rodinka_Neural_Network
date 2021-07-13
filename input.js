//require
const fs = require('fs');
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');

//do a little trolling...
function loadImageFromBuffer(buffer, newShape=[32,32]){
    log(0, `Loading image from buffer`);
    var tensor = tf.node.decodeImage(buffer)
                    .resizeNearestNeighbor(newShape)
                    .toFloat()
                    .div(tf.scalar(255.0))
                    .expandDims()
    log(0, "Loaded image");
    return tensor;
}

function loadImage(path, newShape=[32, 32]){
    log(0,`Loading image ${__dirname+path}`)
    var buffer = fs.readFileSync(path);
    var tensor = tf.node.decodeImage(buffer)
                    .resizeNearestNeighbor(newShape)
                    .toFloat()
                    .div(tf.scalar(255.0))
                    .expandDims();
    log(0,"Loaded image");
    let output = tensor;
    return output;
}

function loadImages(dir, newShape=[32, 32], limit = null){
    log(0,`Loading images from ${__dirname+dir}`)
    var unsorted = fs.readdirSync(dir);
    var sorted = [];
    var tensors=[];

    unsorted.forEach(e=>{
        if(/(.+).png/.test(e.toLowerCase())||/(.+).jpg/.test(e.toLowerCase())||/(.+).jpeg/.test(e.toLowerCase())){
            if(typeof(limit)=="number"){
                if(sorted.length<limit){
                    sorted.push(e);
                }
            } else {
                sorted.push(e);
            }
        }
    });
    sorted.forEach(e=>{
        let img_buffer = fs.readFileSync(`${dir}/${e}`);
        let img_tensor = tf.node.decodeImage(img_buffer)
                            .resizeNearestNeighbor(newShape)
                            .toFloat()
                            .div(tf.scalar(255.0))
                            .expandDims();
        tensors.push(img_tensor);
    });
    log(0,"Loaded images");
    let output = tf.concat(tensors);
    return output;
}

//exports
exports.loadImages = loadImages;
exports.loadImage = loadImage;
exports.loadImageFromBuffer = loadImageFromBuffer;