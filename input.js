//require
const fs = require('fs');
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');

//do a little trolling...

//A LITTLE TROLLING
function loadImages(dir){
    var unsorted = fs.readdirSync(dir);
    var sorted = [];
    var tensors=[];
    log(0,`Loading images from ${__dirname+dir}`)
    unsorted.forEach(e=>{
        if(/(.+).png/.test(e.toLowerCase())||/(.+).jpg/.test(e.toLowerCase())||/(.+).jpeg/.test(e.toLowerCase())){
            sorted.push(e);
        }
    });
    sorted.forEach(e=>{
        let img_buffer = fs.readFileSync(`${dir}/${e}`);
        let img_tensor = tf.node.decodeImage(img_buffer)
                            .resizeNearestNeighbor([128,128])
                            .toFloat()
                            .div(tf.scalar(255.0))
        tensors.push(img_tensor);
    });
    log(0,"Loaded images");
    return tf.concat(tensors);
}

//maybe?
/*
function splitData(data){

}
*/

//mamba
(async()=>{
    log(3,loadImages('./not_ok'))
})()

//exports
exports.laodImages = loadImages;