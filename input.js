//require
const fs = require('fs');
const log = require('./logger.js').log;
const tf = require('@tensorflow/tfjs-node');

//do a little trolling...

//A LITTLE TROLLING
function loadImages(dir, limit = null){
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
                            .resizeNearestNeighbor([32,32])
                            .toFloat()
                            .div(tf.scalar(255.0))
                            .expandDims();
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
/*
(async()=>{
    log(3,loadImages('./data/not_ok'))
})()/*/

//exports
exports.loadImages = loadImages;