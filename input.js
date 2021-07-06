//require
const fs = require('fs');
const log = require('./logger.js').log; //custom logger xd
const tf = require('@tensorflow/tfjs-node');

//do a little trolling...

//tezt  
/*var img_buffer = fs.readFileSync('./not_ok/test.jpg');
var img_tensor = tf.node.decodeImage(img_buffer);
log(2, img_tensor.shape);
var img_tensor2 = tf.image.resizeNearestNeighbor(img_tensor, [100,100]);
log(2, img_tensor2.shape);
log(0,img_tensor);*/

//A LITTLE TROLLING
async function loadImages(dir){
    var unsorted = fs.readdirSync(dir);
    var sorted = [];
    var tensors=[];
    await unsorted.forEach(e=>{
        if(/(.+).png/.test(e)||/(.+).jpg/.test(e)||/(.+).jpeg/.test(e)){
            sorted.push(e);
        }
    });
    await sorted.forEach(e=>{
        let img_buffer = fs.readFileSync(`${dir}/${e}`);
        let img_tensor = tf.node.decodeImage(img_buffer);
        let img_tensor_reshaped = img_tensor.resizeNearestNeighbor([512,512]);
        tensors.push(img_tensor_reshaped);
    });
    return tensors
}

//mamba
(async()=>{
    let mamba = await loadImages('./not_ok');
    await log(0,mamba);
})()

//exports
exports.loadImages = loadImages;