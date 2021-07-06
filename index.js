//require
const fs = require('fs');
const log = require('./logger.js').log; //custom logger xd
const tf = require('@tensorflow/tfjs-node');

//do a little trolling...

//tezt 
var img_buffer = fs.readFileSync('./not_ok/test.jpg');
var img_tensor = tf.node.decodeImage(img_buffer);
log(2, img_tensor.shape);
var img_tensor2 = tf.image.resizeNearestNeighbor(img_tensor, [100,100]);
log(2, img_tensor2.shape);
log(0,img_tensor);

//A LITTLE TROLLING
async function loadImages(dir){
    var unsorted = fs.readdirSync(dir);
    var sorted = [];
    await unsorted.forEach(e=>{
        if(/(.+).png/.test(e)||/(.+).jpg/.test(e)||/(.+).jpeg/.test(e)){
            sorted.push(e);
        }
    });
    return sorted
}

///sfasdfasdfs
(async()=>{
    let prik = await loadImages('./ok');
    log(2, prik);
})()
