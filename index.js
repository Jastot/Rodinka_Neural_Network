//require
const fs = require('fs');
const log = require('./logger.js').log; //custom logger xd
const tf = require('@tensorflow/tfjs-node');

//code
var ok_unsorted = fs.readdirSync('./ok');
var ok = [];

var not_ok_unsorted = fs.readdirSync('./not_ok');
var not_ok = [];

//do a little trolling...
var img_buffer = fs.readFileSync('./not_ok/test.jpg');
var img_tensor = tf.node.decodeImage(img_buffer);
log(0,img_tensor);

///sfasdfasdfs
(async()=>{
    await ok_unsorted.forEach(e=>{
        if(/(.+).png/.test(e)||/(.+).jpg/.test(e)||/(.+).jpeg/.test(e)){
            ok.push(e);
        }
    })
    await not_ok_unsorted.forEach(e=>{
        if(/(.+).png/.test(e)||/(.+).jpg/.test(e)||/(.+).jpeg/.test(e)){
            not_ok.push(e);
        }
    })
    console.log(ok);
    console.log(not_ok);
})();