const fs = require('fs');

const log = require('./logger.js').log;

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

var ok_unsorted = fs.readdirSync('./ok');
var ok = [];

var not_ok_unsorted = fs.readdirSync('./not_ok');
var not_ok = [];

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



