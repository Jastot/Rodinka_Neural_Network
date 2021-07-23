const log = require('../js_common/logger.js').log;
const loadImageFromBuffer = require('../js_common/input.js').loadImageFromBuffer;

const express = require('express');
const app = express();
const morgan = require('morgan');
const tfjs = require('@tensorflow/tfjs-node');

const path = `file://${__dirname}/../models/model_2_2_7k/js/model.json`;
const port = 5005;
const shape=[224,224]

tfjs.loadLayersModel(path).then(model=>{
    app.use(morgan('combined'));
    app.use(express.json({limit:'24mb'}));
    app.post('/', (req, res)=>{
        if (req.is('application/json')){
            try {
                let body = req.body;
                let img = loadImageFromBuffer(Buffer.from(body.image,'base64'), newShape=shape);
                let modelResponse = model.predict(img).dataSync();
                if(req.accepts('json')){
                    res.json({
                        "isBenign":modelResponse[0],
                        "isMalignant":modelResponse[1]
                    })
                    return;
                }

            } catch(err) {
                err = err.toString();
                if(req.accepts('json')){
                    res.json({error:err});
                    return;
                }
                res.type('txt').send(err);
            }
        } else {
            if(req.accepts('json')){
                res.json({error:'use json; if already, set content-type header to application/json'});
                return;
            }
            res.type('txt').send('use json; if already, set content-type header to application/json');
        }
    });
    app.use('/', (req, res)=>{
        if(req.accepts('html')){
            res.sendStatus(403);
            return;
        }
        if(req.accepts('json')){
            res.json({error:"forbidden"});
            return;
        }
        res.type('txt').send('forbidden');
    })
    app.listen(port, 'localhost');
})
