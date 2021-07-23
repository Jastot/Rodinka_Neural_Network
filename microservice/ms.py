import json
import os
import sys
import base64
from io import BytesIO
from PIL import Image
import http.server
import socketserver
import numpy as np

import tensorflow as tf
import visualkeras


# functions
def load(path):
    model = tf.keras.models.load_model(path)
    return model

model_path=f"./../models/model_2_2_3k/py"
shape=(224,224)
port = 5005

model = load(model_path)

class Handler(http.server.BaseHTTPRequestHandler):
    def usecnn(self, data):
        imgbytes = Image.open(BytesIO(base64.b64decode(data)))
        imgbytes.load()
        data = np.asarray(imgbytes, dtype="float32")
        data = data[:,:,::-1]
        data = data/255
        data = np.expand_dims(data, axis=0)
        tensor = tf.convert_to_tensor(data)
        tensor = tf.image.resize(tensor, size=shape, method='nearest')    
        resp = model.predict(tensor)
        return resp

    def do_POST(self):
        try:
            cl = int(self.headers.get('Content-Length'))
            data = json.loads(self.rfile.read(cl).decode('utf-8'))
            if ('image' in data.keys()):
                print('in')
                cnnresp = self.usecnn(data['image'])
                response = json.dumps({
                    "isBenign":float(cnnresp[0,0]),
                    "isMalignant":float(cnnresp[0,1])
                }).encode('utf-8')
                self.send_response(200, 'AMOGUS')
                self.send_header('content-type','application/json; charset=utf-8')
                self.send_header('content-length',len(str(response)))
                self.end_headers()
                self.wfile.write(response)
                return
        except Exception as e:
            print(e)
        self.send_response(400, 'ABOBUS')
        self.end_headers()
        self.wfile.write('badreq'.encode('utf-8'))
        return

def run(server_class=http.server.HTTPServer, handler_class=Handler):
    server_address = ('localhost', port)
    httpd = server_class(server_address, handler_class)

    httpd.serve_forever()

run()