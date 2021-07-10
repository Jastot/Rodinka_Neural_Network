import os
import json

tf_config = {
    "cluster":{
        "worker":["localhost:1234"]
    },
    "task":{
        "type":"worker",
        "index":0
    }
}

os.environ['TF_CONFIG']=json.dumps(tf_config)