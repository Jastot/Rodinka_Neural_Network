import os
import json

tf_config = {
    "cluster":{
        "worker":["192.168.0.1:5555", "192.168.0.2:5555", "192.168.0.3:5555"]
    },
    "task":{
        "type":"worker",
        "index":0
    }
}
md_config = {
    "epochs":5,
    "spe":10
}

print(json.dumps(tf_config))
print(json.dumps(md_config))

os.environ['TF_CONFIG']=json.dumps(tf_config)
os.environ['MD_CONFIG']=json.dumps(tf_config)

print(os.environ['TF_CONFIG'])
print(os.environ['MD_CONFIG'])