import os
import sys
import json
import socket

HOST = '127.0.0.1'
PORT = 65433

if len(sys.argv) != 2:
    print('usage:', sys.argv[0], 'model.onnx')

model_path = sys.argv[1]
model_path = os.path.abspath(model_path)

params = {
    'model' : model_path,
    'output_dir' : os.path.dirname(model_path),
}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(bytes(json.dumps(params), 'utf-8'))
    ret = s.recv(1024)
    ret = json.loads(ret.decode())

print(f'Received {ret}')

