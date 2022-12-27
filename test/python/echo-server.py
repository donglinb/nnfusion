import os
import sys
import json
import socket
import subprocess
from contextlib import contextmanager
from wslpath import wslpath


if len(sys.argv) != 2:
    print('usage:', sys.argv[0], ' path-to-nnfusion-executable')
    sys.exit(0)

HOST = '127.0.0.1'
PORT = 65433

cmd_options = [
    '-f onnx',
    '-p "batch_size:1"',
    '-fmulti_shape=false',
    '-fort_folding=false',
    '-fdefault_device=HLSL',
    '-fhlsl_codegen_type=cpp',
    '-fantares_mode=true',
    '-fblockfusion_level=0',
    '-fkernel_fusion_level=0',
    '-fkernel_tuning_steps=0',
    '-ffold_where=0',
    '-fsymbolic=0',
    '-fsplit_softmax=0',
    '-fhost_entry=0',
    '-fir_based_fusion=1',
    '-fextern_result_memory=1',
    '-fuse_cpuprofiler=1',
    '-ftuning_platform="win64"',
    '-fantares_codegen_server=127.0.0.1:8880',
]
exec_path = os.path.abspath(sys.argv[1])

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def execute(command, redirect_stderr=True, shell=True, **kwargs):
    stderr = subprocess.STDOUT if redirect_stderr else None
    try:
        output = subprocess.check_output(command,
                                         stderr=stderr,
                                         shell=shell,
                                         encoding="utf8",
                                         **kwargs)
    except subprocess.CalledProcessError as e:
        raise e
    return output


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            params = ''
            print(f'Connected by {addr}')
            while True:
                data = conn.recv(10240)
                if not data:
                    break
                params += data.decode()
                ret = {
                    'ret' : True,
                    'error' : '',
                }
                try:
                    params = json.loads(params)
                    model_path = wslpath(params['model'])
                    output_dir = wslpath(params['output_dir'])
                    with cd(output_dir):
                        cmd = ' '.join([exec_path, model_path] + cmd_options)
                        # subprocess.run(cmd, shell = True)
                        out = execute(cmd)
                    print('model_path:', model_path)
                    print('output_dir:', output_dir)
                    print('params:', params)
                    print(out)
                except Exception as e:
                    print(e)
                    ret['ret'] = False
                    ret['error'] = str(e)
                conn.sendall(bytes(json.dumps(ret), 'utf-8'))
            
