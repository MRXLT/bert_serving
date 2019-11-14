# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import tarfile
import bert_serving
import subprocess
import imp

try:
    imp.find_module('paddlehub')
    paddlehub_found = True
    print('Working with paddlehub')
except ImportError:
    paddlehub_found = False


class BertServer():
    def __init__(self, with_gpu=True):
        os.chdir(self.get_path())
        self.with_gpu_flag = with_gpu
        self.p_list = []
        self.use_other_model = False
        self.model_url = 'https://paddle-serving.bj.bcebos.com/data/bert'
        self.cpu_run_cmd = './bin/serving-cpu --bthread_min_concurrency=4 --bthread_concurrency=4 '
        self.gpu_run_cmd = './bin/serving-gpu --bthread_min_concurrency=4 --bthread_concurrency=4 '
        os.system(
            'cp ./conf/model_toolkit.prototxt.bk ./conf/model_toolkit.prototxt')

        if self.with_gpu_flag:
            with open('./conf/model_toolkit.prototxt', 'r') as f:
                conf_str = f.read()
            conf_str = re.sub('CPU', 'GPU', conf_str)
            conf_str = re.sub('}', '  enable_memory_optimization: 1\n}',
                              conf_str)
            open('./conf/model_toolkit.prototxt', 'w').write(conf_str)

    def help(self):
        print("hello")

    def run(self, gpu_index=0, port=8010):
        if self.with_gpu_flag == True:
            gpu_msg = '--gpuid=' + str(gpu_index) + ' '
            run_cmd = self.gpu_run_cmd + gpu_msg
            run_cmd += '--port=' + str(port) + ' '
            print('Start serving on gpu ' + str(gpu_index) + ' port = ' + str(
                port))
        else:
            re = subprocess.Popen(
                'cat /usr/local/cuda/version.txt > tmp 2>&1', shell=True)
            re.wait()
            if re.returncode == 0:
                run_cmd = self.gpu_run_cmd + '--port=' + str(port) + ' '
            else:
                run_cmd = self.cpu_run_cmd + '--port=' + str(port) + ' '

        process = subprocess.Popen(run_cmd, shell=True)
        self.p_list.append(process)

    def run_multi(self, gpu_index_list=[], port_list=[]):
        if len(port_list) < 1:
            print('Please set one port at least.')
            return -1
        if self.with_gpu_flag == True:
            if len(gpu_index_list) != len(port_list):
                print('Expect same length of gpu_index_list and port_list.')
                return -1
            for gpu_index, port in zip(gpu_index_list, port_list):
                self.run(gpu_index=gpu_index, port=port)
        else:
            for port in port_list:
                self.run(port=port)

    def stop(self):
        for p in self.p_list:
            p.kill()

    def show_conf(self):
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        print(conf_str)

    def with_model(self, model_name, model_url=None):
        '''
        if model_url != None:
            self.mode_url = model_url
            self.use_other_model = True
        '''
        os.chdir(self.get_path())
        self.get_model(model_name)

    def get_path(self):
        py_path = os.path.dirname(bert_serving.__file__)
        server_path = os.path.join(py_path, 'server')
        return server_path

    def get_model(self, model_name):
        server_path = self.get_path()
        if not paddlehub_found or self.use_other_model:
            tar_name = model_name + '.tar.gz'
            model_url = self.model_url + '/' + tar_name

            model_path = os.path.join(server_path, 'data/model/paddle/fluid')
            if not os.path.exists(model_path):
                os.makedirs('data/model/paddle/fluid')
            os.chdir(model_path)
            if os.path.exists(model_name):
                pass
            else:
                os.system('wget ' + model_url + ' --no-check-certificate')
                tar = tarfile.open(tar_name)
                tar.extractall()
                tar.close()
                os.remove(tar_name)

            model_path_str = r'model_data_path: "./data/model/paddle/fluid/' + model_name + r'"'

        else:
            import paddlehub as hub
            import paddle.fluid as fluid

            paddlehub_modules_path = os.path.expanduser('~/.paddlehub')
            paddlehub_bert_path = os.path.join(paddlehub_modules_path,
                                               'bert_service')
            model_path = os.path.join(paddlehub_bert_path, model_name)
            model_path_str = r'model_data_path: "' + model_path + r'"'

            if not os.path.exists(model_path):
                print('Save model for serving ...')
                os.makedirs(model_path)
                module = hub.Module(name=model_name)
                inputs, outputs, program = module.context(
                    trainable=True, max_seq_len=128)
                place = fluid.core_avx.CPUPlace()
                exe = fluid.Executor(place)
                input_ids = inputs["input_ids"]
                position_ids = inputs["position_ids"]
                segment_ids = inputs["segment_ids"]
                input_mask = inputs["input_mask"]
                feed_var_names = [
                    input_ids.name, position_ids.name, segment_ids.name,
                    input_mask.name
                ]
                target_vars = [
                    outputs["pooled_output"], outputs["sequence_output"]
                ]
                fluid.io.save_inference_model(
                    feeded_var_names=feed_var_names,
                    target_vars=target_vars,
                    main_program=program,
                    executor=exe,
                    dirname=model_path)

        os.chdir(server_path)
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        open('./conf/model_toolkit.prototxt', 'w').write(
            re.sub('model_data_path.*"', model_path_str, conf_str))
