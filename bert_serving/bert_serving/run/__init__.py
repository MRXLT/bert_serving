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


class BertServer():
    def __init__(self, port=8010):
        os.chdir(self.get_path())
        self.with_gpu_flag = False
        self.gpuid = [0]
        self.port = [port]
        self.model_url = 'https://paddle-serving.bj.bcebos.com/data/bert'
        self.cpu_run_cmd = './bin/serving-cpu --bthread_min_concurrency=4 --bthread_concurrency=4 '
        self.gpu_run_cmd = './bin/serving-gpu --bthread_min_concurrency=4 --bthread_concurrency=4 '
        self.p_list = []
        os.system(
            'cp ./conf/model_toolkit.prototxt.bk ./conf/model_toolkit.prototxt')

    def help(self):
        print("hello")

    def run(self):
        cmd_list = []
        if self.with_gpu_flag == True:
            for index, gpuid in enumerate(self.gpuid):
                gpu_msg = '--gpuid=' + str(gpuid) + ' '
                run_cmd = self.gpu_run_cmd + gpu_msg
                run_cmd += '--port=' + str(self.port[0] + index) + ' '
                cmd_list.append(run_cmd)
                print('Start serving on gpu ' + str(gpuid) + ' port = ' + str(
                    self.port[0] + index))
        else:
            re = subprocess.Popen(
                'cat /usr/local/cuda/version.txt > tmp 2>&1', shell=True)
            re.wait()
            if re.returncode == 0:
                cmd_list.append(self.gpu_run_cmd)
            else:
                cmd_list.append(self.cpu_run_cmd)
            print('Start serving on cpu port = ' + str(self.port[0]))
        for cmd in cmd_list:
            process = subprocess.Popen(cmd, shell=True)
            self.p_list.append(process)

    def stop(self):
        for p in self.p_list:
            p.kill()

    def set_model_url(self, url):
        self.model_url = url

    def show_conf(self):
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        print(conf_str)

    def with_model(self, model_name):
        os.chdir(self.get_path())
        self.get_model(model_name)

    def with_gpu(self, gpuid=0):
        self.with_gpu_flag = True
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        conf_str = re.sub('CPU', 'GPU', conf_str)
        conf_str = re.sub('}', '  enable_memory_optimization: 1\n}', conf_str)
        open('./conf/model_toolkit.prototxt', 'w').write(conf_str)
        if type(gpuid) == int:
            self.gpuid = [gpuid]
        if type(gpuid) == list:
            self.gpuid = gpuid

    def get_path(self):
        py_path = os.path.dirname(bert_serving.__file__)
        server_path = os.path.join(py_path, 'server')
        return server_path

    def get_model(self, model_name):
        tar_name = model_name + '.tar.gz'
        model_url = self.model_url + '/' + tar_name

        server_path = self.get_path()
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

        os.chdir(server_path)
        model_path_str = r'model_data_path: "./data/model/paddle/fluid/' + model_name + r'"'
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        open('./conf/model_toolkit.prototxt', 'w').write(
            re.sub('model_data_path.*"', model_path_str, conf_str))
