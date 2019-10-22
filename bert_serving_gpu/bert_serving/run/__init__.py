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


class BertServer():
    def __init__(self, port=8010):
        os.chdir(self.get_path())
        self.with_gpu_flag = False
        self.gpuid = 0
        self.port = port
        self.model_url = '127.0.0.1:8099'
        os.system(
            'cp ./conf/model_toolkit.prototxt.bk ./conf/model_toolkit.prototxt')

    def help(self):
        print("hello")

    def set_model_url(self, url):
        self.model_url = url

    def show_conf(self):
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        print(conf_str)

    def with_model(self, model_name):
        os.chdir(self.get_path())
        self.get_model(model_name)

        run_cmd = './bin/serving --bthread_min_concurrency=4 --bthread_concurrency=4 '
        run_cmd += '--port=' + str(self.port) + ' '

        if self.with_gpu_flag == True:
            gpu_msg = '--gpuid=' + str(self.gpuid) + ' '
            run_cmd += gpu_msg

        os.system(run_cmd)

    def with_gpu(self, gpuid=0):
        self.with_gpu_flag = True
        self.gpuid = gpuid
        with open('./conf/model_toolkit.prototxt', 'r') as f:
            conf_str = f.read()
        conf_str = re.sub('CPU', 'GPU', conf_str)
        conf_str = re.sub('}', '  enable_memory_optimization: 1\n}', conf_str)
        open('./conf/model_toolkit.prototxt', 'w').write(conf_str)

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
            os.system('wget ' + model_url)
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
