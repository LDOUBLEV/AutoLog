# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import time
import pynvml
import psutil
import GPUtil
import os
import paddle
from pathlib import Path
import logging
import sys
from .env import get_env_info
from .utils import Times
from .device import MemInfo, SubprocessGetMem


def get_infer_gpuid():
    cmd = "env | grep CUDA_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


class RunConfig:
    def __init(self,
               run_devices="cpu",
               ir_optim=False,
               enable_tensorrt=False,
               enable_mkldnn=False,
               cpu_threads=0,
               enable_mem_optim=True):

        self.run_devices = run_devices
        self.ir_optim = ir_optim
        self.enable_mkldnn = enable_mkldnn
        self.enable_tensorrt = enable_tensorrt
        self.cpu_math_library_num_threads = cpu_threads
        self.enable_mem_optim = enable_mem_optim


class AutoLogger(RunConfig):
    def __init__(self,
                 model_name,
                 model_precision="fp32",
                 batch_size=1,
                 data_shape="dynamic",
                 save_path=None,
                 inference_config=None,
                 pids=None,
                 process_name=None,
                 gpu_ids=None,
                 time_keys=[
                     'preprocess_time', 'inference_time', 'postprocess_time'
                 ],
                 warmup=0,
                 logger=None,
                 **kwargs):
        super(AutoLogger, self).__init__(**kwargs)
        self.autolog_version = 1.0
        self.save_path = save_path
        self.model_name = model_name
        self.precision = model_precision
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.paddle_infer_config = inference_config

        self.config_status = self.parse_config(self.paddle_infer_config)

        self.time_keys = time_keys
        self.times = Times(keys=time_keys, warmup=warmup)

        self.get_paddle_info()

        self.logger = self.init_logger() if logger is None else logger

        if pids is None:
            pids = os.getpid()
        self.pids = pids
        if gpu_ids == "auto":
            gpu_ids = get_infer_gpuid()
        self.gpu_ids = gpu_ids

        self.get_mem = SubprocessGetMem(pid=pids, gpu_id=gpu_ids)
        self.start_subprocess_get_mem()

    def start_subprocess_get_mem(self):
        self.get_mem.get_mem_subprocess_run(0.2)

    def end_subprocess_get_mem(self):
        self.get_mem.get_mem_subprocess_end()
        cpu_infos = self.get_mem.cpu_infos
        gpu_infos = self.get_mem.gpu_infos
        self.cpu_infos = cpu_infos[str(self.pids)]
        if self.gpu_ids is None:
            self.gpu_infos = {}
        else:
            self.gpu_infos = gpu_infos[str(self.gpu_ids)]
        return self.cpu_infos, self.gpu_infos

    def init_logger(self, name='root', log_level=logging.DEBUG):
        log_file = self.save_path

        logger = logging.getLogger(name)

        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
            datefmt="%Y/%m/%d %H:%M:%S")

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_file is not None:
            dir_name = os.path.dirname(log_file)
            if len(dir_name) > 0 and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.setLevel(log_level)
        return logger

    def parse_config(self, config) -> dict:
        """
        parse paddle predictor config
        args:
            config(paddle.inference.Config): paddle inference config
        return:
            config_status(dict): dict style config info
        """
        config_status = {}
        if config is not None and type(config) is not dict:
            config_status['runtime_device'] = "gpu" if config.use_gpu(
            ) else "cpu"
            config_status['ir_optim'] = config.ir_optim()
            config_status['enable_tensorrt'] = config.tensorrt_engine_enabled()
            config_status['precision'] = self.precision
            config_status['enable_mkldnn'] = config.mkldnn_enabled()
            config_status[
                'cpu_math_library_num_threads'] = config.cpu_math_library_num_threads(
                )
        elif type(config) is dict:
            config_status['runtime_device'] = config[
                'runtime_device'] if 'runtime_device' in config else None
            config_status['ir_optim'] = config[
                'ir_optim'] if 'ir_optim' in config else None
            config_status['enable_tensorrt'] = config[
                'enable_tensorrt'] if 'enable_tensorrt' in config else None
            config_status['precision'] = config[
                'precision'] if 'precision' in config else None
            config_status['enable_mkldnn'] = config[
                'enable_mkldnn'] if 'enable_mkldnn' in config else None
            config_status['cpu_math_library_num_threads'] = config[
                'cpu_math_library_num_threads'] if 'cpu_math_library_num_threads' in config else None
        else:
            config_status['runtime_device'] = "None"
            config_status['ir_optim'] = "None"
            config_status['enable_tensorrt'] = "None"
            config_status['precision'] = self.precision
            config_status['enable_mkldnn'] = "None"
            config_status['cpu_math_library_num_threads'] = "None"
        return config_status

    def get_paddle_info(self):
        self.paddle_version = paddle.__version__
        self.paddle_commit = paddle.__git_commit__

    def report(self, identifier=None):
        #TODO: support multi-model report
        """
        print log report
        args:
            identifier(string): identify log
        """
        if identifier:
            identifier = f"[{identifier}]"
        else:
            identifier = ""

        # report time
        _times_value = self.times.value(key=self.time_keys, mode='mean')
        preprocess_time_ms = round(_times_value['preprocess_time'] * 1000, 4)
        inference_time_ms = round(_times_value['inference_time'] * 1000, 4)
        postprocess_time_ms = round(_times_value['postprocess_time'] * 1000, 4)
        data_num = self.times._num_counts()
        total_time_s = round(self.times._report_total_time(mode='sum'), 4)

        # report memory
        cpu_infos, gpu_infos = self.end_subprocess_get_mem()

        cpu_rss_mb = self.cpu_infos['cpu_rss']
        gpu_rss_mb = self.gpu_infos[
            'memory.used'] if self.gpu_ids is not None else None
        gpu_util = self.gpu_infos[
            'utilization.gpu'] if self.gpu_ids is not None else None

        # report env
        envs = get_env_info()

        self.logger.info("\n")
        self.logger.info(
            "---------------------- Env info ----------------------")
        # envs['nvidia_driver_version'] envs['cudnn_version']envs['cuda_version'] envs['os_info']
        self.logger.info(f"{identifier} OS_version: {envs['os_info']}")
        self.logger.info(f"{identifier} CUDA_version: {envs['cuda_version']}")
        self.logger.info(
            f"{identifier} CUDNN_version: {envs['cudnn_version']}")
        self.logger.info(
            f"{identifier} drivier_version: {envs['nvidia_driver_version']}")
        self.logger.info(
            "---------------------- Paddle info ----------------------")
        self.logger.info(f"{identifier} paddle_version: {self.paddle_version}")
        self.logger.info(f"{identifier} paddle_commit: {self.paddle_commit}")
        self.logger.info(
            f"{identifier} log_api_version: {self.autolog_version}")
        self.logger.info(
            "----------------------- Conf info -----------------------")
        self.logger.info(
            f"{identifier} runtime_device: {self.config_status['runtime_device']}"
        )
        self.logger.info(
            f"{identifier} ir_optim: {self.config_status['ir_optim']}")
        self.logger.info(f"{identifier} enable_memory_optim: {True}")
        self.logger.info(
            f"{identifier} enable_tensorrt: {self.config_status['enable_tensorrt']}"
        )
        self.logger.info(
            f"{identifier} enable_mkldnn: {self.config_status['enable_mkldnn']}"
        )
        self.logger.info(
            f"{identifier} cpu_math_library_num_threads: {self.config_status['cpu_math_library_num_threads']}"
        )
        self.logger.info(
            "----------------------- Model info ----------------------")
        self.logger.info(f"{identifier} model_name: {self.model_name}")
        self.logger.info(f"{identifier} precision: {self.precision}")
        self.logger.info(
            "----------------------- Data info -----------------------")
        self.logger.info(f"{identifier} batch_size: {self.batch_size}")
        self.logger.info(f"{identifier} input_shape: {self.data_shape}")
        self.logger.info(f"{identifier} data_num: {data_num}")
        self.logger.info(
            "----------------------- Perf info -----------------------")
        self.logger.info(
            f"{identifier} cpu_rss(MB): {cpu_rss_mb}, gpu_rss(MB): {gpu_rss_mb}, gpu_util: {gpu_util}%"
        )
        self.logger.info(f"{identifier} total time spent(s): {total_time_s}")
        self.logger.info(
            f"{identifier} preprocess_time(ms): {preprocess_time_ms}, inference_time(ms): {inference_time_ms}, postprocess_time(ms): {postprocess_time_ms}"
        )

    def print_help(self):
        """
        print function help
        """
        print("""Usage: 
            ==== Print inference benchmark logs. ====
            config = paddle.inference.Config()
            model_info = {'model_name': 'resnet50'
                          'precision': 'fp32'}
            data_info = {'batch_size': 1
                         'shape': '3,224,224'
                         'data_num': 1000}
            perf_info = {'preprocess_time_s': 1.0
                         'inference_time_s': 2.0
                         'postprocess_time_s': 1.0
                         'total_time_s': 4.0}
            resource_info = {'cpu_rss_mb': 100
                             'gpu_rss_mb': 100
                             'gpu_util': 60}
            log = PaddleInferBenchmark(config, model_info, data_info, perf_info, resource_info)
            log('Test')
            """)


# if __name__ == "__main__":
#     get_os_info()
#     print(envs['os_info'])
#     get_cudnn_info()
#     print(envs['cudnn_version'])
