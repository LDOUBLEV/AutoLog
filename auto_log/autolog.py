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
from .env import get_env_info
from .utils import Times
from .device import MemInfo


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
        self.cpu_math_library_num_threads = self.cpu_threads
        self.enable_mem_optim = enable_mem_optim


class AutoLogger(RunConfig):
    def __init__(self,
                 model_name,
                 model_precision,
                 batch_size,
                 data_shape,
                 save_path,
                 inference_config=None,
                 pids=None, 
                 process_name=None, 
                 gpu_ids=None, 
                 time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
                 warmup=0,
                 **kwargs):
        super(AutoLogger, self).__init__()
        self.autolog_version = 1.0
        self.save_path = save_path
        self.model_name = model_name
        self.precision = model_precision
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.paddle_infer_config = inference_config

        self.config_status = self.parse_config(self.paddle_infer_config)

        self.time_keys = time_keys
        self.times = Times(keys=time_keys,warmup=warmup)

        self.get_paddle_info()
        self.init_logger()
        self.mem_info = MemInfo(pids=pids, gpu_id=gpu_ids)

    def init_logger(self):
        """
        benchmark logger
        """
        # Init logger
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_output = f"{self.save_path}/{self.model_name}.log"
        Path(f"{self.save_path}").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=FORMAT,
            handlers=[
                logging.FileHandler(
                    filename=log_output, mode='w'),
                logging.StreamHandler(),
            ])
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Paddle Inference benchmark log will be saved to {log_output}")
    
    def get_avg_mem_mb(self):
        self.cpu_infos, self.gpu_infos = self.mem_info.get_avg_mem_mb()

    def reset(self):
        pass

    def get_mem(self, pid, gpu_id, q, interval=1.0):
        mem_info = MemInfo([pid], [gpu_id])
        while True:
            cpu_infos, gpu_infos = mem_info.summary_mem()
            q.put()
            time.sleep(interval)
        return 

    
    def parse_config(self, config) -> dict:
        """
        parse paddle predictor config
        args:
            config(paddle.inference.Config): paddle inference config
        return:
            config_status(dict): dict style config info
        """
        config_status = {}
        config_status['runtime_device'] = "gpu" if config.use_gpu() else "cpu"
        config_status['ir_optim'] = config.ir_optim()
        config_status['enable_tensorrt'] = config.tensorrt_engine_enabled()
        config_status['precision'] = self.precision
        config_status['enable_mkldnn'] = config.mkldnn_enabled()
        config_status[
            'cpu_math_library_num_threads'] = config.cpu_math_library_num_threads(
            )
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

        # cpu_rss_mb, gpu_rss_mb, gpu_util = self.GetMem.report()

        _times_value = self.times.value(key=self.time_keys, mode='mean')
        preprocess_time_ms = _times_value['preprocess_time'] * 1000
        inference_time_ms = _times_value['inference_time'] * 1000
        postprocess_time_ms = _times_value['postprocess_time'] * 1000
        data_num = self.times._num_counts()
        total_time_s = self.times._report_total_time(mode='sum')

        envs = get_env_info()

        self.logger.info("\n")
        self.logger.info(
            "---------------------- Env info ----------------------")
            # envs['nvidia_driver_version'] envs['cudnn_version']envs['cuda_version'] envs['os_info']
        self.logger.info(f"{identifier} OS_version: {envs['os_info']}")
        self.logger.info(f"{identifier} CUDA_version: {envs['cuda_version']}")
        self.logger.info(f"{identifier} CUDNN_version: {envs['cudnn_version']}")
        self.logger.info(f"{identifier} drivier_version: {envs['nvidia_driver_version']}")
        self.logger.info(
            "---------------------- Paddle info ----------------------")
        self.logger.info(f"{identifier} paddle_version: {self.paddle_version}")
        self.logger.info(f"{identifier} paddle_commit: {self.paddle_commit}")
        self.logger.info(f"{identifier} log_api_version: {self.autolog_version}")
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
            f"{identifier} enable_mkldnn: {self.config_status['enable_mkldnn']}")
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
        # self.logger.info(
        #     f"{identifier} cpu_rss(MB): {cpu_rss_mb}, gpu_rss(MB): {gpu_rss_mb}, gpu_util: {gpu_util}%"
        # )
        self.logger.info(
            f"{identifier} total time spent(s): {total_time_s}")
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
    