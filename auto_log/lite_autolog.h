// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>
#include <numeric>


class AutoLogger {
public:
    AutoLogger(std::string model_name, 
               bool use_gpu,
               bool enable_tensorrt,
               bool enable_mkldnn,
               int cpu_threads,
               int batch_size,
               std::string input_shape,
               std::string model_precision,
               std::string power_mode,
               std::vector<double> time_info,
               int img_num) {
        this->model_name_ = model_name;
        this->use_gpu_ = use_gpu;
        this->enable_tensorrt_ = enable_tensorrt;
        this->enable_mkldnn_ = enable_mkldnn;
        this->cpu_threads_ = cpu_threads;
        this->power_mode = power_mode;
        this->batch_size_ = batch_size;
        this->input_shape_ = input_shape;
        this->model_precision_ = model_precision;
        this->time_info_ = time_info;
        this->img_num_ = img_num;
    }
    void report() {
        std::cout << "----------------------- Config info -----------------------" << std::endl;
        std::cout << "runtime_device: " << (this->use_gpu_ ? "gpu" : "cpu") << std::endl;
        std::cout << "ir_optim: " << "True" << std::endl;
        std::cout << "enable_memory_optim: " << "True" << std::endl;
        std::cout << "enable_tensorrt: " << this->enable_tensorrt_ << std::endl;
        std::cout << "enable_mkldnn: " << (this->enable_mkldnn_ ? "True" : "False") << std::endl;
        std::cout << "power_mode: " << this->power_mode << std::endl;
        std::cout << "cpu_math_library_num_threads: " << this->cpu_threads_ << std::endl;
        std::cout << "----------------------- Data info -----------------------" << std::endl;
        std::cout << "batch_size: " << this->batch_size_ << std::endl;
        std::cout << "input_shape: " << this->input_shape_ << std::endl;
        std::cout << "data_num: " << this->img_num_ << std::endl;
        std::cout << "----------------------- Model info -----------------------" << std::endl;
        std::cout << "model_name: " << this->model_name_ << std::endl;
        std::cout << "precision: " << this->model_precision_ << std::endl;
        std::cout << "----------------------- Perf info ------------------------" << std::endl;
        std::cout << "Total time spent(ms): " << std::accumulate(this->time_info_.begin(), this->time_info_.end(), 0) << std::endl;
        std::cout << "preprocess_time(ms): " << this->time_info_[0] / this->img_num_
                  << ", inference_time(ms): " << this->time_info_[1] / this->img_num_
                  << ", postprocess_time(ms): " << this->time_info_[2] / this->img_num_ << std::endl;
    }
        
private:
    std::string model_name_;
    bool use_gpu_ = false;
    bool enable_tensorrt_ = false;
    bool enable_mkldnn_ = true;
    int cpu_threads_ = 10;
    int batch_size_ = 1;
    std::string power_mode = power_mode;
    std::string input_shape_ = "dynamic";
    std::string model_precision_ = "fp32";
    std::vector<double> time_info_;
    int img_num_;
};
