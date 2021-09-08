
# AutoLog

包含自动计时，统计CPU内存、GPU显存等信息，自动生成日志等功能。
```
├── auto_log
│   ├── autolog.py  # 包含AutoLogger类，规范日志
│   ├── device.py   # 包含统计CPU内存，GPU显存的函数
│   ├── ens.py      # 包含获取环境信息的函数
│   ├── util.py     # 包含自动计时的类

```

依赖环境：
```
python3
GPUtil
psutil
pynvml
distro
```

autolog编译安装：
```
git clone https://github.com/LDOUBLEV/AutoLog
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
```


使用方式可以参考[PR](https://github.com/PaddlePaddle/PaddleOCR/pull/3182/files)：

AutoLogger类参数说明：
```
model_name="det",       #  string 模型名字，可自定义
model_precision=args.precision,  # string 精度，'fp32', 'fp16', 'int8'
batch_size=1,           # int, batchsize大小
data_shape="dynamic",   # string, list, tuple, 模型输入的shape
save_path="./output/auto_log.lpg",  # string, 日志保存的路径
inference_config=self.config,       # paddle.infer.Config类，用于获取enable_mkldnn, enable_trt等信息 
pids=pid,               # int,  当前进程的pid，可用os.getpid()获取
process_name=None,      # string 当前进程的的名字，比如'Python',用于获取当前进程的pid号，默认为None
gpu_ids=0,              # int,  当前进程的GPU卡号，默认为None,
time_keys=None,         # list,  统计时间的键值，默认为['preprocess_time', 'inference_time', 'postprocess_time']
warmup=10               # int, warmup times,默认为0，warmup次数内，不会统计时间
```

# Updates
- 2021.8.5: 增加获取GPU信息的类GpuInfoV2，从nvidia-smi中获取GPU显存占用，可以不需要引入pynuml和GPUtil 这两个依赖

