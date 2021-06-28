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

import time
import numpy as np

class Times(object):
    def __init__(self, keys=['preprocess_time', 'inference_time', 'postprocess_time'],
                warmup=0):
        super(Times, self).__init__()
        
        self.keys = keys
        self.warmup = warmup
        self.time_stamps = []
        self.stamp_count = 0
        self.reset()

    def reset(self):
        self.time_stamps = []
        self.times = {}
        for k in self.keys:
            self.times[k] = []
        self.init()
        self.total_time = []
        self.time_stamps = 0

    def init(self):
        self.stamp_count = 0
        self.time_stamps = []
    
    def start(self, stamp=True):
        self.init()
        if stamp is True:
            self.time_stamps.append(time.time())

    def end(self, stamp=False):
        if stamp is True:
            self.stamp()
        self.total_time.append(self.time_stamps[-1] - self.time_stamps[0])
        self.init()

    def stamp(self):
        
        if len(self.time_stamps) <1:
            self.start()
        
        self.stamp_count += 1

        self.time_stamps.append(time.time())

        if self.stamp_count > len(self.keys):
            self.keys.append(f"stamps_{self.stamp_count}")

        idx = (self.stamp_count - 1) % len(self.keys)
        _k = self.keys[idx] 
        if _k in self.times.keys():
            self.times[_k].append(self.time_stamps[-1] - self.time_stamps[-2])
        else:
            self.times[_k] = [self.time_stamps[-1] - self.time_stamps[-2]]

    def set_keys(self, keys):
        self.keys = keys
        self.init()
    
    def _num_counts(self):
        return len(self.total_time)

    def _report_total_time(self, mode='avg'):
        if mode=='avg':
            return self.mean(self.total_time)
        elif mode=='sum':
            return self.sum(self.total_time)
        else:
            return self.total_time

    def value(self, key=None, mode=None):
        #TODO: 
        # assert key in self.keys, f"Please set key:{key} as one of self.keys:{self.keys}"
        res = {}
        
        for k in self.keys:
            if mode=="mean":
                res[k] = self.mean(self.times[k])
            elif mode=="sum":
                res[k] = self.sum(self.times[k])
            else:
                res[k] = self.times[k]
        
        if key is None:
            return res

        for k in key:
            assert k in self.keys, f"Expect the element of {key} in self.keys: {self.keys}"
        return res

    def mean(self, lists):
        if len(lists) <= self.warmup:
            raise ValueError(f"The number {len(lists)} of time stamps must be larger than warmup: {self.warmup}")
        if len(lists) < 1:
            return 0.
        else:
            return np.mean(lists[self.warmup:])
    
    def sum(self, lists):
        if len(lists) <= self.warmup:
            raise ValueError(f"The number {len(lists)} of time stamps must be larger than warmup: {self.warmup}")
        if len(lists) < 1:
            return 0.
        else:
            return np.sum(lists[self.warmup:])


# if __name__ == "__main__":
#     mytime = Times()
#     for i in range(5):
#         mytime.start()
#         time.sleep(0.1)
#         mytime.stamp() # 1
#         time.sleep(0.2)
#         mytime.stamp() # 2
#         time.sleep(0.3)
#         mytime.stamp() # 3
#         time.sleep(0.4)
#         mytime.stamp() # 4
#         mytime.end(stamp=True) # 5
#     print(mytime.times)
#     print("keys: ", mytime.keys)
#     print(mytime.value(mode="mean"))
#     print(mytime.value(mode="sum"))
#     print(mytime.value(mode=None))
