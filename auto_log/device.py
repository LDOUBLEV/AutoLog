import psutil  
import os 
import json
import time
import multiprocessing
import numpy as np


class CpuInfo(object):
    @staticmethod
    def get_disk_info(path):
        G = 1024*1024
        diskinfo = psutil.disk_usage(path)
        info = "path:%s  total:%dG,  used:%dG,  free:%dG,  used_percent:%d%%"%(path,
                                    diskinfo.total/G, diskinfo.used/G, diskinfo.free/G, diskinfo.percent)
        return info

    @staticmethod
    def get_disk_partitions():
        return psutil.disk_partitions()

    @staticmethod
    def get_current_process_pid():
        pids = psutil.pids()
        return pids

    @staticmethod
    def get_process_info(pid):
        p = psutil.Process(pid) 
        info = "name:{}  pid:{}  \nstatus:{}  \ncreate_time:{}  \ncpu_times:{}  \nmemory_percent:{}  \nmemory_info:{}  \nio_counters：{}  \nnum_threads：{}".format(p.name(), 
                        pid, p.status(), p.create_time(), p.cpu_times(), p.memory_percent(), p.memory_info(), p.io_counters(), p.num_threads())
        return info
    
    @staticmethod
    def get_cpu_current_memory_mb(pid):
        info = {}
        p = psutil.Process(pid) 
        mem_info = p.memory_info()
        info['cpu_rss'] = round(mem_info.rss/1024/1024, 4)
        info['memory_percent'] = round(p.memory_percent(), 4)
        return info


class GpuInfo(object):
    def __init__(self):
        import pynvml  
        import GPUtil
        # init
        pynvml.nvmlInit()
    
    def get_gpu_name(self):
        name = pynvml.nvmlDeviceGetName(handle)
        return name.decode('utf-8')

    def get_gpu_device(self):
        deviceCount = pynvml.nvmlDeviceGetCount()
        gpu_list = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_list.append(i)
        return gpu_list

    def get_free_rate(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_rate = int((info.free / info.total) * 100)
        return free_rate

    def get_used_rate(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_rate = int((info.used / info.total) * 100)
        return used_rate
    
    def get_gpu_util(self, gpu_id):
        GPUs = GPUtil.getGPUs()
        gpu_util = GPUs[gpu_id].load
        return gpu_util

    def get_gpu_info(self, gpu_id):
        self.init()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        M = 1024*1024
        gpu_info = {}
        gpu_info['total'] = info.total/M
        gpu_info['free'] = info.free/M
        gpu_info['used'] = info.used/M
        gpu_info['util'] = self.get_gpu_util(gpu_id)
        return gpu_info

    def release(self):
        pynvml.nvmlShutdown()

class GpuInfoV2(object):
    def __init__(self):
        self.gpu_info = {}
        self.default_att = (
                    'index',
                    'uuid',
                    'name',
                    'timestamp',
                    'memory.total',
                    'memory.free',
                    'memory.used',
                    'utilization.gpu',
                    'utilization.memory'
                )

    def get_gpu_info(self, gpu_id, nvidia_smi_path='nvidia-smi', no_units=True):
        """
        The run time of get_gpu_info are about 70ms.
        """
        keys = self.default_att
        nu_opt = '' if not no_units else ',nounits'
        cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
        f = os.popen(cmd)
        lines = f.readlines()
        lines = [ line.strip() for line in lines if line.strip() != '' ]

        gpu_info_list = [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

        gpu_info={}
        gpu_info["memory.used"] = float(gpu_info_list[gpu_id]["memory.used"])
        gpu_info["utilization.gpu"] = float(gpu_info_list[gpu_id]["utilization.gpu"])
        return gpu_info

class MemInfo(CpuInfo):
    def __init__(self, pids=None, gpu_id=None):

        self.pids = self.check_pid(pids)
        if gpu_id is not None:
            self.gpuinfo = GpuInfoV2()
            self.gpu_id = self.check_gpu_id(gpu_id)
            # self.gpuinfo.init()
        else:
            self.gpuinfo = None
            self.gpu_id = self.check_gpu_id(gpu_id)

        self.cpu_infos = {}
        self.gpu_infos = {}
    
    def get_cpu_mem(self):
        cpu_mem = {}
        for pid in self.pids:
            cpu_mem[pid] = self.get_cpu_current_memory_mb(pid)
        return cpu_mem

    def get_gpu_mem(self):
        if gpuinfo is None:
            return None
        else:
            gpu_info = {}
            for id in self.gpu_id:
                gpu_info[id] = self.gpuinfo.get_gpu_info(id)
            return gpu_info

    def check_pid(self, pids):
        if type(pids)==int:
            return [pids]
        elif type(pids)==tuple:
            return [p for p in pids]
        elif type(pids)==list:
            return pids
        else:
            raise ValueError("Expect list of int input about pids, but get {} with type {}".format(pids, type(pids)))
    
    def check_gpu_id(self, gpu_id):
        if gpu_id is None:
            return []
        elif type(gpu_id)==int:
            return [gpu_id]
        elif type(gpu_id)==tuple:
            return [p for p in gpu_id]
        elif type(gpu_id)==list:
            return gpu_id
        else:
            raise ValueError("Expect list of int input about gpu_id, but get {} with type {}".format(gpu_id, type(gpu_id)))

    def summary_mem(self, return_str=True):
        cpu_infos = {}
        for p in self.pids:
            cpu_infos[str(p)] = self.get_cpu_current_memory_mb(p)
        
        gpu_infos = {}
        for g in self.gpu_id:
            gpu_infos[str(g)] = self.gpuinfo.get_gpu_info(g)
        
        if return_str is False:
            return cpu_infos, gpu_infos
        else:
            return json.dumps(cpu_infos) + "\n" + json.dumps(gpu_infos)
    
    def get_avg_mem_mb(self):
        cpu_infos, gpu_infos = self.summary_mem(return_str=False)
        if len(self.cpu_infos.keys()) < 1:
            self.cpu_infos = cpu_infos
        else:
            for p in self.pids:
                for k in cpu_infos[str(p)].keys():
                    v = cpu_infos[str(p)][k]
                    self.cpu_infos[str(p)][k] = np.mean([v, self.cpu_infos[str(p)][k]])
        if len(self.gpu_infos.keys()) < 1:
            self.gpu_infos = gpu_infos
        else:
            for g in self.gpu_id:
                for k in gpu_infos[str(g)].keys():
                    #self.gpu_infos[str(g)][k] = np.mean([v, self.gpu_infos[str(g)][k]])
                    self.gpu_infos[str(g)][k] = np.max([v, self.gpu_infos[str(g)][k]])
        return self.cpu_infos, self.gpu_infos


class SubprocessGetMem(object):
    def __init__(self, pid, gpu_id):
        self.mem_info = MemInfo(pid, gpu_id)

    def get_mem_subprocess_start(self, q, interval=0.0):
        while True:
            cpu_infos, gpu_infos = self.mem_info.get_avg_mem_mb()
            pid = os.getpid()
            q.put([cpu_infos, gpu_infos, pid])
            time.sleep(interval)
        return
    
    def get_mem_subprocess_init(self, interval=0.0):
        ctx = multiprocessing.get_context('spawn')
        self.mem_q = ctx.Queue()

        results = []
        self.mem_p = ctx.Process(target=self.get_mem_subprocess_start, args=(self.mem_q, interval))
        self.mem_p.start()
    
    def get_mem_subprocess_run(self, interval=0.0):
        self.get_mem_subprocess_init(interval=interval)
    
    def get_mem_subprocess_end(self):
        self.cpu_infos, self.gpu_infos, subpid = self.mem_q.get()
        #self.mem_p.terminate()
        try: 
            self.mem_p.kill() # python>=3.7 needed
        except:
            import os, signal
            import subprocess  
            try:
                subprocess.Popen("kill -s 9  %i"%subpid , shell=True)   # for linux
            except: 
                subprocess.Popen("cmd.exe /k taskkill /F /T /PID %i"%subpid , shell=True) # for win




# if __name__ == "__main__":
#     gpu_info = GpuInfoV2()
#     res = gpu_info.get_gpu_info(0)
#     print(res)

#     # print("----------------------------")
#     mem_info = MemInfo(11227, [0])
#     res = mem_info.summary_mem()
#     print(res)

