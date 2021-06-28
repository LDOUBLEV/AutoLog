
import os
import time
from device import MemInfo
import multiprocessing
from multiprocessing import Process
import subprocess


def test(pid, gpu_id, q):
    mem_info = MemInfo([pid], [gpu_id])
    _t = time.time()

    while True:
        res = mem_info.summary_mem(return_str=False)
        pid = os.getpid()
        t_ = time.time() - _t
        print(f"子进程: pid: {pid} time: {t_}", res)
        q.put(res)

        time.sleep(1)


if __name__ == "__main__":

    pid = os.getpid()
    
    multiprocessing.set_start_method('spawn')
    q = multiprocessing.Queue()

    st = time.time()
    results = []
    p = Process(target=test, args=(pid, 0, q))
    p.start()

    while True:
        time.sleep(1)
        print(q.get())
        t = time.time() - st
        print(f"主进程：pid: {pid}, time: {t}")
        
        print("=="*30)
        # print(f"{len(results)}="*30)
        if t>10:
            p.terminate()
            break
    
    # f = open("/dev/shm/_mem.txt", 'r') 
    # res = f.readlines()
    # print("*"*100)
    # print(len(res), res[0])

