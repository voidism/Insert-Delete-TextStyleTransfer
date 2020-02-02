import GPUtil
import sys
import time
import os

script = sys.argv[1]

while True:
    gpus = GPUtil.getAvailable()
    if len(gpua) > 0:
        os.popen(scripts.format(gpu=gpus[0]))
        break
    time.sleep(5)
