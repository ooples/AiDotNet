"""Load the GPU with large matrix multiplications and report the SM clock it sustains, as JSON on stdout.

run_matrix.py runs this before CUDA measurements. A laptop GPU can be held far below its rated clock by the platform
(a low-wattage charger, a quiet or battery performance profile) while reporting no thermal or power-limit reason; the
benchmark refuses to measure such a GPU, because both frameworks' times would describe the cap, not the frameworks.
"""
import json
import statistics
import subprocess
import threading
import time

import torch

WARM_SECONDS, SAMPLES, INTERVAL = 4.0, 8, 1.0


def query():
    fields = "clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu"
    line = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
    return dict(zip(fields.split(","), (float(v) for v in line.splitlines()[0].split(","))))


def main():
    stop = threading.Event()

    def load():
        a = torch.randn(4096, 4096, device="cuda")
        while not stop.is_set():
            a = (a @ a).clamp_(-1, 1)
            torch.cuda.synchronize()

    worker = threading.Thread(target=load)
    worker.start()
    time.sleep(WARM_SECONDS)
    samples = []
    for _ in range(SAMPLES):
        samples.append(query())
        time.sleep(INTERVAL)
    stop.set()
    worker.join()
    clocks = [s["clocks.sm"] for s in samples]
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), maxSmClock=samples[0]["clocks.max.sm"],
                          medianSmClock=statistics.median(clocks), minSmClock=min(clocks), samples=samples)))


if __name__ == "__main__":
    main()
