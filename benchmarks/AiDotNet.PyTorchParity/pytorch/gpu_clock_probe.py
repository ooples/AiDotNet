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

from benchmark import cuda_smi_id

WARM_SECONDS, SAMPLES, INTERVAL = 4.0, 8, 1.0


def query(gpu_id):
    """One telemetry row for the GPU the load runs on, selected by UUID rather than nvidia-smi's first row."""
    fields = "clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu"
    line = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
                                    f"--id={gpu_id}"], text=True)
    values = [v.strip() for v in line.splitlines()[0].split(",")]
    row = dict(zip(fields.split(","), (_number(v) for v in values)))
    # Only the two clocks decide the gate; a GPU may print [N/A] for power, temperature or utilization.
    missing = [name for name in ("clocks.sm", "clocks.max.sm") if row.get(name) is None]
    if missing:
        raise RuntimeError(f"nvidia-smi reports no {', '.join(missing)} for GPU {gpu_id}, so its clock cannot be gated.")
    return row


def _number(text):
    """A telemetry value, or None for the [N/A] (or any non-numeric) placeholder nvidia-smi prints."""
    try:
        return float(text)
    except ValueError:
        return None


def main():
    device = torch.device("cuda", torch.cuda.current_device())
    # CUDA_VISIBLE_DEVICES can make torch's default device a GPU other than nvidia-smi's row 0, so the
    # telemetry must name the loaded GPU, or the gate would judge an idle one.
    gpu_id = cuda_smi_id(device)
    if gpu_id is None:
        raise RuntimeError(f"Cannot resolve the UUID of {device}, so its clock cannot be sampled.")
    stop = threading.Event()
    failure = []

    def load():
        try:
            a = torch.randn(4096, 4096, device=device)
            while not stop.is_set():
                a = (a @ a).clamp_(-1, 1)
                torch.cuda.synchronize(device)
        except BaseException as error:  # an unhandled thread exception would not reach join()
            failure.append(error)

    worker = threading.Thread(target=load)
    worker.start()
    samples = []
    try:
        time.sleep(WARM_SECONDS)
        for _ in range(SAMPLES):
            if failure:
                break
            samples.append(query(gpu_id))
            time.sleep(INTERVAL)
    finally:
        # A telemetry error must not leave the non-daemon worker running, or the process never exits.
        stop.set()
        worker.join()
    if failure:
        raise RuntimeError("The load worker failed, so the clocks were not sampled under load.") from failure[0]
    clocks = [s["clocks.sm"] for s in samples]
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(device), maxSmClock=samples[0]["clocks.max.sm"],
                          medianSmClock=statistics.median(clocks), minSmClock=min(clocks), samples=samples)))

if __name__ == "__main__":
    main()
