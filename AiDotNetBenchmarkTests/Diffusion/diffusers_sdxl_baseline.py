"""
PyTorch / diffusers SDXL baseline for the head-to-head comparison in
#1272 acceptance criterion #1.

Invoked as a subprocess by SDXLEndToEndBenchmark.cs. Output protocol:
emit a single JSON line to stdout in the form
    {"wall_ms": <float>}
representing the wall-clock time of one
StableDiffusionXLPipeline.__call__ at fp32, CPU, batch_size=1.

The script intentionally excludes its own startup (Python interpreter
launch + diffusers/torch import + pipeline weight load) from the
measured window — those are amortised at the process boundary, not at
the per-call boundary. The C# benchmark spawns this script once per
benchmark iteration, so warmup-once + measure-once is the steady-state
behaviour.

Usage:
    python diffusers_sdxl_baseline.py --steps 50 --width 256 --height 256 \
        --prompt "a photograph of an astronaut riding a horse on mars"

Requirements:
    pip install diffusers transformers accelerate torch --upgrade

The C# benchmark probes for `python --version` at GlobalSetup time. If
Python or diffusers is unavailable the PyTorch baseline is skipped via
BenchmarkSkippedException; the AiDotNet method still runs and reports
its absolute wall time without a comparison ratio.
"""
import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--model', type=str,
                        default='stabilityai/stable-diffusion-xl-base-1.0')
    args = parser.parse_args()

    # Lazy imports so import time isn't inside the measured window.
    import torch
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float32, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cpu')

    # Warmup the kernel-selection cache before timing — first generation
    # is 2-3x slower due to MKL-DNN tile-size autotune.
    _ = pipe(args.prompt, num_inference_steps=4,
             height=args.height, width=args.width).images

    # Measured run.
    t0 = time.perf_counter()
    _ = pipe(args.prompt, num_inference_steps=args.steps,
             height=args.height, width=args.width).images
    t1 = time.perf_counter()

    print(json.dumps({'wall_ms': (t1 - t0) * 1000.0}))


if __name__ == '__main__':
    main()
