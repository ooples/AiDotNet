# AiDotNet vs PyTorch training scoreboard

Generated 2026-10-07T06:14:54+00:00 by `pytorch/scoreboard.py`. Training **ms per step** (forward + backward + clip + AdamW), batch 64, 20 steps/epoch, 5 epochs per process (epoch 0 is warmup). Each cell is the **median of 9 process runs** with p25-p75; every run held `Global\AiDotNetBenchLock` on its own, ran at above-normal priority (both sides), and was quiet-gated: a run during which other processes used more than 25% of the CPUs was discarded and redone (up to 2x; see *Machine noise*).

PyTorch is scored in its **fastest mode for that cell**: every candidate mode ran 3x first, then the best 2 (plus eager) were re-measured 9x from scratch, interleaved with AiDotNet, and the lowest final median is the bar (see *Mode selection*).

| family | device | AiDotNet ms/step | best PyTorch mode | PyTorch ms/step | ours / PyTorch | verdict | PyTorch eager ms/step | ours / eager |
|---|---|---|---|---|---|---|---|---|
| mlp | cpu | 6.779 [6.719-7.040] | eager+fused-adamw | 2.809 [2.721-2.949] | 2.413x | LOSE | 3.778 [3.568-4.178] | 1.794x |
| cnn | cpu | 23.291 [22.105-23.717] | eager | 7.590 [7.179-8.529] | 3.069x | LOSE | 7.590 [7.179-8.529] | 3.069x |
| lstm | cpu | 27.879 [27.046-28.062] | eager+fused-adamw | 5.064 [4.590-5.443] | 5.506x | LOSE | 5.145 [4.986-5.526] | 5.418x |
| transformer | cpu | 44.161 [43.132-44.884] | eager+fused-adamw | 15.262 [14.925-17.210] | 2.894x | LOSE | 17.152 [16.403-18.330] | 2.575x |
| mlp | cuda | 1.983 [1.972-2.099] | compile[cudagraphs]+fused-adamw | 2.188 [2.087-2.347] | 0.906x | win | 2.588 [2.464-2.618] | 0.766x |
| cnn | cuda | 3.513 [3.413-3.693] | compile[cudagraphs]+fused-adamw | 2.524 [2.377-2.574] | 1.392x | LOSE | 3.646 [3.230-3.851] | 0.964x |
| lstm | cuda | 10.963 [10.906-11.140] | compile[inductor]+fused-adamw | 3.407 [3.171-3.549] | 3.218x | LOSE | 3.369 [3.055-3.675] | 3.255x |
| transformer | cuda | 6.561 [6.311-6.624] | compile[inductor/reduce-overhead]+fused-adamw | 3.026 [2.936-3.347] | 2.168x | LOSE | 9.016 [8.606-9.151] | 0.728x |

**0 decisive wins, 7 decisive losses, 1 within noise** (verdicts as in `compare.py`: WIN/LOSE only when the run IQRs do not overlap; lowercase = medians differ but IQRs overlap).

## Mode selection

Median ms/step of the 3 selection runs; for the re-measured modes, `-> ` the median of the 9 scored runs. Bold = the mode scored against.

| family | device | compile[cudagraphs] | compile[cudagraphs]+fused-adamw | compile[inductor/reduce-overhead] | compile[inductor/reduce-overhead]+fused-adamw | compile[inductor] | compile[inductor]+fused-adamw | eager | eager+fused-adamw |
|---|---|---|---|---|---|---|---|---|---|
| mlp | cpu | - | - | - | - | 4.705 | 4.948 | 3.864 -> 3.778 | **2.680 -> 2.809** |
| cnn | cpu | - | - | - | - | 13.554 | 12.906 | **8.091 -> 7.590** | 8.010 -> 7.957 |
| lstm | cpu | - | - | - | - | 5.466 | 5.092 -> 5.164 | 5.404 -> 5.145 | **4.910 -> 5.064** |
| transformer | cpu | - | - | - | - | 16.692 -> 18.370 | 17.332 | 17.029 -> 17.152 | **15.445 -> 15.262** |
| mlp | cuda | 2.688 | **2.367 -> 2.188** | 2.836 | 2.121 -> 2.418 | 3.102 | 3.198 | 2.564 -> 2.588 | 2.542 |
| cnn | cuda | 2.732 | **2.650 -> 2.524** | 2.683 -> 2.888 | 2.693 | 3.677 | 3.171 | 3.212 -> 3.646 | 3.093 |
| lstm | cuda | 3.761 | 3.273 -> 3.456 | 3.554 | 3.411 | 3.581 | **3.114 -> 3.407** | 3.560 -> 3.369 | 3.287 |
| transformer | cuda | 4.061 | 3.173 -> 3.137 | 3.699 | **3.121 -> 3.026** | 6.908 | 6.378 | 9.502 -> 9.016 | 8.508 |

## Modes skipped or failed

- `compile[inductor/max-autotune*]`: autotuning benchmarks GEMM/conv templates for minutes per process while holding the shared bench lock, and for these small shapes it targets the same kernels reduce-overhead already graphs; not run.
- `compile[inductor/reduce-overhead], compile[cudagraphs] on cpu`: CUDA graphs exist only on CUDA.
- `compile[aot_eager]`: a debugging backend that runs the eager kernels through AOTAutograd; never faster than eager.

## Machine noise

CPU used by OTHER processes during each kept run (% of all logical CPUs), and runs discarded and redone because it exceeded the threshold.

| family | device | contender | kept runs median background % | kept runs max background % | kept runs over 25% | discarded runs |
|---|---|---|---|---|---|---|
| mlp | cpu | AiDotNet | 12.8 | 22.8 | 0 | 0 |
| mlp | cpu | PyTorch eager+fused-adamw | 18.4 | 73.5 | 1 | 2 |
| mlp | cpu | PyTorch eager | 17.0 | 23.7 | 0 | 2 |
| mlp | cpu | PyTorch eager (selection) | 22.9 | 23.3 | 0 | 2 |
| mlp | cpu | PyTorch compile[inductor] (selection) | 16.9 | 21.2 | 0 | 0 |
| mlp | cpu | PyTorch eager+fused-adamw (selection) | 17.7 | 20.8 | 0 | 0 |
| mlp | cpu | PyTorch compile[inductor]+fused-adamw (selection) | 19.1 | 24.3 | 0 | 1 |
| cnn | cpu | AiDotNet | 13.9 | 20.8 | 0 | 1 |
| cnn | cpu | PyTorch eager+fused-adamw | 11.8 | 24.9 | 0 | 0 |
| cnn | cpu | PyTorch eager | 17.1 | 27.3 | 1 | 3 |
| cnn | cpu | PyTorch eager (selection) | 19.4 | 20.0 | 0 | 0 |
| cnn | cpu | PyTorch compile[inductor] (selection) | 14.6 | 16.3 | 0 | 0 |
| cnn | cpu | PyTorch eager+fused-adamw (selection) | 17.8 | 21.3 | 0 | 0 |
| cnn | cpu | PyTorch compile[inductor]+fused-adamw (selection) | 15.9 | 19.3 | 0 | 0 |
| lstm | cpu | AiDotNet | 12.3 | 18.0 | 0 | 0 |
| lstm | cpu | PyTorch eager+fused-adamw | 12.2 | 24.4 | 0 | 2 |
| lstm | cpu | PyTorch compile[inductor]+fused-adamw | 15.3 | 24.1 | 0 | 4 |
| lstm | cpu | PyTorch eager | 12.8 | 21.6 | 0 | 1 |
| lstm | cpu | PyTorch eager (selection) | 12.3 | 12.6 | 0 | 2 |
| lstm | cpu | PyTorch compile[inductor] (selection) | 13.1 | 61.1 | 1 | 2 |
| lstm | cpu | PyTorch eager+fused-adamw (selection) | 13.3 | 19.6 | 0 | 0 |
| lstm | cpu | PyTorch compile[inductor]+fused-adamw (selection) | 15.0 | 17.0 | 0 | 1 |
| transformer | cpu | AiDotNet | 13.6 | 23.8 | 0 | 1 |
| transformer | cpu | PyTorch eager+fused-adamw | 15.2 | 24.0 | 0 | 2 |
| transformer | cpu | PyTorch compile[inductor] | 20.4 | 24.8 | 0 | 2 |
| transformer | cpu | PyTorch eager | 16.6 | 20.3 | 0 | 2 |
| transformer | cpu | PyTorch eager (selection) | 16.1 | 24.2 | 0 | 0 |
| transformer | cpu | PyTorch compile[inductor] (selection) | 15.3 | 16.5 | 0 | 1 |
| transformer | cpu | PyTorch eager+fused-adamw (selection) | 16.6 | 33.3 | 1 | 3 |
| transformer | cpu | PyTorch compile[inductor]+fused-adamw (selection) | 17.9 | 20.1 | 0 | 0 |
| mlp | cuda | AiDotNet | 10.0 | 17.8 | 0 | 1 |
| mlp | cuda | PyTorch compile[inductor/reduce-overhead]+fused-adamw | 18.0 | 24.1 | 0 | 0 |
| mlp | cuda | PyTorch compile[cudagraphs]+fused-adamw | 14.8 | 19.0 | 0 | 2 |
| mlp | cuda | PyTorch eager | 10.5 | 19.5 | 0 | 2 |
| mlp | cuda | PyTorch eager (selection) | 15.4 | 19.9 | 0 | 0 |
| mlp | cuda | PyTorch compile[inductor] (selection) | 18.0 | 20.1 | 0 | 0 |
| mlp | cuda | PyTorch compile[inductor/reduce-overhead] (selection) | 16.6 | 18.3 | 0 | 1 |
| mlp | cuda | PyTorch compile[cudagraphs] (selection) | 13.8 | 22.7 | 0 | 0 |
| mlp | cuda | PyTorch eager+fused-adamw (selection) | 18.8 | 23.1 | 0 | 0 |
| mlp | cuda | PyTorch compile[inductor]+fused-adamw (selection) | 20.4 | 21.3 | 0 | 0 |
| mlp | cuda | PyTorch compile[inductor/reduce-overhead]+fused-adamw (selection) | 18.3 | 20.9 | 0 | 0 |
| mlp | cuda | PyTorch compile[cudagraphs]+fused-adamw (selection) | 13.9 | 15.8 | 0 | 1 |
| cnn | cuda | AiDotNet | 15.9 | 29.9 | 1 | 3 |
| cnn | cuda | PyTorch compile[cudagraphs]+fused-adamw | 17.8 | 24.6 | 0 | 2 |
| cnn | cuda | PyTorch compile[inductor/reduce-overhead] | 17.5 | 21.3 | 0 | 1 |
| cnn | cuda | PyTorch eager | 18.6 | 24.7 | 0 | 3 |
| cnn | cuda | PyTorch eager (selection) | 10.4 | 14.3 | 0 | 0 |
| cnn | cuda | PyTorch compile[inductor] (selection) | 23.7 | 71.5 | 1 | 2 |
| cnn | cuda | PyTorch compile[inductor/reduce-overhead] (selection) | 20.4 | 22.8 | 0 | 1 |
| cnn | cuda | PyTorch compile[cudagraphs] (selection) | 18.4 | 22.5 | 0 | 0 |
| cnn | cuda | PyTorch eager+fused-adamw (selection) | 14.0 | 21.8 | 0 | 0 |
| cnn | cuda | PyTorch compile[inductor]+fused-adamw (selection) | 13.2 | 46.9 | 1 | 3 |
| cnn | cuda | PyTorch compile[inductor/reduce-overhead]+fused-adamw (selection) | 20.8 | 78.3 | 1 | 3 |
| cnn | cuda | PyTorch compile[cudagraphs]+fused-adamw (selection) | 14.1 | 22.8 | 0 | 1 |
| lstm | cuda | AiDotNet | 13.2 | 22.5 | 0 | 0 |
| lstm | cuda | PyTorch compile[inductor]+fused-adamw | 13.8 | 20.1 | 0 | 0 |
| lstm | cuda | PyTorch compile[cudagraphs]+fused-adamw | 12.9 | 24.7 | 0 | 1 |
| lstm | cuda | PyTorch eager | 14.2 | 19.2 | 0 | 1 |
| lstm | cuda | PyTorch eager (selection) | 15.5 | 19.4 | 0 | 0 |
| lstm | cuda | PyTorch compile[inductor] (selection) | 11.6 | 18.0 | 0 | 1 |
| lstm | cuda | PyTorch compile[inductor/reduce-overhead] (selection) | 14.4 | 15.2 | 0 | 1 |
| lstm | cuda | PyTorch compile[cudagraphs] (selection) | 11.3 | 12.4 | 0 | 0 |
| lstm | cuda | PyTorch eager+fused-adamw (selection) | 11.8 | 12.2 | 0 | 0 |
| lstm | cuda | PyTorch compile[inductor]+fused-adamw (selection) | 12.4 | 12.6 | 0 | 0 |
| lstm | cuda | PyTorch compile[inductor/reduce-overhead]+fused-adamw (selection) | 11.8 | 12.2 | 0 | 0 |
| lstm | cuda | PyTorch compile[cudagraphs]+fused-adamw (selection) | 11.0 | 14.3 | 0 | 0 |
| transformer | cuda | AiDotNet | 12.4 | 15.5 | 0 | 1 |
| transformer | cuda | PyTorch compile[inductor/reduce-overhead]+fused-adamw | 11.8 | 17.2 | 0 | 0 |
| transformer | cuda | PyTorch compile[cudagraphs]+fused-adamw | 12.7 | 20.0 | 0 | 0 |
| transformer | cuda | PyTorch eager | 13.1 | 15.1 | 0 | 1 |
| transformer | cuda | PyTorch eager (selection) | 12.5 | 13.1 | 0 | 0 |
| transformer | cuda | PyTorch compile[inductor] (selection) | 21.8 | 23.1 | 0 | 1 |
| transformer | cuda | PyTorch compile[inductor/reduce-overhead] (selection) | 14.6 | 16.6 | 0 | 0 |
| transformer | cuda | PyTorch compile[cudagraphs] (selection) | 14.4 | 16.8 | 0 | 0 |
| transformer | cuda | PyTorch eager+fused-adamw (selection) | 11.4 | 11.7 | 0 | 0 |
| transformer | cuda | PyTorch compile[inductor]+fused-adamw (selection) | 13.6 | 13.7 | 0 | 0 |
| transformer | cuda | PyTorch compile[inductor/reduce-overhead]+fused-adamw (selection) | 12.2 | 14.4 | 0 | 0 |
| transformer | cuda | PyTorch compile[cudagraphs]+fused-adamw (selection) | 12.5 | 13.5 | 0 | 0 |

## Provenance

- Machine: AMD Ryzen 7 4800H with Radeon Graphics (16 logical), GPU NVIDIA GeForce GTX 1660 Ti, Windows-11-10.0.26200-SP0
- PyTorch 2.14.1+cu126 (CUDA 12.6, 8 intra-op threads), Python 3.13.3, triton 3.8.0, MSVC: vcvars64: C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat
- AiDotNet harness: `C:\Users\yolan\source\repos\_bench\baseline-bin` (run from a copy taken at the start) (.NET SDK 10.0.401); thread pin: none (each side at its default)
  - AiDotNet.dll: 0.204.0+2cc2dfa919100532b255ae6951e7dc80b4ce44b6 sha256 f2753a572ff74feb
  - AiDotNet.Tensors.dll: 1.0.0-preview+efb63d404bb1e6ecbd118249ceddfb53f1eb8c30 sha256 ed3f114b504f0d53
  - AiDotNet.PyTorchParity.dll: 1.0.0+c14b4dc4275c3d2529c9221aac6725d66a54c22a sha256 ad4fc342254f93bc

Raw per-run values: `scoreboard.json`.
