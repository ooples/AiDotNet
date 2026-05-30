# Inference baseline vs `torch.compile` (Phase 0)

Establishes the per-row baseline the compiled-inference engine work
(`plans/distributed-forging-spindle.md`) must beat. CPU, 8 threads, bs64 train,
200 inference iters / 30 warmup. PyTorch run with `--compile` (TorchInductor)
under MSVC `vcvars64.bat`. Gate = **p95(AiDotNet) < mean(torch.compile)**.

Reproduce:
```
# AiDotNet
set AIDOTNET_BLAS_THREADS=8
dotnet run -c Release --project benchmarks/AiDotNet.PyTorchParity -- \
  --models mlp,mlp-fused,cnn,lstm,transformer --epochs 1 --train-batches 5 \
  --batch-size 64 --inference-iterations 200 --warmup-iterations 30 \
  --output benchmarks/AiDotNet.PyTorchParity/results/aidotnet_baseline.json
# PyTorch (needs cl.exe — run inside a VS x64 Native Tools prompt / vcvars64.bat)
cd benchmarks/AiDotNet.PyTorchParity/pytorch
python benchmark.py --models mlp,cnn,lstm,transformer --device cpu --threads 8 \
  --epochs 1 --train-batches 5 --batch-size 64 --inference-iterations 200 \
  --warmup-iterations 30 --compile --output ../results/pytorch_compiled_baseline.json
python compare.py ../results/aidotnet_baseline.json ../results/pytorch_compiled_baseline.json
```

## Inference latency (AiDotNet p95 vs torch.compile mean, this machine)

| model | bs | AiDotNet p95 ms | torch.compile mean ms | ratio | verdict |
|---|---|---|---|---|---|
| cnn | 1 | 7.613 | 0.420 | 18.1× | lose |
| cnn | 8 | 11.291 | 0.918 | 12.3× | lose |
| cnn | 32 | 7.232 | 1.660 | 4.4× | lose |
| cnn | 128 | 23.500 | 5.054 | 4.7× | lose |
| lstm | 1 | 0.432 | 0.515 | 0.84× | **WIN** |
| lstm | 8 | 1.557 | 0.754 | 2.1× | lose |
| lstm | 32 | 2.955 | 1.255 | 2.4× | lose |
| lstm | 128 | 8.513 | 2.823 | 3.0× | lose |
| mlp | 1 | 1.042 | 0.267 | 3.9× | lose |
| mlp | 8 | 2.146 | 0.761 | 2.8× | lose |
| mlp | 32 | **136.708** | 0.855 | 159.9× | lose ⚠ anomaly |
| mlp | 128 | 3.029 | 1.300 | 2.3× | lose |
| transformer | 1 | 2.656 | 6.501 | 0.41× | **WIN** |
| transformer | 8 | 10.121 | 2.204 | 4.6× | lose |
| transformer | 32 | 18.159 | 4.645 | 3.9× | lose |
| transformer | 128 | 64.445 | 11.329 | 5.7× | lose |

**Wins: 2/16** (transformer bs1, lstm bs1).

## Targets / reading

- **CNN is the worst gap (4–18×)** — the conv inference path, not just GEMM. New priority alongside MLP.
- **Transformer/MLP/LSTM lose 2–6× at bs ≥ 8** — consistent with the diagnosed causes: separate bias+act pass, per-call weight repack, per-layer intermediate materialization, dispatch.
- **`mlp` bs32 = 136.7 ms is an anomaly** (bs1=1.0, bs128=3.0) — a single-sample p95 spike (likely GC pause or a parallel-dispatch cliff at that exact shape). Investigate first in Phase 1; do not treat as representative.
- **Where we already win (bs1 transformer/lstm)** shows the fused/low-dispatch path pays off when GEMM is small and dispatch dominates — exactly the regime the tiny-model resident kernel (Phase 4) targets.

## Training (compiled training plan) — context, not the inference gate
Measured separately this session: AiDotNet fused ~0.014 s/epoch vs torch.compile ~0.084 s/epoch (steady-state) — **AiDotNet ~6× faster** on MLP training. The inference gap above is the focus of this plan.
