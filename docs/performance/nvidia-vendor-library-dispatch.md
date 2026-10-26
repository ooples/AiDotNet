# NVIDIA vendor-library dispatch (cuBLAS, cuDNN)

Which NVIDIA libraries AiDotNet's layers actually reach on a CUDA host, why, and how to check on your own machine (#1159).

AiDotNet does not choose between vendor libraries itself. Layers call `Engine.Conv2D`, `Engine.BatchNorm` and `Engine.MatMul`. On an NVIDIA GPU the engine is `DirectGpuTensorEngine` with the CUDA backend, and the backend decides per operation. So no AiDotNet-side dispatcher is needed, and there is none.

## The rule: our kernels where they measure faster, the vendor library elsewhere

The goal is to need no external libraries. The vendor libraries stay only as measured fallbacks, on the shapes where our generated kernels have not yet beaten them.

| Operation | Route | Vendor library's role |
|---|---|---|
| GEMM: `DenseLayer`, `MatMul`, attention projections | A shape with a measured verdict runs the tuned generated kernel. Skinny and small shapes run the backend's narrow kernels. Any other shape goes to cuBLAS. | cuBLAS is **optional**: when it cannot load or is turned off, the generated GEMM serves every shape, in FP32, FP16, BF16 and FP64. That coverage is AiDotNet.Tensors #1316's result, which added the FP16, BF16 and FP64 generated paths; it is not this page's. The probe below measured one float `DenseLayer` shape on one host. |
| Conv2D forward and backward | The tuned convolution slot measures every candidate per shape and keeps the fastest (as `cudnn.benchmark` does in PyTorch). | cuDNN's algorithms are **candidates only** (#1457): never chosen by default, only when they win the measurement. |
| BatchNorm | The backend's kernels, or cuDNN batch normalization when cuDNN loads. | cuDNN optional. |

Why the vendor libraries are not deleted yet. Measured on an RTX 3080:
- **cuBLAS:** our best strict-FP32 SGEMM runs at 0.85–0.97× of cuBLAS at 2048–4096³ (#1310). Inside CUDA graphs it runs at 0.55× on small shapes (#1322).
- **cuDNN:** adding its algorithms as candidates made ResNet 3×3 convolutions 1.5–2.0× faster than our routes (#1457).

Removing either now would slow those shapes down by those factors. They are removed once evolved kernels win those shapes.

Opt-outs:
- `AIDOTNET_DISABLE_CUBLAS=1`, or `TensorCodecOptions.UseCublas = false`, runs every GEMM on the generated kernels.
- `TensorCodecOptions.UseCudnn = false` keeps convolutions and batch norm off cuDNN.

## What ships

The `AiDotNet.Native.CUDA` package ships `cudart`, cuBLAS, cuBLASLt and NVRTC. **It does not ship cuDNN**, so cuDNN is a candidate only on machines that have `cudnn64_9.dll` (or `cudnn64_8.dll`) on `PATH`.

## Measured on a GTX 1660 Ti (sm_75), driver 617.42, no cuDNN installed (2026-10-26)

The probe ran three layers on `DirectGpuTensorEngine` with `CudaBackend`:
- a `ConvolutionalLayer<float>`: 64 filters, 3x3, on [8, 64, 56, 56];
- an eval-mode `BatchNormalizationLayer<float>`;
- a `DenseLayer<float>`: 1024 outputs on [256, 1024].

After each, it recorded the vendor libraries loaded into the process and the backend's `LastGemmDispatchPath`.
The `AIDOTNET_DISABLE_CUBLAS=1` row ran in a **fresh process**, with the variable set before launch, rather than
continuing the process that had just run the normal Dense case. It has to: that process already had cuBLAS
loaded, so a `none` reading taken in it would say nothing about whether the GEMM needed cuBLAS.

| After | Vendor libraries loaded | GEMM route |
|---|---|---|
| Conv2D | none | `NarrowGemm` (our kernel) |
| BatchNorm (eval) | none | n/a |
| Dense | `cublas64_13.dll`, `cublasLt64_13.dll` | `Cublas`: an untuned shape |
| Dense, `AIDOTNET_DISABLE_CUBLAS=1` | none | the generated GEMM, identical output |

`LastGemmDispatchPath` is a `CudaGemmDispatchPath`, whose values are `None`, `Cublas`, `DirectPtxSgemm`,
`DirectPtxTensorCore`, `NarrowGemm` and `DeclaredProgram`. With cuBLAS off the reading is therefore whichever
generated route served the shape, not a distinct "fallback" value.

## Checking your own machine

1. Run each operation once.
2. List the loaded modules with `Process.GetCurrentProcess().Modules`, looking for `cublas*` and `cudnn*`.
3. Read `CudaBackend.LastGemmDispatchPath`.

Watching from outside with Nsight Systems does not work for this. `nsys profile -t cuda,cuDNN,cublas dotnet app.dll` traced only `cuInit` here, because the GPU work runs in a context it did not follow.
