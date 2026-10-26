# NVIDIA vendor-library dispatch (cuBLAS, cuDNN)

Which NVIDIA libraries AiDotNet's layers actually reach on a CUDA host, and how to check on your own machine (#1159).

AiDotNet does not choose between vendor libraries itself. Layers call `Engine.Conv2D`, `Engine.BatchNorm` and `Engine.MatMul`. On an NVIDIA GPU the engine is `DirectGpuTensorEngine` with the CUDA backend, and the backend makes the routing decision per operation. So no AiDotNet-side dispatcher is needed, and there is none.

## The routing rules (AiDotNet.Tensors CUDA backend)

| Operation | Vendor route | When it is taken | Otherwise |
|---|---|---|---|
| GEMM: `DenseLayer`, `MatMul`, attention projections | cuBLAS / cuBLASLt | Whenever cuBLAS loads. It loads lazily on the first GEMM. | The backend's generated GEMM kernels, which solve the same problem. |
| Conv2D forward | cuDNN's algorithms, as candidates of the tuned convolution slot | cuDNN loads and `TensorCodecOptions.UseCudnn` is on (the default where it loads). Every cuDNN algorithm competes with the backend's own routes under the slot's measured evidence, and the fastest wins per shape, as `cudnn.benchmark` does in PyTorch. | The backend's own convolution routes. |
| Conv2D backward | im2col + cuBLAS GEMM, competing with cuDNN's backward algorithms when cuDNN loads | Always available with cuBLAS. | The direct backward kernels (`AIDOTNET_CONV_GEMM_BACKWARD=0`). |
| BatchNorm | cuDNN batch normalization | cuDNN loads. | The backend's own kernels. |

Precision and determinism follow the backend's policy: cuDNN descriptors use strict fp32 (FMA) math unless reduced-precision TF32 is opted into, and under `GpuDeterminism` the atomics-based cuDNN backward algorithms are never considered.

Opt-outs: `AIDOTNET_DISABLE_CUBLAS=1` (or `TensorCodecOptions.UseCublas = false`) runs every GEMM on the generated kernels; `TensorCodecOptions.UseCudnn = false` keeps convolutions and batch norm off cuDNN.

## What ships, and what that means

The `AiDotNet.Native.CUDA` package ships `cudart`, cuBLAS, cuBLASLt and NVRTC, so **cuBLAS is always available** on an NVIDIA machine with no CUDA Toolkit installed.

**It does not ship cuDNN.** cuDNN is used only when `cudnn64_9.dll` (or `cudnn64_8.dll`) can be loaded from the machine: a cuDNN install on `PATH`, or alongside the application. Without it, Conv2D and BatchNorm run on the backend's own tuned kernels, correctly and without a warning.

## Measured on a GTX 1660 Ti (sm_75), driver 617.42, no cuDNN installed (2026-10-26)

A probe ran a `ConvolutionalLayer<float>` (64 filters, 3x3, on [8, 64, 56, 56]), an eval-mode `BatchNormalizationLayer<float>` and a `DenseLayer<float>` (1024 outputs on [256, 1024]) on `DirectGpuTensorEngine` with `CudaBackend`, and listed the vendor libraries loaded into the process after each:

| After | Vendor libraries loaded |
|---|---|
| Conv2D | none |
| BatchNorm (eval) | none |
| Dense | `cublas64_13.dll`, `cublasLt64_13.dll` |

So on this host GEMMs reach cuBLAS, and convolution and batch norm use the backend's kernels because cuDNN is not present, exactly as the rules above say.

## Checking your own machine

List the loaded modules after running each operation once (`Process.GetCurrentProcess().Modules`, looking for `cublas*` and `cudnn*`). Watching with Nsight Systems from outside does not work for this: `nsys profile -t cuda,cuDNN,cublas dotnet app.dll` traced only `cuInit` here, because the GPU work runs in a context it did not follow.

To get cuDNN's convolution algorithms, install cuDNN 9 for your CUDA version so that `cudnn64_9.dll` is on `PATH`; the convolution slot then measures them against the backend's routes and keeps whichever is faster per shape.
