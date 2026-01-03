# GPU Thread Safety - Phase B

## Overview
DirectGpu backends are not guaranteed thread-safe. Treat each DirectGpuEngine/DirectGpuTensorEngine instance as single-threaded unless you provide external synchronization.

## Recommended usage
- Use one engine instance per thread, or
- Use a shared engine with a lock around GPU calls.

## Example
```csharp
private readonly object _gpuLock = new object();

lock (_gpuLock)
{
    var result = _engine.TensorMatMul(a, b);
}
```

## Notes
- Avoid sharing GPU buffers across threads without synchronization.
- CPU fallback operations remain thread-safe.
- Multi-GPU support is not implemented; use separate processes if you need parallel GPUs.
