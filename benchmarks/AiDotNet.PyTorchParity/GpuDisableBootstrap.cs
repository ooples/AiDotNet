using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.PyTorchParity;

/// <summary>
/// Disables AiDotNet.Tensors' GPU auto-detection for this CPU-vs-PyTorch-CPU parity
/// harness, the right way: from a module initializer on THIS (entry) assembly, which
/// the runtime runs at startup — before <c>Main</c> and before the Tensors assembly is
/// first touched (the <c>ResetToCpu()</c> call in Program.cs). Tensors ships a
/// <c>[ModuleInitializer]</c> (GpuAutoDetectModuleInit) that, at Tensors-assembly load,
/// auto-detects a GPU/OpenCL device and compiles ~600 OpenCL kernels. Even though
/// Program.cs immediately calls <c>ResetToCpu()</c>, that's too late — the kernels are
/// already compiled and the OpenCL/GPU polling threads are spun up, and they then
/// compete for CPU cores and inflate the very inference numbers this harness measures
/// (clean-CPU CNN inference measured ~271 µs in a GPU-free process vs ~635 µs here).
///
/// <para>Tensors' auto-detect honors the <c>AIDOTNET_DISABLE_GPU</c> env var, checked at
/// its module-init time. Setting it here — from the entry assembly's initializer, which
/// runs before Tensors is loaded and touches no Tensors type itself — guarantees the env
/// var is in place before that check, so GPU auto-detect (and the OpenCL kernel compile)
/// is skipped entirely. Idempotent and respects an existing value, so a caller can still
/// set it before process start.</para>
/// </summary>
internal static class GpuDisableBootstrap
{
    [ModuleInitializer]
    internal static void DisableGpuForCpuParity()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU")))
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
    }
}
