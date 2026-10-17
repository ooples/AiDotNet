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
        // `--device cuda` is the GPU-vs-PyTorch-CUDA run: leave auto-detect alone so the
        // DirectGpu engine is adopted. Read from the raw command line because this runs
        // before Main, so BenchmarkOptions has not been parsed yet.
        if (BenchmarkDeviceArg.Parse(Environment.GetCommandLineArgs()) == BenchmarkDevice.Cuda)
            return;
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU")))
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
    }
}

/// <summary>The device a parity run measures; matches the PyTorch twin's --device.</summary>
internal enum BenchmarkDevice
{
    Cpu,
    Cuda,
}

internal static class BenchmarkDeviceArg
{
    /// <summary>Reads `--device cpu|cuda` (default cpu); any other value is an error, not a silent CPU run.</summary>
    public static BenchmarkDevice Parse(string[] args)
    {
        for (var i = 0; i < args.Length; i++)
        {
            if (!string.Equals(args[i], "--device", StringComparison.OrdinalIgnoreCase)) continue;
            if (i + 1 >= args.Length)
                throw new ArgumentException("--device needs a value: cpu or cuda.");
            // The two names only: Enum.TryParse also takes "1" (Cuda) and "7" (an undefined value every caller
            // reads as CPU), either of which would put a CPU row against PyTorch's CUDA results.
            string value = args[i + 1];
            if (string.Equals(value, "cpu", StringComparison.OrdinalIgnoreCase)) return BenchmarkDevice.Cpu;
            if (string.Equals(value, "cuda", StringComparison.OrdinalIgnoreCase)) return BenchmarkDevice.Cuda;
            throw new ArgumentException($"--device must be cpu or cuda, got '{value}'.");
        }
        return BenchmarkDevice.Cpu;
    }
}
