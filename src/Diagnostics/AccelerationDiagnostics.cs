using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Diagnostics;

/// <summary>
/// Snapshots the live acceleration environment so users can see which SIMD, GPU, and
/// native BLAS paths are actually engaged at runtime, instead of assuming from config.
/// </summary>
/// <remarks>
/// <para>
/// Wraps Tensors' <see cref="PlatformDetector"/> and <see cref="NativeLibraryDetector"/>
/// into a single facade-friendly report that can be logged at builder time and surfaced
/// on <c>PredictionModelResult</c> for production observability.
/// </para>
/// </remarks>
public static class AccelerationDiagnostics
{
    public static string GetReport()
    {
        var caps = PlatformDetector.Capabilities;
        var nativeStatus = NativeLibraryDetector.Status;
        var sb = new StringBuilder();

        sb.AppendLine("=== AiDotNet Acceleration Report ===");
        sb.AppendLine($"  Engine:           {AiDotNetEngine.Current.Name}");
        sb.AppendLine($"  Deterministic:    {AiDotNetEngine.DeterministicMode}");
        sb.AppendLine($"  Framework:        {caps.FrameworkDescription}");
        sb.AppendLine($"  OS:               {caps.OSDescription}");
        sb.AppendLine($"  Arch:             {caps.Architecture} ({(caps.Is64BitProcess ? "64-bit" : "32-bit")} process)");
        sb.AppendLine($"  Processors:       {caps.ProcessorCount}");
        sb.AppendLine($"  L1/L2/L3 cache:   {caps.L1CacheSize / 1024}KB / {caps.L2CacheSize / 1024}KB / {caps.L3CacheSize / 1024}KB");
        sb.AppendLine($"  Best SIMD:        {caps.GetBestSimdSet()}");

        sb.AppendLine("  x86 SIMD:         " +
            $"SSE={caps.HasSSE} SSE2={caps.HasSSE2} SSE3={caps.HasSSE3} SSSE3={caps.HasSSSE3} " +
            $"SSE4.1={caps.HasSSE41} SSE4.2={caps.HasSSE42} AVX={caps.HasAVX} AVX2={caps.HasAVX2} FMA={caps.HasFMA}");
        sb.AppendLine("  AVX-512:          " +
            $"F={caps.HasAVX512F} BW={caps.HasAVX512BW} DQ={caps.HasAVX512DQ} VL={caps.HasAVX512VL}");
        sb.AppendLine("  ARM SIMD:         " +
            $"NEON={caps.HasNeon} AES={caps.HasArmAes} CRC32={caps.HasArmCrc32} DP={caps.HasArmDp}");

        sb.AppendLine("  GPU backends:     " +
            $"CUDA={caps.HasCudaSupport} OpenCL={caps.HasOpenCLSupport} HIP={caps.HasHipSupport}");
        sb.AppendLine("  Native BLAS:      " +
            $"OpenBLAS={nativeStatus.HasOpenBlas} CLBlast={nativeStatus.HasClBlast} " +
            $"MKL={nativeStatus.HasMkl} CpuBLAS={nativeStatus.HasCpuBlas}");
        sb.AppendLine("  Native GPU libs:  " +
            $"CUDA={nativeStatus.HasCuda} HIP={nativeStatus.HasHip} OpenCL={nativeStatus.HasOpenCl}");

        sb.AppendLine("  Autotune cache:   " +
            $"path={AutotuneCache.DefaultCachePath}");
        sb.AppendLine("  Autotune HW fp:   " +
            $"{AutotuneCache.CurrentHardwareFingerprint}");

        sb.Append(NativeLibraryDetector.GetStatusSummary());
        return sb.ToString();
    }

    /// <summary>
    /// Gets a structured snapshot of the current acceleration environment.
    /// Intended for programmatic checks (assertions in tests, automated CI reports).
    /// </summary>
    public static AccelerationSnapshot GetSnapshot()
    {
        var caps = PlatformDetector.Capabilities;
        var status = NativeLibraryDetector.Status;
        return new AccelerationSnapshot
        {
            EngineName = AiDotNetEngine.Current.Name,
            DeterministicMode = AiDotNetEngine.DeterministicMode,
            BestSimdSet = caps.GetBestSimdSet(),
            HasAvx2 = caps.HasAVX2,
            HasAvx512F = caps.HasAVX512F,
            HasFma = caps.HasFMA,
            HasNeon = caps.HasNeon,
            HasCuda = caps.HasCudaSupport,
            HasOpenCl = caps.HasOpenCLSupport,
            HasHip = caps.HasHipSupport,
            HasOpenBlas = status.HasOpenBlas,
            HasClBlast = status.HasClBlast,
            HasMkl = status.HasMkl,
            HasGpuAcceleration = status.HasGpuAcceleration,
            ProcessorCount = caps.ProcessorCount,
            L1CacheKB = caps.L1CacheSize / 1024,
            L2CacheKB = caps.L2CacheSize / 1024,
            L3CacheKB = caps.L3CacheSize / 1024,
            AutotuneCachePath = AutotuneCache.DefaultCachePath,
            AutotuneHardwareFingerprint = AutotuneCache.CurrentHardwareFingerprint,
        };
    }
}

/// <summary>
/// Immutable snapshot of acceleration state at a point in time.
/// </summary>
public sealed class AccelerationSnapshot
{
    public string EngineName { get; init; } = "";
    public bool DeterministicMode { get; init; }
    public string BestSimdSet { get; init; } = "";
    public bool HasAvx2 { get; init; }
    public bool HasAvx512F { get; init; }
    public bool HasFma { get; init; }
    public bool HasNeon { get; init; }
    public bool HasCuda { get; init; }
    public bool HasOpenCl { get; init; }
    public bool HasHip { get; init; }
    public bool HasOpenBlas { get; init; }
    public bool HasClBlast { get; init; }
    public bool HasMkl { get; init; }
    public bool HasGpuAcceleration { get; init; }
    public int ProcessorCount { get; init; }
    public int L1CacheKB { get; init; }
    public int L2CacheKB { get; init; }
    public int L3CacheKB { get; init; }
    public string AutotuneCachePath { get; init; } = "";
    public string AutotuneHardwareFingerprint { get; init; } = "";
}
