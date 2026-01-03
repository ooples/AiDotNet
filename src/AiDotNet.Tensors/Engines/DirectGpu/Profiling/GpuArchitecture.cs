// Copyright (c) 2024 AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Profiling;

/// <summary>
/// Defines GPU architecture families and their resource limits.
/// Used for occupancy calculations and performance analysis.
/// </summary>
public enum GpuArchitectureFamily
{
    /// <summary>Unknown architecture - uses conservative defaults.</summary>
    Unknown,

    /// <summary>AMD GCN (Graphics Core Next) - Polaris, Vega.</summary>
    AmdGcn,

    /// <summary>AMD RDNA 1st generation - RX 5000 series.</summary>
    AmdRdna1,

    /// <summary>AMD RDNA 2nd generation - RX 6000 series.</summary>
    AmdRdna2,

    /// <summary>AMD RDNA 3rd generation - RX 7000 series.</summary>
    AmdRdna3,

    /// <summary>AMD CDNA - MI100, MI200 series (datacenter).</summary>
    AmdCdna,

    /// <summary>NVIDIA Turing architecture - RTX 20 series.</summary>
    NvidiaTuring,

    /// <summary>NVIDIA Ampere architecture - RTX 30 series.</summary>
    NvidiaAmpere,

    /// <summary>NVIDIA Ada Lovelace architecture - RTX 40 series.</summary>
    NvidiaAda,

    /// <summary>Intel Arc architecture.</summary>
    IntelArc
}

/// <summary>
/// GPU architecture specifications for occupancy and performance calculations.
/// </summary>
public sealed class GpuArchitectureSpec
{
    /// <summary>Architecture family identifier.</summary>
    public GpuArchitectureFamily Family { get; init; }

    /// <summary>Human-readable architecture name.</summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>Vector GPRs (VGPRs) per SIMD unit.</summary>
    public int VgprsPerSimd { get; init; }

    /// <summary>Scalar GPRs (SGPRs) per compute unit.</summary>
    public int SgprsPerCu { get; init; }

    /// <summary>Local Data Share (LDS) per compute unit in bytes.</summary>
    public int LdsPerCuBytes { get; init; }

    /// <summary>Maximum wavefronts per SIMD unit.</summary>
    public int MaxWavesPerSimd { get; init; }

    /// <summary>Number of SIMD units per compute unit.</summary>
    public int SimdsPerCu { get; init; }

    /// <summary>Native wavefront size (32 for RDNA Wave32, 64 for GCN/Wave64).</summary>
    public int WavefrontSize { get; init; }

    /// <summary>Whether Wave32 mode is supported (RDNA only).</summary>
    public bool SupportsWave32 { get; init; }

    /// <summary>L1 cache size per CU in bytes (0 if not available).</summary>
    public int L1CachePerCuBytes { get; init; }

    /// <summary>L2 cache size total in bytes.</summary>
    public int L2CacheTotalBytes { get; init; }

    /// <summary>Infinity Cache size in bytes (AMD RDNA2+ only, 0 otherwise).</summary>
    public long InfinityCacheBytes { get; init; }

    /// <summary>Typical kernel launch overhead in microseconds.</summary>
    public int TypicalLaunchOverheadUs { get; init; }

    /// <summary>Typical boost clock speed in GHz for peak GFLOPS calculations.</summary>
    public double TypicalBoostClockGHz { get; init; }

    /// <summary>FP32 ALUs per Compute Unit (64 for AMD, varies for NVIDIA).</summary>
    public int Fp32AlusPerCu { get; init; } = 64;

    /// <summary>
    /// Gets the maximum wavefronts per compute unit.
    /// </summary>
    public int MaxWavesPerCu => MaxWavesPerSimd * SimdsPerCu;

    /// <summary>
    /// Calculates estimated peak GFLOPS based on compute units and architecture.
    /// Formula: CUs * ALUs/CU * 2 (FMA) * clockGHz
    /// </summary>
    public double CalculatePeakGflops(int computeUnits)
    {
        return computeUnits * Fp32AlusPerCu * 2 * TypicalBoostClockGHz;
    }

    /// <summary>
    /// Predefined specifications for common AMD architectures.
    /// </summary>
    public static class Amd
    {
        /// <summary>AMD GCN (Polaris, Vega) specifications.</summary>
        public static readonly GpuArchitectureSpec Gcn = new()
        {
            Family = GpuArchitectureFamily.AmdGcn,
            Name = "AMD GCN (Polaris/Vega)",
            VgprsPerSimd = 1024,
            SgprsPerCu = 800,
            LdsPerCuBytes = 65536,
            MaxWavesPerSimd = 10,
            SimdsPerCu = 4,
            WavefrontSize = 64,
            SupportsWave32 = false,
            L1CachePerCuBytes = 16384,
            L2CacheTotalBytes = 4 * 1024 * 1024,
            InfinityCacheBytes = 0,
            TypicalLaunchOverheadUs = 300,
            TypicalBoostClockGHz = 1.5,  // Polaris ~1.3 GHz, Vega ~1.6 GHz
            Fp32AlusPerCu = 64
        };

        /// <summary>AMD RDNA1 (RX 5000 series) specifications.</summary>
        public static readonly GpuArchitectureSpec Rdna1 = new()
        {
            Family = GpuArchitectureFamily.AmdRdna1,
            Name = "AMD RDNA1 (RX 5000)",
            VgprsPerSimd = 1024,
            SgprsPerCu = 800,
            LdsPerCuBytes = 65536,
            MaxWavesPerSimd = 20,  // Wave32 mode
            SimdsPerCu = 2,        // RDNA uses dual-CU (WGP)
            WavefrontSize = 32,
            SupportsWave32 = true,
            L1CachePerCuBytes = 16384,
            L2CacheTotalBytes = 4 * 1024 * 1024,
            InfinityCacheBytes = 0,
            TypicalLaunchOverheadUs = 300,
            TypicalBoostClockGHz = 1.845,  // RX 5500 XT: 1.845 GHz, RX 5700 XT: 1.905 GHz
            Fp32AlusPerCu = 64
        };

        /// <summary>AMD RDNA2 (RX 6000 series) specifications.</summary>
        public static readonly GpuArchitectureSpec Rdna2 = new()
        {
            Family = GpuArchitectureFamily.AmdRdna2,
            Name = "AMD RDNA2 (RX 6000)",
            VgprsPerSimd = 1024,
            SgprsPerCu = 800,
            LdsPerCuBytes = 65536,
            MaxWavesPerSimd = 20,  // Wave32 mode
            SimdsPerCu = 2,
            WavefrontSize = 32,
            SupportsWave32 = true,
            L1CachePerCuBytes = 16384,
            L2CacheTotalBytes = 4 * 1024 * 1024,
            InfinityCacheBytes = 128 * 1024 * 1024,  // 128 MB Infinity Cache
            TypicalLaunchOverheadUs = 300,
            TypicalBoostClockGHz = 2.25,  // RX 6800 XT: 2.25 GHz, RX 6900 XT: 2.25 GHz
            Fp32AlusPerCu = 64
        };

        /// <summary>AMD RDNA3 (RX 7000 series) specifications.</summary>
        public static readonly GpuArchitectureSpec Rdna3 = new()
        {
            Family = GpuArchitectureFamily.AmdRdna3,
            Name = "AMD RDNA3 (RX 7000)",
            VgprsPerSimd = 1536,   // Increased from RDNA2
            SgprsPerCu = 800,
            LdsPerCuBytes = 65536,
            MaxWavesPerSimd = 16,  // Wave32 mode
            SimdsPerCu = 2,
            WavefrontSize = 32,
            SupportsWave32 = true,
            L1CachePerCuBytes = 32768,  // Doubled
            L2CacheTotalBytes = 6 * 1024 * 1024,
            InfinityCacheBytes = 96 * 1024 * 1024,  // 96 MB (reduced)
            TypicalLaunchOverheadUs = 250,
            TypicalBoostClockGHz = 2.5,  // RX 7900 XTX: 2.5 GHz
            Fp32AlusPerCu = 64
        };

        /// <summary>AMD CDNA (MI100, MI200) specifications.</summary>
        public static readonly GpuArchitectureSpec Cdna = new()
        {
            Family = GpuArchitectureFamily.AmdCdna,
            Name = "AMD CDNA (MI100/MI200)",
            VgprsPerSimd = 1024,
            SgprsPerCu = 800,
            LdsPerCuBytes = 65536,
            MaxWavesPerSimd = 8,
            SimdsPerCu = 4,
            WavefrontSize = 64,
            SupportsWave32 = false,
            L1CachePerCuBytes = 16384,
            L2CacheTotalBytes = 8 * 1024 * 1024,
            InfinityCacheBytes = 0,
            TypicalLaunchOverheadUs = 200,
            TypicalBoostClockGHz = 1.5,  // MI100: 1.5 GHz, MI250: 1.7 GHz
            Fp32AlusPerCu = 64
        };
    }

    /// <summary>
    /// Detects GPU architecture from device name string.
    /// </summary>
    public static GpuArchitectureSpec DetectFromDeviceName(string deviceName)
    {
        if (string.IsNullOrEmpty(deviceName))
            return Amd.Gcn;  // Conservative default

        var lower = deviceName.ToLowerInvariant();

        // AMD RDNA detection by gfx version
        if (lower.Contains("gfx110") || lower.Contains("gfx1100") || lower.Contains("gfx1101") ||
            lower.Contains("gfx1102") || lower.Contains("gfx1103") || lower.Contains("navi 3"))
            return Amd.Rdna3;

        if (lower.Contains("gfx103") || lower.Contains("gfx1030") || lower.Contains("gfx1031") ||
            lower.Contains("gfx1032") || lower.Contains("gfx1034") || lower.Contains("navi 2"))
            return Amd.Rdna2;

        if (lower.Contains("gfx101") || lower.Contains("gfx1010") || lower.Contains("gfx1011") ||
            lower.Contains("gfx1012") || lower.Contains("navi"))
            return Amd.Rdna1;

        if (lower.Contains("gfx90") || lower.Contains("mi100") || lower.Contains("mi200") ||
            lower.Contains("mi250") || lower.Contains("mi300"))
            return Amd.Cdna;

        if (lower.Contains("vega") || lower.Contains("gfx9") || lower.Contains("polaris") ||
            lower.Contains("rx 5") && !lower.Contains("rx 50") && !lower.Contains("rx 55") && !lower.Contains("rx 56") && !lower.Contains("rx 58"))
            return Amd.Gcn;

        // Default to RDNA1 for unknown AMD GPUs (most common modern consumer)
        if (lower.Contains("amd") || lower.Contains("radeon"))
            return Amd.Rdna1;

        return Amd.Gcn;  // Conservative fallback
    }
}
