// Copyright (c) AiDotNet. All rights reserved.
// Auto-tuning framework for GEMM kernel selection and parameter optimization.
// Implements Bayesian optimization for efficient kernel parameter search.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// GEMM kernel configuration parameters.
/// </summary>
public readonly struct GemmConfig
{
    public int TileM { get; init; }
    public int TileN { get; init; }
    public int TileK { get; init; }
    public int ThreadTileM { get; init; }
    public int ThreadTileN { get; init; }
    public int VectorWidthM { get; init; }
    public int VectorWidthN { get; init; }
    public bool UseDoubleBuffering { get; init; }
    public bool UseVectorizedLoads { get; init; }
    public string KernelName { get; init; }

    // CLBlast-style parameters for higher performance
    public int KReg { get; init; }           // Register tiling in K dimension (1, 2, 4)
    public int KUnroll { get; init; }         // K-loop unroll factor (1, 2, 4, 8)
    public bool UseSubgroupOps { get; init; } // Use wavefront shuffle operations

    // CLBlast stride parameters for local memory bank conflict avoidance
    public bool StrideM { get; init; }  // STRM: Use strided access for A tile stores
    public bool StrideN { get; init; }  // STRN: Use strided access for B tile stores

    // CLBlast local memory caching parameters (SA/SB)
    public bool CacheA { get; init; }   // SA: Cache A tile in local memory (GlobalToLocalA pattern)
    public bool CacheB { get; init; }   // SB: Cache B tile in local memory (GlobalToLocalB pattern)

    // CLBlast workgroup decomposition parameters
    public int MdimaSize { get; init; } // MDIMA: Workgroup rows for A tile (8, 16, 32)
    public int NdimbSize { get; init; } // NDIMB: Workgroup cols for B tile (8, 16, 32)

    // True CLBlast-style vectorized LDS (THE KEY TO 2500+ GFLOPS)
    public bool UseTrueVectorLDS { get; init; }  // Use vectorized LDS arrays instead of scalar
    public bool UseColumnMajorA { get; init; }   // Treat A as column-major (packed/transpose path)

    /// <summary>
    /// Generates a unique cache key for this configuration.
    /// Used by DynamicGemmKernel to cache compiled kernels.
    /// </summary>
    public string ToKey() =>
        $"{(string.IsNullOrWhiteSpace(KernelName) ? "default" : KernelName)}_{TileM}_{TileN}_{TileK}_{ThreadTileM}_{ThreadTileN}_{VectorWidthM}_{VectorWidthN}_{UseDoubleBuffering}_{UseVectorizedLoads}_{KReg}_{KUnroll}_{UseSubgroupOps}_{StrideM}_{StrideN}_{CacheA}_{CacheB}_{MdimaSize}_{NdimbSize}_{UseTrueVectorLDS}_{UseColumnMajorA}";

    public override string ToString() =>
        $"{KernelName}[{TileM}x{TileN}x{TileK}, TT:{ThreadTileM}x{ThreadTileN}, VW:{VectorWidthM}x{VectorWidthN}, K:{KReg}x{KUnroll}, SG:{UseSubgroupOps}, SA/B:{(CacheA ? 1 : 0)}/{(CacheB ? 1 : 0)}, MD:{MdimaSize}x{NdimbSize}, ACol:{(UseColumnMajorA ? 1 : 0)}]";
}

/// <summary>
/// Result of a tuning run.
/// </summary>
public readonly struct TuningResult
{
    public GemmConfig Config { get; init; }
    public double GFlops { get; init; }
    public double TimeMs { get; init; }
    public bool IsValid { get; init; }
    public string? Error { get; init; }
}

/// <summary>
/// Auto-tuner for GEMM kernel selection and parameter optimization.
/// Uses Bayesian-inspired heuristics to quickly find optimal configurations.
/// </summary>
public sealed class GemmAutoTuner
{
    private readonly Dictionary<(int M, int N, int K), GemmConfig> _cache = new();
    private readonly object _cacheLock = new();

    /// <summary>
    /// Enable verbose diagnostic output for debugging tuning progress.
    /// </summary>
    public static bool EnableDiagnostics { get; set; } = false;

    /// <summary>
    /// Optional logger for live tuning diagnostics.
    /// </summary>
    public static ILogger? Logger { get; set; }

    /// <summary>
    /// Show progress every N trials (default: 10).
    /// </summary>
    public static int ProgressInterval { get; set; } = 10;

    /// <summary>
    /// Emit progress output during tuning and benchmarks.
    /// </summary>
    public static bool EnableProgress { get; set; } = true;

    /// <summary>
    /// Emit a heartbeat log for long-running trials (seconds). Set to 0 to disable.
    /// </summary>
    public static int TrialHeartbeatSeconds { get; set; } = 0;

    /// <summary>
    /// Log file path for diagnostic output. If null, logs to console.
    /// Shared with DynamicGemmKernel.
    /// </summary>
    public static string? LogFilePath
    {
        get => DynamicGemmKernel.LogFilePath;
        set => DynamicGemmKernel.LogFilePath = value;
    }

    /// <summary>
    /// CSV log file path for trial features and results.
    /// </summary>
    public static string? TrialLogFilePath { get; set; }

    private static readonly object _trialLogLock = new();
    private static string? _trialLogHeader;

    /// <summary>
    /// Logs a diagnostic message if diagnostics are enabled.
    /// Uses same log file as DynamicGemmKernel if configured.
    /// </summary>
    private static void LogDiag(string message)
    {
        if (!EnableDiagnostics && Logger == null && string.IsNullOrEmpty(LogFilePath))
            return;

        string logLine = $"[{DateTime.Now:HH:mm:ss.fff}] [GemmTuner] {message}";

        if (Logger != null)
        {
            Logger.LogInformation(logLine);
        }

        if (!string.IsNullOrEmpty(LogFilePath))
        {
            if (EnableDiagnostics)
                DynamicGemmKernel.EnableDiagnostics = true;
            try
            {
                using var sw = new System.IO.StreamWriter(LogFilePath, append: true);
                sw.WriteLine(logLine);
            }
            catch
            {
                if (EnableDiagnostics && Logger == null)
                    Console.WriteLine(logLine);
            }
        }
        else if (EnableDiagnostics && Logger == null)
        {
            WriteConsoleWithColor(logLine);
        }
    }

    private static void LogProgress(string message)
    {
        if (!EnableProgress)
            return;

        if (Logger != null)
        {
            Logger.LogInformation(message);
            return;
        }

        WriteConsoleWithColor(message);
    }

    private static void WriteConsoleWithColor(string message)
    {
        if (Console.IsOutputRedirected)
        {
            Console.WriteLine(message);
            return;
        }

        var color = GetDiagColor(message);
        if (color.HasValue)
        {
            var previous = Console.ForegroundColor;
            Console.ForegroundColor = color.Value;
            Console.WriteLine(message);
            Console.ForegroundColor = previous;
        }
        else
        {
            Console.WriteLine(message);
        }
    }

    private static ConsoleColor? GetDiagColor(string message)
    {
        if (message.Contains("Diag[HIGH]", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("LIKELY COMPUTE BOUND", StringComparison.OrdinalIgnoreCase))
            return ConsoleColor.Red;
        if (message.Contains("Diag[MED]", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("LIKELY MEMORY BOUND", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("MODERATE EFFICIENCY", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("WARNING", StringComparison.OrdinalIgnoreCase))
            return ConsoleColor.Yellow;
        if (message.Contains("Diag[LOW]", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("GOOD EFFICIENCY", StringComparison.OrdinalIgnoreCase))
            return ConsoleColor.Green;
        return null;
    }

    private static string GetTrialLogHeader()
    {
        if (_trialLogHeader != null)
            return _trialLogHeader;

        lock (_trialLogLock)
        {
            if (_trialLogHeader != null)
                return _trialLogHeader;

            var headers = new List<string>
            {
                "TimestampUtc",
                "TrialIndex",
                "Phase",
                "Strategy",
                "M",
                "N",
                "K",
                "KernelName",
                "IsValid",
                "GFlops",
                "TimeMs",
                "Error",
                "ConfigKey",
                "TileM",
                "TileN",
                "TileK",
                "ThreadTileM",
                "ThreadTileN",
                "VectorWidthM",
                "VectorWidthN",
                "UseDoubleBuffering",
                "UseVectorizedLoads",
                "KReg",
                "KUnroll",
                "UseSubgroupOps",
                "StrideM",
                "StrideN",
                "CacheA",
                "CacheB",
                "MdimaSize",
                "NdimbSize",
                "UseTrueVectorLDS",
                "UseColumnMajorA",
                "Diag_OccupancyEst",
                "Diag_LdsUsageKb",
                "Diag_LdsLimitKb",
                "Diag_RegistersEst",
                "Diag_RegisterLimit",
                "Diag_WorkgroupSize",
                "Diag_WavefrontSize",
                "Diag_WaveUtilization",
                "Diag_ComputeIntensity",
                "Diag_VectorBandwidth",
                "Diag_IlpFactor",
                "Diag_PadRatio",
                "Diag_BottleneckHints",
                "Diag_BottleneckSeverity",
                "Diag_BottleneckSummary"
            };

            foreach (var name in GemmFeatureBayesianTuner.FeatureNames)
                headers.Add($"Feature_{name}");

            _trialLogHeader = string.Join(",", headers);
            return _trialLogHeader;
        }
    }

    private static void LogTrialCsv(int trialIndex, string phase, string strategy, int M, int N, int K,
        TuningResult result, GpuCapabilities? capabilities)
    {
        if (string.IsNullOrWhiteSpace(TrialLogFilePath))
            return;

        try
        {
            string path = TrialLogFilePath!;
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            var config = result.Config;
            var values = new List<string>
            {
                DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                trialIndex.ToString(CultureInfo.InvariantCulture),
                phase,
                strategy,
                M.ToString(CultureInfo.InvariantCulture),
                N.ToString(CultureInfo.InvariantCulture),
                K.ToString(CultureInfo.InvariantCulture),
                config.KernelName ?? string.Empty,
                result.IsValid ? "1" : "0",
                result.GFlops.ToString("G17", CultureInfo.InvariantCulture),
                result.TimeMs.ToString("G17", CultureInfo.InvariantCulture),
                result.Error ?? string.Empty,
                config.ToKey(),
                config.TileM.ToString(CultureInfo.InvariantCulture),
                config.TileN.ToString(CultureInfo.InvariantCulture),
                config.TileK.ToString(CultureInfo.InvariantCulture),
                config.ThreadTileM.ToString(CultureInfo.InvariantCulture),
                config.ThreadTileN.ToString(CultureInfo.InvariantCulture),
                config.VectorWidthM.ToString(CultureInfo.InvariantCulture),
                config.VectorWidthN.ToString(CultureInfo.InvariantCulture),
                config.UseDoubleBuffering ? "1" : "0",
                config.UseVectorizedLoads ? "1" : "0",
                config.KReg.ToString(CultureInfo.InvariantCulture),
                config.KUnroll.ToString(CultureInfo.InvariantCulture),
                config.UseSubgroupOps ? "1" : "0",
                config.StrideM ? "1" : "0",
                config.StrideN ? "1" : "0",
                config.CacheA ? "1" : "0",
                config.CacheB ? "1" : "0",
                config.MdimaSize.ToString(CultureInfo.InvariantCulture),
                config.NdimbSize.ToString(CultureInfo.InvariantCulture),
                config.UseTrueVectorLDS ? "1" : "0",
                config.UseColumnMajorA ? "1" : "0"
            };

            var diag = AnalyzeBottlenecks(config, capabilities ?? GpuCapabilities.CreateDefault(), M, N, K);
            values.Add(diag.OccupancyEst.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.LdsUsageKb.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.LdsLimitKb.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.RegistersEst.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.RegisterLimit.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.WorkgroupSize.ToString(CultureInfo.InvariantCulture));
            values.Add(diag.WavefrontSize.ToString(CultureInfo.InvariantCulture));
            values.Add(diag.WaveUtilization.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.ComputeIntensity.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.VectorBandwidth.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.IlpFactor.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.PadRatio.ToString("G17", CultureInfo.InvariantCulture));
            values.Add(diag.Hints);
            values.Add(diag.Severity);
            values.Add(diag.Summary);

            var featureValues = GemmFeatureBayesianTuner.ExtractFeatureVector(config);
            foreach (var feature in featureValues)
            {
                values.Add(feature.ToString("G17", CultureInfo.InvariantCulture));
            }

            string line = string.Join(",", values.Select(EscapeCsv));

            lock (_trialLogLock)
            {
                bool needsHeader = !File.Exists(path) || new FileInfo(path).Length == 0;
                using var sw = new StreamWriter(path, append: true);
                if (needsHeader)
                    sw.WriteLine(GetTrialLogHeader());
                sw.WriteLine(line);
            }
        }
        catch
        {
            // Ignore CSV logging failures to avoid interrupting tuning.
        }
    }

    private static string EscapeCsv(string value)
    {
        if (value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0)
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        return value;
    }

    private readonly struct BottleneckDiagnostics
    {
        public double OccupancyEst { get; init; }
        public double LdsUsageKb { get; init; }
        public double LdsLimitKb { get; init; }
        public double RegistersEst { get; init; }
        public double RegisterLimit { get; init; }
        public int WorkgroupSize { get; init; }
        public int WavefrontSize { get; init; }
        public double WaveUtilization { get; init; }
        public double ComputeIntensity { get; init; }
        public double VectorBandwidth { get; init; }
        public double IlpFactor { get; init; }
        public double PadRatio { get; init; }
        public string Hints { get; init; }
        public string Severity { get; init; }
        public string Summary { get; init; }
    }

    private static BottleneckDiagnostics AnalyzeBottlenecks(GemmConfig config, GpuCapabilities capabilities, int M, int N, int K)
    {
        static int CeilDiv(int value, int divisor) => (value + divisor - 1) / divisor;

        int tileM = Math.Max(1, config.TileM);
        int tileN = Math.Max(1, config.TileN);
        int tileK = Math.Max(1, config.TileK);
        int threadTileM = Math.Max(1, config.ThreadTileM > 0 ? config.ThreadTileM : 8);
        int threadTileN = Math.Max(1, config.ThreadTileN > 0 ? config.ThreadTileN : 8);
        int vwm = Math.Max(1, config.VectorWidthM > 0 ? config.VectorWidthM : 1);
        int vwn = Math.Max(1, config.VectorWidthN > 0 ? config.VectorWidthN : 1);
        int kreg = Math.Max(1, config.KReg > 0 ? config.KReg : 1);
        int kunroll = Math.Max(1, config.KUnroll > 0 ? config.KUnroll : 1);
        bool usesClBlastBaselineK0 = !string.IsNullOrWhiteSpace(config.KernelName) &&
            config.KernelName.StartsWith("clblast_baseline_k0", StringComparison.OrdinalIgnoreCase);

        int mwi = Math.Max(1, tileM / threadTileM);
        int nwi = Math.Max(1, tileN / threadTileN);
        double computeIntensity = mwi * nwi;
        int workgroupSize = Math.Max(1, threadTileM * threadTileN);
        int wavefrontSize = Math.Max(1, capabilities.WavefrontSize);
        double waveCount = Math.Ceiling((double)workgroupSize / wavefrontSize);
        double waveUtilization = wavefrontSize > 0 ? workgroupSize / (waveCount * wavefrontSize) : 1.0;

        double ldsUsageKb = (tileK * (tileM + 1) + tileK * (tileN + 1)) * sizeof(float) / 1024.0;
        if (config.UseDoubleBuffering)
            ldsUsageKb *= 2.0;

        double ldsLimitKb = Math.Max(1.0, capabilities.LocalMemoryBytes / 1024.0);
        double ldsOccupancy = Math.Min(1.0, ldsLimitKb / Math.Max(0.001, ldsUsageKb));

        double registersEst = computeIntensity + mwi + nwi + 8.0;
        bool isNvidia = capabilities.VendorName.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0;
        double registerLimit = isNvidia ? 255.0 : 256.0;
        double regOccupancy = Math.Min(1.0, registerLimit / Math.Max(1.0, registersEst));
        double occupancyEst = Math.Min(ldsOccupancy, regOccupancy);

        double vecBandwidth = vwm * vwn;
        double ilpFactor = kreg * kunroll;

        double padRatio = 1.0;
        if (M > 0 && N > 0 && K > 0)
        {
            int kUnit = tileK * kreg;
            int mPad = CeilDiv(M, tileM) * tileM;
            int nPad = CeilDiv(N, tileN) * tileN;
            int kPad = CeilDiv(K, kUnit) * kUnit;
            double baseElems = (double)M * K + (double)K * N + (double)M * N;
            double paddedElems = (double)mPad * kPad + (double)kPad * nPad + (double)mPad * nPad;
            if (baseElems > 0)
                padRatio = paddedElems / baseElems;
        }

        var hints = new List<string>();
        var summaryParts = new List<string>();
        if (workgroupSize < wavefrontSize)
        {
            hints.Add("workgroup smaller than wavefront");
            summaryParts.Add("not enough threads per wave");
        }
        else if (workgroupSize % wavefrontSize != 0)
        {
            hints.Add("workgroup not multiple of wavefront");
            summaryParts.Add("workgroup doesn't align with wavefront");
        }
        if (waveUtilization < 0.9)
        {
            hints.Add("wave underutilization");
            summaryParts.Add("wavefronts underutilized");
        }
        if (occupancyEst < 0.5)
        {
            hints.Add("low occupancy");
            summaryParts.Add("low occupancy limits throughput");
        }
        if (ldsUsageKb > ldsLimitKb * 0.8)
        {
            hints.Add("lds near limit");
            summaryParts.Add("shared memory pressure");
        }
        if (registersEst > registerLimit * 0.8)
        {
            hints.Add("register pressure");
            summaryParts.Add("register pressure limits occupancy");
        }
        if (computeIntensity < 8)
        {
            hints.Add("low compute intensity");
            summaryParts.Add("likely memory-bound");
        }
        if (vecBandwidth < 2)
        {
            hints.Add("low vector width");
            summaryParts.Add("low vectorization");
        }
        if (!config.UseVectorizedLoads)
        {
            hints.Add("scalar loads");
            summaryParts.Add("scalar memory loads");
        }
        if (!config.CacheA || !config.CacheB)
        {
            hints.Add("no lds cache for A/B");
            summaryParts.Add("no local cache for A/B tiles");
        }
        if (!config.UseTrueVectorLDS && (config.CacheA || config.CacheB))
        {
            hints.Add("scalar lds");
            summaryParts.Add("scalar LDS path");
        }
        if (config.UseColumnMajorA)
        {
            hints.Add("packed A path");
            summaryParts.Add("extra packing/transpose overhead");
        }
        if (usesClBlastBaselineK0)
        {
            hints.Add("packed C path");
            summaryParts.Add("extra C transpose overhead");
        }
        if (padRatio > 1.05)
        {
            hints.Add(string.Format(CultureInfo.InvariantCulture, "padding +{0:F0}%", (padRatio - 1.0) * 100.0));
            summaryParts.Add("padding overhead");
        }
        if (ilpFactor < 2)
        {
            hints.Add("low ilp");
            summaryParts.Add("low instruction-level parallelism");
        }

        int severityScore = 0;
        if (occupancyEst < 0.4)
            severityScore += 2;
        else if (occupancyEst < 0.7)
            severityScore += 1;
        if (padRatio > 1.2)
            severityScore += 2;
        else if (padRatio > 1.05)
            severityScore += 1;
        if (waveUtilization < 0.8)
            severityScore += 1;
        if (ldsUsageKb > ldsLimitKb * 0.85)
            severityScore += 2;
        else if (ldsUsageKb > ldsLimitKb * 0.7)
            severityScore += 1;
        if (registersEst > registerLimit * 0.9)
            severityScore += 2;
        else if (registersEst > registerLimit * 0.75)
            severityScore += 1;
        if (!config.UseVectorizedLoads || vecBandwidth < 2)
            severityScore += 1;
        if (computeIntensity < 8)
            severityScore += 1;

        string severity = severityScore >= 5 ? "HIGH" : severityScore >= 3 ? "MED" : "LOW";
        string summary = summaryParts.Count == 0
            ? "no obvious bottleneck"
            : $"likely bottlenecks: {string.Join(", ", summaryParts.Take(3))}";

        string hintText = hints.Count == 0 ? "none" : string.Join(";", hints);

        return new BottleneckDiagnostics
        {
            OccupancyEst = occupancyEst,
            LdsUsageKb = ldsUsageKb,
            LdsLimitKb = ldsLimitKb,
            RegistersEst = registersEst,
            RegisterLimit = registerLimit,
            WorkgroupSize = workgroupSize,
            WavefrontSize = wavefrontSize,
            WaveUtilization = waveUtilization,
            ComputeIntensity = computeIntensity,
            VectorBandwidth = vecBandwidth,
            IlpFactor = ilpFactor,
            PadRatio = padRatio,
            Hints = hintText,
            Severity = severity,
            Summary = summary
        };
    }

    private static void LogBottleneckDiagnostics(GemmConfig config, GpuCapabilities capabilities, int M, int N, int K)
    {
        if (!EnableDiagnostics)
            return;

        var diag = AnalyzeBottlenecks(config, capabilities, M, N, K);
        LogDiag($"    Diag[{diag.Severity}]: occ={diag.OccupancyEst:F2} (lds={diag.LdsUsageKb:F1}KB/{diag.LdsLimitKb:F0}KB, " +
                $"reg={diag.RegistersEst:F0}/{diag.RegisterLimit:F0}) wg={diag.WorkgroupSize} wave={diag.WavefrontSize} " +
                $"util={diag.WaveUtilization:F2} ci={diag.ComputeIntensity:F0} vec={diag.VectorBandwidth:F0} " +
                $"ilp={diag.IlpFactor:F0} pad={diag.PadRatio:F2} summary={diag.Summary} hints={diag.Hints}");
    }

    private sealed class TrialHeartbeat : IDisposable
    {
        private static readonly char[] Spinner = new[] { '|', '/', '-', '\\' };
        private readonly string _label;
        private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
        private readonly System.Threading.Timer _timer;
        private int _spinnerIndex;
        private bool _disposed;

        public TrialHeartbeat(string label, int intervalSeconds)
        {
            _label = label;
            int intervalMs = Math.Max(1, intervalSeconds) * 1000;
            _timer = new System.Threading.Timer(_ => Tick(), null, intervalMs, intervalMs);
        }

        private void Tick()
        {
            if (_disposed)
                return;

            double elapsed = _stopwatch.Elapsed.TotalSeconds;
            char spinner = Spinner[_spinnerIndex++ % Spinner.Length];
            LogProgress($"    [{spinner}] running {_label}, elapsed {elapsed:F1}s");
        }

        public void Dispose()
        {
            _disposed = true;
            _timer.Dispose();
        }
    }

    // Predefined configurations for different matrix sizes (AMD RDNA1/RDNA2 optimized via A/B testing)
    // These configs achieved 78% efficiency on RX 5500 XT (gfx1012) - will need per-GPU tuning
    private static readonly GemmConfig[] _smallConfigs = new[]
    {
        // Small matrices (512): tile64x64_k32 - 25% efficiency (memory/launch limited)
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_small"
        },
        // Fallback for very small matrices
        new GemmConfig { TileM = 16, TileN = 16, TileK = 16, ThreadTileM = 2, ThreadTileN = 2, UseDoubleBuffering = false, UseVectorizedLoads = false, KernelName = "gemm_small" },
    };

    private static readonly GemmConfig[] _mediumConfigs = new[]
    {
        // Medium matrices (768-1024): tile64x64_k16_ku4 - 42-54% efficiency
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_medium"
        },
        // Original baseline for comparison
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0"
        },
    };

    private static readonly GemmConfig[] _largeConfigs = new[]
    {
        // Large matrices (1024+): tile64x128_k8_ku8 - 78% efficiency (A/B tested winner)
        // TileM=64, TileN=128 provides optimal memory coalescing on RDNA
        // TileK=8 with KUnroll=8 reduces LDS pressure while maintaining compute efficiency
        new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 8, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 16,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_large"
        },
        // Original CLBlast baseline for fallback/comparison
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0"
        },
        // Alternative: 128x64 for M-dominant shapes
        new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 8, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_large_m"
        },
    };

    /// <summary>
    /// Selects the best kernel configuration based on matrix dimensions and GPU capabilities.
    /// Uses heuristics for instant selection without runtime tuning.
    /// </summary>
    public GemmConfig SelectConfig(int M, int N, int K, GpuCapabilities capabilities)
    {
        // Check cache first
        lock (_cacheLock)
        {
            if (_cache.TryGetValue((M, N, K), out var cached))
                return cached;
        }

        var config = SelectConfigHeuristic(M, N, K, capabilities);

        // Cache the result
        lock (_cacheLock)
        {
            _cache[(M, N, K)] = config;
        }

        return config;
    }

    /// <summary>
    /// Heuristic-based configuration selection (no runtime tuning needed).
    /// Based on A/B testing results on AMD RDNA1 (RX 5500 XT gfx1012).
    /// Thresholds optimized for maximum efficiency at each size range.
    /// </summary>
    private GemmConfig SelectConfigHeuristic(int M, int N, int K, GpuCapabilities capabilities)
    {
        long totalOps = 2L * M * N * K;  // FLOPs for GEMM
        int maxDim = Math.Max(Math.Max(M, N), K);

        // Very small matrices: overhead-sensitive, use simple kernel
        if (maxDim <= 64 || totalOps < 100_000)
        {
            if (maxDim <= 32)
                return _smallConfigs[1];  // gemm_small fallback

            return _smallConfigs[1];  // gemm_small for tiny matrices
        }

        // Small matrices (65-512): tile64x64_k32 - 25% efficiency (launch overhead limited)
        // A/B tested: TileK=32 with minimal unroll works best for memory-bound small sizes
        if (maxDim <= 512)
        {
            return _smallConfigs[0];  // tile64x64_k32
        }

        // Medium matrices (513-1023): tile64x64_k16_ku4 - 42-54% efficiency
        // A/B tested: TileK=16 with KUnroll=4 balances LDS pressure and compute
        if (maxDim < 1024)
        {
            return _mediumConfigs[0];  // tile64x64_k16_ku4
        }

        // Large matrices (1024+): tile64x128_k8_ku8 - 73-78% efficiency
        // A/B tested winner: asymmetric 64x128 tiles with aggressive K unrolling
        // TileK=8 reduces LDS pressure, KUnroll=8 maintains high throughput
        return _largeConfigs[0];  // tile64x128_k8_ku8 - THE PERFORMANCE WINNER
    }

    /// <summary>
    /// Runs actual benchmark to find optimal configuration using exhaustive search.
    /// Use this for production workloads that will run many times.
    /// </summary>
    public TuningResult[] TuneForDimensions(int M, int N, int K, GpuCapabilities capabilities,
        Func<GemmConfig, double> benchmarkFunc, int warmupRuns = 2, int benchmarkRuns = 5, bool useFullSearchSpace = false)
    {
        var candidates = useFullSearchSpace
            ? GenerateConfigurationSpace(M, N, K, capabilities)
            : GetCandidateConfigs(M, N, K, capabilities);
        if (useFullSearchSpace)
        {
            var unique = new Dictionary<string, GemmConfig>(StringComparer.Ordinal);
            foreach (var config in candidates)
            {
                var key = config.ToKey();
                if (!unique.ContainsKey(key))
                    unique[key] = config;
            }
            candidates = unique.Values.ToArray();
        }
        var results = new List<TuningResult>();
        int trialIndex = 0;
        int totalCandidates = candidates.Length;

        foreach (var config in candidates)
        {
            trialIndex++;
            string progressLabel = $"trial {trialIndex}/{totalCandidates} exhaustive {config.KernelName ?? "kernel"}";
            var result = BenchmarkConfig(config, benchmarkFunc, warmupRuns, benchmarkRuns, M, N, K, capabilities, progressLabel);
            results.Add(result);
            LogTrialCsv(trialIndex, "exhaustive", "exhaustive", M, N, K, result, capabilities);

            if (trialIndex % ProgressInterval == 0 || trialIndex == totalCandidates)
            {
                double pct = totalCandidates > 0 ? (trialIndex * 100.0 / totalCandidates) : 100.0;
                LogProgress($"[Progress] exhaustive {trialIndex}/{totalCandidates} ({pct:F1}%)");
            }
        }

        // Sort by GFLOPS (descending)
        results.Sort((a, b) => b.GFlops.CompareTo(a.GFlops));

        // Cache the best result
        if (results.Count > 0 && results[0].IsValid)
        {
            lock (_cacheLock)
            {
                _cache[(M, N, K)] = results[0].Config;
            }
        }

        return results.ToArray();
    }

    /// <summary>
    /// Runs Bayesian optimization to efficiently find optimal configuration.
    /// More efficient than exhaustive search when the configuration space is large.
    /// Uses Gaussian Process regression with Expected Improvement acquisition function.
    /// </summary>
    /// <remarks>
    /// This implementation is inspired by the AiDotNet HyperparameterOptimization/BayesianOptimizer.
    /// It uses a simplified GP model optimized for kernel parameter tuning.
    /// </remarks>
    public TuningResult[] TuneWithBayesianOptimization(
        int M, int N, int K,
        GpuCapabilities capabilities,
        Func<GemmConfig, double> benchmarkFunc,
        int maxTrials = 20,
        int initialRandomSamples = 5,
        int warmupRuns = 2,
        int benchmarkRuns = 3,
        int? seed = null)
    {
        var tuningStopwatch = Stopwatch.StartNew();
        var allConfigs = GenerateConfigurationSpace(M, N, K, capabilities);
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var bayesian = new GemmFeatureBayesianTuner(allConfigs, seed);

        // Count valid configurations for diagnostics
        int validConfigs = 0;
        int invalidConfigs = 0;
        foreach (var cfg in allConfigs)
        {
            if (DynamicGemmKernel.ValidateConfig(cfg) == null)
                validConfigs++;
            else
                invalidConfigs++;
        }

        var stratifiedSamples = GetStratifiedSamples(allConfigs, random);
        initialRandomSamples = Math.Min(maxTrials, Math.Max(initialRandomSamples, stratifiedSamples.Count));
        initialRandomSamples = Math.Min(initialRandomSamples, allConfigs.Length);

        LogDiag($"=== Bayesian Optimization for {M}x{N}x{K} ===");
        LogDiag($"Configuration space: {allConfigs.Length} total, {validConfigs} valid, {invalidConfigs} invalid");
        LogDiag($"Max trials: {maxTrials}, Random samples: {initialRandomSamples}");
        LogDiag($"Warmup runs: {warmupRuns}, Benchmark runs: {benchmarkRuns}");

        if (allConfigs.Length <= initialRandomSamples)
        {
            LogDiag("Config space too small, using exhaustive search");
            return TuneForDimensions(M, N, K, capabilities, benchmarkFunc, warmupRuns, benchmarkRuns);
        }

        var results = new List<TuningResult>();
        var testedIndices = new HashSet<int>();
        double bestGflops = 0;
        string bestConfig = "";
        int failedTrials = 0;
        int trialsSinceImprovement = 0;
        int earlyStopPatience = Math.Max(15, maxTrials / 5);  // Stop after N trials without improvement

        // Phase 1: Stratified sampling for kernel type diversity
        LogDiag($"\n--- Phase 1: Stratified Exploration ({initialRandomSamples} trials) ---");

        for (int i = 0; i < initialRandomSamples && i < allConfigs.Length; i++)
        {
            int idx;
            string strategy;
            if (i < stratifiedSamples.Count)
            {
                // Use stratified sample for the first samples (one from each category)
                idx = stratifiedSamples[i];
                if (testedIndices.Contains(idx))
                {
                    idx = bayesian.SampleRandomIndex(allConfigs.Length, testedIndices);
                    strategy = "random";
                }
                else
                {
                    strategy = "stratified";
                }
            }
            else
            {
                // After covering all categories, use random sampling
                idx = bayesian.SampleRandomIndex(allConfigs.Length, testedIndices);
                strategy = "random";
            }
            testedIndices.Add(idx);

            var config = allConfigs[idx];
            LogDiag($"Trial {i + 1}/{maxTrials}: {config.KernelName} [{strategy}]");
            string progressLabel = $"trial {i + 1}/{maxTrials} {strategy} {config.KernelName ?? "kernel"}";
            var result = BenchmarkConfig(config, benchmarkFunc, warmupRuns, benchmarkRuns, M, N, K, capabilities, progressLabel);
            results.Add(result);
            LogTrialCsv(i + 1, "explore", strategy, M, N, K, result, capabilities);

            if (result.IsValid)
            {
                bayesian.AddObservation(idx, result.GFlops);
                if (result.GFlops > bestGflops)
                {
                    bestGflops = result.GFlops;
                    bestConfig = config.ToString();
                    LogDiag($"  NEW BEST: {bestGflops:F2} GFLOPS");
                }
            }
            else
            {
                failedTrials++;
                bayesian.AddObservation(idx, 0);  // Record failure
            }

            int progressTrial = i + 1;
            if (progressTrial % ProgressInterval == 0 || progressTrial == initialRandomSamples)
            {
                double pct = maxTrials > 0 ? (progressTrial * 100.0 / maxTrials) : 100.0;
                LogProgress($"[Progress] explore {progressTrial}/{maxTrials} ({pct:F1}%) best={bestGflops:F2} failed={failedTrials}");
            }
        }

        if (testedIndices.Count >= allConfigs.Length)
        {
            LogDiag("\nAll configurations tested during stratified exploration.");
        }
        else
        {
            // Phase 2: Bayesian optimization
            LogDiag($"\n--- Phase 2: Bayesian Optimization ({maxTrials - initialRandomSamples} trials) ---");
            LogDiag($"Current best: {bestGflops:F2} GFLOPS - {bestConfig}");
            LogDiag($"Early stopping patience: {earlyStopPatience} trials without improvement");

            for (int trial = initialRandomSamples; trial < maxTrials && testedIndices.Count < allConfigs.Length; trial++)
            {
                // Check early stopping
                if (trialsSinceImprovement >= earlyStopPatience)
                {
                    LogDiag($"\n*** EARLY STOPPING: No improvement for {earlyStopPatience} trials ***");
                    LogDiag($"Stopped at trial {trial}/{maxTrials}, saved {maxTrials - trial} trials");
                    break;
                }

                int? coverageIdx = TrySelectCoverageCandidate(allConfigs, testedIndices, random);
                string strategy;
                int nextIdx;

                if (coverageIdx.HasValue)
                {
                    nextIdx = coverageIdx.Value;
                    strategy = "coverage";
                }
                else
                {
                    // Update GP model
                    bayesian.UpdateModel();

                    int phaseIndex = trial - initialRandomSamples;
                    int selector = phaseIndex % 10;
                    if (selector == 0)
                    {
                        strategy = "random";
                        nextIdx = bayesian.SampleRandomIndex(allConfigs.Length, testedIndices);
                    }
                    else if (selector == 1)
                    {
                        strategy = "uncertainty";
                        nextIdx = bayesian.SelectNextPointByUncertainty(allConfigs.Length, testedIndices);
                    }
                    else if (selector == 2)
                    {
                        strategy = "ucb";
                        nextIdx = bayesian.SelectNextPointByUcb(allConfigs.Length, testedIndices, 1.5);
                    }
                    else
                    {
                        strategy = "ei";
                        nextIdx = bayesian.SelectNextPoint(allConfigs.Length, testedIndices);
                    }
                }

                testedIndices.Add(nextIdx);

                var config = allConfigs[nextIdx];

                // Show progress at intervals
                if ((trial + 1) % ProgressInterval == 0 || trial == maxTrials - 1)
                {
                    double elapsed = tuningStopwatch.Elapsed.TotalSeconds;
                    double trialsPerSec = elapsed > 0 ? (trial + 1) / elapsed : 0.0;
                    double eta = trialsPerSec > 0 ? (maxTrials - trial - 1) / trialsPerSec : 0.0;
                    double pct = maxTrials > 0 ? ((trial + 1) * 100.0 / maxTrials) : 100.0;
                    LogProgress($"[Progress] bayes {trial + 1}/{maxTrials} ({pct:F1}%) best={bestGflops:F2} failed={failedTrials} noImprove={trialsSinceImprovement}/{earlyStopPatience} eta={eta:F1}s");
                }

                string progressLabel = $"trial {trial + 1}/{maxTrials} {strategy} {config.KernelName ?? "kernel"}";
                var result = BenchmarkConfig(config, benchmarkFunc, warmupRuns, benchmarkRuns, M, N, K, capabilities, progressLabel);
                results.Add(result);
                LogTrialCsv(trial + 1, "bayes", strategy, M, N, K, result, capabilities);

                if (result.IsValid)
                {
                    bayesian.AddObservation(nextIdx, result.GFlops);
                    if (result.GFlops > bestGflops)
                    {
                        bestGflops = result.GFlops;
                        bestConfig = config.ToString();
                        trialsSinceImprovement = 0;  // Reset counter on improvement
                        LogDiag($"  NEW BEST at trial {trial + 1}: {bestGflops:F2} GFLOPS - {config.KernelName}");
                    }
                    else
                    {
                        trialsSinceImprovement++;  // No improvement
                    }
                }
                else
                {
                    failedTrials++;
                    trialsSinceImprovement++;  // Failed trial counts as no improvement
                    bayesian.AddObservation(nextIdx, 0);
                }
            }
        }

        tuningStopwatch.Stop();

        // Sort by GFLOPS (descending)
        results.Sort((a, b) => b.GFlops.CompareTo(a.GFlops));

        // Log summary
        int successfulTrials = results.Count(r => r.IsValid);
        LogDiag($"\n=== Tuning Complete ===");
        LogDiag($"Elapsed: {tuningStopwatch.Elapsed.TotalSeconds:F1}s");
        LogDiag($"Trials: {results.Count} total, {successfulTrials} successful, {failedTrials} failed");
        LogDiag($"Best: {bestGflops:F2} GFLOPS - {bestConfig}");

        if (results.Count > 0)
        {
            LogDiag("\nTop 5 configurations:");
            for (int i = 0; i < Math.Min(5, results.Count); i++)
            {
                var r = results[i];
                if (r.IsValid)
                {
                    LogDiag($"  {i + 1}. {r.GFlops:F2} GFLOPS - {r.Config}");
                }
            }
        }

        // Cache the best result
        if (results.Count > 0 && results[0].IsValid)
        {
            lock (_cacheLock)
            {
                _cache[(M, N, K)] = results[0].Config;
            }
        }

        return results.ToArray();
    }

    private static bool IsValidTime(double timeMs)
    {
        return !double.IsNaN(timeMs) &&
               !double.IsInfinity(timeMs) &&
               timeMs > 0 &&
               timeMs < double.MaxValue / 4;
    }

    private TuningResult BenchmarkConfig(GemmConfig config, Func<GemmConfig, double> benchmarkFunc,
        int warmupRuns, int benchmarkRuns, int M, int N, int K, GpuCapabilities capabilities,
        string? progressLabel = null)
    {
        // First validate the configuration
        var validationError = DynamicGemmKernel.ValidateConfig(config);
        if (validationError != null)
        {
            LogDiag($"  Config invalid: {validationError}");
            return new TuningResult
            {
                Config = config,
                GFlops = 0,
                TimeMs = double.MaxValue,
                IsValid = false,
                Error = validationError
            };
        }

        var sw = Stopwatch.StartNew();
        TrialHeartbeat? heartbeat = null;
        try
        {
            if (TrialHeartbeatSeconds > 0 && (EnableDiagnostics || EnableProgress))
            {
                // Ensure label is non-null for TrialHeartbeat constructor
                string label = string.IsNullOrWhiteSpace(progressLabel)
                    ? $"{config.KernelName ?? "kernel"} {M}x{N}x{K}"
                    : progressLabel ?? string.Empty;
                heartbeat = new TrialHeartbeat(label, TrialHeartbeatSeconds);
            }

            // Warmup
            for (int i = 0; i < warmupRuns; i++)
            {
                double time = benchmarkFunc(config);
                if (!IsValidTime(time))
                {
                    LogDiag($"  {config.KernelName}: invalid warmup time {time}");
                    return new TuningResult
                    {
                        Config = config,
                        GFlops = 0,
                        TimeMs = double.MaxValue,
                        IsValid = false,
                        Error = $"Invalid warmup time: {time}"
                    };
                }
            }

            // Benchmark
            double totalTimeMs = 0;
            double minTime = double.MaxValue;
            double maxTime = 0;
            for (int i = 0; i < benchmarkRuns; i++)
            {
                double time = benchmarkFunc(config);
                if (!IsValidTime(time))
                {
                    LogDiag($"  {config.KernelName}: invalid benchmark time {time}");
                    return new TuningResult
                    {
                        Config = config,
                        GFlops = 0,
                        TimeMs = double.MaxValue,
                        IsValid = false,
                        Error = $"Invalid benchmark time: {time}"
                    };
                }
                totalTimeMs += time;
                minTime = Math.Min(minTime, time);
                maxTime = Math.Max(maxTime, time);
            }

            double avgTimeMs = totalTimeMs / benchmarkRuns;
            if (!IsValidTime(avgTimeMs))
            {
                LogDiag($"  {config.KernelName}: invalid average time {avgTimeMs}");
                return new TuningResult
                {
                    Config = config,
                    GFlops = 0,
                    TimeMs = double.MaxValue,
                    IsValid = false,
                    Error = $"Invalid average time: {avgTimeMs}"
                };
            }
            double gflops = (2.0 * M * N * K) / (avgTimeMs * 1e6);
            sw.Stop();

            LogDiag($"  {config.KernelName}: {gflops:F2} GFLOPS (avg: {avgTimeMs:F3} ms, min: {minTime:F3}, max: {maxTime:F3})");
            LogBottleneckDiagnostics(config, capabilities, M, N, K);

            return new TuningResult
            {
                Config = config,
                GFlops = gflops,
                TimeMs = avgTimeMs,
                IsValid = true,
                Error = null
            };
        }
        catch (Exception ex)
        {
            sw.Stop();
            LogDiag($"  Config {config.KernelName} failed: {ex.Message}");
            return new TuningResult
            {
                Config = config,
                GFlops = 0,
                TimeMs = double.MaxValue,
                IsValid = false,
                Error = ex.Message
            };
        }
        finally
        {
            heartbeat?.Dispose();
        }
    }

    /// <summary>
    /// Generates the full configuration space for Bayesian optimization.
    /// Includes CLBlast-proven configurations for AMD RDNA GPUs.
    /// </summary>
    private GemmConfig[] GenerateConfigurationSpace(int M, int N, int K, GpuCapabilities capabilities)
    {
        var configs = new List<GemmConfig>();

        // ============================================================
        // TRUE CLBLAST-STYLE VECTORIZED LDS KERNELS (THE KEY TO 2500+ GFLOPS)
        // These use UseTrueVectorLDS=true for fully vectorized local memory
        // ============================================================

        // CLBlast RX 5700 XT TRUE VECTORIZED: MWG=64, NWG=64, VWM=2, VWN=2     
        // Based on analysis: VWM=2 works best on RDNA1
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 8,
            UseTrueVectorLDS = true,  // THE KEY!
            UseColumnMajorA = true,
            KernelName = "clblast_true_vec_64x64"
        });

        // CLBlast BASELINE (exact XGEMM kernel 0 configuration for RDNA1)
        // GEMMK=0, MWG=64, NWG=64, KWG=16, MDIMC=8, NDIMC=8, VWM=2, VWN=2
        // STRM=0, STRN=1, SA=1, SB=1, MDIMA=16, NDIMB=8, KWI=2, KREG=1
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0"
        });

        // CLBlast BASELINE (kernel 1: 2D register tiling, row-major A/B/C)
        // GEMMK=1 expects KWG=1 and uses KREG for K blocking, no LDS caching
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 1,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 4, KUnroll = 1, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = false, CacheB = false,
            MdimaSize = 8, NdimbSize = 8,
            UseTrueVectorLDS = false,
            UseColumnMajorA = false,
            KernelName = "clblast_baseline_k1"
        });

        // CLBlast RX 5700 TRUE VECTORIZED: MWG=128, NWG=64, VWM=4, VWN=1
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 8,  // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 1,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = false,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 16,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_true_vec_128x64"
        });

        // 64x128 TRUE VECTORIZED - our best config with vectorized LDS
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,  // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "true_vec_64x128"
        });

        // 32x64 TRUE VECTORIZED - smaller tiles for 512x512
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,   // MWI=4, NWI=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "true_vec_32x64"
        });

        // 64x64 TRUE VECTORIZED with float4
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,   // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "true_vec_64x64_v4"
        });

        // ============================================================
        // CLBlast-optimal configurations for AMD RDNA1/RDNA2 GPUs
        // These are proven high-performance configurations from CLBlast
        // EXACT parameters from CLBlast's tuned database for gfx10xx
        // ============================================================

        // ============================================================
        // CLBlast EXACT CONFIGS - These are the actual tuned values!
        // From: https://github.com/CNugteren/CLBlast/blob/master/src/database/kernels/xgemm/
        // ============================================================

        // CLBlast RX 5700 XT exact (gfx1010 DB): MWG=64, NWG=64, KWG=16, MDIMC=8, NDIMC=8, VWM=2, VWN=2
        // RX 5500 XT (gfx1012) has no CLBlast entry; reuse this seed.
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,  // MDIMC=8, NDIMC=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = true,  // STRM=0, STRN=1
            CacheA = true, CacheB = true,     // SA=1, SB=1 local memory caching
            MdimaSize = 16, NdimbSize = 8,    // MDIMA=16, NDIMB=8
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_rx5700xt_exact"
        });

        // Same config without stride for comparison
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,     // SA=1, SB=1 local memory caching
            MdimaSize = 16, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_rx5700xt_nostride"
        });

        // CLBlast RX 5700 exact (gfx1010 DB): MWG=128, NWG=64, KWG=32, MDIMC=16, NDIMC=8, VWM=4, VWN=1
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 1,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = false,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 16,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_rx5700_exact"
        });

        // CLBlast gfx10 default (gfx1010 DB): MWG=64, NWG=64, KWG=32, MDIMC=16, NDIMC=16, VWM=4, VWN=4
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = false, CacheB = false,
            MdimaSize = 16, NdimbSize = 16,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_gfx10_default"
        });

        // ============================================================
        // VECTORIZED CONFIGS WITHOUT KREG (simpler kernel, often faster!)
        // Based on our cached best results that achieved 1913 GFLOPS
        // ============================================================

        // Our best cached config for 2048x2048: VW:2x2, no KREG
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 0, UseSubgroupOps = false,  // NO KREG - simpler is faster!
            KernelName = "simple_vec_64x128"
        });

        // ============================================================
        // 64x128 VARIANTS - Testing different vector widths
        // Base config is best for 2048x2048, try VWN=4 variants
        // ============================================================

        // 64x128 with VWN=4 (higher N vectorization like CLBlast)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            KernelName = "vec_64x128_v2x4"
        });

        // 64x128 with VW:4x4 (both high)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            KernelName = "vec_64x128_v4x4"
        });

        // 64x128 with smaller K=8 but VWN=4
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            KernelName = "vec_64x128_k8_v2x4"
        });

        // 64x128 with VW:1x4 (minimal M vectorization, high N)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 1, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            KernelName = "vec_64x128_v1x4"
        });

        // 64x256 with VW:2x4 (wider N tile)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 256, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 32,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            KernelName = "vec_64x256_v2x4"
        });

        // 128x128 with VW:2x4 and no KREG
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 8,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            KernelName = "vec_128x128_k8"
        });

        // Our best for 1024x1024: VW:1x2, no KREG
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 128, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 16,
            VectorWidthM = 1, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 0, UseSubgroupOps = false,
            KernelName = "simple_vec_32x128"
        });

        // ============================================================
        // LARGER K-TILE CONFIGURATIONS
        // Key insight: Larger K-tiles reduce K-loop iterations and sync overhead
        // For 2048x2048: TileK=8 → 256 iterations, TileK=16 → 128, TileK=32 → 64
        // ============================================================

        // Best config with larger K tile (16)
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 128, TileK = 16,  // K=16 instead of 8
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=4, NWI=8 (same pattern)
            VectorWidthM = 1, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            KernelName = "simple_vec_32x128_k16"
        });

        // Best config with even larger K tile (32)
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 128, TileK = 32,  // K=32 for maximum K-loop reduction
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=4, NWI=8 (same pattern)
            VectorWidthM = 2, VectorWidthN = 4,  // Higher vector widths for larger K
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 8, UseSubgroupOps = false,
            KernelName = "simple_vec_32x128_k32"
        });

        // 64x128 with larger K tile (32) - more work per K iteration
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 32,  // K=32
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 8, UseSubgroupOps = false,
            KernelName = "simple_vec_64x128_k32"
        });

        // 64x64 with large K tile (32) - balanced config
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 8,    // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 8, UseSubgroupOps = false,
            KernelName = "simple_64x64_k32"
        });

        // 128x64 asymmetric for M-heavy matrices (K=16)
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 8,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            KernelName = "simple_128x64_k16"
        });

        // ============================================================
        // CLBLAST AMD-STYLE CONFIGURATIONS
        // Based on actual CLBlast database for AMD GPUs:
        // - KWG=32, KWI=2 (large K with small unroll)
        // - Smaller MWG/NWG (64) to compensate for LDS usage
        // - VWM=4 common for AMD
        // ============================================================

        // CLBlast Fiji-style: 64x64 with K=32, VWM=4, VWN=4
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=4, NWI=4
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "clblast_64x64_k32_v4"
        });

        // CLBlast Vega-style: 64x64 with K=32, VWM=2, VWN=4
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 8,    // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "clblast_64x64_k32_v2x4"
        });

        // CLBlast RX580-style: 128x128 with K=16, VWM=2, VWN=4, STRM/STRN
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=8, NWI=8 (256 threads)
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = true,      // STRM=1, STRN=1 like CLBlast
            CacheA = true, CacheB = true,
            KernelName = "clblast_128x128_str"
        });

        // Smaller tiles with larger K for better K-loop efficiency
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 64, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=4, NWI=4
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            KernelName = "small_tile_k32"
        });

        // VWM=4 variant of best config (simple_vec_32x128)
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 128, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=4, NWI=8
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            KernelName = "simple_vec_32x128_v4"
        });

        // ============================================================
        // OPTIMIZED CONFIGS BASED ON CLBLAST ANALYSIS
        // Key findings from CLBlast source analysis:
        // 1. STRM/STRN for bank conflict avoidance
        // 2. Double-buffering requires MWI*NWI <= 16
        // 3. KREG=2 with proper K unrolling
        // ============================================================

        // Best config WITH STRM/STRN stride patterns (bank conflict avoidance)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,      // STRM=1, STRN=1 for bank conflict avoidance!
            CacheA = true, CacheB = true,
            KernelName = "vec_64x128_stride"
        });

        // 32x128 with STRM/STRN
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 128, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=4, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            KernelName = "vec_32x128_stride"
        });

        // HIGH-OCCUPANCY with TRUE DOUBLE-BUFFERING (MWI*NWI=16 triggers ping-pong!)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=4, NWI=4 → 16 outputs/thread
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,  // TRIGGERS PING-PONG!
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            KernelName = "double_buf_64x64_4x4"
        });

        // Even smaller tile for maximum double-buffering efficiency
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=4, NWI=4 → 16 outputs/thread
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            KernelName = "double_buf_32x64"
        });

        // KREG=2 variant to test register tiling
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = false,  // KREG=2!
            StrideM = true, StrideN = true,
            KernelName = "vec_64x128_kreg2"
        });

        // CLBlast AMD pattern: MWG>NWG, VWM=4
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 8,   // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = false,
            KernelName = "amd_128x64_v4x2"
        });

        // Float4 vectorized WITHOUT KREG
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 0, UseSubgroupOps = false,
            KernelName = "simple_vec4_64x128"
        });

        // Float8 vectorized WITHOUT KREG (test if KREG is the bottleneck)
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 8, VectorWidthN = 8,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 0, UseSubgroupOps = false,
            KernelName = "simple_vec8_128x128"
        });

        // ============================================================
        // HIGH-PERFORMANCE CONFIGS WITH KREG AND SUBGROUP OPERATIONS
        // These match CLBlast's actual implementation more closely
        // ============================================================

        // RX 5700 XT optimal with KREG=2 and subgroup ops (RDNA1)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            KernelName = "clblast_rdna1_optimal"
        });

        // High KREG variant for maximum compute density
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 4, KUnroll = 8, UseSubgroupOps = capabilities.SupportsSubgroups,
            KernelName = "clblast_high_kreg"
        });

        // RX 5700 variant with more aggressive unrolling
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            KernelName = "clblast_rdna1_alt1"
        });

        // AMD default with KREG
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            KernelName = "clblast_amd_default"
        });

        // Large tile with subgroup shuffling
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            KernelName = "clblast_large_tile"
        });

        // Extremely aggressive configuration for maximum throughput
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 4, KUnroll = 8, UseSubgroupOps = capabilities.SupportsSubgroups,
            KernelName = "clblast_max_throughput"
        });

        // Balanced config for medium matrices
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = false,  // No subgroup for baseline
            KernelName = "clblast_balanced"
        });

        // ============================================================
        // LOW-REGISTER CONFIGURATIONS for better occupancy
        // Target: 40-60 registers per thread to allow 60-80% occupancy
        // MWI×NWI = 4×4 = 16 outputs → ~20-30 registers
        // ============================================================

        // Low register config 1: 4×4 outputs, 16 threads
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 32, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,  // MWI=NWI=4
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            KernelName = "low_reg_4x4"
        });

        // Low register config 2: 4×4 outputs, larger tiles
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=NWI=4
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            KernelName = "low_reg_4x4_64"
        });

        // Low register config 3: 2×4 outputs, asymmetric
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=2, NWI=4
            VectorWidthM = 1, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            KernelName = "low_reg_2x4"
        });

        // Low register config 4: 2×2 outputs, minimal registers
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 32, TileK = 8,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=NWI=2
            VectorWidthM = 1, VectorWidthN = 1,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 1, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "low_reg_2x2"
        });

        // ============================================================
        // HIGH-OCCUPANCY DOUBLE-BUFFERED CONFIGURATIONS
        // KEY TO SURPASSING CLBlast: True ping-pong buffers for 100% latency hiding
        // Requirements: MWI*NWI <= 16 for high occupancy, UseDoubleBuffering=true
        // ============================================================

        // High-occupancy 4x4 with 64x64 tiles (256 threads, 16 outputs/thread)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 8,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=NWI=4
            VectorWidthM = 1, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,  // ENABLES PING-PONG!
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x4_64"
        });

        // High-occupancy 4x4 with 128x128 tiles, 32 threads/dim
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 8,
            ThreadTileM = 32, ThreadTileN = 32,  // MWI=NWI=4, 1024 threads BUT max is 256
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x4_128_invalid"  // Will be filtered out by validation
        });

        // High-occupancy 4x2 with 64x64 tiles (128 threads)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 8,
            ThreadTileM = 16, ThreadTileN = 32,  // MWI=4, NWI=2
            VectorWidthM = 2, VectorWidthN = 1,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x2_64"
        });

        // High-occupancy 2x4 with 64x64 tiles (128 threads)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 8,
            ThreadTileM = 32, ThreadTileN = 16,  // MWI=2, NWI=4
            VectorWidthM = 1, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_2x4_64"
        });

        // High-occupancy 2x2 with 64x64 tiles (64 threads - very high occupancy)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 32, ThreadTileN = 32,  // MWI=NWI=2, BUT 1024 threads - INVALID
            VectorWidthM = 1, VectorWidthN = 1,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_2x2_64_invalid"
        });

        // High-occupancy 4x4 with 32x32 tiles (64 threads, very high occupancy)
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 32, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,  // MWI=NWI=4, 64 threads
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x4_32"
        });

        // High-occupancy 4x4 with 64x64 tiles, larger K tile
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=NWI=4
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x4_64_k16"
        });

        // High-occupancy 4x4 with 128x64 tiles (asymmetric)
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 8,
            ThreadTileM = 32, ThreadTileN = 16,  // MWI=4, NWI=4, 512 threads - INVALID
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x4_128x64_invalid"
        });

        // High-occupancy with 16x8 = 128 threads, 4x4 outputs
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 32, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 8,  // MWI=4, NWI=4, 128 threads
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x4_64x32"
        });

        // High-occupancy with 8x16 = 128 threads, 4x4 outputs
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,  // MWI=4, NWI=4, 128 threads
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "high_occ_4x4_32x64"
        });

        // ============================================================
        // HIGH-VWM CONFIGURATIONS - KEY TO SURPASSING CLBlast
        // Use TRUE vectorized A loads (vload4/vload8) to maximize memory bandwidth
        // CLBlast uses VWM=8, VWN=8 for their best performance
        // ============================================================

        // VWM=4, VWN=4 - 4x memory bandwidth for both A and B
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,   // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 4,  // 4x bandwidth for A and B
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec4_64x64"
        });

        // VWM=8, VWN=8 - Maximum vectorization (CLBlast style)
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,   // MWI=8, NWI=8
            VectorWidthM = 8, VectorWidthN = 8,  // 8x bandwidth - CLBlast's secret!
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec8_64x64"
        });

        // VWM=4, VWN=4 with larger tiles
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec4_128x128"
        });

        // VWM=8, VWN=8 with larger tiles - matching CLBlast exactly
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=8, NWI=8
            VectorWidthM = 8, VectorWidthN = 8,  // Maximum vectorization
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec8_128x128"
        });

        // VWM=4, VWN=4 with balanced tiles
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec4_64x128"
        });

        // VWM=8, VWN=4 - Heavy A vectorization
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 8, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec8x4_64x128"
        });

        // VWM=4, VWN=8 - Heavy B vectorization
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 8,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec4x8_64x128"
        });

        // ============================================================
        // CLBLAST-STYLE TILES WITHOUT KREG
        // Testing if KREG is causing performance issues
        // ============================================================

        // CLBlast 128x128 tiles WITHOUT KREG
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,  // NO KREG!
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "clblast_128x128_nokr"
        });

        // CLBlast 128x128 tiles WITHOUT KREG, higher VWM
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MWI=8, NWI=8
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "clblast_128x128_v4_nokr"
        });

        // 64x128 with larger K tile
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 16,  // MWI=8, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 8, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec2x4_64x128_k32"
        });

        // Best-performing config variant with larger K tile
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,  // MWI=4, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec2x4_32x128_k16"
        });

        // Asymmetric tile with VWM=2
        configs.Add(new GemmConfig
        {
            TileM = 32, TileN = 256, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 32,  // MWI=4, NWI=8
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,
            KernelName = "vec2x4_32x256"
        });

        // ============================================================
        // Additional vectorized configurations to explore
        // ============================================================

        // EXPANDED configuration space for Bayesian optimization
        // Includes CLBlast parameters PLUS additional options to explore beyond their tunings
        int[] tileSizes = { 32, 64, 128, 256 };  // Added 256 for very large matrices
        int[] tileSizesK = { 8, 16, 32, 64 };    // Added 64 for deeper K unrolling
        int[] threadTiles = { 4, 8, 16 };        // Added 4 for finer granularity
        int[] vectorWidths = { 1, 2, 4, 8 };     // ADDED 8 - CLBlast's secret to 2500 GFLOPS!
        int[] kregValues = { 0, 1, 2, 4 };       // KREG: register tiling in K (0=disable, 1,2,4)
        int[] kwiValues = { 1, 2, 4 };           // KWI: K-loop unroll factor
        bool[] cacheOptions = { false, true };   // SA/SB: local memory caching
        // MDIMA/NDIMB multipliers for cooperative loading (1 = no coop, 2 = 2x loading threads)
        int[] coopMultipliers = { 1, 2 };        // 1 = MDIMA=MDIMC, 2 = MDIMA=2*MDIMC for cooperative loading

        foreach (int tileM in tileSizes)
        {
            foreach (int tileN in tileSizes)
            {
                // Skip oversized tiles for small matrices
                if (tileM > M * 2 || tileN > N * 2) continue;

                foreach (int tileK in tileSizesK)
                {
                    foreach (int ttM in threadTiles)
                    {
                        foreach (int ttN in threadTiles)
                        {
                            // Validate work group size (max 256 threads)
                            if (ttM * ttN > 256) continue;

                            // Validate tile divisibility
                            if (tileM % ttM != 0 || tileN % ttN != 0) continue;

                            int mwi = tileM / ttM;
                            int nwi = tileN / ttN;

                            // Skip if output per thread is too small
                            if (mwi < 2 || nwi < 2) continue;

                            foreach (int vwm in vectorWidths)
                            {
                                // Vector width must divide MWI
                                if (mwi % vwm != 0) continue;

                                foreach (int vwn in vectorWidths)
                                {
                                    // Vector width must divide NWI
                                    if (nwi % vwn != 0) continue;

                                    // Skip duplicate of CLBlast configs
                                    if (tileM == 64 && tileN == 64 && tileK == 16 &&
                                        ttM == 8 && ttN == 8 && vwm == 2 && vwn == 2) continue;

                                    // Add configs with different KREG/KWI combinations for high-perf setups
                                    foreach (int kreg in kregValues)
                                    {
                                        foreach (int kwi in kwiValues)
                                        {
                                            // Validate KREG/KWI compatibility with TileK
                                            if (kreg > 0 && tileK % (kwi * kreg) != 0) continue;

                                            // For large vector widths, test both SA/SB options
                                            foreach (bool cacheA in cacheOptions)
                                            {
                                                foreach (bool cacheB in cacheOptions)
                                                {
                                                    // Only test both SA/SB options for high-perf configs
                                                    // For simpler configs, just use both caching enabled
                                                    if (vwm < 4 && vwn < 4 && !(cacheA && cacheB)) continue;

                                                    // Test with different MDIMA/NDIMB cooperative loading multipliers
                                                    foreach (int coopM in coopMultipliers)
                                                    {
                                                        foreach (int coopN in coopMultipliers)
                                                        {
                                                            // Calculate MDIMA/NDIMB based on multiplier
                                                            int mdima = ttM * coopM;
                                                            int ndimb = ttN * coopN;

                                                            // Skip if MDIMA/NDIMB would exceed tile size
                                                            if (mdima > tileM || ndimb > tileN) continue;

                                                            // Only test cooperative loading (coopM>1 or coopN>1) for high-perf configs
                                                            // to avoid exploding the config space
                                                            if ((coopM > 1 || coopN > 1) && vwm < 2 && vwn < 2) continue;

                                                            string coopSuffix = (coopM > 1 || coopN > 1) ?
                                                                $"_coop{coopM}x{coopN}" : "";

                                                            configs.Add(new GemmConfig
                                                            {
                                                                TileM = tileM,
                                                                TileN = tileN,
                                                                TileK = tileK,
                                                                ThreadTileM = ttM,
                                                                ThreadTileN = ttN,
                                                                VectorWidthM = vwm,
                                                                VectorWidthN = vwn,
                                                                UseDoubleBuffering = tileM >= 64,
                                                                UseVectorizedLoads = tileK >= 16,
                                                                KReg = kreg,
                                                                KUnroll = kwi,
                                                                UseSubgroupOps = capabilities.SupportsSubgroups && vwm >= 2,
                                                                StrideM = true,
                                                                StrideN = true,
                                                                CacheA = cacheA,
                                                                CacheB = cacheB,
                                                                MdimaSize = mdima,  // MDIMA for cooperative loading
                                                                NdimbSize = ndimb,  // NDIMB for cooperative loading
                                                                KernelName = $"gemm_{tileM}x{tileN}x{tileK}_v{vwm}x{vwn}_k{kreg}x{kwi}{coopSuffix}"
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ============================================================
        // COOPERATIVE LOADING CONFIGURATIONS (CLBlast-style MDIMA/NDIMB)
        // The KEY insight: Use different thread organization for loading vs computing
        // MDIMA != MDIMC or NDIMB != NDIMC enables cooperative loading kernel
        // This is how CLBlast achieves maximum memory bandwidth on large matrices!
        // ============================================================

        // 128x128 tile with cooperative loading: 8 threads load A (MDIMA=8), 8 threads load B (NDIMB=8)
        // But compute uses 16x16 threads - more threads for loading bandwidth
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,  // MDIMC=16, NDIMC=16 (256 threads for compute)
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 8,  // MDIMA=8, NDIMB=8 (different from MDIMC/NDIMC!)
            KernelName = "coop_128x128_md8_nd8"
        });

        // 128x128 with MDIMA=16, NDIMB=16 (same as MDIMC/NDIMC but wider vectors)
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 32, NdimbSize = 32,  // Larger than MDIMC/NDIMC - more cooperative loading
            KernelName = "coop_128x128_md32_nd32"
        });

        // 64x128 with cooperative loading
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 16,   // MDIMC=8, NDIMC=16 (128 threads)
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 32,  // MDIMA=16, NDIMB=32 (more threads for B loading)
            KernelName = "coop_64x128_md16_nd32"
        });

        // 128x64 with cooperative loading (M-heavy)
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 64, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 8,   // MDIMC=16, NDIMC=8 (128 threads)
            VectorWidthM = 4, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 32, NdimbSize = 16,  // MDIMA=32 (more threads for A loading)
            KernelName = "coop_128x64_md32_nd16"
        });

        // 64x64 with cooperative loading
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,    // MDIMC=8, NDIMC=8 (64 threads)
            VectorWidthM = 2, VectorWidthN = 2,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 16,  // MDIMA=16, NDIMB=16 (2x loading threads)
            KernelName = "coop_64x64_md16_nd16"
        });

        // Large matrix optimized: 128x128 with VWN=8 and cooperative loading
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 4, VectorWidthN = 8,  // Heavy B vectorization
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 32, NdimbSize = 64,  // Very cooperative B loading
            KernelName = "coop_128x128_v8_md32_nd64"
        });

        // High-compute density: 256x256 tiles with cooperative loading
        // (May exceed LDS for some GPUs, validation will filter)
        configs.Add(new GemmConfig
        {
            TileM = 256, TileN = 256, TileK = 8,  // Small K to fit in LDS
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 4, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 0, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 32, NdimbSize = 32,
            KernelName = "coop_256x256_md32_nd32"
        });

        // CLBlast RX 5700 style: cooperative loading variant with explicit MDIMA/NDIMB
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 8,   // MDIMC=16, NDIMC=8 (128 threads)
            VectorWidthM = 8, VectorWidthN = 4,
            UseDoubleBuffering = false, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 16,  // CLBlast-style MDIMA/NDIMB
            KernelName = "clblast_rx5700_coop_md16"
        });

        // Add mixed precision configs if supported
        if (capabilities.SupportsFP16)
        {
            configs.Add(new GemmConfig
            {
                TileM = 64, TileN = 64, TileK = 32,
                ThreadTileM = 8, ThreadTileN = 8,
                VectorWidthM = 4, VectorWidthN = 4,
                UseDoubleBuffering = true, UseVectorizedLoads = true,
                KernelName = "gemm_mixed_precision"
            });
            configs.Add(new GemmConfig
            {
                TileM = 128, TileN = 128, TileK = 32,
                ThreadTileM = 16, ThreadTileN = 16,
                VectorWidthM = 4, VectorWidthN = 4,
                UseDoubleBuffering = true, UseVectorizedLoads = true,
                KernelName = "gemm_mixed_precision_large"
            });
        }

        return configs.ToArray();
    }

    /// <summary>
    /// Gets stratified samples across kernel categories and key tuning features.
    /// </summary>
    private List<int> GetStratifiedSamples(GemmConfig[] configs, Random random)
    {
        var result = new List<int>();
        var used = new HashSet<int>();

        var baseCategories = new Dictionary<string, List<int>>
        {
            ["TrueVectorized"] = new List<int>(),
            ["CooperativeLoading"] = new List<int>(),
            ["StridedAccess"] = new List<int>(),
            ["SimpleVectorized"] = new List<int>(),
            ["Basic"] = new List<int>()
        };

        var featureCategories = new Dictionary<string, List<int>>
        {
            ["ColumnMajorA"] = new List<int>(),
            ["RowMajorA"] = new List<int>(),
            ["SubgroupOps"] = new List<int>(),
            ["NoSubgroupOps"] = new List<int>(),
            ["DoubleBuffering"] = new List<int>(),
            ["SingleBuffer"] = new List<int>(),
            ["KReg>1"] = new List<int>(),
            ["KReg=1"] = new List<int>(),
            ["KUnroll>1"] = new List<int>(),
            ["KUnroll=1"] = new List<int>(),
            ["VecWidthM>1"] = new List<int>(),
            ["VecWidthM=1"] = new List<int>(),
            ["VecWidthN>1"] = new List<int>(),
            ["VecWidthN=1"] = new List<int>(),
            ["StrideM"] = new List<int>(),
            ["StrideN"] = new List<int>()
        };

        for (int i = 0; i < configs.Length; i++)
        {
            var cfg = configs[i];
            if (cfg.UseTrueVectorLDS)
                baseCategories["TrueVectorized"].Add(i);
            else if (cfg.MdimaSize > 0 || cfg.NdimbSize > 0)
                baseCategories["CooperativeLoading"].Add(i);
            else if (cfg.StrideM || cfg.StrideN)
                baseCategories["StridedAccess"].Add(i);
            else if (cfg.VectorWidthM > 1 || cfg.VectorWidthN > 1)
                baseCategories["SimpleVectorized"].Add(i);
            else
                baseCategories["Basic"].Add(i);

            if (cfg.UseColumnMajorA)
                featureCategories["ColumnMajorA"].Add(i);
            else
                featureCategories["RowMajorA"].Add(i);

            if (cfg.UseSubgroupOps)
                featureCategories["SubgroupOps"].Add(i);
            else
                featureCategories["NoSubgroupOps"].Add(i);

            if (cfg.UseDoubleBuffering)
                featureCategories["DoubleBuffering"].Add(i);
            else
                featureCategories["SingleBuffer"].Add(i);

            if (cfg.KReg > 1)
                featureCategories["KReg>1"].Add(i);
            else
                featureCategories["KReg=1"].Add(i);

            if (cfg.KUnroll > 1)
                featureCategories["KUnroll>1"].Add(i);
            else
                featureCategories["KUnroll=1"].Add(i);

            if (cfg.VectorWidthM > 1)
                featureCategories["VecWidthM>1"].Add(i);
            else
                featureCategories["VecWidthM=1"].Add(i);

            if (cfg.VectorWidthN > 1)
                featureCategories["VecWidthN>1"].Add(i);
            else
                featureCategories["VecWidthN=1"].Add(i);

            if (cfg.StrideM)
                featureCategories["StrideM"].Add(i);
            if (cfg.StrideN)
                featureCategories["StrideN"].Add(i);
        }

        void AddSample(List<int> indices)
        {
            if (indices.Count == 0)
                return;

            int idx = indices[random.Next(indices.Count)];
            if (used.Add(idx))
                result.Add(idx);
        }

        foreach (var indices in baseCategories.Values)
            AddSample(indices);
        foreach (var indices in featureCategories.Values)
            AddSample(indices);

        Shuffle(result, random);

        LogDiag($"  Category distribution: TrueVec={baseCategories["TrueVectorized"].Count}, " +
               $"Coop={baseCategories["CooperativeLoading"].Count}, " +
               $"Stride={baseCategories["StridedAccess"].Count}, " +
               $"SimpleVec={baseCategories["SimpleVectorized"].Count}, " +
               $"Basic={baseCategories["Basic"].Count}");
        LogDiag($"  Feature coverage: ACol={featureCategories["ColumnMajorA"].Count}, " +
               $"Subgroup={featureCategories["SubgroupOps"].Count}, " +
               $"DoubleBuf={featureCategories["DoubleBuffering"].Count}, " +
               $"KReg>1={featureCategories["KReg>1"].Count}, " +
               $"KUnroll>1={featureCategories["KUnroll>1"].Count}");

        return result;
    }

    private int? TrySelectCoverageCandidate(GemmConfig[] configs, HashSet<int> testedIndices, Random random)
    {
        var categories = new List<(string Name, Func<GemmConfig, bool> Predicate)>
        {
            ("TrueVectorized", cfg => cfg.UseTrueVectorLDS),
            ("NonTrueVectorized", cfg => !cfg.UseTrueVectorLDS),
            ("ColumnMajorA", cfg => cfg.UseColumnMajorA),
            ("RowMajorA", cfg => !cfg.UseColumnMajorA),
            ("SubgroupOps", cfg => cfg.UseSubgroupOps),
            ("DoubleBuffering", cfg => cfg.UseDoubleBuffering),
            ("KReg>1", cfg => cfg.KReg > 1),
            ("KUnroll>1", cfg => cfg.KUnroll > 1),
            ("StrideM", cfg => cfg.StrideM),
            ("StrideN", cfg => cfg.StrideN),
            ("CooperativeLoading", cfg => cfg.MdimaSize > 0 || cfg.NdimbSize > 0),
            ("VecWidthM>1", cfg => cfg.VectorWidthM > 1),
            ("VecWidthN>1", cfg => cfg.VectorWidthN > 1)
        };

        foreach (var category in categories)
        {
            bool covered = false;
            foreach (var idx in testedIndices)
            {
                if (category.Predicate(configs[idx]))
                {
                    covered = true;
                    break;
                }
            }

            if (covered)
                continue;

            var candidates = new List<int>();
            for (int i = 0; i < configs.Length; i++)
            {
                if (testedIndices.Contains(i))
                    continue;
                if (category.Predicate(configs[i]))
                    candidates.Add(i);
            }

            if (candidates.Count > 0)
                return candidates[random.Next(candidates.Count)];
        }

        return null;
    }

    private static void Shuffle<T>(IList<T> list, Random random)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    /// <summary>
    /// Gets candidate configurations for a given matrix size.
    /// </summary>
    private GemmConfig[] GetCandidateConfigs(int M, int N, int K, GpuCapabilities capabilities)
    {
        var candidates = new List<GemmConfig>();
        int maxDim = Math.Max(Math.Max(M, N), K);

        if (maxDim <= 64)
            candidates.AddRange(_smallConfigs);
        else if (maxDim <= 512)
            candidates.AddRange(_mediumConfigs);
        else
            candidates.AddRange(_largeConfigs);

        // Add mixed precision if supported
        if (capabilities.SupportsFP16)
        {
            candidates.Add(new GemmConfig
            {
                TileM = 64,
                TileN = 64,
                TileK = 32,
                ThreadTileM = 8,
                ThreadTileN = 8,
                UseDoubleBuffering = true,
                UseVectorizedLoads = true,
                KernelName = "gemm_mixed_precision"
            });
        }

        return candidates.ToArray();
    }

    /// <summary>
    /// Clears the tuning cache.
    /// </summary>
    public void ClearCache()
    {
        lock (_cacheLock)
        {
            _cache.Clear();
        }
    }

    /// <summary>
    /// Gets the number of cached configurations.
    /// </summary>
    public int CachedConfigCount
    {
        get
        {
            lock (_cacheLock)
            {
                return _cache.Count;
            }
        }
    }
}

/// <summary>
/// GPU capabilities for tuning decisions.
/// </summary>
public sealed class GpuCapabilities
{
    public int ComputeUnits { get; init; }
    public long GlobalMemoryBytes { get; init; }
    public int LocalMemoryBytes { get; init; }
    public int MaxWorkGroupSize { get; init; }
    public int WavefrontSize { get; init; }  // 32 for NVIDIA, 64 for AMD
    public bool SupportsFP16 { get; init; }
    public bool SupportsSubgroups { get; init; }
    public bool SupportsMFMA { get; init; }  // AMD Matrix Cores
    public string VendorName { get; init; } = "";
    public string DeviceName { get; init; } = "";

    /// <summary>
    /// Detects GPU capabilities from OpenCL device info.
    /// </summary>
    public static GpuCapabilities Detect(
        int computeUnits,
        long globalMemory,
        int localMemory,
        int maxWorkGroupSize,
        string vendor,
        string device,
        string extensions)
    {
        bool isAmd = vendor.Contains("AMD", StringComparison.OrdinalIgnoreCase) ||
                     vendor.Contains("Advanced Micro Devices", StringComparison.OrdinalIgnoreCase);
        bool isNvidia = vendor.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase);

        return new GpuCapabilities
        {
            ComputeUnits = computeUnits,
            GlobalMemoryBytes = globalMemory,
            LocalMemoryBytes = localMemory,
            MaxWorkGroupSize = maxWorkGroupSize,
            WavefrontSize = isAmd ? 64 : (isNvidia ? 32 : 32),
            SupportsFP16 = extensions.Contains("cl_khr_fp16"),
            SupportsSubgroups = extensions.Contains("cl_khr_subgroups") ||
                               extensions.Contains("cl_intel_subgroups"),
            SupportsMFMA = extensions.Contains("cl_amd_mfma") ||
                          device.Contains("MI100") || device.Contains("MI200") ||
                          device.Contains("MI300") || device.Contains("gfx90"),
            VendorName = vendor,
            DeviceName = device
        };
    }

    /// <summary>
    /// Creates default capabilities for unknown GPU.
    /// </summary>
    public static GpuCapabilities CreateDefault()
    {
        return new GpuCapabilities
        {
            ComputeUnits = 32,
            GlobalMemoryBytes = 4L * 1024 * 1024 * 1024,
            LocalMemoryBytes = 64 * 1024,
            MaxWorkGroupSize = 256,
            WavefrontSize = 64,
            SupportsFP16 = false,
            SupportsSubgroups = false,
            SupportsMFMA = false,
            VendorName = "Unknown",
            DeviceName = "Unknown"
        };
    }

    /// <summary>
    /// Gets a diagnostic string summarizing GPU capabilities.
    /// </summary>
    public string GetDiagnosticString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"GPU: {DeviceName}");
        sb.AppendLine($"Vendor: {VendorName}");
        sb.AppendLine($"Compute Units: {ComputeUnits}");
        sb.AppendLine($"Global Memory: {GlobalMemoryBytes / (1024 * 1024)} MB");
        sb.AppendLine($"Local Memory: {LocalMemoryBytes / 1024} KB");
        sb.AppendLine($"Max Work Group Size: {MaxWorkGroupSize}");
        sb.AppendLine($"Wavefront Size: {WavefrontSize}");
        sb.AppendLine($"Features: FP16={SupportsFP16}, Subgroups={SupportsSubgroups}, MFMA={SupportsMFMA}");

        // Add performance expectations based on hardware
        double theoreticalGflops = ComputeUnits * WavefrontSize * 2.0 * 1.5; // Rough estimate: CU * wave * FMA * boost
        sb.AppendLine($"Theoretical Peak (estimate): {theoreticalGflops:F0} GFLOPS");

        return sb.ToString();
    }

    /// <summary>
    /// Gets a short summary for logging.
    /// </summary>
    public override string ToString()
    {
        return $"{DeviceName} ({ComputeUnits} CUs, {GlobalMemoryBytes / (1024 * 1024)} MB, FP16={SupportsFP16})";
    }
}

/// <summary>
/// Lightweight Bayesian tuner for GEMM kernel configuration selection.
/// Uses Gaussian Process regression with RBF kernel and Expected Improvement acquisition.
/// Inspired by AiDotNet.HyperparameterOptimization.BayesianOptimizer.
/// </summary>
internal sealed class GemmBayesianTuner
{
    private readonly Random _random;
    private readonly List<int> _observedIndices;
    private readonly List<double> _observedValues;
    private double[,]? _covarianceMatrix;
    private double[,]? _covarianceMatrixInverse;

    // GP hyperparameters
    private double _lengthScale = 2.0;
    private double _signalVariance = 1.0;
    private const double NoiseVariance = 0.01;

    public double BestObservedValue => _observedValues.Count > 0 ? _observedValues.Max() : 0;

    public GemmBayesianTuner(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _observedIndices = new List<int>();
        _observedValues = new List<double>();
    }

    public int SampleRandomIndex(int totalConfigs, HashSet<int> excluded)
    {
        int idx;
        int attempts = 0;
        do
        {
            idx = _random.Next(totalConfigs);
            attempts++;
        } while (excluded.Contains(idx) && attempts < totalConfigs * 2);

        // If we couldn't find one randomly, find first available
        if (excluded.Contains(idx))
        {
            for (int i = 0; i < totalConfigs; i++)
            {
                if (!excluded.Contains(i))
                    return i;
            }
        }

        return idx;
    }

    public void AddObservation(int configIndex, double gflops)
    {
        _observedIndices.Add(configIndex);
        _observedValues.Add(gflops);
    }

    public void UpdateModel()
    {
        if (_observedIndices.Count < 2)
            return;

        // Normalize observations for better GP behavior
        double mean = _observedValues.Average();
        double std = Math.Max(0.01, Math.Sqrt(_observedValues.Sum(v => (v - mean) * (v - mean)) / _observedValues.Count));

        // Update covariance matrix
        int n = _observedIndices.Count;
        _covarianceMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _covarianceMatrix[i, j] = RbfKernel(_observedIndices[i], _observedIndices[j]);
                if (i == j)
                    _covarianceMatrix[i, j] += NoiseVariance;
            }
        }

        _covarianceMatrixInverse = InvertMatrixCholesky(_covarianceMatrix);

        // Optimize length scale periodically
        if (n % 5 == 0 && n >= 5)
        {
            OptimizeLengthScale();
        }
    }

    public int SelectNextPoint(int totalConfigs, HashSet<int> excluded)
    {
        if (_observedIndices.Count < 2 || _covarianceMatrixInverse == null)
        {
            return SampleRandomIndex(totalConfigs, excluded);
        }

        double bestAcquisition = double.NegativeInfinity;
        int bestIndex = -1;

        // Evaluate acquisition function at all untested points
        for (int idx = 0; idx < totalConfigs; idx++)
        {
            if (excluded.Contains(idx))
                continue;

            double acquisition = ComputeExpectedImprovement(idx);
            if (acquisition > bestAcquisition)
            {
                bestAcquisition = acquisition;
                bestIndex = idx;
            }
        }

        return bestIndex >= 0 ? bestIndex : SampleRandomIndex(totalConfigs, excluded);
    }

    private double ComputeExpectedImprovement(int candidateIdx)
    {
        var (mean, variance) = PredictGP(candidateIdx);
        double std = Math.Sqrt(Math.Max(0, variance) + 1e-9);

        if (std < 1e-9)
            return 0;

        double bestValue = BestObservedValue;
        double z = (mean - bestValue) / std;

        // EI = std * (z * Phi(z) + phi(z))
        double phi = NormalPdf(z);
        double Phi = NormalCdf(z);

        return std * (z * Phi + phi);
    }

    private (double mean, double variance) PredictGP(int candidateIdx)
    {
        if (_observedIndices.Count == 0 || _covarianceMatrixInverse == null)
        {
            return (0.0, _signalVariance);
        }

        int n = _observedIndices.Count;
        var kStar = new double[n];

        for (int i = 0; i < n; i++)
        {
            kStar[i] = RbfKernel(candidateIdx, _observedIndices[i]);
        }

        // Mean prediction: k* @ K^-1 @ y
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * _observedValues[j];
            }
            mean += kStar[i] * sum;
        }

        // Variance prediction: k** - k* @ K^-1 @ k*^T
        double kStarStar = RbfKernel(candidateIdx, candidateIdx);
        double variance = kStarStar;

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * kStar[j];
            }
            variance -= kStar[i] * sum;
        }

        return (mean, Math.Max(0, variance));
    }

    private double RbfKernel(int idx1, int idx2)
    {
        // For discrete indices, use distance in index space
        double diff = idx1 - idx2;
        return _signalVariance * Math.Exp(-diff * diff / (2 * _lengthScale * _lengthScale));
    }

    private void OptimizeLengthScale()
    {
        double bestLs = _lengthScale;
        double bestLl = double.NegativeInfinity;

        foreach (double ls in new[] { 1.0, 2.0, 5.0, 10.0, 20.0 })
        {
            _lengthScale = ls;
            UpdateCovarianceForLengthScale();
            double ll = ComputeLogMarginalLikelihood();

            if (ll > bestLl)
            {
                bestLl = ll;
                bestLs = ls;
            }
        }

        _lengthScale = bestLs;
        UpdateCovarianceForLengthScale();
    }

    private void UpdateCovarianceForLengthScale()
    {
        if (_observedIndices.Count < 2)
            return;

        int n = _observedIndices.Count;
        _covarianceMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _covarianceMatrix[i, j] = RbfKernel(_observedIndices[i], _observedIndices[j]);
                if (i == j)
                    _covarianceMatrix[i, j] += NoiseVariance;
            }
        }

        _covarianceMatrixInverse = InvertMatrixCholesky(_covarianceMatrix);
    }

    private double ComputeLogMarginalLikelihood()
    {
        if (_covarianceMatrix == null || _covarianceMatrixInverse == null)
            return double.NegativeInfinity;

        int n = _observedIndices.Count;
        double[] y = _observedValues.ToArray();

        double dataFit = 0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * y[j];
            }
            dataFit += y[i] * sum;
        }

        double logDet = LogDeterminant(_covarianceMatrix);

        return -0.5 * (dataFit + logDet + n * Math.Log(2 * Math.PI));
    }

    #region Math Helpers

    private static double NormalPdf(double x)
    {
        return Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
    }

    private static double NormalCdf(double x)
    {
        return 0.5 * (1 + Erf(x / Math.Sqrt(2)));
    }

    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }

    private static double[,] InvertMatrixCholesky(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var L = new double[n, n];
        var inverse = new double[n, n];

        // Cholesky decomposition: A = L * L^T
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double diag = matrix[i, i] - sum;
                    L[i, j] = Math.Sqrt(Math.Max(1e-10, diag));
                }
                else
                {
                    L[i, j] = (matrix[i, j] - sum) / L[j, j];
                }
            }
        }

        // Invert L
        var Linv = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            Linv[i, i] = 1.0 / L[i, i];
            for (int j = 0; j < i; j++)
            {
                double sum = 0;
                for (int k = j; k < i; k++)
                {
                    sum += L[i, k] * Linv[k, j];
                }
                Linv[i, j] = -sum / L[i, i];
            }
        }

        // A^-1 = L^-T * L^-1
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = i; k < n; k++)
                {
                    sum += Linv[k, i] * Linv[k, j];
                }
                inverse[i, j] = sum;
                inverse[j, i] = sum;
            }
        }

        return inverse;
    }

    private static double LogDeterminant(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var L = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double diag = matrix[i, i] - sum;
                    L[i, j] = Math.Sqrt(Math.Max(1e-10, diag));
                }
                else
                {
                    L[i, j] = (matrix[i, j] - sum) / L[j, j];
                }
            }
        }

        double logDet = 0;
        for (int i = 0; i < n; i++)
        {
            logDet += Math.Log(Math.Max(1e-10, L[i, i]));
        }

        return 2 * logDet;
    }

    #endregion
}
