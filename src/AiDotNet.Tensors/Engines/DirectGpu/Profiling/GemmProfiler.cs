// Copyright (c) 2024 AiDotNet. All rights reserved.

using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines.DirectGpu.Profiling;

/// <summary>
/// Configuration for GEMM profiling sessions.
/// </summary>
public sealed class ProfilerConfig
{
    /// <summary>Matrix sizes to profile (M=N=K for square matrices).</summary>
    public int[] Sizes { get; init; } = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096];

    /// <summary>Number of warmup runs before timing.</summary>
    public int WarmupRuns { get; init; } = 3;

    /// <summary>Number of benchmark runs for averaging.</summary>
    public int BenchmarkRuns { get; init; } = 5;

    /// <summary>Peak GPU GFLOPS (0 = auto-detect).</summary>
    public double PeakGflops { get; init; }

    /// <summary>Peak memory bandwidth in GB/s (0 = auto-detect).</summary>
    public double PeakBandwidthGBs { get; init; }

    /// <summary>Typical kernel launch overhead in microseconds.</summary>
    public int LaunchOverheadUs { get; init; } = 300;

    /// <summary>Whether to print progress to console.</summary>
    public bool Verbose { get; init; } = true;

    /// <summary>Whether to include occupancy analysis.</summary>
    public bool IncludeOccupancy { get; init; } = true;

    /// <summary>Output file path for JSON export (null = no export).</summary>
    public string? JsonOutputPath { get; init; }

    /// <summary>Output file path for CSV export (null = no export).</summary>
    public string? CsvOutputPath { get; init; }

    /// <summary>
    /// Default configuration for quick profiling.
    /// </summary>
    public static ProfilerConfig Default => new();

    /// <summary>
    /// Quick configuration with fewer sizes.
    /// </summary>
    public static ProfilerConfig Quick => new()
    {
        Sizes = [256, 512, 1024, 2048, 4096],
        WarmupRuns = 1,
        BenchmarkRuns = 3,
        Verbose = true
    };

    /// <summary>
    /// Comprehensive configuration with many sizes.
    /// </summary>
    public static ProfilerConfig Comprehensive => new()
    {
        Sizes = [64, 128, 192, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 8192],
        WarmupRuns = 3,
        BenchmarkRuns = 10,
        IncludeOccupancy = true,
        Verbose = true
    };
}

/// <summary>
/// Profiles GEMM operations across multiple sizes and generates comprehensive reports.
/// </summary>
public sealed class GemmProfiler
{
    private readonly OpenClBackend _backend;
    private readonly RooflineAnalyzer _roofline;
    private readonly GpuArchitectureSpec _arch;
    private readonly ProfilerConfig _config;

    /// <summary>
    /// Creates a GEMM profiler for the given backend.
    /// </summary>
    /// <param name="backend">OpenCL backend to profile.</param>
    /// <param name="config">Profiler configuration.</param>
    public GemmProfiler(OpenClBackend backend, ProfilerConfig? config = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _config = config ?? ProfilerConfig.Default;

        // Detect architecture from device name
        _arch = GpuArchitectureSpec.DetectFromDeviceName(backend.DeviceName);

        // Get peak performance values
        double peakGflops = _config.PeakGflops;
        double peakBandwidth = _config.PeakBandwidthGBs;

        // Auto-detect if not specified
        if (peakGflops <= 0)
        {
            // Estimate from CUs: CUs * 64 FP32 ALUs * 2 (FMA) * ~1.8 GHz
            peakGflops = backend.ComputeUnits * 64 * 2 * 1.8;

            // Check environment variable override
            var envPeak = Environment.GetEnvironmentVariable("AIDOTNET_GPU_PEAK_GFLOPS");
            if (int.TryParse(envPeak, out int envPeakInt) && envPeakInt > 0)
                peakGflops = envPeakInt;
        }

        if (peakBandwidth <= 0)
        {
            // Default estimate for GDDR6
            peakBandwidth = 224.0;

            // Check environment variable override
            var envBw = Environment.GetEnvironmentVariable("AIDOTNET_GPU_BANDWIDTH_GBS");
            if (int.TryParse(envBw, out int envBwInt) && envBwInt > 0)
                peakBandwidth = envBwInt;
        }

        _roofline = new RooflineAnalyzer(peakGflops, peakBandwidth);
    }

    /// <summary>
    /// Runs a full profiling session across all configured sizes.
    /// </summary>
    /// <returns>Complete profiling result.</returns>
    public ProfileResult RunFullProfile()
    {
        var startTime = DateTime.Now;
        var stopwatch = Stopwatch.StartNew();
        var entries = new List<GemmProfileEntry>();

        if (_config.Verbose)
        {
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("GEMM PROFILING SESSION");
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine($"Device: {_backend.DeviceName}");
            Console.WriteLine($"Architecture: {_arch.Name}");
            Console.WriteLine($"Peak: {_roofline.PeakGflops:F0} GFLOPS, {_roofline.PeakBandwidthGBs:F0} GB/s");
            Console.WriteLine($"Ridge Point: {_roofline.RidgePoint:F1} FLOPS/byte");
            Console.WriteLine($"Sizes: {string.Join(", ", _config.Sizes)}");
            Console.WriteLine("-".PadRight(80, '-'));
        }

        foreach (var size in _config.Sizes)
        {
            try
            {
                var entry = ProfileSize(size, size, size);
                entries.Add(entry);

                if (_config.Verbose)
                {
                    Console.WriteLine(entry.GetSummary());
                }
            }
            catch (Exception ex)
            {
                if (_config.Verbose)
                {
                    Console.WriteLine($"{size}x{size}x{size}: ERROR - {ex.Message}");
                }
            }
        }

        stopwatch.Stop();

        var result = new ProfileResult
        {
            DeviceName = _backend.DeviceName,
            Architecture = _arch,
            PeakGflops = _roofline.PeakGflops,
            PeakBandwidthGBs = _roofline.PeakBandwidthGBs,
            RidgePoint = _roofline.RidgePoint,
            ProfileStartTime = startTime,
            ProfileDurationSeconds = stopwatch.Elapsed.TotalSeconds,
            Entries = entries
        };

        if (_config.Verbose)
        {
            Console.WriteLine("-".PadRight(80, '-'));
            Console.WriteLine($"Profiling complete in {result.ProfileDurationSeconds:F1}s");
            Console.WriteLine($"Best: {result.BestGflops:F0} GFLOPS ({result.BestEfficiencyPercent:F1}% efficiency)");
            Console.WriteLine("=".PadRight(80, '='));
        }

        // Export if configured
        if (_config.JsonOutputPath is { Length: > 0 })
        {
            ProfileExporter.ToJsonFile(result, _config.JsonOutputPath);
            if (_config.Verbose)
                Console.WriteLine($"JSON exported to: {_config.JsonOutputPath}");
        }

        if (_config.CsvOutputPath is { Length: > 0 })
        {
            ProfileExporter.ToCsvFile(result, _config.CsvOutputPath);
            if (_config.Verbose)
                Console.WriteLine($"CSV exported to: {_config.CsvOutputPath}");
        }

        return result;
    }

    /// <summary>
    /// Profiles a single matrix size.
    /// </summary>
    public GemmProfileEntry ProfileSize(int m, int n, int k)
    {
        // Allocate matrices
        var random = new Random(42);
        var dataA = new float[m * k];
        var dataB = new float[k * n];
        for (int i = 0; i < dataA.Length; i++) dataA[i] = (float)(random.NextDouble() - 0.5);
        for (int i = 0; i < dataB.Length; i++) dataB[i] = (float)(random.NextDouble() - 0.5);

        var bufA = _backend.AllocateBuffer(dataA);
        var bufB = _backend.AllocateBuffer(dataB);
        var bufC = _backend.AllocateBuffer(m * n);

        try
        {
            // Warmup
            for (int i = 0; i < _config.WarmupRuns; i++)
            {
                _backend.Gemm(bufA, bufB, bufC, m, n, k, 1.0f, 0.0f);
                _backend.Synchronize();
            }

            // Benchmark with timing
            var totalStopwatch = Stopwatch.StartNew();
            var execStopwatch = new Stopwatch();

            for (int i = 0; i < _config.BenchmarkRuns; i++)
            {
                execStopwatch.Start();
                _backend.Gemm(bufA, bufB, bufC, m, n, k, 1.0f, 0.0f);
                _backend.Synchronize();
                execStopwatch.Stop();
            }
            totalStopwatch.Stop();

            // Use TotalMilliseconds * 1000 for net471 compatibility (TotalMicroseconds is .NET 7+)
            double avgExecTimeUs = (execStopwatch.Elapsed.TotalMilliseconds * 1000.0) / _config.BenchmarkRuns;
            double avgTotalTimeUs = (totalStopwatch.Elapsed.TotalMilliseconds * 1000.0) / _config.BenchmarkRuns;
            double queueTimeUs = avgTotalTimeUs - avgExecTimeUs;

            // Calculate metrics
            long flops = 2L * m * n * k;
            double execTimeS = avgExecTimeUs / 1_000_000.0;
            double gflops = execTimeS > 0 ? flops / (execTimeS * 1e9) : 0;

            // Memory calculations
            long bytes = (long)(m * k + k * n + m * n) * sizeof(float);
            double bwGBs = execTimeS > 0 ? bytes / (execTimeS * 1e9) : 0;
            double ai = RooflineAnalyzer.CalculateGemmArithmeticIntensity(m, n, k);

            // Roofline analysis
            var rooflineAnalysis = _roofline.AnalyzeGemm(m, n, k, gflops, avgExecTimeUs, avgTotalTimeUs, _config.LaunchOverheadUs);

            // Occupancy analysis (if enabled)
            OccupancyResult? occupancy = null;
            if (_config.IncludeOccupancy)
            {
                // Estimate based on typical CLBlast parameters
                int vgprs = OccupancyCalculator.EstimateGemmVgprs(64, 64, 16, 8, 8);
                int ldsBytes = OccupancyCalculator.EstimateGemmLds(64, 64, 16);
                int workgroupSize = 64; // Typical 8x8 workgroup

                occupancy = OccupancyCalculator.Calculate(_arch, vgprs, ldsBytes, workgroupSize);
            }

            return new GemmProfileEntry
            {
                M = m,
                N = n,
                K = k,
                QueueTimeUs = queueTimeUs > 0 ? queueTimeUs : 0,
                SubmitToStartUs = 0, // Would need OpenCL profiling events for accurate measurement
                ExecutionTimeUs = avgExecTimeUs,
                TotalTimeUs = avgTotalTimeUs,
                Gflops = gflops,
                EfficiencyPercent = 100.0 * gflops / _roofline.PeakGflops,
                MemoryBandwidthGBs = bwGBs,
                ArithmeticIntensity = ai,
                RooflineLimitGflops = rooflineAnalysis.RooflineLimitGflops,
                Bottleneck = rooflineAnalysis.Bottleneck,
                RecommendedAction = rooflineAnalysis.RecommendedAction,
                Occupancy = occupancy
            };
        }
        finally
        {
            bufA.Dispose();
            bufB.Dispose();
            bufC.Dispose();
        }
    }

    /// <summary>
    /// Profiles rectangular matrix sizes (M != N != K).
    /// </summary>
    public ProfileResult RunRectangularProfile(int[] mSizes, int[] nSizes, int[] kSizes)
    {
        var startTime = DateTime.Now;
        var stopwatch = Stopwatch.StartNew();
        var entries = new List<GemmProfileEntry>();

        if (_config.Verbose)
        {
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("RECTANGULAR GEMM PROFILING");
            Console.WriteLine("=".PadRight(80, '='));
        }

        foreach (var m in mSizes)
        {
            foreach (var n in nSizes)
            {
                foreach (var k in kSizes)
                {
                    try
                    {
                        var entry = ProfileSize(m, n, k);
                        entries.Add(entry);

                        if (_config.Verbose)
                        {
                            Console.WriteLine(entry.GetSummary());
                        }
                    }
                    catch (Exception ex)
                    {
                        if (_config.Verbose)
                        {
                            Console.WriteLine($"{m}x{n}x{k}: ERROR - {ex.Message}");
                        }
                    }
                }
            }
        }

        stopwatch.Stop();

        return new ProfileResult
        {
            DeviceName = _backend.DeviceName,
            Architecture = _arch,
            PeakGflops = _roofline.PeakGflops,
            PeakBandwidthGBs = _roofline.PeakBandwidthGBs,
            RidgePoint = _roofline.RidgePoint,
            ProfileStartTime = startTime,
            ProfileDurationSeconds = stopwatch.Elapsed.TotalSeconds,
            Entries = entries
        };
    }

    /// <summary>
    /// Finds the matrix size where GPU becomes faster than CPU.
    /// </summary>
    /// <param name="cpuGflops">CPU baseline GFLOPS for comparison.</param>
    /// <returns>Minimum size where GPU wins, or -1 if never.</returns>
    public int FindGpuCrossoverSize(double cpuGflops)
    {
        // Binary search for crossover point
        int minSize = 32;
        int maxSize = 1024;

        // First check if GPU ever wins at large sizes
        var largeEntry = ProfileSize(maxSize, maxSize, maxSize);
        if (largeEntry.Gflops <= cpuGflops)
            return -1; // GPU never wins

        // Binary search
        while (minSize < maxSize)
        {
            int midSize = (minSize + maxSize) / 2;
            var entry = ProfileSize(midSize, midSize, midSize);

            if (entry.Gflops > cpuGflops)
                maxSize = midSize;
            else
                minSize = midSize + 1;
        }

        return minSize;
    }

    /// <summary>
    /// Generates a compact report suitable for printing.
    /// </summary>
    public static string GenerateCompactReport(ProfileResult result)
    {
        return ProfileExporter.ToCompactSummary(result);
    }

    /// <summary>
    /// Gets the roofline analyzer for external use.
    /// </summary>
    public RooflineAnalyzer RooflineAnalyzer => _roofline;

    /// <summary>
    /// Gets the detected GPU architecture.
    /// </summary>
    public GpuArchitectureSpec Architecture => _arch;
}
