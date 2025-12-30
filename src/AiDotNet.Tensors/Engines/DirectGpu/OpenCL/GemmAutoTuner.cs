// Copyright (c) AiDotNet. All rights reserved.
// Auto-tuning framework for GEMM kernel selection and parameter optimization.
// Implements Bayesian optimization for efficient kernel parameter search.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

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

    /// <summary>
    /// Generates a unique cache key for this configuration.
    /// Used by DynamicGemmKernel to cache compiled kernels.
    /// </summary>
    public string ToKey() =>
        $"{TileM}_{TileN}_{TileK}_{ThreadTileM}_{ThreadTileN}_{VectorWidthM}_{VectorWidthN}_{UseDoubleBuffering}_{UseVectorizedLoads}_{KReg}_{KUnroll}_{UseSubgroupOps}_{StrideM}_{StrideN}_{CacheA}_{CacheB}_{MdimaSize}_{NdimbSize}";

    public override string ToString() =>
        $"{KernelName}[{TileM}x{TileN}x{TileK}, TT:{ThreadTileM}x{ThreadTileN}, VW:{VectorWidthM}x{VectorWidthN}, K:{KReg}x{KUnroll}, SG:{UseSubgroupOps}, SA/B:{(CacheA ? 1 : 0)}/{(CacheB ? 1 : 0)}, MD:{MdimaSize}x{NdimbSize}]";
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
    /// Show progress every N trials (default: 10).
    /// </summary>
    public static int ProgressInterval { get; set; } = 10;

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
    /// Logs a diagnostic message if diagnostics are enabled.
    /// Uses same log file as DynamicGemmKernel if configured.
    /// </summary>
    private static void LogDiag(string message)
    {
        if (!EnableDiagnostics)
            return;

        string logLine = $"[{DateTime.Now:HH:mm:ss.fff}] [GemmTuner] {message}";

        if (!string.IsNullOrEmpty(LogFilePath))
        {
            // Use DynamicGemmKernel's logging mechanism
            DynamicGemmKernel.EnableDiagnostics = true;
            try
            {
                using var sw = new System.IO.StreamWriter(LogFilePath, append: true);
                sw.WriteLine(logLine);
            }
            catch
            {
                Console.WriteLine(logLine);
            }
        }
        else
        {
            Console.WriteLine(logLine);
        }
    }

    // Predefined configurations for different matrix sizes (AMD-tuned)
    private static readonly GemmConfig[] _smallConfigs = new[]
    {
        new GemmConfig { TileM = 16, TileN = 16, TileK = 16, ThreadTileM = 2, ThreadTileN = 2, UseDoubleBuffering = false, UseVectorizedLoads = false, KernelName = "gemm_small" },
        new GemmConfig { TileM = 32, TileN = 32, TileK = 8, ThreadTileM = 2, ThreadTileN = 2, UseDoubleBuffering = false, UseVectorizedLoads = false, KernelName = "gemm_warp_small" },
    };

    private static readonly GemmConfig[] _mediumConfigs = new[]
    {
        new GemmConfig { TileM = 32, TileN = 32, TileK = 16, ThreadTileM = 4, ThreadTileN = 4, UseDoubleBuffering = true, UseVectorizedLoads = false, KernelName = "gemm_double_buffered" },
        new GemmConfig { TileM = 64, TileN = 64, TileK = 16, ThreadTileM = 4, ThreadTileN = 4, UseDoubleBuffering = true, UseVectorizedLoads = true, KernelName = "gemm_double_buffered" },
        new GemmConfig { TileM = 64, TileN = 32, TileK = 32, ThreadTileM = 4, ThreadTileN = 2, UseDoubleBuffering = true, UseVectorizedLoads = true, KernelName = "gemm_double_buffered" },
    };

    private static readonly GemmConfig[] _largeConfigs = new[]
    {
        new GemmConfig { TileM = 128, TileN = 64, TileK = 16, ThreadTileM = 8, ThreadTileN = 4, UseDoubleBuffering = true, UseVectorizedLoads = true, KernelName = "gemm_double_buffered" },
        new GemmConfig { TileM = 128, TileN = 128, TileK = 16, ThreadTileM = 8, ThreadTileN = 8, UseDoubleBuffering = true, UseVectorizedLoads = true, KernelName = "gemm_double_buffered" },
        new GemmConfig { TileM = 64, TileN = 64, TileK = 32, ThreadTileM = 8, ThreadTileN = 8, UseDoubleBuffering = true, UseVectorizedLoads = true, KernelName = "gemm_mixed_precision" },
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
    /// Based on AMD GCN/RDNA architecture characteristics.
    /// </summary>
    private GemmConfig SelectConfigHeuristic(int M, int N, int K, GpuCapabilities capabilities)
    {
        long totalOps = 2L * M * N * K;  // FLOPs for GEMM
        int maxDim = Math.Max(Math.Max(M, N), K);

        // Small matrices: overhead-sensitive
        if (maxDim <= 64 || totalOps < 100_000)
        {
            // Very small: use warp-level kernel
            if (maxDim <= 32)
                return _smallConfigs[1];  // gemm_warp_small

            return _smallConfigs[0];  // gemm_small
        }

        // Medium matrices: balance between occupancy and register usage
        if (maxDim <= 512 || totalOps < 10_000_000)
        {
            // Prefer smaller tiles for better occupancy on mid-range GPUs
            if (capabilities.ComputeUnits < 40)
                return _mediumConfigs[0];

            // More CUs can handle larger tiles
            return _mediumConfigs[1];
        }

        // Large matrices: maximize throughput
        // Check if mixed precision is available and beneficial
        if (capabilities.SupportsFP16 && K >= 256)
        {
            return _largeConfigs[2];  // gemm_mixed_precision
        }

        // Select based on matrix shape
        if (M > N * 2)
        {
            // Tall matrix: prefer M-heavy tiles
            return _largeConfigs[0];  // 128x64
        }
        else if (N > M * 2)
        {
            // Wide matrix: prefer N-heavy tiles (transpose of above)
            return new GemmConfig
            {
                TileM = 64,
                TileN = 128,
                TileK = 16,
                ThreadTileM = 4,
                ThreadTileN = 8,
                UseDoubleBuffering = true,
                UseVectorizedLoads = true,
                KernelName = "gemm_double_buffered"
            };
        }

        // Square-ish: use balanced tiles
        return _largeConfigs[1];  // 128x128
    }

    /// <summary>
    /// Runs actual benchmark to find optimal configuration using exhaustive search.
    /// Use this for production workloads that will run many times.
    /// </summary>
    public TuningResult[] TuneForDimensions(int M, int N, int K, GpuCapabilities capabilities,
        Func<GemmConfig, double> benchmarkFunc, int warmupRuns = 2, int benchmarkRuns = 5)
    {
        var candidates = GetCandidateConfigs(M, N, K, capabilities);
        var results = new List<TuningResult>();

        foreach (var config in candidates)
        {
            try
            {
                // Warmup
                for (int i = 0; i < warmupRuns; i++)
                    benchmarkFunc(config);

                // Benchmark
                double totalTimeMs = 0;
                for (int i = 0; i < benchmarkRuns; i++)
                    totalTimeMs += benchmarkFunc(config);

                double avgTimeMs = totalTimeMs / benchmarkRuns;
                double gflops = (2.0 * M * N * K) / (avgTimeMs * 1e6);

                results.Add(new TuningResult
                {
                    Config = config,
                    GFlops = gflops,
                    TimeMs = avgTimeMs,
                    IsValid = true
                });
            }
            catch
            {
                results.Add(new TuningResult
                {
                    Config = config,
                    GFlops = 0,
                    TimeMs = double.MaxValue,
                    IsValid = false
                });
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

        // Phase 1: Initial random sampling
        LogDiag($"\n--- Phase 1: Random Exploration ({initialRandomSamples} trials) ---");
        for (int i = 0; i < initialRandomSamples && i < allConfigs.Length; i++)
        {
            int idx = bayesian.SampleRandomIndex(allConfigs.Length, testedIndices);
            testedIndices.Add(idx);

            var config = allConfigs[idx];
            LogDiag($"Trial {i + 1}/{maxTrials}: {config.KernelName}");
            var result = BenchmarkConfig(config, benchmarkFunc, warmupRuns, benchmarkRuns, M, N, K);
            results.Add(result);

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
        }

        // Phase 2: Bayesian optimization
        LogDiag($"\n--- Phase 2: Bayesian Optimization ({maxTrials - initialRandomSamples} trials) ---");
        LogDiag($"Current best: {bestGflops:F2} GFLOPS - {bestConfig}");

        for (int trial = initialRandomSamples; trial < maxTrials && testedIndices.Count < allConfigs.Length; trial++)
        {
            // Update GP model
            bayesian.UpdateModel();

            // Find next point using Expected Improvement
            int nextIdx = bayesian.SelectNextPoint(allConfigs.Length, testedIndices);
            testedIndices.Add(nextIdx);

            var config = allConfigs[nextIdx];

            // Show progress at intervals
            if ((trial + 1) % ProgressInterval == 0 || trial == maxTrials - 1)
            {
                double elapsed = tuningStopwatch.Elapsed.TotalSeconds;
                double trialsPerSec = (trial + 1) / elapsed;
                double eta = (maxTrials - trial - 1) / trialsPerSec;
                LogDiag($"Trial {trial + 1}/{maxTrials}: Best={bestGflops:F2} GFLOPS, Failed={failedTrials}, ETA={eta:F1}s");
            }

            var result = BenchmarkConfig(config, benchmarkFunc, warmupRuns, benchmarkRuns, M, N, K);
            results.Add(result);

            if (result.IsValid)
            {
                bayesian.AddObservation(nextIdx, result.GFlops);
                if (result.GFlops > bestGflops)
                {
                    bestGflops = result.GFlops;
                    bestConfig = config.ToString();
                    LogDiag($"  NEW BEST at trial {trial + 1}: {bestGflops:F2} GFLOPS - {config.KernelName}");
                }
            }
            else
            {
                failedTrials++;
                bayesian.AddObservation(nextIdx, 0);
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

    private TuningResult BenchmarkConfig(GemmConfig config, Func<GemmConfig, double> benchmarkFunc,
        int warmupRuns, int benchmarkRuns, int M, int N, int K)
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
                IsValid = false
            };
        }

        var sw = Stopwatch.StartNew();
        try
        {
            // Warmup
            for (int i = 0; i < warmupRuns; i++)
                benchmarkFunc(config);

            // Benchmark
            double totalTimeMs = 0;
            double minTime = double.MaxValue;
            double maxTime = 0;
            for (int i = 0; i < benchmarkRuns; i++)
            {
                double time = benchmarkFunc(config);
                totalTimeMs += time;
                minTime = Math.Min(minTime, time);
                maxTime = Math.Max(maxTime, time);
            }

            double avgTimeMs = totalTimeMs / benchmarkRuns;
            double gflops = (2.0 * M * N * K) / (avgTimeMs * 1e6);
            sw.Stop();

            LogDiag($"  {config.KernelName}: {gflops:F2} GFLOPS (avg: {avgTimeMs:F3} ms, min: {minTime:F3}, max: {maxTime:F3})");

            return new TuningResult
            {
                Config = config,
                GFlops = gflops,
                TimeMs = avgTimeMs,
                IsValid = true
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
                IsValid = false
            };
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
        // CLBlast-optimal configurations for AMD RDNA1/RDNA2 GPUs
        // These are proven high-performance configurations from CLBlast
        // EXACT parameters from CLBlast's tuned database for gfx10xx
        // ============================================================

        // ============================================================
        // CLBlast EXACT CONFIGS - These are the actual tuned values!
        // From: https://github.com/CNugteren/CLBlast/blob/master/src/database/kernels/xgemm/
        // ============================================================

        // CLBlast RX 5700 XT exact: MWG=128, NWG=128, KWG=32, MDIMC=8, NDIMC=16, VWM=8, VWN=8
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 16,  // MDIMC=8, NDIMC=16
            VectorWidthM = 8, VectorWidthN = 8,  // VWM=8, VWN=8 - KEY TO 2500 GFLOPS!
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            StrideM = true, StrideN = true,  // Bank conflict avoidance
            CacheA = true, CacheB = true,     // SA=1, SB=1 local memory caching
            MdimaSize = 8, NdimbSize = 16,    // MDIMA=8, NDIMB=16
            KernelName = "clblast_rx5700xt_exact"
        });

        // Same config without stride for comparison
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 16,
            VectorWidthM = 8, VectorWidthN = 8,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            StrideM = false, StrideN = false,
            CacheA = true, CacheB = true,     // SA=1, SB=1 local memory caching
            MdimaSize = 8, NdimbSize = 16,
            KernelName = "clblast_rx5700xt_nostride"
        });

        // CLBlast RX 5700 variant: MWG=128, NWG=128, KWG=16, MDIMC=16, NDIMC=8, VWM=8, VWN=4
        configs.Add(new GemmConfig
        {
            TileM = 128, TileN = 128, TileK = 16,
            ThreadTileM = 16, ThreadTileN = 8,
            VectorWidthM = 8, VectorWidthN = 4,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 8,
            KernelName = "clblast_rx5700_variant"
        });

        // CLBlast gfx10 default: MWG=64, NWG=64, KWG=32, MDIMC=16, NDIMC=16, VWM=2, VWN=4
        configs.Add(new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 16, ThreadTileN = 16,
            VectorWidthM = 2, VectorWidthN = 4,
            UseDoubleBuffering = true, UseVectorizedLoads = true,
            KReg = 2, KUnroll = 4, UseSubgroupOps = capabilities.SupportsSubgroups,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 16, NdimbSize = 16,
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
                                                        MdimaSize = ttM,  // MDIMA = ThreadTileM
                                                        NdimbSize = ttN,  // NDIMB = ThreadTileN
                                                        KernelName = $"gemm_{tileM}x{tileN}x{tileK}_v{vwm}x{vwn}_k{kreg}x{kwi}"
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
