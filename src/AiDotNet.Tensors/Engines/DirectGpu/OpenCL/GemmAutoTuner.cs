// Copyright (c) AiDotNet. All rights reserved.
// Auto-tuning framework for GEMM kernel selection and parameter optimization.
// Implements Bayesian optimization for efficient kernel parameter search.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

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
    public bool UseDoubleBuffering { get; init; }
    public bool UseVectorizedLoads { get; init; }
    public string KernelName { get; init; }

    public override string ToString() =>
        $"{KernelName}[{TileM}x{TileN}x{TileK}, TT:{ThreadTileM}x{ThreadTileN}, DB:{UseDoubleBuffering}, VL:{UseVectorizedLoads}]";
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
        var bayesian = new GemmBayesianTuner(seed);
        var allConfigs = GenerateConfigurationSpace(M, N, K, capabilities);

        if (allConfigs.Length <= initialRandomSamples)
        {
            // Configuration space is small, just do exhaustive search
            return TuneForDimensions(M, N, K, capabilities, benchmarkFunc, warmupRuns, benchmarkRuns);
        }

        var results = new List<TuningResult>();
        var testedIndices = new HashSet<int>();

        // Phase 1: Initial random sampling
        for (int i = 0; i < initialRandomSamples && i < allConfigs.Length; i++)
        {
            int idx = bayesian.SampleRandomIndex(allConfigs.Length, testedIndices);
            testedIndices.Add(idx);

            var config = allConfigs[idx];
            var result = BenchmarkConfig(config, benchmarkFunc, warmupRuns, benchmarkRuns, M, N, K);
            results.Add(result);

            bayesian.AddObservation(idx, result.GFlops);
        }

        // Phase 2: Bayesian optimization
        for (int trial = initialRandomSamples; trial < maxTrials && testedIndices.Count < allConfigs.Length; trial++)
        {
            // Update GP model
            bayesian.UpdateModel();

            // Find next point using Expected Improvement
            int nextIdx = bayesian.SelectNextPoint(allConfigs.Length, testedIndices);
            testedIndices.Add(nextIdx);

            var config = allConfigs[nextIdx];
            var result = BenchmarkConfig(config, benchmarkFunc, warmupRuns, benchmarkRuns, M, N, K);
            results.Add(result);

            bayesian.AddObservation(nextIdx, result.GFlops);

            // Early stopping if we found a very good config
            if (result.GFlops > bayesian.BestObservedValue * 0.99)
            {
                // We're within 1% of best, likely near optimal
                break;
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

    private TuningResult BenchmarkConfig(GemmConfig config, Func<GemmConfig, double> benchmarkFunc,
        int warmupRuns, int benchmarkRuns, int M, int N, int K)
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

            return new TuningResult
            {
                Config = config,
                GFlops = gflops,
                TimeMs = avgTimeMs,
                IsValid = true
            };
        }
        catch
        {
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
    /// </summary>
    private GemmConfig[] GenerateConfigurationSpace(int M, int N, int K, GpuCapabilities capabilities)
    {
        var configs = new List<GemmConfig>();

        // Tile sizes to try
        int[] tileSizes = { 16, 32, 64, 128 };
        int[] tileSizesK = { 8, 16, 32 };
        int[] threadTiles = { 2, 4, 8 };

        foreach (int tileM in tileSizes)
        {
            foreach (int tileN in tileSizes)
            {
                foreach (int tileK in tileSizesK)
                {
                    foreach (int ttM in threadTiles)
                    {
                        foreach (int ttN in threadTiles)
                        {
                            // Skip invalid configurations
                            if (tileM < ttM * 4 || tileN < ttN * 4) continue;
                            if (tileM > M * 2 || tileN > N * 2) continue;

                            // Add base config
                            configs.Add(new GemmConfig
                            {
                                TileM = tileM,
                                TileN = tileN,
                                TileK = tileK,
                                ThreadTileM = ttM,
                                ThreadTileN = ttN,
                                UseDoubleBuffering = tileM >= 64,
                                UseVectorizedLoads = tileK >= 16,
                                KernelName = tileM >= 64 ? "gemm_double_buffered" : "gemm_small"
                            });
                        }
                    }
                }
            }
        }

        // Add mixed precision configs if supported
        if (capabilities.SupportsFP16)
        {
            foreach (int tileM in new[] { 64, 128 })
            {
                foreach (int tileN in new[] { 64, 128 })
                {
                    configs.Add(new GemmConfig
                    {
                        TileM = tileM,
                        TileN = tileN,
                        TileK = 32,
                        ThreadTileM = 8,
                        ThreadTileN = 8,
                        UseDoubleBuffering = true,
                        UseVectorizedLoads = true,
                        KernelName = "gemm_mixed_precision"
                    });
                }
            }
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
