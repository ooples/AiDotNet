// Copyright (c) 2024 AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Profiling;

/// <summary>
/// Roofline model analysis for GPU performance characterization.
/// </summary>
/// <remarks>
/// The roofline model bounds achievable performance by two limits:
/// 1. Compute roof: Peak GFLOPS of the hardware
/// 2. Memory roof: Peak bandwidth * arithmetic intensity
///
/// For GEMM: AI = 2*M*N*K / (4*(M*K + K*N + M*N)) for float32
/// Ridge point = Peak GFLOPS / Peak Bandwidth (FLOPS/byte)
/// </remarks>
public sealed class RooflineAnalyzer
{
    /// <summary>Peak compute performance in GFLOPS.</summary>
    public double PeakGflops { get; }

    /// <summary>Peak memory bandwidth in GB/s.</summary>
    public double PeakBandwidthGBs { get; }

    /// <summary>Ridge point where compute roof meets memory roof.</summary>
    public double RidgePoint { get; }

    /// <summary>
    /// Creates a roofline analyzer with the given hardware limits.
    /// </summary>
    /// <param name="peakGflops">Peak compute performance in GFLOPS.</param>
    /// <param name="peakBandwidthGBs">Peak memory bandwidth in GB/s.</param>
    public RooflineAnalyzer(double peakGflops, double peakBandwidthGBs)
    {
        PeakGflops = peakGflops;
        PeakBandwidthGBs = peakBandwidthGBs;
        RidgePoint = peakGflops / peakBandwidthGBs;
    }

    /// <summary>
    /// Creates a roofline analyzer from GPU architecture specifications.
    /// </summary>
    /// <param name="arch">GPU architecture spec.</param>
    /// <param name="numCus">Number of compute units on the GPU.</param>
    /// <param name="clockMhz">GPU clock speed in MHz.</param>
    /// <param name="memoryBandwidthGBs">Memory bandwidth in GB/s (from device query).</param>
    /// <returns>Configured roofline analyzer.</returns>
    public static RooflineAnalyzer FromArchitecture(
        GpuArchitectureSpec arch,
        int numCus,
        int clockMhz,
        double memoryBandwidthGBs)
    {
        // Peak GFLOPS = CUs * SIMDs/CU * wavefront_size * 2 (FMA) * clock_GHz
        // For RDNA1 (Wave32): CUs * 2 * 32 * 2 * clock_GHz = CUs * 128 * clock_GHz
        double clockGhz = clockMhz / 1000.0;
        double fmaOpsPerClock = arch.SimdsPerCu * arch.WavefrontSize * 2; // 2 for FMA
        double peakGflops = numCus * fmaOpsPerClock * clockGhz;

        return new RooflineAnalyzer(peakGflops, memoryBandwidthGBs);
    }

    /// <summary>
    /// Calculates the arithmetic intensity for GEMM operation.
    /// </summary>
    /// <param name="m">M dimension.</param>
    /// <param name="n">N dimension.</param>
    /// <param name="k">K dimension.</param>
    /// <param name="bytesPerElement">Bytes per element (4 for float, 8 for double).</param>
    /// <returns>Arithmetic intensity in FLOPS per byte.</returns>
    public static double CalculateGemmArithmeticIntensity(int m, int n, int k, int bytesPerElement = 4)
    {
        // FLOPS = 2 * M * N * K (multiply-add per output element)
        long flops = 2L * m * n * k;

        // Bytes = (A: M*K + B: K*N + C: M*N) * bytesPerElement
        // Assuming alpha=1, beta=0 (common case, no C read)
        long bytes = (long)(m * k + k * n + m * n) * bytesPerElement;

        return (double)flops / bytes;
    }

    /// <summary>
    /// Calculates theoretical arithmetic intensity accounting for tiling.
    /// </summary>
    /// <param name="m">M dimension.</param>
    /// <param name="n">N dimension.</param>
    /// <param name="k">K dimension.</param>
    /// <param name="mwg">M-dimension tile size.</param>
    /// <param name="nwg">N-dimension tile size.</param>
    /// <param name="kwg">K-dimension tile size.</param>
    /// <param name="bytesPerElement">Bytes per element.</param>
    /// <returns>Effective arithmetic intensity with tiling.</returns>
    public static double CalculateTiledGemmArithmeticIntensity(
        int m, int n, int k,
        int mwg, int nwg, int kwg,
        int bytesPerElement = 4)
    {
        // With tiling, each tile of A (MWG x KWG) is reused NWG/KWG times
        // and each tile of B (KWG x NWG) is reused MWG/KWG times

        long flops = 2L * m * n * k;

        // Effective memory traffic with LDS tiling
        // A: (M/MWG) * (K/KWG) * MWG * KWG * bytes = M * K * bytes (no reduction)
        // But with register tiling, each LDS load feeds multiple FMAs
        // Effective reuse factor = tile size / wavefront width

        int tilesM = (m + mwg - 1) / mwg;
        int tilesN = (n + nwg - 1) / nwg;
        int tilesK = (k + kwg - 1) / kwg;

        // Each workgroup loads: MWG*KWG (A) + KWG*NWG (B) per K-tile
        // Total loads: tilesM * tilesN * tilesK * (MWG*KWG + KWG*NWG) * bytes
        long bytesLoaded = (long)tilesM * tilesN * tilesK *
                           (mwg * kwg + kwg * nwg) * bytesPerElement;

        // Output writes: M * N * bytes
        bytesLoaded += (long)m * n * bytesPerElement;

        return (double)flops / bytesLoaded;
    }

    /// <summary>
    /// Gets the roofline-limited performance for a given arithmetic intensity.
    /// </summary>
    /// <param name="arithmeticIntensity">FLOPS per byte.</param>
    /// <returns>Maximum achievable GFLOPS.</returns>
    public double GetRooflineLimitGflops(double arithmeticIntensity)
    {
        // Performance = min(compute_roof, memory_roof * AI)
        double memoryBound = PeakBandwidthGBs * arithmeticIntensity;
        return Math.Min(PeakGflops, memoryBound);
    }

    /// <summary>
    /// Determines if an operation is memory or compute bound.
    /// </summary>
    /// <param name="arithmeticIntensity">FLOPS per byte.</param>
    /// <returns>True if memory bound, false if compute bound.</returns>
    public bool IsMemoryBound(double arithmeticIntensity)
    {
        return arithmeticIntensity < RidgePoint;
    }

    /// <summary>
    /// Calculates the efficiency relative to the roofline limit.
    /// </summary>
    /// <param name="achievedGflops">Actually achieved GFLOPS.</param>
    /// <param name="arithmeticIntensity">Arithmetic intensity of the operation.</param>
    /// <returns>Efficiency as a percentage (0-100).</returns>
    public double CalculateRooflineEfficiency(double achievedGflops, double arithmeticIntensity)
    {
        double limit = GetRooflineLimitGflops(arithmeticIntensity);
        return limit > 0 ? 100.0 * achievedGflops / limit : 0;
    }

    /// <summary>
    /// Analyzes GEMM performance against the roofline model.
    /// </summary>
    /// <param name="m">M dimension.</param>
    /// <param name="n">N dimension.</param>
    /// <param name="k">K dimension.</param>
    /// <param name="achievedGflops">Measured GFLOPS.</param>
    /// <param name="executionTimeUs">Kernel execution time in microseconds.</param>
    /// <param name="totalTimeUs">Total time including overhead.</param>
    /// <param name="launchOverheadUs">Typical launch overhead for this GPU.</param>
    /// <returns>Analysis result with bottleneck identification.</returns>
    public RooflineAnalysisResult AnalyzeGemm(
        int m, int n, int k,
        double achievedGflops,
        double executionTimeUs,
        double totalTimeUs,
        double launchOverheadUs = 300)
    {
        double ai = CalculateGemmArithmeticIntensity(m, n, k);
        double rooflineLimit = GetRooflineLimitGflops(ai);
        double rooflineEfficiency = CalculateRooflineEfficiency(achievedGflops, ai);

        // Calculate actual memory bandwidth achieved
        long bytes = (long)(m * k + k * n + m * n) * sizeof(float);
        double executionTimeS = executionTimeUs / 1_000_000.0;
        double achievedBandwidthGBs = executionTimeS > 0 ? bytes / (executionTimeS * 1e9) : 0;

        // Identify bottleneck
        BottleneckType bottleneck;
        OptimizationAction action;
        string explanation;

        // Check if launch overhead dominates
        double overheadRatio = totalTimeUs > 0 ? (totalTimeUs - executionTimeUs) / totalTimeUs : 0;
        double sizeThreshold = launchOverheadUs * PeakGflops * 1e-6 / 0.1; // Size where overhead = 10%

        long matrixOps = 2L * m * n * k;
        double effectiveSize = Math.Pow(matrixOps, 1.0 / 3.0);

        if (overheadRatio > 0.3 || effectiveSize < sizeThreshold)
        {
            bottleneck = BottleneckType.LaunchOverhead;
            action = effectiveSize < 256 ? OptimizationAction.UseCpu : OptimizationAction.BatchOperations;
            explanation = $"Launch overhead is {overheadRatio * 100:F0}% of total time. " +
                          $"Consider batching operations or using CPU for size {effectiveSize:F0}.";
        }
        else if (ai < RidgePoint && rooflineEfficiency > 70)
        {
            // Memory bound but achieving good roofline efficiency
            bottleneck = BottleneckType.MemoryBandwidth;
            action = OptimizationAction.OptimizeTiling;
            explanation = $"Memory bound (AI={ai:F2} < ridge={RidgePoint:F1}). " +
                          $"Achieving {achievedBandwidthGBs:F0}/{PeakBandwidthGBs:F0} GB/s " +
                          $"({100 * achievedBandwidthGBs / PeakBandwidthGBs:F0}% bandwidth utilization).";
        }
        else if (ai >= RidgePoint && rooflineEfficiency < 50)
        {
            // Should be compute bound but not achieving good performance
            bottleneck = BottleneckType.Occupancy;
            action = OptimizationAction.IncreaseOccupancy;
            explanation = $"Compute bound (AI={ai:F2}) but only {rooflineEfficiency:F0}% efficiency. " +
                          "Low occupancy may be hiding latency poorly.";
        }
        else if (rooflineEfficiency >= 70)
        {
            bottleneck = BottleneckType.Compute;
            action = OptimizationAction.None;
            explanation = $"Good compute efficiency ({rooflineEfficiency:F0}%). Near optimal for current configuration.";
        }
        else
        {
            bottleneck = BottleneckType.Unknown;
            action = OptimizationAction.AutoTune;
            explanation = $"Mixed bottleneck. AI={ai:F2}, Efficiency={rooflineEfficiency:F0}%. Try auto-tuning.";
        }

        return new RooflineAnalysisResult
        {
            ArithmeticIntensity = ai,
            RooflineLimitGflops = rooflineLimit,
            AchievedGflops = achievedGflops,
            RooflineEfficiency = rooflineEfficiency,
            AchievedBandwidthGBs = achievedBandwidthGBs,
            IsMemoryBound = ai < RidgePoint,
            Bottleneck = bottleneck,
            RecommendedAction = action,
            Explanation = explanation
        };
    }

    /// <summary>
    /// Generates roofline plot data points for visualization.
    /// </summary>
    /// <param name="minAI">Minimum arithmetic intensity.</param>
    /// <param name="maxAI">Maximum arithmetic intensity.</param>
    /// <param name="points">Number of data points.</param>
    /// <returns>Array of (AI, GFLOPS) pairs representing the roofline.</returns>
    public (double ArithmeticIntensity, double Gflops)[] GenerateRooflineCurve(
        double minAI = 0.1,
        double maxAI = 100,
        int points = 50)
    {
        var result = new (double, double)[points];
        double logMin = Math.Log10(minAI);
        double logMax = Math.Log10(maxAI);
        double step = (logMax - logMin) / (points - 1);

        for (int i = 0; i < points; i++)
        {
            double ai = Math.Pow(10, logMin + i * step);
            double gflops = GetRooflineLimitGflops(ai);
            result[i] = (ai, gflops);
        }

        return result;
    }
}

/// <summary>
/// Result of roofline analysis for a single operation.
/// </summary>
public sealed class RooflineAnalysisResult
{
    /// <summary>Arithmetic intensity in FLOPS per byte.</summary>
    public double ArithmeticIntensity { get; init; }

    /// <summary>Maximum achievable GFLOPS according to roofline model.</summary>
    public double RooflineLimitGflops { get; init; }

    /// <summary>Actually achieved GFLOPS.</summary>
    public double AchievedGflops { get; init; }

    /// <summary>Efficiency relative to roofline limit (percentage).</summary>
    public double RooflineEfficiency { get; init; }

    /// <summary>Achieved memory bandwidth in GB/s.</summary>
    public double AchievedBandwidthGBs { get; init; }

    /// <summary>Whether the operation is memory bound.</summary>
    public bool IsMemoryBound { get; init; }

    /// <summary>Identified bottleneck.</summary>
    public BottleneckType Bottleneck { get; init; }

    /// <summary>Recommended optimization action.</summary>
    public OptimizationAction RecommendedAction { get; init; }

    /// <summary>Human-readable explanation of the analysis.</summary>
    public string Explanation { get; init; } = string.Empty;
}
