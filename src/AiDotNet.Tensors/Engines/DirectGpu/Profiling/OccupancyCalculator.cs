// Copyright (c) 2024 AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Profiling;

/// <summary>
/// Result of GPU occupancy calculation.
/// </summary>
public sealed class OccupancyResult
{
    /// <summary>Theoretical occupancy as a percentage (0-100).</summary>
    public double TheoreticalOccupancy { get; init; }

    /// <summary>Number of wavefronts that can run per SIMD.</summary>
    public int WavesPerSimd { get; init; }

    /// <summary>Maximum wavefronts per SIMD for this architecture.</summary>
    public int MaxWavesPerSimd { get; init; }

    /// <summary>The resource that limits occupancy.</summary>
    public OccupancyLimitingFactor LimitingFactor { get; init; }

    /// <summary>VGPR-limited wavefronts per SIMD.</summary>
    public int VgprLimitedWaves { get; init; }

    /// <summary>LDS-limited wavefronts per SIMD.</summary>
    public int LdsLimitedWaves { get; init; }

    /// <summary>SGPR-limited wavefronts per SIMD.</summary>
    public int SgprLimitedWaves { get; init; }

    /// <summary>Thread-limited wavefronts per SIMD.</summary>
    public int ThreadLimitedWaves { get; init; }

    /// <summary>VGPRs used per thread (input parameter).</summary>
    public int VgprsPerThread { get; init; }

    /// <summary>LDS bytes used per workgroup (input parameter).</summary>
    public int LdsBytesPerWorkgroup { get; init; }

    /// <summary>Workgroup size in threads (input parameter).</summary>
    public int WorkgroupSize { get; init; }

    /// <summary>
    /// Gets recommendations for improving occupancy.
    /// </summary>
    public List<string> GetRecommendations()
    {
        var recommendations = new List<string>();

        if (TheoreticalOccupancy >= 80)
        {
            recommendations.Add("Occupancy is good (>= 80%). Focus on other optimizations.");
            return recommendations;
        }

        switch (LimitingFactor)
        {
            case OccupancyLimitingFactor.Vgpr:
                recommendations.Add($"VGPR-limited: Using {VgprsPerThread} VGPRs per thread.");
                recommendations.Add("Consider: Reduce register usage with compiler flags.");
                recommendations.Add("Consider: Use smaller data types (half instead of float).");
                recommendations.Add("Consider: Increase loop unrolling to reduce live variables.");
                if (VgprsPerThread > 64)
                    recommendations.Add("Warning: High VGPR usage (>64) - check for register spilling.");
                break;

            case OccupancyLimitingFactor.Lds:
                recommendations.Add($"LDS-limited: Using {LdsBytesPerWorkgroup} bytes per workgroup.");
                recommendations.Add("Consider: Reduce tile sizes to use less LDS.");
                recommendations.Add("Consider: Use register tiling instead of LDS for some data.");
                recommendations.Add("Consider: Increase workgroup count, decrease LDS per group.");
                break;

            case OccupancyLimitingFactor.Sgpr:
                recommendations.Add($"SGPR-limited: This is unusual - check kernel for excess uniform values.");
                recommendations.Add("Consider: Move some scalar values to constant memory.");
                break;

            case OccupancyLimitingFactor.Threads:
                recommendations.Add($"Thread-limited: Workgroup size {WorkgroupSize} limits parallelism.");
                recommendations.Add("Consider: Increase workgroup size to better fill SIMDs.");
                if (WorkgroupSize < 64)
                    recommendations.Add("Warning: Workgroup size < 64 - cannot fill even one wavefront.");
                break;
        }

        if (TheoreticalOccupancy < 50)
        {
            recommendations.Add("Critical: Occupancy below 50% - significant performance impact.");
            recommendations.Add("This severely limits memory latency hiding capability.");
        }

        return recommendations;
    }

    public override string ToString()
    {
        return $"Occupancy: {TheoreticalOccupancy:F1}% ({WavesPerSimd}/{MaxWavesPerSimd} waves/SIMD) - Limited by: {LimitingFactor}";
    }
}

/// <summary>
/// The resource that limits GPU occupancy.
/// </summary>
public enum OccupancyLimitingFactor
{
    /// <summary>No limitation - maximum occupancy achieved.</summary>
    None,

    /// <summary>Vector General Purpose Registers limit occupancy.</summary>
    Vgpr,

    /// <summary>Local Data Share (shared memory) limits occupancy.</summary>
    Lds,

    /// <summary>Scalar General Purpose Registers limit occupancy.</summary>
    Sgpr,

    /// <summary>Thread/workgroup size limits occupancy.</summary>
    Threads
}

/// <summary>
/// Calculates GPU occupancy based on kernel resource usage.
/// </summary>
public static class OccupancyCalculator
{
    /// <summary>
    /// Calculates theoretical occupancy for a kernel on AMD GPUs.
    /// </summary>
    /// <param name="arch">GPU architecture specifications.</param>
    /// <param name="vgprsPerThread">Number of VGPRs used per thread.</param>
    /// <param name="ldsBytesPerWorkgroup">LDS bytes used per workgroup.</param>
    /// <param name="workgroupSize">Number of threads per workgroup.</param>
    /// <param name="sgprsPerThread">Number of SGPRs used per thread (optional, default 16).</param>
    /// <returns>Occupancy calculation result.</returns>
    public static OccupancyResult Calculate(
        GpuArchitectureSpec arch,
        int vgprsPerThread,
        int ldsBytesPerWorkgroup,
        int workgroupSize,
        int sgprsPerThread = 16)
    {
        // Calculate wavefronts per workgroup
        int wavesPerWorkgroup = (workgroupSize + arch.WavefrontSize - 1) / arch.WavefrontSize;

        // VGPR-limited waves per SIMD
        // VGPRs are allocated per wavefront: vgprsPerThread * wavefrontSize
        int vgprsPerWave = vgprsPerThread * arch.WavefrontSize;
        int vgprLimitedWaves = vgprsPerWave > 0 ? arch.VgprsPerSimd / vgprsPerWave : arch.MaxWavesPerSimd;
        vgprLimitedWaves = Math.Min(vgprLimitedWaves, arch.MaxWavesPerSimd);

        // LDS-limited waves per SIMD
        // LDS is shared per CU, so we calculate workgroups per CU first
        int maxWorkgroupsPerCu = ldsBytesPerWorkgroup > 0
            ? arch.LdsPerCuBytes / ldsBytesPerWorkgroup
            : int.MaxValue;

        // Convert to waves per SIMD (CU has multiple SIMDs)
        int maxWavesFromLds = maxWorkgroupsPerCu * wavesPerWorkgroup;
        int ldsLimitedWavesPerSimd = maxWavesFromLds / arch.SimdsPerCu;
        ldsLimitedWavesPerSimd = Math.Min(ldsLimitedWavesPerSimd, arch.MaxWavesPerSimd);
        if (ldsBytesPerWorkgroup == 0)
            ldsLimitedWavesPerSimd = arch.MaxWavesPerSimd;

        // SGPR-limited waves per SIMD
        int sgprsPerWave = sgprsPerThread; // SGPRs are per-wavefront, not per-thread
        int sgprLimitedWaves = sgprsPerWave > 0 ? arch.SgprsPerCu / (sgprsPerWave * arch.SimdsPerCu) : arch.MaxWavesPerSimd;
        sgprLimitedWaves = Math.Min(sgprLimitedWaves, arch.MaxWavesPerSimd);

        // Thread-limited waves per SIMD
        int maxThreadsPerCu = 2048;  // Typical limit
        int maxWavesFromThreads = maxThreadsPerCu / arch.WavefrontSize;
        int threadLimitedWavesPerSimd = maxWavesFromThreads / arch.SimdsPerCu;
        threadLimitedWavesPerSimd = Math.Min(threadLimitedWavesPerSimd, arch.MaxWavesPerSimd);

        // Determine the actual limit
        int wavesPerSimd = Math.Min(
            Math.Min(vgprLimitedWaves, ldsLimitedWavesPerSimd),
            Math.Min(sgprLimitedWaves, threadLimitedWavesPerSimd));

        // Determine limiting factor
        OccupancyLimitingFactor limitingFactor;
        if (wavesPerSimd == vgprLimitedWaves && vgprLimitedWaves < arch.MaxWavesPerSimd)
            limitingFactor = OccupancyLimitingFactor.Vgpr;
        else if (wavesPerSimd == ldsLimitedWavesPerSimd && ldsLimitedWavesPerSimd < arch.MaxWavesPerSimd)
            limitingFactor = OccupancyLimitingFactor.Lds;
        else if (wavesPerSimd == sgprLimitedWaves && sgprLimitedWaves < arch.MaxWavesPerSimd)
            limitingFactor = OccupancyLimitingFactor.Sgpr;
        else if (wavesPerSimd == threadLimitedWavesPerSimd && threadLimitedWavesPerSimd < arch.MaxWavesPerSimd)
            limitingFactor = OccupancyLimitingFactor.Threads;
        else
            limitingFactor = OccupancyLimitingFactor.None;

        double occupancy = 100.0 * wavesPerSimd / arch.MaxWavesPerSimd;

        return new OccupancyResult
        {
            TheoreticalOccupancy = occupancy,
            WavesPerSimd = wavesPerSimd,
            MaxWavesPerSimd = arch.MaxWavesPerSimd,
            LimitingFactor = limitingFactor,
            VgprLimitedWaves = vgprLimitedWaves,
            LdsLimitedWaves = ldsLimitedWavesPerSimd,
            SgprLimitedWaves = sgprLimitedWaves,
            ThreadLimitedWaves = threadLimitedWavesPerSimd,
            VgprsPerThread = vgprsPerThread,
            LdsBytesPerWorkgroup = ldsBytesPerWorkgroup,
            WorkgroupSize = workgroupSize
        };
    }

    /// <summary>
    /// Estimates VGPR usage for CLBlast XGEMM kernel based on tile parameters.
    /// </summary>
    /// <param name="mwg">M-dimension work-group tile size.</param>
    /// <param name="nwg">N-dimension work-group tile size.</param>
    /// <param name="kwg">K-dimension work-group tile size.</param>
    /// <param name="mdimc">M-dimension workgroup threads.</param>
    /// <param name="ndimc">N-dimension workgroup threads.</param>
    /// <returns>Estimated VGPRs per thread.</returns>
    public static int EstimateGemmVgprs(int mwg, int nwg, int kwg, int mdimc, int ndimc)
    {
        // Each thread computes a (MWG/MDIMC) x (NWG/NDIMC) tile
        int threadTileM = mwg / mdimc;
        int threadTileN = nwg / ndimc;

        // Accumulator registers: threadTileM * threadTileN floats
        int accumulators = threadTileM * threadTileN;

        // Register tile for A and B loads (typically vectorized)
        int aTile = threadTileM;  // One column of A tile
        int bTile = threadTileN;  // One row of B tile

        // Miscellaneous: loop counters, addresses, etc.
        int misc = 8;

        // Total estimate (conservative, actual depends on compiler)
        return accumulators + aTile + bTile + misc;
    }

    /// <summary>
    /// Estimates LDS usage for CLBlast XGEMM kernel.
    /// </summary>
    /// <param name="mwg">M-dimension work-group tile size.</param>
    /// <param name="nwg">N-dimension work-group tile size.</param>
    /// <param name="kwg">K-dimension work-group tile size.</param>
    /// <param name="ldsPad">LDS padding for bank conflict avoidance.</param>
    /// <returns>LDS bytes per workgroup.</returns>
    public static int EstimateGemmLds(int mwg, int nwg, int kwg, int ldsPad = 1)
    {
        // LDS stores tiles of A and B
        // A tile: MWG x KWG floats
        // B tile: KWG x NWG floats
        int aTileSize = mwg * (kwg + ldsPad) * sizeof(float);
        int bTileSize = kwg * (nwg + ldsPad) * sizeof(float);

        return aTileSize + bTileSize;
    }

    /// <summary>
    /// Calculates occupancy specifically for CLBlast-style GEMM kernels.
    /// </summary>
    public static OccupancyResult CalculateForGemm(
        GpuArchitectureSpec arch,
        int mwg, int nwg, int kwg,
        int mdimc, int ndimc,
        int ldsPad = 1)
    {
        int workgroupSize = mdimc * ndimc;
        int vgprs = EstimateGemmVgprs(mwg, nwg, kwg, mdimc, ndimc);
        int ldsBytes = EstimateGemmLds(mwg, nwg, kwg, ldsPad);

        return Calculate(arch, vgprs, ldsBytes, workgroupSize);
    }
}
