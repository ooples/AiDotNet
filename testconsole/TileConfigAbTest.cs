// Comprehensive Tile Configuration A/B Testing
// Tests multiple tile configurations to find optimal parameters per matrix size

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNetTestConsole;

public static class TileConfigAbTest
{
    /// <summary>
    /// Round 8: Focus on small matrices (512, 768, 1024)
    /// Small matrices need smaller tiles with more parallelism
    /// Goal: 80%+ efficiency for ALL sizes
    /// </summary>
    private static readonly GemmConfig[] TestConfigs = new[]
    {
        // Config 0: Best for large (reference)
        new GemmConfig
        {
            TileM = 64, TileN = 128, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 8, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 16,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile64x128_k8_ku8"
        },
        // Config 1: Small tiles 32x32 (more work groups for small matrices)
        // WG = 4 * 4 = 16 threads
        new GemmConfig
        {
            TileM = 32, TileN = 32, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 4, NdimbSize = 4,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile32x32"
        },
        // Config 2: Small tiles 32x64 (WG = 4*8 = 32 threads)
        new GemmConfig
        {
            TileM = 32, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 4, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile32x64"
        },
        // Config 3: 64x64 k16 ku4 (balanced for small)
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile64x64_k16_ku4"
        },
        // Config 4: 64x64 k32 ku2 (larger K tile for less LDS swaps)
        // LDS = (32*64 + 32*64) * 4 = 16KB
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 32,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 2, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile64x64_k32"
        },
        // Config 5: 64x64 k8 ku8 with VW2
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 2, VectorWidthN = 2,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 8, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile64x64_k8_ku8_vw2"
        },
        // Config 6: 32x128 (wide N for coalescing, WG = 4*16 = 64)
        new GemmConfig
        {
            TileM = 32, TileN = 128, TileK = 16,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 4, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 4, NdimbSize = 16,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile32x128"
        },
        // Config 7: 64x64 k8 ku16 (very aggressive unroll)
        new GemmConfig
        {
            TileM = 64, TileN = 64, TileK = 8,
            ThreadTileM = 8, ThreadTileN = 8,
            VectorWidthM = 4, VectorWidthN = 4,
            UseVectorizedLoads = true,
            KReg = 1, KUnroll = 16, UseSubgroupOps = false,
            StrideM = true, StrideN = true,
            CacheA = true, CacheB = true,
            MdimaSize = 8, NdimbSize = 8,
            UseTrueVectorLDS = true,
            UseColumnMajorA = true,
            KernelName = "clblast_baseline_k0_tile64x64_k8_ku16"
        },
    };

    public static void Run()
    {
        Console.WriteLine("=== Tile Configuration A/B Testing ===");
        Console.WriteLine();
        Console.WriteLine("This test compares different tile configurations to find");
        Console.WriteLine("the optimal parameters for maximum GFLOPS at each matrix size.");
        Console.WriteLine();

        try
        {
            using var backend = new OpenClBackend();

            Console.WriteLine($"GPU: {backend.DeviceName}");
            Console.WriteLine($"Vendor: {backend.DeviceVendor}");
            Console.WriteLine($"Compute Units: {backend.ComputeUnits}");
            Console.WriteLine();

            // Calculate theoretical peak
            var arch = AiDotNet.Tensors.Engines.DirectGpu.Profiling.GpuArchitectureSpec.DetectFromDeviceName(backend.DeviceName);
            double peakGflops = arch.CalculatePeakGflops(backend.ComputeUnits);
            Console.WriteLine($"Architecture: {arch.Name}");
            Console.WriteLine($"Theoretical Peak: {peakGflops:F0} GFLOPS");
            Console.WriteLine();

            // CRITICAL: Initialize CLBlast parameters by calling Gemm once
            // This sets _clblastMinIndirectSize and other required parameters
            Console.WriteLine("Initializing CLBlast parameters...");
            using (var initA = backend.AllocateBuffer(new float[64 * 64]))
            using (var initB = backend.AllocateBuffer(new float[64 * 64]))
            using (var initC = backend.AllocateBuffer(64 * 64))
            {
                backend.Gemm(initA, initB, initC, 64, 64, 64);
            }
            Console.WriteLine();

            // Test sizes focusing on compute-bound region
            var sizes = new[] { 512, 768, 1024, 1536, 2048, 3072, 4096 };

            Console.WriteLine("Testing configurations:");
            foreach (var cfg in TestConfigs)
            {
                Console.WriteLine($"  {cfg.KernelName}");
            }
            Console.WriteLine();

            var results = new Dictionary<int, List<(string Config, double Gflops, double Efficiency, bool Correct)>>();
            var winners = new Dictionary<int, (string Config, double Gflops, double Efficiency)>();

            foreach (var size in sizes)
            {
                Console.WriteLine($"--- Size: {size}x{size}x{size} ---");

                int M = size, N = size, K = size;
                long flops = 2L * M * N * K;

                // Allocate test matrices
                var random = new Random(42);
                var dataA = new float[M * K];
                var dataB = new float[K * N];
                for (int i = 0; i < dataA.Length; i++) dataA[i] = (float)(random.NextDouble() - 0.5);
                for (int i = 0; i < dataB.Length; i++) dataB[i] = (float)(random.NextDouble() - 0.5);

                var bufA = backend.AllocateBuffer(dataA);
                var bufB = backend.AllocateBuffer(dataB);
                var bufC = backend.AllocateBuffer(M * N);

                try
                {
                    var sizeResults = new List<(string Config, double Gflops, double Efficiency, bool Correct)>();
                    float[]? referenceResult = null;

                    int configIdx = 0;
                    foreach (var config in TestConfigs)
                    {
                        try
                        {
                            // Clear output buffer
                            ((DirectOpenClGpuBuffer)bufC).Buffer.CopyFromHost(new float[M * N]);

                            // Skip configs with work group size > 256
                            // CLBlast work group size = (TileM/ThreadTileM) * (TileN/ThreadTileN)
                            int threadsM = config.TileM / config.ThreadTileM;
                            int threadsN = config.TileN / config.ThreadTileN;
                            int threadCount = threadsM * threadsN;
                            if (threadCount > 256)
                            {
                                Console.WriteLine($"  {config.KernelName}: SKIP (workgroup={threadsM}x{threadsN}={threadCount} > 256)");
                                continue;
                            }

                            // Skip configs where tile doesn't fit in LDS (64KB limit)
                            // LDS = (TileK * TileM + TileK * TileN) * sizeof(float) + padding
                            int ldsBytes = (config.TileK * config.TileM + config.TileK * config.TileN) * 4 + 1024;
                            if (ldsBytes > 65536)
                            {
                                Console.WriteLine($"  {config.KernelName}: SKIP (LDS={ldsBytes / 1024}KB > 64KB)");
                                continue;
                            }

                            // Use XOR swizzle variant for all tests (variant 1)
                            DynamicGemmKernel.KernelVariant = 1;

                            // Warmup
                            for (int w = 0; w < 3; w++)
                            {
                                backend.GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, config);
                            }

                            // Benchmark
                            var sw = Stopwatch.StartNew();
                            for (int r = 0; r < 10; r++)
                            {
                                backend.GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, config);
                            }
                            sw.Stop();

                            double timeMs = sw.Elapsed.TotalMilliseconds / 10;
                            double gflops = flops / (timeMs * 1e6);
                            double efficiency = 100.0 * gflops / peakGflops;

                            // Verify correctness against first config
                            bool correct = true;
                            if (configIdx == 0)
                            {
                                referenceResult = ((DirectOpenClGpuBuffer)bufC).Download();
                            }
                            else if (referenceResult != null)
                            {
                                var currentResult = ((DirectOpenClGpuBuffer)bufC).Download();
                                double maxDiff = 0;
                                for (int i = 0; i < Math.Min(currentResult.Length, referenceResult.Length); i++)
                                {
                                    maxDiff = Math.Max(maxDiff, Math.Abs(currentResult[i] - referenceResult[i]));
                                }
                                correct = maxDiff < 0.01f;
                            }

                            sizeResults.Add((config.KernelName, gflops, efficiency, correct));
                            Console.WriteLine($"  {config.KernelName}: {gflops,7:F0} GFLOPS ({efficiency,5:F1}%) {timeMs,6:F2}ms {(correct ? "OK" : "WRONG!")}");
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"  {config.KernelName}: FAILED - {ex.Message}");
                        }

                        configIdx++;
                    }

                    results[size] = sizeResults;

                    // Find winner
                    var valid = sizeResults.Where(r => r.Correct && r.Gflops > 0).ToList();
                    if (valid.Any())
                    {
                        var best = valid.OrderByDescending(r => r.Gflops).First();
                        winners[size] = (best.Config, best.Gflops, best.Efficiency);
                        Console.WriteLine($"  ** Winner: {best.Config} ({best.Gflops:F0} GFLOPS, {best.Efficiency:F1}%) **");
                    }
                    Console.WriteLine();
                }
                finally
                {
                    bufA.Dispose();
                    bufB.Dispose();
                    bufC.Dispose();
                }
            }

            // Summary
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("SUMMARY: BEST CONFIGURATION PER SIZE");
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine();
            Console.WriteLine($"{"Size",-8} {"Winner",-35} {"GFLOPS",-10} {"Efficiency",-10}");
            Console.WriteLine("-".PadRight(70, '-'));

            foreach (var size in sizes)
            {
                if (winners.TryGetValue(size, out var w))
                {
                    Console.WriteLine($"{size,-8} {w.Config,-35} {w.Gflops,7:F0}    {w.Efficiency,5:F1}%");
                }
            }

            // Calculate average efficiency
            if (winners.Any())
            {
                double avgEff = winners.Values.Average(w => w.Efficiency);
                double maxEff = winners.Values.Max(w => w.Efficiency);
                Console.WriteLine();
                Console.WriteLine($"Average Efficiency: {avgEff:F1}%");
                Console.WriteLine($"Best Efficiency: {maxEff:F1}%");
                Console.WriteLine($"Theoretical Peak: {peakGflops:F0} GFLOPS");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
