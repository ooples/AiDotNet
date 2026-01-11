// Copyright (c) AiDotNet. All rights reserved.
// GEMM Benchmark to validate 25,000+ GFLOPS target on AMD GPUs.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Benchmark results for GEMM performance.
/// </summary>
public readonly struct GemmBenchmarkResult
{
    public int M { get; init; }
    public int N { get; init; }
    public int K { get; init; }
    public double TimeMs { get; init; }
    public double GFlops { get; init; }
    public double TFlops => GFlops / 1000.0;
    public string KernelName { get; init; }
    public bool MetTarget { get; init; }

    public override string ToString() =>
        $"[{M}x{K}] × [{K}x{N}] = {GFlops:F1} GFLOPS ({TFlops:F2} TFLOPS) in {TimeMs:F2}ms - {KernelName} - {(MetTarget ? "PASS" : "BELOW TARGET")}";
}

/// <summary>
/// GEMM benchmark to validate performance targets.
/// Target: 25,000+ GFLOPS on AMD RX 6800+ (vs CLBlast's ~2,500 GFLOPS)
/// </summary>
public sealed class GemmBenchmark
{
    private readonly DirectGpuEngine _engine;
    private readonly double _targetGFlops;

    /// <summary>
    /// Creates benchmark with target performance.
    /// </summary>
    /// <param name="engine">DirectGpuEngine instance.</param>
    /// <param name="targetGFlops">Target GFLOPS (default: 25,000).</param>
    public GemmBenchmark(DirectGpuEngine engine, double targetGFlops = 25000.0)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _targetGFlops = targetGFlops;
    }

    /// <summary>
    /// Runs a single benchmark for given dimensions.
    /// </summary>
    public GemmBenchmarkResult BenchmarkSingle(int M, int N, int K, int warmupRuns = 3, int benchmarkRuns = 10)
    {
        if (!_engine.IsAvailable)
        {
            return new GemmBenchmarkResult
            {
                M = M,
                N = N,
                K = K,
                TimeMs = 0,
                GFlops = 0,
                KernelName = "N/A",
                MetTarget = false
            };
        }

        // Allocate test matrices
        var A = new float[M * K];
        var B = new float[K * N];

        // Initialize with random values
        var rand = new Random(42);
        for (int i = 0; i < A.Length; i++) A[i] = (float)(rand.NextDouble() - 0.5) * 2;
        for (int i = 0; i < B.Length; i++) B[i] = (float)(rand.NextDouble() - 0.5) * 2;

        // Warmup runs
        for (int i = 0; i < warmupRuns; i++)
        {
            _engine.MatMul(A, B, M, K, N);
        }

        // Benchmark runs
        var sw = new Stopwatch();
        double totalMs = 0;

        for (int i = 0; i < benchmarkRuns; i++)
        {
            sw.Restart();
            _engine.MatMul(A, B, M, K, N);
            sw.Stop();
            totalMs += sw.Elapsed.TotalMilliseconds;
        }

        double avgMs = totalMs / benchmarkRuns;
        double flops = 2.0 * M * N * K;  // 2 ops per multiply-add
        double gflops = flops / (avgMs * 1e6);

        // Determine kernel name from backend - varies by vendor/architecture
        string kernelName = _engine.BackendName switch
        {
            string b when b.Contains("MFMA") => "mfma_gemm_tiled",        // AMD MI-series with matrix cores
            string b when b.Contains("RDNA3") => "gemm_wmma_gfx11",       // AMD RDNA3 with WMMA
            string b when b.Contains("RDNA") => "gemm_double_buffered",    // AMD RDNA/RDNA2
            string b when b.Contains("HIP") => "gemm_tiled",               // Generic HIP
            string b when b.Contains("CUDA") => "gemm_tensor_core",        // NVIDIA with tensor cores
            string b when b.Contains("OpenCL") => "gemm_double_buffered",  // OpenCL fallback
            _ => "gemm_default"
        };

        return new GemmBenchmarkResult
        {
            M = M,
            N = N,
            K = K,
            TimeMs = avgMs,
            GFlops = gflops,
            KernelName = kernelName,
            MetTarget = gflops >= _targetGFlops
        };
    }

    /// <summary>
    /// Runs comprehensive benchmark suite with various matrix sizes.
    /// </summary>
    public GemmBenchmarkResult[] RunSuite()
    {
        var results = new System.Collections.Generic.List<GemmBenchmarkResult>();

        // Standard square matrices
        int[] sizes = { 256, 512, 1024, 2048, 4096, 8192 };
        foreach (var size in sizes)
        {
            var result = BenchmarkSingle(size, size, size);
            results.Add(result);
            Console.WriteLine(result);
        }

        // Neural network layer sizes (typical Dense layer dimensions)
        var nnSizes = new (int M, int N, int K)[]
        {
            (64, 512, 768),     // Batch 64, BERT hidden
            (64, 3072, 768),    // BERT FFN expansion
            (64, 768, 3072),    // BERT FFN projection
            (256, 1024, 1024),  // Large batch, medium layer
            (1024, 4096, 1024), // Transformer-XL style
            (2048, 2048, 2048), // Large square
        };

        foreach (var (M, N, K) in nnSizes)
        {
            var result = BenchmarkSingle(M, N, K);
            results.Add(result);
            Console.WriteLine(result);
        }

        return results.ToArray();
    }

    /// <summary>
    /// Generates a performance comparison report.
    /// </summary>
    public string GenerateReport(GemmBenchmarkResult[] results)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("╔══════════════════════════════════════════════════════════════════════════════╗");
        sb.AppendLine("║                    DirectGpu GEMM Performance Report                         ║");
        sb.AppendLine("╠══════════════════════════════════════════════════════════════════════════════╣");
        sb.AppendLine($"║ Target: {_targetGFlops:N0} GFLOPS ({_targetGFlops / 1000:F1} TFLOPS)");
        sb.AppendLine($"║ Comparison: CLBlast ~2,500 GFLOPS");
        sb.AppendLine("╠══════════════════════════════════════════════════════════════════════════════╣");
        sb.AppendLine("║  M      N      K     Time(ms)   GFLOPS   TFLOPS  Status                      ║");
        sb.AppendLine("╠══════════════════════════════════════════════════════════════════════════════╣");

        int passed = 0;
        double maxGFlops = 0;

        foreach (var r in results)
        {
            string status = r.MetTarget ? "[PASS]" : "[MISS]";
            if (r.MetTarget) passed++;
            if (r.GFlops > maxGFlops) maxGFlops = r.GFlops;

            sb.AppendLine($"║ {r.M,5} {r.N,5} {r.K,5}  {r.TimeMs,8:F2}  {r.GFlops,8:F1}  {r.TFlops,6:F2}  {status,-8}                    ║");
        }

        sb.AppendLine("╠══════════════════════════════════════════════════════════════════════════════╣");
        sb.AppendLine($"║ Summary: {passed}/{results.Length} passed target, Peak: {maxGFlops:N0} GFLOPS ({maxGFlops / 1000:F2} TFLOPS)");
        sb.AppendLine("║                                                                              ║");

        // Speedup comparison
        double clblastSpeedup = maxGFlops / 2500.0;

        sb.AppendLine($"║ vs CLBlast: {clblastSpeedup,5:F1}x faster");
        sb.AppendLine("╚══════════════════════════════════════════════════════════════════════════════╝");

        return sb.ToString();
    }

    /// <summary>
    /// Quick test to verify GPU is working and estimate peak performance.
    /// </summary>
    public static void QuickTest()
    {
        Console.WriteLine("DirectGpu GEMM Quick Test");
        Console.WriteLine("=========================");

        using var engine = new DirectGpuEngine();

        if (!engine.IsAvailable)
        {
            Console.WriteLine("ERROR: DirectGpu not available.");
            Console.WriteLine("Make sure OpenCL is installed and a compatible GPU is present.");
            return;
        }

        Console.WriteLine($"GPU: {engine.DeviceName}");
        Console.WriteLine($"Vendor: {engine.DeviceVendor}");
        Console.WriteLine($"Compute Units: {engine.ComputeUnits}");
        Console.WriteLine($"Global Memory: {engine.GlobalMemoryGB:F1} GB");
        Console.WriteLine();

        var benchmark = new GemmBenchmark(engine);

        // Quick test with 2048x2048
        Console.WriteLine("Running quick benchmark (2048x2048x2048)...");
        var result = benchmark.BenchmarkSingle(2048, 2048, 2048);

        Console.WriteLine();
        Console.WriteLine($"Result: {result.GFlops:N0} GFLOPS ({result.TFlops:F2} TFLOPS)");
        Console.WriteLine($"Target: 25,000 GFLOPS (25.0 TFLOPS)");
        Console.WriteLine($"Status: {(result.MetTarget ? "TARGET MET!" : "Below target")}");

        if (result.GFlops > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"vs CLBlast (2,500 GFLOPS): {result.GFlops / 2500:F1}x faster");
        }
    }
}
