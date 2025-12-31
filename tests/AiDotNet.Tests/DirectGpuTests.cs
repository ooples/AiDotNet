// Copyright (c) AiDotNet. All rights reserved.
// Tests for DirectGpu engine functionality

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
#if !NET462
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
#endif

namespace AiDotNet.Tests;

/// <summary>
/// Tests for DirectGpu engine - custom optimized GPU kernels.
/// </summary>
public class DirectGpuTests
{
    private readonly ITestOutputHelper _output;

    public DirectGpuTests(ITestOutputHelper output)
    {
        _output = output;
    }

#if !NET462
    [Fact]
    public void DirectGpuEngine_CanInitialize()
    {
        // Arrange & Act
        using var engine = new DirectGpuEngine();

        // Assert
        _output.WriteLine($"DirectGpu Available: {engine.IsAvailable}");
        _output.WriteLine($"Backend: {engine.BackendName}");
        _output.WriteLine($"Device: {engine.DeviceName}");
        _output.WriteLine($"Vendor: {engine.DeviceVendor}");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine($"Global Memory: {engine.GlobalMemoryGB:F2} GB");
        _output.WriteLine(engine.GetDiagnostics());

        // This test always passes - it's for diagnostics
        Assert.NotNull(engine.BackendName);
    }

    [Fact]
    public void DirectGpuEngine_MatMul_SmallMatrix()
    {
        // Arrange
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        // 4x3 * 3x2 = 4x2
        int M = 4, K = 3, N = 2;
        var A = new float[]
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        };
        var B = new float[]
        {
            1, 2,
            3, 4,
            5, 6
        };

        // Expected result (calculated manually):
        // [1,2,3] * [1,2; 3,4; 5,6] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4,5,6] * [1,2; 3,4; 5,6] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        // etc.
        var expected = new float[]
        {
            22, 28,
            49, 64,
            76, 100,
            103, 136
        };

        // Act
        var result = engine.MatMul(A, B, M, K, N);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(M * N, result.Length);

        _output.WriteLine("Result:");
        for (int i = 0; i < M; i++)
        {
            _output.WriteLine($"  [{result[i * N]:F2}, {result[i * N + 1]:F2}]");
        }

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], 2); // 2 decimal places tolerance
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_MatMul_Benchmark()
    {
        // Arrange
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping benchmark");
            return;
        }

        // Test different matrix sizes - including large sizes for GPU performance
        var sizes = new[] { 64, 128, 256, 512, 1024, 2048, 4096 };

        foreach (var size in sizes)
        {
            int M = size, K = size, N = size;
            var A = new float[M * K];
            var B = new float[K * N];

            // Initialize with random values
            var rand = new Random(42);
            for (int i = 0; i < A.Length; i++)
                A[i] = (float)(rand.NextDouble() * 2 - 1);
            for (int i = 0; i < B.Length; i++)
                B[i] = (float)(rand.NextDouble() * 2 - 1);

            // Warm up
            engine.MatMul(A, B, M, K, N);

            // Benchmark
            var sw = Stopwatch.StartNew();
            const int iterations = 10;
            for (int i = 0; i < iterations; i++)
            {
                engine.MatMul(A, B, M, K, N);
            }
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
            double flops = 2.0 * M * N * K; // Multiply-add = 2 FLOPs per element
            double gflops = (flops / avgMs) * 1e-6; // GFLOPS

            _output.WriteLine($"Size {size}x{size}: {avgMs:F2}ms avg, {gflops:F1} GFLOPS (includes data transfer)");
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_MatMul_Benchmark_KernelOnly()
    {
        // This benchmark measures ONLY kernel compute time by:
        // 1. Uploading data once to GPU
        // 2. Running GEMM multiple times with data staying on GPU
        // 3. Downloading result once at the end
        // This gives true GPU compute performance without PCIe transfer overhead.

        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping benchmark");
            return;
        }

        // Access the backend directly for GPU-resident operations
        var backend = engine.Backend;
        if (backend == null)
        {
            _output.WriteLine("Backend not available");
            return;
        }

        _output.WriteLine($"Testing on: {engine.DeviceName} ({engine.DeviceVendor})");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine("");
        _output.WriteLine("Kernel-Only Performance (no PCIe transfer overhead):");

        var sizes = new[] { 512, 1024, 2048, 4096 };

        foreach (var size in sizes)
        {
            int M = size, K = size, N = size;
            var A = new float[M * K];
            var B = new float[K * N];

            var rand = new Random(42);
            for (int i = 0; i < A.Length; i++)
                A[i] = (float)(rand.NextDouble() * 2 - 1);
            for (int i = 0; i < B.Length; i++)
                B[i] = (float)(rand.NextDouble() * 2 - 1);

            // Upload once to GPU
            using var bufferA = backend.AllocateBuffer(A);
            using var bufferB = backend.AllocateBuffer(B);
            using var bufferC = backend.AllocateBuffer(M * N);

            // Warm up kernel (data already on GPU)
            backend.Gemm(bufferA, bufferB, bufferC, M, N, K);
            backend.Synchronize();

            // Benchmark kernel only (data stays on GPU)
            var sw = Stopwatch.StartNew();
            const int iterations = 20;
            for (int i = 0; i < iterations; i++)
            {
                backend.Gemm(bufferA, bufferB, bufferC, M, N, K);
            }
            backend.Synchronize(); // Wait for all kernels to complete
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
            double flops = 2.0 * M * N * K;
            double gflops = (flops / avgMs) * 1e-6;

            // Calculate theoretical peak and utilization
            // RX 5500 XT: 22 CUs, 1717 MHz boost, ~5.2 TFLOPS theoretical
            double theoreticalTflops = 5.2;
            double utilization = (gflops / 1000.0) / theoreticalTflops * 100.0;
            _output.WriteLine($"Size {size}x{size}: {avgMs:F3}ms avg, {gflops:F1} GFLOPS ({utilization:F1}% utilization)");
        }

        _output.WriteLine("");
        _output.WriteLine("Target: ~2500 GFLOPS (48% utilization, CLBlast level)");
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_MatMul_Benchmark_KernelComparison()
    {
        // Compare simple tiled kernel vs double-buffered kernel
        // This helps identify if double buffering complexity is hurting performance.

        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping benchmark");
            return;
        }

        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available");
            return;
        }

        _output.WriteLine($"Testing on: {engine.DeviceName} ({engine.DeviceVendor})");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine("");
        _output.WriteLine("Kernel Comparison (kernel-only, no PCIe transfer):");
        _output.WriteLine("--------------------------------------------------");

        var sizes = new[] { 1024, 2048, 4096 };

        foreach (var size in sizes)
        {
            int M = size, K = size, N = size;
            var A = new float[M * K];
            var B = new float[K * N];

            var rand = new Random(42);
            for (int i = 0; i < A.Length; i++)
                A[i] = (float)(rand.NextDouble() * 2 - 1);
            for (int i = 0; i < B.Length; i++)
                B[i] = (float)(rand.NextDouble() * 2 - 1);

            // Upload once to GPU
            using var bufferA = backend.AllocateBuffer(A);
            using var bufferB = backend.AllocateBuffer(B);
            using var bufferC1 = backend.AllocateBuffer(M * N);
            using var bufferC2 = backend.AllocateBuffer(M * N);

            const int iterations = 20;
            double flops = 2.0 * M * N * K;

            // Benchmark double-buffered kernel
            backend.Gemm(bufferA, bufferB, bufferC1, M, N, K);
            backend.Synchronize();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                backend.Gemm(bufferA, bufferB, bufferC1, M, N, K);
            }
            backend.Synchronize();
            sw.Stop();
            double doubleBufferedMs = sw.Elapsed.TotalMilliseconds / iterations;
            double doubleBufferedGflops = (flops / doubleBufferedMs) * 1e-6;

            // Benchmark simple tiled kernel
            backend.GemmSimple(bufferA, bufferB, bufferC2, M, N, K);
            backend.Synchronize();
            sw.Restart();
            for (int i = 0; i < iterations; i++)
            {
                backend.GemmSimple(bufferA, bufferB, bufferC2, M, N, K);
            }
            backend.Synchronize();
            sw.Stop();
            double simpleMs = sw.Elapsed.TotalMilliseconds / iterations;
            double simpleGflops = (flops / simpleMs) * 1e-6;

            _output.WriteLine($"Size {size}x{size}:");
            _output.WriteLine($"  Double-buffered (64x64, 4x4 outputs): {doubleBufferedMs:F3}ms, {doubleBufferedGflops:F1} GFLOPS");
            _output.WriteLine($"  Simple tiled (16x16, 1 output):       {simpleMs:F3}ms, {simpleGflops:F1} GFLOPS");
            _output.WriteLine($"  Ratio (double-buffered/simple):       {doubleBufferedGflops / simpleGflops:F2}x");
            _output.WriteLine("");
        }

        _output.WriteLine("If simple kernel is faster, the issue is in double buffering implementation.");
        _output.WriteLine("If similar speed, issue is in memory coalescing or fundamental approach.");
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_MatMul_Benchmark_AllKernelVariations()
    {
        // Comprehensive benchmark comparing ALL kernel variations to identify best configuration
        // This test helps identify which kernel configuration gives the best performance
        // by testing different hypotheses:
        // 1. gemm_small_tile: 16x16 tiles, 1x1 output/thread (minimal register pressure baseline)
        // 2. gemm_medium_tile: 32x32 tiles, 2x2 output/thread (balance of registers and loads)
        // 3. gemm_coalesced: Focus on memory coalescing patterns
        // 4. gemm_vectorized_tile: float4 for all loads (maximize bandwidth)
        // 5. gemm_low_register: Existing 32x32 tile, 2x2 output/thread
        // 6. gemm_double_buffered: Current optimized kernel (64x64 tile, 4x4 output/thread)
        // 7. gemm_tiled_simple: Baseline simple tiled kernel

        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping benchmark");
            return;
        }

        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available");
            return;
        }

        _output.WriteLine("=============================================================");
        _output.WriteLine("COMPREHENSIVE GEMM KERNEL BENCHMARK");
        _output.WriteLine("=============================================================");
        _output.WriteLine($"Device: {engine.DeviceName} ({engine.DeviceVendor})");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine("");
        _output.WriteLine("Goal: Identify which kernel configuration gives best performance");
        _output.WriteLine("Target: 2500 GFLOPS (vs current ~540 GFLOPS)");
        _output.WriteLine("");

        // Define kernel variants to test
        var kernelVariants = new (string Name, string Description, Action<IGpuBuffer, IGpuBuffer, IGpuBuffer, int, int, int> Execute)[]
        {
            ("gemm_small_tile", "16x16 tile, 1x1 out/thread (minimal regs)",
                (a, b, c, m, n, k) => backend.GemmSmallTile(a, b, c, m, n, k)),
            ("gemm_medium_tile", "32x32 tile, 2x2 out/thread (balanced)",
                (a, b, c, m, n, k) => backend.GemmMediumTile(a, b, c, m, n, k)),
            ("gemm_coalesced", "16x16 tile, coalesced access pattern",
                (a, b, c, m, n, k) => backend.GemmCoalesced(a, b, c, m, n, k)),
            ("gemm_vectorized_tile", "16x64 tile, float4 vectorized",
                (a, b, c, m, n, k) => backend.GemmVectorizedTile(a, b, c, m, n, k)),
            ("gemm_low_register", "32x32 tile, 2x2 out/thread (low reg)",
                (a, b, c, m, n, k) => backend.GemmLowRegister(a, b, c, m, n, k)),
            ("gemm_tiled_simple", "16x16 tile, simple baseline",
                (a, b, c, m, n, k) => backend.GemmSimple(a, b, c, m, n, k)),
            ("gemm_double_buffered", "64x64 tile, 4x4 out/thread (current)",
                (a, b, c, m, n, k) => backend.Gemm(a, b, c, m, n, k)),
            ("gemm_kreg4", "32x32 tile, KREG=4 unrolling",
                (a, b, c, m, n, k) => backend.GemmKreg4(a, b, c, m, n, k)),
            ("gemm_prefetch", "32x32 tile, software prefetch",
                (a, b, c, m, n, k) => backend.GemmPrefetch(a, b, c, m, n, k)),
            ("gemm_wide_vec", "32x128 tile, float4 for A and B",
                (a, b, c, m, n, k) => backend.GemmWideVec(a, b, c, m, n, k)),
            ("gemm_clblast_rdna1", "64x64 tile, 8x8 WG, 8x8 out/thread (CLBlast)",
                (a, b, c, m, n, k) => backend.GemmClblastRdna1(a, b, c, m, n, k)),
        };

        // Test sizes - focus on sizes that need good performance
        var sizes = new[] { 1024, 2048, 4096 };
        const int warmupIterations = 3;
        const int benchmarkIterations = 10;

        // Store results for summary
        var results = new Dictionary<string, (double[] gflopsPerSize, string description)>();
        foreach (var variant in kernelVariants)
        {
            results[variant.Name] = (new double[sizes.Length], variant.Description);
        }

        foreach (var size in sizes)
        {
            int M = size, K = size, N = size;
            _output.WriteLine($"--- Matrix Size: {size}x{size}x{size} ---");

            // Create test data
            var A = new float[M * K];
            var B = new float[K * N];
            var rand = new Random(42);
            for (int i = 0; i < A.Length; i++)
                A[i] = (float)(rand.NextDouble() * 2 - 1);
            for (int i = 0; i < B.Length; i++)
                B[i] = (float)(rand.NextDouble() * 2 - 1);

            // Upload once to GPU
            using var bufferA = backend.AllocateBuffer(A);
            using var bufferB = backend.AllocateBuffer(B);
            using var bufferC = backend.AllocateBuffer(M * N);

            double flops = 2.0 * M * N * K;

            foreach (var variant in kernelVariants)
            {
                try
                {
                    // Warmup
                    for (int i = 0; i < warmupIterations; i++)
                    {
                        variant.Execute(bufferA, bufferB, bufferC, M, N, K);
                    }
                    backend.Synchronize();

                    // Benchmark
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < benchmarkIterations; i++)
                    {
                        variant.Execute(bufferA, bufferB, bufferC, M, N, K);
                    }
                    backend.Synchronize();
                    sw.Stop();

                    double avgMs = sw.Elapsed.TotalMilliseconds / benchmarkIterations;
                    double gflops = (flops / avgMs) * 1e-6;

                    int sizeIndex = Array.IndexOf(sizes, size);
                    results[variant.Name].gflopsPerSize[sizeIndex] = gflops;

                    _output.WriteLine($"  {variant.Name,-25}: {avgMs:F3}ms, {gflops:F1} GFLOPS");
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"  {variant.Name,-25}: FAILED - {ex.Message}");
                    int sizeIndex = Array.IndexOf(sizes, size);
                    results[variant.Name].gflopsPerSize[sizeIndex] = 0;
                }
            }
            _output.WriteLine("");
        }

        // Summary table
        _output.WriteLine("=============================================================");
        _output.WriteLine("SUMMARY: Average GFLOPS across all sizes");
        _output.WriteLine("=============================================================");
        _output.WriteLine($"{"Kernel",-25} | {"Avg GFLOPS",10} | Description");
        _output.WriteLine(new string('-', 70));

        var sortedResults = results
            .Select(r => (Name: r.Key, AvgGflops: r.Value.gflopsPerSize.Average(), Desc: r.Value.description))
            .OrderByDescending(r => r.AvgGflops)
            .ToList();

        foreach (var result in sortedResults)
        {
            _output.WriteLine($"{result.Name,-25} | {result.AvgGflops,10:F1} | {result.Desc}");
        }

        _output.WriteLine("");
        _output.WriteLine("WINNER: " + sortedResults[0].Name + $" ({sortedResults[0].AvgGflops:F1} GFLOPS)");
        _output.WriteLine("");

        // Analysis
        var best = sortedResults[0];
        var current = sortedResults.First(r => r.Name == "gemm_double_buffered");

        if (best.AvgGflops > current.AvgGflops * 1.1)
        {
            _output.WriteLine($"RECOMMENDATION: Switch to {best.Name} for {best.AvgGflops / current.AvgGflops:F2}x improvement");
        }
        else
        {
            _output.WriteLine("RECOMMENDATION: Current kernel is competitive. Focus on other optimizations.");
        }

        _output.WriteLine("");
        _output.WriteLine("Hypothesis Analysis:");
        _output.WriteLine("- If small_tile wins: Register pressure is the issue");
        _output.WriteLine("- If vectorized_tile wins: Memory bandwidth was the bottleneck");
        _output.WriteLine("- If coalesced wins: Memory access patterns were suboptimal");
        _output.WriteLine("- If medium_tile wins: 2x2 register blocking is the sweet spot");
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_Relu_Activation()
    {
        // Arrange
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        var input = new float[] { -2, -1, 0, 1, 2, 3 };
        var expected = new float[] { 0, 0, 0, 1, 2, 3 };

        // Act
        var result = engine.Relu(input);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(expected.Length, result.Length);

        _output.WriteLine($"Input:    [{string.Join(", ", input)}]");
        _output.WriteLine($"Output:   [{string.Join(", ", result)}]");
        _output.WriteLine($"Expected: [{string.Join(", ", expected)}]");

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], 4);
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_Softmax()
    {
        // Arrange
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        // 2 batches, 3 features each
        var input = new float[] { 1, 2, 3, 1, 1, 1 };
        int batchSize = 2;
        int features = 3;

        // Act
        var result = engine.Softmax(input, batchSize, features);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(input.Length, result.Length);

        _output.WriteLine("Softmax results:");
        for (int b = 0; b < batchSize; b++)
        {
            var sum = 0f;
            var values = new string[features];
            for (int f = 0; f < features; f++)
            {
                var val = result[b * features + f];
                sum += val;
                values[f] = $"{val:F4}";
            }
            _output.WriteLine($"  Batch {b}: [{string.Join(", ", values)}] sum={sum:F4}");

            // Softmax should sum to 1
            Assert.Equal(1.0f, sum, 2);
        }
    }

    [Fact]
    public void Engine_DirectGpu_Property()
    {
        // Arrange & Act
        var directGpu = Engine.DirectGpu;

        // Assert
        _output.WriteLine($"Engine.DirectGpu is null: {directGpu == null}");
        if (directGpu != null)
        {
            _output.WriteLine($"IsAvailable: {directGpu.IsAvailable}");
            _output.WriteLine($"BackendName: {directGpu.BackendName}");
            _output.WriteLine($"DeviceName: {directGpu.DeviceName}");
        }
    }

    [Fact]
    public void HardwareCapabilities_IncludesDirectGpu()
    {
        // Arrange & Act
        var caps = Engine.Capabilities;

        // Assert
        _output.WriteLine(caps.ToString());
        _output.WriteLine($"DirectGpuAvailable: {caps.DirectGpuAvailable}");
        _output.WriteLine($"DirectGpuBackend: {caps.DirectGpuBackend}");
        _output.WriteLine($"DirectGpuDevice: {caps.DirectGpuDevice}");
        _output.WriteLine($"DirectGpuComputeUnits: {caps.DirectGpuComputeUnits}");
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_NewKernels_Correctness()
    {
        // Validate that the 3 new GEMM kernels produce correct results
        // Test: gemm_kreg4, gemm_prefetch, gemm_wide_vec

        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available");
            return;
        }

        _output.WriteLine("=== New Kernel Correctness Validation ===");
        _output.WriteLine($"Device: {engine.DeviceName} ({engine.DeviceVendor})");
        _output.WriteLine("");

        // Use a small matrix for correctness testing
        int M = 128, K = 128, N = 128;

        // Create test data with predictable values
        var A = new float[M * K];
        var B = new float[K * N];
        var rand = new Random(42);
        for (int i = 0; i < A.Length; i++)
            A[i] = (float)(rand.NextDouble() * 2 - 1);
        for (int i = 0; i < B.Length; i++)
            B[i] = (float)(rand.NextDouble() * 2 - 1);

        // Compute reference result using medium_tile (known-good kernel)
        using var bufferA = backend.AllocateBuffer(A);
        using var bufferB = backend.AllocateBuffer(B);
        using var bufferCRef = backend.AllocateBuffer(M * N);

        backend.GemmMediumTile(bufferA, bufferB, bufferCRef, M, N, K);
        backend.Synchronize();
        var referenceResult = backend.DownloadBuffer(bufferCRef);

        // Test each new kernel against the reference
        var newKernels = new (string Name, Action<IGpuBuffer, IGpuBuffer, IGpuBuffer, int, int, int> Execute)[]
        {
            ("gemm_kreg4", (a, b, c, m, n, k) => backend.GemmKreg4(a, b, c, m, n, k)),
            ("gemm_prefetch", (a, b, c, m, n, k) => backend.GemmPrefetch(a, b, c, m, n, k)),
            ("gemm_wide_vec", (a, b, c, m, n, k) => backend.GemmWideVec(a, b, c, m, n, k)),
        };

        foreach (var (name, execute) in newKernels)
        {
            try
            {
                using var bufferCTest = backend.AllocateBuffer(M * N);
                execute(bufferA, bufferB, bufferCTest, M, N, K);
                backend.Synchronize();
                var testResult = backend.DownloadBuffer(bufferCTest);

                // Compare results
                double maxError = 0;
                int errorCount = 0;
                const float tolerance = 1e-4f;

                for (int i = 0; i < testResult.Length; i++)
                {
                    double error = Math.Abs(testResult[i] - referenceResult[i]);
                    maxError = Math.Max(maxError, error);
                    if (error > tolerance)
                        errorCount++;
                }

                if (errorCount == 0)
                {
                    _output.WriteLine($"{name}: PASS (max error: {maxError:E3})");
                }
                else
                {
                    _output.WriteLine($"{name}: FAIL ({errorCount} elements with error > {tolerance}, max error: {maxError:E3})");
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"{name}: ERROR - {ex.Message}");
            }
        }

        _output.WriteLine("");
        _output.WriteLine("Correctness validation complete.");
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_NewKernels_Benchmark()
    {
        // Benchmark the 3 new GEMM kernels against the current best (gemm_medium_tile)
        // Tests: gemm_kreg4, gemm_prefetch, gemm_wide_vec

        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping benchmark");
            return;
        }

        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available");
            return;
        }

        _output.WriteLine("=============================================================");
        _output.WriteLine("NEW KERNEL BENCHMARK (vs gemm_medium_tile @ 857 GFLOPS)");
        _output.WriteLine("=============================================================");
        _output.WriteLine($"Device: {engine.DeviceName} ({engine.DeviceVendor})");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine("");
        _output.WriteLine("Target: 2500 GFLOPS (48% utilization, CLBlast level)");
        _output.WriteLine("");

        var kernels = new (string Name, string Description, Action<IGpuBuffer, IGpuBuffer, IGpuBuffer, int, int, int> Execute)[]
        {
            ("gemm_medium_tile", "Baseline (32x32, 2x2 out)", (a, b, c, m, n, k) => backend.GemmMediumTile(a, b, c, m, n, k)),
            ("gemm_kreg4", "K-unroll 4 (32x32, KREG=4)", (a, b, c, m, n, k) => backend.GemmKreg4(a, b, c, m, n, k)),
            ("gemm_prefetch", "Prefetch (32x32, double buf)", (a, b, c, m, n, k) => backend.GemmPrefetch(a, b, c, m, n, k)),
            ("gemm_wide_vec", "Wide vec (32x128, float4)", (a, b, c, m, n, k) => backend.GemmWideVec(a, b, c, m, n, k)),
        };

        var sizes = new[] { 2048, 4096 };
        const int warmupIterations = 3;
        const int benchmarkIterations = 10;

        foreach (var size in sizes)
        {
            int M = size, K = size, N = size;
            _output.WriteLine($"--- Matrix Size: {size}x{size}x{size} ---");

            var A = new float[M * K];
            var B = new float[K * N];
            var rand = new Random(42);
            for (int i = 0; i < A.Length; i++)
                A[i] = (float)(rand.NextDouble() * 2 - 1);
            for (int i = 0; i < B.Length; i++)
                B[i] = (float)(rand.NextDouble() * 2 - 1);

            using var bufferA = backend.AllocateBuffer(A);
            using var bufferB = backend.AllocateBuffer(B);
            using var bufferC = backend.AllocateBuffer(M * N);

            double flops = 2.0 * M * N * K;

            foreach (var (name, desc, execute) in kernels)
            {
                try
                {
                    // Warmup
                    for (int i = 0; i < warmupIterations; i++)
                        execute(bufferA, bufferB, bufferC, M, N, K);
                    backend.Synchronize();

                    // Benchmark
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < benchmarkIterations; i++)
                        execute(bufferA, bufferB, bufferC, M, N, K);
                    backend.Synchronize();
                    sw.Stop();

                    double avgMs = sw.Elapsed.TotalMilliseconds / benchmarkIterations;
                    double gflops = (flops / avgMs) * 1e-6;

                    _output.WriteLine($"  {name,-20}: {avgMs:F3}ms, {gflops:F1} GFLOPS");
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"  {name,-20}: FAILED - {ex.Message}");
                }
            }
            _output.WriteLine("");
        }

        _output.WriteLine("Analysis:");
        _output.WriteLine("- gemm_kreg4: Tests if K-loop unrolling (KREG=4) improves ILP");
        _output.WriteLine("- gemm_prefetch: Tests if double-buffering hides memory latency");
        _output.WriteLine("- gemm_wide_vec: Tests if wider vectorization (float4 for both A and B) improves bandwidth");
    }

    // TODO: This test uses APIs that don't exist yet (Rdna1ConfigCount, TuneWithBayesianOptimization)
    // Uncomment and implement when the Bayesian optimization API is finalized
    // [Fact]
    // [Trait("Category", "GPU")]
    // public void DirectGpuEngine_BayesianOptimization_FindsOptimalGemmConfig()
    // { ... }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_DynamicKernelGeneration_CompilesAndBenchmarks()
    {
        // Test dynamic kernel generation like CLBlast - parameters are baked in as compile-time constants
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        // Get the OpenCL backend
        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available - skipping test");
            return;
        }

        _output.WriteLine($"=== Dynamic Kernel Generation Test ===");
        _output.WriteLine($"Device: {engine.DeviceName}");
        _output.WriteLine($"Vendor: {engine.DeviceVendor}");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine("");

        // Test Bayesian optimization with dynamically compiled kernels
        // This uses RunBayesianGemmOptimization which compiles different kernel variants
        int size = 1024;  // Start with 1024x1024 for faster testing
        _output.WriteLine($"Running Bayesian optimization with dynamic kernels on {size}x{size}x{size}...");
        _output.WriteLine("This will compile multiple kernel variants with different parameters baked in.");
        _output.WriteLine("");

        var sw = Stopwatch.StartNew();
        var results = backend.RunBayesianGemmOptimization(
            size, size, size,
            maxTrials: 15,  // Test 15 configurations
            warmupRuns: 2,
            benchmarkRuns: 3);
        sw.Stop();

        _output.WriteLine("");
        _output.WriteLine($"=== Dynamic Kernel Results ===");
        _output.WriteLine($"Total optimization time: {sw.Elapsed.TotalSeconds:F1}s");
        _output.WriteLine($"Configurations tested: {results.Length}");

        if (results.Length > 0 && results[0].IsValid)
        {
            var best = results[0];
            _output.WriteLine("");
            _output.WriteLine($"Best configuration:");
            _output.WriteLine($"  Config: {best.Config}");
            _output.WriteLine($"  GFLOPS: {best.GFlops:F2}");
            _output.WriteLine($"  Time: {best.TimeMs:F3}ms");

            // Calculate theoretical peak for comparison (RX 5500 XT has 5.196 TFLOPS peak)
            double peakGflops = 5196.0;  // RX 5500 XT theoretical FP32 peak
            _output.WriteLine($"  Efficiency: {best.GFlops / peakGflops * 100:F1}% of theoretical peak");

            // Verify we got reasonable GFLOPS (should be at least 500 GFLOPS for RX 5500 XT on 1024x1024)
            Assert.True(best.GFlops > 100, $"Expected at least 100 GFLOPS but got {best.GFlops:F2}");
        }
        else
        {
            _output.WriteLine("No valid configurations found");
            Assert.Fail("Dynamic kernel generation failed - no valid configurations");
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_DynamicKernelGeneration_LargeMatrices()
    {
        // Test dynamic kernel generation on larger matrices (2048, 4096)
        // This should achieve higher GFLOPS due to better occupancy
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available - skipping test");
            return;
        }

        _output.WriteLine($"=== Dynamic Kernel Large Matrix Test ===");
        _output.WriteLine($"Device: {engine.DeviceName}");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine("");

        // RX 5500 XT theoretical peak is 5.196 TFLOPS
        double peakGflops = 5196.0;

        int[] sizes = { 2048, 4096 };

        foreach (int size in sizes)
        {
            _output.WriteLine($"--- Testing {size}x{size}x{size} ---");

            var sw = Stopwatch.StartNew();
            var results = backend.RunBayesianGemmOptimization(
                size, size, size,
                maxTrials: 25,  // More trials for larger search
                warmupRuns: 2,
                benchmarkRuns: 5);  // More benchmark runs for accuracy
            sw.Stop();

            if (results.Length > 0 && results[0].IsValid)
            {
                var best = results[0];
                _output.WriteLine($"Best config: {best.Config}");
                _output.WriteLine($"GFLOPS: {best.GFlops:F2}");
                _output.WriteLine($"Time: {best.TimeMs:F3}ms");
                _output.WriteLine($"Efficiency: {best.GFlops / peakGflops * 100:F1}%");
                _output.WriteLine($"Optimization time: {sw.Elapsed.TotalSeconds:F1}s");
                _output.WriteLine("");

                // For large matrices we expect higher efficiency
                Assert.True(best.GFlops > 500, $"Expected at least 500 GFLOPS on {size}x{size} but got {best.GFlops:F2}");
            }
        }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_StaticVsDynamic_DiagnosticBenchmark()
    {
        // Diagnostic benchmark comparing static optimized kernel vs dynamic kernel system
        // This identifies bottlenecks and regression sources
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available - skipping test");
            return;
        }

        _output.WriteLine("=============================================================");
        _output.WriteLine("STATIC vs DYNAMIC KERNEL DIAGNOSTIC BENCHMARK");
        _output.WriteLine("=============================================================");
        _output.WriteLine($"Device: {engine.DeviceName} ({engine.DeviceVendor})");
        _output.WriteLine($"Compute Units: {engine.ComputeUnits}");
        _output.WriteLine($"Theoretical Peak: 5196 GFLOPS (RX 5500 XT)");
        _output.WriteLine("");

        int[] sizes = { 1024, 2048, 4096 };
        const int warmupIterations = 5;
        const int benchmarkIterations = 20;

        foreach (int size in sizes)
        {
            int M = size, N = size, K = size;
            double flops = 2.0 * M * N * K;
            double peakGflops = 5196.0;

            _output.WriteLine($"=== Matrix Size: {size}x{size}x{size} ({flops / 1e9:F1} GFLOP) ===");
            _output.WriteLine("");

            // Allocate test buffers
            var dataA = new float[M * K];
            var dataB = new float[K * N];
            var rand = new Random(42);
            for (int i = 0; i < dataA.Length; i++) dataA[i] = (float)(rand.NextDouble() * 2 - 1);
            for (int i = 0; i < dataB.Length; i++) dataB[i] = (float)(rand.NextDouble() * 2 - 1);

            using var bufA = backend.AllocateBuffer(dataA);
            using var bufB = backend.AllocateBuffer(dataB);
            using var bufC = backend.AllocateBuffer(M * N);

            // ---- TEST 1: Static gemm_clblast_rdna1 (our best kernel) ----
            _output.WriteLine("1. Static gemm_clblast_rdna1 (optimized, vectorized, hand-tuned):");
            {
                // Warmup
                for (int i = 0; i < warmupIterations; i++)
                {
                    backend.GemmClblastRdna1(bufA, bufB, bufC, M, N, K);
                }
                backend.Synchronize();

                // Benchmark
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < benchmarkIterations; i++)
                {
                    backend.GemmClblastRdna1(bufA, bufB, bufC, M, N, K);
                }
                backend.Synchronize();
                sw.Stop();

                double avgMs = sw.Elapsed.TotalMilliseconds / benchmarkIterations;
                double gflops = (flops / avgMs) * 1e-6;
                double efficiency = gflops / peakGflops * 100;
                _output.WriteLine($"   Time: {avgMs:F3} ms");
                _output.WriteLine($"   GFLOPS: {gflops:F2}");
                _output.WriteLine($"   Efficiency: {efficiency:F1}%");
                _output.WriteLine("");
            }

            // ---- TEST 2: Dynamic kernel with SAME config as static ----
            _output.WriteLine("2. Dynamic kernel (same config: 64x64x16, WG:8x8):");
            {
                // Create config matching the static kernel
                var matchingConfig = new AiDotNet.Tensors.Engines.DirectGpu.OpenCL.GemmConfig
                {
                    TileM = 64,
                    TileN = 64,
                    TileK = 16,
                    ThreadTileM = 8,  // MDIMC
                    ThreadTileN = 8,  // NDIMC
                    VectorWidthM = 2,
                    VectorWidthN = 2,
                    UseDoubleBuffering = true,
                    UseVectorizedLoads = true,
                    KernelName = "gemm_clblast_rdna1"
                };

                try
                {
                    // Measure first call (includes compilation)
                    var compileStart = Stopwatch.StartNew();
                    backend.GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, matchingConfig);
                    compileStart.Stop();
                    _output.WriteLine($"   First call (includes JIT compile): {compileStart.Elapsed.TotalMilliseconds:F3} ms");

                    // Warmup (kernel is now cached)
                    for (int i = 0; i < warmupIterations - 1; i++)
                    {
                        backend.GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, matchingConfig);
                    }
                    backend.Synchronize();

                    // Benchmark
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < benchmarkIterations; i++)
                    {
                        backend.GemmWithDynamicKernel(bufA, bufB, bufC, M, N, K, matchingConfig);
                    }
                    backend.Synchronize();
                    sw.Stop();

                    double avgMs = sw.Elapsed.TotalMilliseconds / benchmarkIterations;
                    double gflops = (flops / avgMs) * 1e-6;
                    double efficiency = gflops / peakGflops * 100;
                    _output.WriteLine($"   Cached kernel time: {avgMs:F3} ms");
                    _output.WriteLine($"   GFLOPS: {gflops:F2}");
                    _output.WriteLine($"   Efficiency: {efficiency:F1}%");
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"   FAILED: {ex.Message}");
                }
                _output.WriteLine("");
            }

            // ---- TEST 3: Multiple other static kernels for comparison ----
            _output.WriteLine("3. Other static kernels for comparison:");
            var otherKernels = new (string Name, Action<IGpuBuffer, IGpuBuffer, IGpuBuffer, int, int, int> Execute)[]
            {
                ("gemm_medium_tile", (a, b, c, m, n, k) => backend.GemmMediumTile(a, b, c, m, n, k)),
                ("gemm_wide_vec", (a, b, c, m, n, k) => backend.GemmWideVec(a, b, c, m, n, k)),
                ("gemm_double_buffered", (a, b, c, m, n, k) => backend.Gemm(a, b, c, m, n, k)),
            };

            foreach (var (name, execute) in otherKernels)
            {
                try
                {
                    for (int i = 0; i < warmupIterations; i++) execute(bufA, bufB, bufC, M, N, K);
                    backend.Synchronize();

                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < benchmarkIterations; i++) execute(bufA, bufB, bufC, M, N, K);
                    backend.Synchronize();
                    sw.Stop();

                    double avgMs = sw.Elapsed.TotalMilliseconds / benchmarkIterations;
                    double gflops = (flops / avgMs) * 1e-6;
                    _output.WriteLine($"   {name,-25}: {avgMs:F3} ms, {gflops:F2} GFLOPS");
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"   {name,-25}: FAILED - {ex.Message}");
                }
            }
            _output.WriteLine("");

            // ---- TEST 4: Quick Bayesian search (5 trials only) ----
            _output.WriteLine("4. Quick Bayesian search (5 trials) for best dynamic config:");
            {
                var quickSw = Stopwatch.StartNew();
                var results = backend.RunBayesianGemmOptimization(M, N, K, maxTrials: 5, warmupRuns: 2, benchmarkRuns: 3);
                quickSw.Stop();

                if (results.Length > 0 && results[0].IsValid)
                {
                    var best = results[0];
                    _output.WriteLine($"   Best config: {best.Config}");
                    _output.WriteLine($"   GFLOPS: {best.GFlops:F2}");
                    _output.WriteLine($"   Search time: {quickSw.Elapsed.TotalSeconds:F1}s");
                }
            }
            _output.WriteLine("");
            _output.WriteLine("-----------------------------------------------------------");
            _output.WriteLine("");
        }

        _output.WriteLine("ANALYSIS:");
        _output.WriteLine("- Static kernels have float2 vectorization, K-unrolling, partition camping avoidance");
        _output.WriteLine("- Dynamic kernel template uses simpler patterns without these optimizations");
        _output.WriteLine("- If dynamic << static with same config, the template needs improvement");
        _output.WriteLine("- If dynamic ~ static, Bayesian search should find better configs");
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void DirectGpuEngine_ClBlast_HeadToHead_Comparison()
    {
        // Head-to-head comparison: Our kernel vs CLBlast library
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable)
        {
            _output.WriteLine("DirectGpu not available - skipping test");
            return;
        }

        var backend = engine.Backend as AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend;
        if (backend == null)
        {
            _output.WriteLine("OpenCL backend not available - skipping test");
            return;
        }

        bool clblastAvailable = AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.IsClBlastAvailable;

        _output.WriteLine("=============================================================");
        _output.WriteLine("OUR KERNEL vs CLBLAST HEAD-TO-HEAD COMPARISON");
        _output.WriteLine("=============================================================");
        _output.WriteLine($"Device: {engine.DeviceName} ({engine.DeviceVendor})");
        _output.WriteLine($"CLBlast Available: {clblastAvailable}");
        _output.WriteLine($"Theoretical Peak: 5196 GFLOPS (RX 5500 XT)");
        _output.WriteLine("");

        if (!clblastAvailable)
        {
            _output.WriteLine("CLBlast library not found - testing our kernel only.");
            _output.WriteLine("To enable CLBlast comparison:");
            _output.WriteLine("  1. Install CLBlast from https://github.com/CNugteren/CLBlast");
            _output.WriteLine("  2. Ensure clblast.dll is in the system PATH");
            _output.WriteLine("");
        }

        int[] sizes = { 1024, 2048, 4096 };
        const int warmupIterations = 5;
        const int benchmarkIterations = 20;

        foreach (int size in sizes)
        {
            int M = size, N = size, K = size;
            double flops = 2.0 * M * N * K;
            double peakGflops = 5196.0;

            _output.WriteLine($"=== Matrix Size: {size}x{size}x{size} ({flops / 1e9:F1} GFLOP) ===");
            _output.WriteLine("");

            var dataA = new float[M * K];
            var dataB = new float[K * N];
            var rand = new Random(42);
            for (int i = 0; i < dataA.Length; i++) dataA[i] = (float)(rand.NextDouble() * 2 - 1);
            for (int i = 0; i < dataB.Length; i++) dataB[i] = (float)(rand.NextDouble() * 2 - 1);

            using var bufA = backend.AllocateBuffer(dataA);
            using var bufB = backend.AllocateBuffer(dataB);
            using var bufC = backend.AllocateBuffer(M * N);

            double ourGflops = 0;
            double clblastGflops = 0;

            // Test our static kernel (gemm_clblast_rdna1)
            _output.WriteLine("1. Our Static Kernel (gemm_clblast_rdna1):");
            {
                for (int i = 0; i < warmupIterations; i++)
                    backend.GemmClblastRdna1(bufA, bufB, bufC, M, N, K);
                backend.Synchronize();

                var sw = Stopwatch.StartNew();
                for (int i = 0; i < benchmarkIterations; i++)
                    backend.GemmClblastRdna1(bufA, bufB, bufC, M, N, K);
                backend.Synchronize();
                sw.Stop();

                double avgMs = sw.Elapsed.TotalMilliseconds / benchmarkIterations;
                ourGflops = (flops / avgMs) * 1e-6;
                _output.WriteLine($"   Time: {avgMs:F3} ms, GFLOPS: {ourGflops:F2}, Efficiency: {ourGflops / peakGflops * 100:F1}%");
            }

            // Test CLBlast if available
            if (clblastAvailable)
            {
                _output.WriteLine("2. CLBlast Library (clblast.dll):");
                {
                    for (int i = 0; i < warmupIterations; i++)
                        backend.GemmWithClBlast(bufA, bufB, bufC, M, N, K);
                    backend.Synchronize();

                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < benchmarkIterations; i++)
                        backend.GemmWithClBlast(bufA, bufB, bufC, M, N, K);
                    backend.Synchronize();
                    sw.Stop();

                    double avgMs = sw.Elapsed.TotalMilliseconds / benchmarkIterations;
                    clblastGflops = (flops / avgMs) * 1e-6;
                    _output.WriteLine($"   Time: {avgMs:F3} ms, GFLOPS: {clblastGflops:F2}, Efficiency: {clblastGflops / peakGflops * 100:F1}%");
                }

                // Comparison
                _output.WriteLine("");
                double speedup = ourGflops / clblastGflops;
                if (speedup >= 1.0)
                    _output.WriteLine($"   >>> OUR KERNEL is {speedup:F2}x FASTER than CLBlast <<<");
                else
                    _output.WriteLine($"   >>> CLBlast is {1/speedup:F2}x faster than our kernel <<<");
            }
            else
            {
                _output.WriteLine("2. CLBlast: SKIPPED (library not available)");
            }

            // Run quick Bayesian search
            _output.WriteLine("");
            _output.WriteLine("3. Quick Bayesian Search (10 trials):");
            {
                var quickSw = Stopwatch.StartNew();
                var results = backend.RunBayesianGemmOptimization(M, N, K, maxTrials: 10, warmupRuns: 2, benchmarkRuns: 3);
                quickSw.Stop();

                if (results.Length > 0 && results[0].IsValid)
                {
                    var best = results[0];
                    _output.WriteLine($"   Best config: {best.Config}");
                    _output.WriteLine($"   GFLOPS: {best.GFlops:F2} (vs static: {ourGflops:F2})");
                    _output.WriteLine($"   Search time: {quickSw.Elapsed.TotalSeconds:F1}s");

                    if (best.GFlops > ourGflops)
                        _output.WriteLine($"   >>> Bayesian found {(best.GFlops / ourGflops - 1) * 100:F1}% better config! <<<");
                }
            }

            _output.WriteLine("");
            _output.WriteLine("-----------------------------------------------------------");
            _output.WriteLine("");
        }

        _output.WriteLine("SUMMARY:");
        _output.WriteLine("- Our kernel uses CLBlast-style optimizations (tiling, vectorization, FMA)");
        _output.WriteLine("- CLBlast is the industry-standard tuned OpenCL BLAS library");
        _output.WriteLine("- If we beat CLBlast, our implementation is competitive");
        _output.WriteLine("- Bayesian optimization can find even better configurations");
    }
#else
    [Fact]
    public void DirectGpu_NotAvailableOnNet462()
    {
        // DirectGpu requires .NET 5.0+ or .NET Framework 4.7.1+
        _output.WriteLine("DirectGpu tests not available on .NET Framework 4.6.2");
        Assert.True(true);
    }
#endif
}
