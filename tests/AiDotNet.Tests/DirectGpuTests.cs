// Copyright (c) AiDotNet. All rights reserved.
// Tests for DirectGpu engine functionality

using System;
using System.Diagnostics;
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

            _output.WriteLine($"Size {size}x{size}: {avgMs:F3}ms avg, {gflops:F1} GFLOPS (kernel only)");
        }
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
