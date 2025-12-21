using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Engines;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.StressTests;

#if NET8_0_OR_GREATER

/// <summary>
/// Memory leak detection tests for GPU acceleration (Phase B: US-GPU-018).
/// Validates that GPU memory pools, managed memory, and native resources are properly released.
/// </summary>
/// <remarks>
/// <para><b>Phase B: US-GPU-018 - Memory Leak Detection</b></para>
/// <para>
/// These tests monitor memory usage over extended operation periods to detect:
/// 1. Managed heap growth (GC pressure)
/// 2. GPU memory pool leaks
/// 3. Native resource leaks (GPU buffers, kernels)
/// 4. Finalizer queue buildup
/// </para>
///
/// <para><b>Memory Leak Criteria:</b></para>
/// - Managed memory growth &lt; 10MB per 1000 iterations
/// - Memory growth rate should plateau (not linear)
/// - GC collections should remain bounded
/// - No unbounded resource accumulation
///
/// <para><b>IMPORTANT: Test Isolation</b></para>
/// These tests use process-wide GC metrics (GC.GetTotalMemory, GC.CollectionCount) which can be
/// affected by parallel test execution. For reliable results, run these tests in isolation or
/// disable parallel execution. Tests are marked with [Trait("Category", "Stress")] for filtering.
///
/// <para><b>Running Memory Leak Tests:</b></para>
/// <code>
/// dotnet test --filter "FullyQualifiedName~MemoryLeakTests"
/// </code>
///
/// <para><b>CI Quarantine:</b></para>
/// This test class is quarantined from CI using [Trait("Category", "GPU")].
/// GPU tests require physical GPU hardware which is not available in CI runners.
/// Run locally with: dotnet test --filter "Category=GPU"
/// </remarks>
[Trait("Category", "Stress")]
[Trait("Category", "GPU")]
public class MemoryLeakTests
{
    private const int LeakDetectionIterations = 5_000;
    private const int SamplingInterval = 500;
    private const long MaxAcceptableMemoryGrowth = 15_000_000; // 15MB
    private static bool ShouldRunStressTests =>
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_RUN_STRESS_TESTS"), "1", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_RUN_STRESS_TESTS"), "true", StringComparison.OrdinalIgnoreCase);

    #region Memory Growth Analysis

    [Fact(DisplayName = "Memory Leak: Matrix Operations Growth Analysis")]
    public void MatrixOperations_5KIterations_LinearGrowthCheck()
    {
        // Arrange
        if (!ShouldRunStressTests)
        {
            return;
        }

        GpuEngine? engine = null;
        try
        {
            engine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            return; // GPU not available
        }

        using var disposableEngine = engine!;

        if (!disposableEngine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var matrixA = CreateRandomMatrix(256, 256);
        var matrixB = CreateRandomMatrix(256, 256);

        var memorySnapshots = new List<MemorySnapshot>();

        // Act - Sample memory every 500 iterations
        for (int i = 0; i < LeakDetectionIterations; i++)
        {
            var result = (Matrix<float>)disposableEngine.MatrixMultiply(matrixA, matrixB);
            Assert.NotNull(result);

            if (i % SamplingInterval == 0)
            {
                memorySnapshots.Add(new MemorySnapshot
                {
                    Iteration = i,
                    ManagedMemory = GC.GetTotalMemory(forceFullCollection: true),
                    Gen0Collections = GC.CollectionCount(0),
                    Gen1Collections = GC.CollectionCount(1),
                    Gen2Collections = GC.CollectionCount(2)
                });
            }
        }

        // Assert - Analyze memory growth pattern
        var firstSnapshot = memorySnapshots.First();
        var lastSnapshot = memorySnapshots.Last();

        var totalGrowth = lastSnapshot.ManagedMemory - firstSnapshot.ManagedMemory;

        // Check for linear growth (leak indicator)
        var isLinearGrowth = AnalyzeGrowthPattern(memorySnapshots);

        Assert.False(isLinearGrowth,
            $"Detected linear memory growth (leak suspected). Total growth: {totalGrowth / 1_000_000.0:F2}MB");

        Assert.True(totalGrowth < MaxAcceptableMemoryGrowth,
            $"Excessive memory growth: {totalGrowth / 1_000_000.0:F2}MB over {LeakDetectionIterations} iterations");
    }

    [Fact(DisplayName = "Memory Leak: Tensor Operations Growth Analysis")]
    public void TensorOperations_5KIterations_PlateauCheck()
    {
        // Arrange
        if (!ShouldRunStressTests)
        {
            return;
        }

        GpuEngine? engine = null;
        try
        {
            engine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            return; // GPU not available
        }

        using var disposableEngine = engine!;

        if (!disposableEngine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var input = CreateRandomTensor(new[] { 4, 32, 28, 28 });
        var kernels = CreateRandomTensor(new[] { 64, 32, 3, 3 });

        var memorySnapshots = new List<MemorySnapshot>();

        // Act - Sample memory periodically
        for (int i = 0; i < LeakDetectionIterations; i++)
        {
            var result = (Tensor<float>)disposableEngine.Conv2D(input, kernels, 1, 1, 1);
            Assert.NotNull(result);

            if (i % SamplingInterval == 0)
            {
                memorySnapshots.Add(new MemorySnapshot
                {
                    Iteration = i,
                    ManagedMemory = GC.GetTotalMemory(forceFullCollection: true),
                    Gen0Collections = GC.CollectionCount(0),
                    Gen1Collections = GC.CollectionCount(1),
                    Gen2Collections = GC.CollectionCount(2)
                });
            }
        }

        // Assert - Memory should plateau after initial allocations
        var hasPlateau = DetectMemoryPlateau(memorySnapshots);

        Assert.True(hasPlateau,
            "Memory did not plateau (leak suspected). Expected stable memory after warmup period.");
    }

    [Fact(DisplayName = "Memory Leak: GC Pressure Analysis")]
    public void GpuOperations_GCPressure_BoundedCollections()
    {
        // Arrange
        if (!ShouldRunStressTests)
        {
            return;
        }

        GpuEngine? engine = null;
        try
        {
            engine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            return; // GPU not available
        }

        using var disposableEngine = engine!;

        if (!disposableEngine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var matrix = CreateRandomMatrix(128, 128);
        var vector = CreateRandomVector(128);

        var initialGen0 = GC.CollectionCount(0);
        var initialGen1 = GC.CollectionCount(1);
        var initialGen2 = GC.CollectionCount(2);

        // Act - Run many operations
        for (int i = 0; i < LeakDetectionIterations; i++)
        {
            var result = (Vector<float>)disposableEngine.MatrixVectorMultiply(matrix, vector);
            Assert.NotNull(result);
        }

        var finalGen0 = GC.CollectionCount(0);
        var finalGen1 = GC.CollectionCount(1);
        var finalGen2 = GC.CollectionCount(2);

        var gen0Delta = finalGen0 - initialGen0;
        var gen1Delta = finalGen1 - initialGen1;
        var gen2Delta = finalGen2 - initialGen2;

        // Assert - GC collections should be bounded
        // GPU memory pooling should minimize Gen 0 collections
        Assert.True(gen0Delta < 100,
            $"Excessive Gen 0 collections: {gen0Delta} (indicates poor memory pooling)");

        Assert.True(gen1Delta < 20,
            $"Excessive Gen 1 collections: {gen1Delta}");

        Assert.True(gen2Delta < 5,
            $"Excessive Gen 2 collections: {gen2Delta} (indicates memory leaks)");
    }

    #endregion

    #region Optimizer Memory Leak Tests

    [Fact(DisplayName = "Memory Leak: Optimizer Vector Updates")]
    public void OptimizerVectorUpdates_5KIterations_NoLeak()
    {
        // Arrange
        if (!ShouldRunStressTests)
        {
            return;
        }

        IEngine? engine = null;
        try
        {
            engine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            engine = new CpuEngine(); // Fallback to CPU
        }

        var parameters = CreateRandomVector(10000); // Large parameter vector
        var gradient = CreateRandomVector(10000);

        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);

        // Act - Simulate optimizer updates
        for (int i = 0; i < LeakDetectionIterations; i++)
        {
            // Simulate gradient update: params = params - 0.01 * gradient
            var scaledGrad = (Vector<float>)engine.Multiply(gradient, 0.01f);
            parameters = (Vector<float>)engine.Subtract(parameters, scaledGrad);
        }

        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert
        Assert.True(memoryGrowth < 10_000_000,
            $"Optimizer leaked memory: {memoryGrowth / 1_000_000.0:F2}MB over {LeakDetectionIterations} updates");
    }

    [Fact(DisplayName = "Memory Leak: Mixed Precision Operations")]
    public void MixedPrecisionOperations_5KIterations_NoLeak()
    {
        // Arrange
        if (!ShouldRunStressTests)
        {
            return;
        }

        GpuEngine? engine = null;
        try
        {
            engine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            return; // GPU not available
        }

        using var disposableEngine = engine!;

        if (!disposableEngine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var memorySnapshots = new List<long>();

        // Act - Mix different operation types
        for (int i = 0; i < LeakDetectionIterations; i++)
        {
            // Alternate between different operation types and sizes
            switch (i % 4)
            {
                case 0:
                    var m1 = CreateRandomMatrix(64, 64);
                    var r1 = (Matrix<float>)disposableEngine.MatrixMultiply(m1, m1);
                    Assert.NotNull(r1);
                    break;

                case 1:
                    var v1 = CreateRandomVector(128);
                    var v2 = CreateRandomVector(128);
                    var r2 = (Vector<float>)disposableEngine.Add(v1, v2);
                    Assert.NotNull(r2);
                    break;

                case 2:
                    var t1 = CreateRandomTensor(new[] { 2, 16, 14, 14 });
                    var r3 = (Tensor<float>)disposableEngine.MaxPool2D(t1, 2, 2, 0);
                    Assert.NotNull(r3);
                    break;

                case 3:
                    var m2 = CreateRandomMatrix(128, 64);
                    var v3 = CreateRandomVector(64);
                    var r4 = (Vector<float>)disposableEngine.MatrixVectorMultiply(m2, v3);
                    Assert.NotNull(r4);
                    break;
            }

            if (i % SamplingInterval == 0)
            {
                memorySnapshots.Add(GC.GetTotalMemory(forceFullCollection: true));
            }
        }

        // Assert - Check memory growth trend
        var firstHalfAvg = memorySnapshots.Take(memorySnapshots.Count / 2).Average();
        var secondHalfAvg = memorySnapshots.Skip(memorySnapshots.Count / 2).Average();
        var growthRate = (secondHalfAvg - firstHalfAvg) / firstHalfAvg;

        Assert.True(growthRate < 0.20,
            $"Excessive memory growth rate: {growthRate * 100:F1}% between halves (leak suspected)");
    }

    #endregion

    #region Resource Cleanup Tests

    [Fact(DisplayName = "Memory Leak: Engine Disposal Cleanup")]
    public void GpuEngine_MultipleCreateDispose_NoResourceLeak()
    {
        if (!ShouldRunStressTests)
        {
            return;
        }

        // Skip if GPU not available
        GpuEngine? probeEngine = null;
        try
        {
            probeEngine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            return; // GPU not available
        }

        if (!probeEngine.SupportsGpu)
        {
            probeEngine.Dispose();
            return; // GPU not available (CPU accelerator selected)
        }

        probeEngine.Dispose();

        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);

        // Act - Create and destroy engines multiple times
        for (int i = 0; i < 100; i++)
        {
            GpuEngine? engine = null;
            try
            {
                engine = new GpuEngine();

                // Use the engine briefly
                var matrix = CreateRandomMatrix(64, 64);
                var result = (Matrix<float>)engine.MatrixMultiply(matrix, matrix);
                Assert.NotNull(result);
            }
            catch (Exception ex) when (ex is InvalidOperationException
                                       or OutOfMemoryException
                                       or ArgumentException
                                       or DllNotFoundException
                                       or PlatformNotSupportedException)
            {
                // GPU might fail - that's ok for this test
            }
            finally
            {
                engine?.Dispose();
            }
        }

        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert - Multiple engine creation/disposal should not leak
        Assert.True(memoryGrowth < 20_000_000,
            $"Engine disposal leaked memory: {memoryGrowth / 1_000_000.0:F2}MB");
    }

    [Fact(DisplayName = "Memory Leak: Tensor Lifecycle Management")]
    public void Tensor_CreateUseDiscard_5KCycles_NoLeak()
    {
        // Arrange
        if (!ShouldRunStressTests)
        {
            return;
        }

        GpuEngine? engine = null;
        try
        {
            engine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            return; // GPU not available
        }

        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);

        // Act - Create, use, and discard tensors repeatedly
        for (int i = 0; i < LeakDetectionIterations; i++)
        {
            // Create new tensors each iteration
            var input = CreateRandomTensor(new[] { 2, 16, 28, 28 });
            var kernels = CreateRandomTensor(new[] { 32, 16, 3, 3 });

            var result = (Tensor<float>)engine.Conv2D(input, kernels, 1, 1, 1);
            Assert.NotNull(result);

            // Tensors go out of scope - should be collected
        }

        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert
        Assert.True(memoryGrowth < MaxAcceptableMemoryGrowth,
            $"Tensor lifecycle leaked memory: {memoryGrowth / 1_000_000.0:F2}MB");
    }

    #endregion

    #region Helper Methods and Classes

    private class MemorySnapshot
    {
        public int Iteration { get; set; }
        public long ManagedMemory { get; set; }
        public int Gen0Collections { get; set; }
        public int Gen1Collections { get; set; }
        public int Gen2Collections { get; set; }
    }

    /// <summary>
    /// Analyzes memory snapshots to detect linear growth (leak indicator).
    /// Returns true if memory growth is linear, false if it plateaus.
    /// </summary>
    private bool AnalyzeGrowthPattern(List<MemorySnapshot> snapshots)
    {
        if (snapshots.Count < 3)
            return false;

        // Calculate correlation coefficient between iteration and memory
        var n = snapshots.Count;
        var sumX = snapshots.Sum(s => (double)s.Iteration);
        var sumY = snapshots.Sum(s => (double)s.ManagedMemory);
        var sumXY = snapshots.Sum(s => (double)s.Iteration * s.ManagedMemory);
        var sumX2 = snapshots.Sum(s => (double)s.Iteration * s.Iteration);
        var sumY2 = snapshots.Sum(s => (double)s.ManagedMemory * s.ManagedMemory);

        var correlation = (n * sumXY - sumX * sumY) /
                         Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        // High correlation (> 0.8) indicates linear growth (leak)
        return Math.Abs(correlation) > 0.8;
    }

    /// <summary>
    /// Detects if memory usage plateaus (stabilizes) after initial allocations.
    /// Returns true if plateau detected, false if continuous growth.
    /// </summary>
    private bool DetectMemoryPlateau(List<MemorySnapshot> snapshots)
    {
        if (snapshots.Count < 4)
            return false;

        // Split into quarters
        var quarterSize = snapshots.Count / 4;
        var firstQuarter = snapshots.Take(quarterSize).ToList();
        var lastQuarter = snapshots.Skip(3 * quarterSize).ToList();

        var firstAvg = firstQuarter.Average(s => s.ManagedMemory);
        var lastAvg = lastQuarter.Average(s => s.ManagedMemory);

        // Calculate variance in last quarter
        var lastVariance = lastQuarter.Average(s => Math.Pow(s.ManagedMemory - lastAvg, 2));
        var lastStdDev = Math.Sqrt(lastVariance);

        // Plateau: last quarter has low variance and modest growth from first quarter
        var growthRate = (lastAvg - firstAvg) / firstAvg;
        var hasLowVariance = lastStdDev / lastAvg < 0.10; // < 10% coefficient of variation
        var hasModestGrowth = growthRate < 0.50; // < 50% growth from start

        return hasLowVariance && hasModestGrowth;
    }

    private static Matrix<float> CreateRandomMatrix(int rows, int cols)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var matrix = new Matrix<float>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = (float)(random.NextDouble() * 2 - 1);
            }
        }
        return matrix;
    }

    private static Vector<float> CreateRandomVector(int size)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var vector = new Vector<float>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return vector;
    }

    private static Tensor<float> CreateRandomTensor(int[] shape)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    #endregion
}
#endif
