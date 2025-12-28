using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.StressTests;

#if NET8_0_OR_GREATER

/// <summary>
/// Stress tests for GPU acceleration infrastructure (Phase B: US-GPU-018).
/// Tests long-running operations, concurrent execution, and stability under load.
/// </summary>
/// <remarks>
/// <para><b>Phase B: US-GPU-018 - Stress Testing and Memory Leak Detection</b></para>
/// <para>
/// These tests validate:
/// 1. GPU stability over extended operation periods
/// 2. Memory pool behavior under repeated allocations
/// 3. Concurrent GPU kernel execution
/// 4. No resource leaks over thousands of iterations
/// 5. Performance degradation monitoring
/// </para>
///
/// <para><b>Running Stress Tests:</b></para>
/// <code>
/// dotnet test --filter "FullyQualifiedName~GpuStressTests"
/// </code>
///
/// <para><b>Expected Results:</b></para>
/// - All operations should complete without crashes
/// - Memory should remain stable (no unbounded growth)
/// - Performance should remain consistent across iterations
/// - GPU memory pool should effectively reuse buffers
///
/// <para><b>CI Quarantine:</b></para>
/// This test class is quarantined from CI using [Trait("Category", "GPU")].
/// GPU tests require physical GPU hardware which is not available in CI runners.
/// Run locally with: dotnet test --filter "Category=GPU"
/// </remarks>
[Trait("Category", "GPU")]
public class GpuStressTests
{
    private const int LongRunIterations = 10_000;
    private const int MediumRunIterations = 1_000;
    private const int ShortRunIterations = 100;
    private const int ConcurrentThreads = 8;
    private static bool ShouldRunStressTests =>
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_RUN_STRESS_TESTS"), "1", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_RUN_STRESS_TESTS"), "true", StringComparison.OrdinalIgnoreCase);

    #region Matrix Operation Stress Tests

    [Fact(DisplayName = "GPU Stress: 10K Matrix Multiplications")]
    public void MatrixMultiply_LongRun_10KIterations_NoMemoryLeak()
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
            // GPU not available - skip test
            return;
        }

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var matrixA = CreateRandomMatrix(256, 256);
        var matrixB = CreateRandomMatrix(256, 256);

        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);
        var stopwatch = Stopwatch.StartNew();

        // Act - Run 10,000 iterations
        Matrix<float>? lastResult = null;
        for (int i = 0; i < LongRunIterations; i++)
        {
            lastResult = (Matrix<float>)engine.MatrixMultiply(matrixA, matrixB);
        }

        stopwatch.Stop();
        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert
        Assert.NotNull(lastResult);

        if (lastResult != null)
        {
            Assert.Equal(256, lastResult.Rows);
            Assert.Equal(256, lastResult.Columns);
        }

        // Memory growth should be minimal (< 10MB for 10K iterations)
        // GPU memory pooling should prevent unbounded growth
        Assert.True(memoryGrowth < 10_000_000,
            $"Memory leaked: {memoryGrowth / 1_000_000.0:F2}MB growth over {LongRunIterations} iterations");

        // Performance should remain consistent (no degradation)
        var avgTimePerOp = stopwatch.ElapsedMilliseconds / (double)LongRunIterations;
        Assert.True(avgTimePerOp < 1.0,
            $"Performance degraded: {avgTimePerOp:F4}ms per operation (expected < 1ms)");
    }

    [Fact(DisplayName = "GPU Stress: Concurrent Matrix Operations")]
    public void MatrixMultiply_Concurrent_8Threads_NoRaceConditions()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var matrices = new Matrix<float>[ConcurrentThreads];
        for (int i = 0; i < ConcurrentThreads; i++)
        {
            matrices[i] = CreateRandomMatrix(128, 128);
        }

        var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        var completedCount = 0;

        // Act - Run concurrent operations
        Parallel.For(0, ConcurrentThreads, threadId =>
        {
            try
            {
                for (int i = 0; i < ShortRunIterations; i++)
                {
                    var result = (Matrix<float>)engine.MatrixMultiply(matrices[threadId], matrices[threadId]);
                    Assert.NotNull(result);
                }
                Interlocked.Increment(ref completedCount);
            }
            catch (Exception ex) when (ex is InvalidOperationException
                                       or ArgumentException
                                       or OutOfMemoryException
                                       or DllNotFoundException
                                       or PlatformNotSupportedException)
            {
                exceptions.Add(ex);
            }
        });

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(ConcurrentThreads, completedCount);
    }

    #endregion

    #region Tensor Operation Stress Tests

    [Fact(DisplayName = "GPU Stress: 1K Conv2D Operations")]
    public void Conv2D_LongRun_1KIterations_StablePerformance()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        // Typical CNN layer sizes
        var input = CreateRandomTensor(new[] { 4, 32, 28, 28 });  // batch=4, channels=32, 28x28
        var kernels = CreateRandomTensor(new[] { 64, 32, 3, 3 }); // 64 filters, 3x3

        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);
        var timings = new List<long>();

        // Act - Run 1000 iterations with timing
        for (int i = 0; i < MediumRunIterations; i++)
        {
            var sw = Stopwatch.StartNew();
            var result = (Tensor<float>)engine.Conv2D(input, kernels, stride: 1, padding: 1, dilation: 1);
            sw.Stop();
            timings.Add(sw.ElapsedMilliseconds);

            Assert.NotNull(result);
        }

        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert - Check performance stability
        var firstQuartileAvg = timings.Take(MediumRunIterations / 4).Average();
        var lastQuartileAvg = timings.Skip(3 * MediumRunIterations / 4).Average();

        // Guard against zero division on very fast hardware
        double performanceDrift = 0;
        if (firstQuartileAvg > 0)
        {
            performanceDrift = Math.Abs(lastQuartileAvg - firstQuartileAvg) / firstQuartileAvg;
        }

        // Performance should not degrade by more than 20%
        Assert.True(performanceDrift < 0.20,
            $"Performance degraded by {performanceDrift * 100:F1}% (first: {firstQuartileAvg:F2}ms, last: {lastQuartileAvg:F2}ms)");

        // Memory growth should be minimal
        Assert.True(memoryGrowth < 20_000_000,
            $"Memory leaked: {memoryGrowth / 1_000_000.0:F2}MB growth over {MediumRunIterations} iterations");
    }

    [Fact(DisplayName = "GPU Stress: Pooling Operations Under Load")]
    public void Pooling_HighFrequency_1KIterations_NoLeaks()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var input = CreateRandomTensor(new[] { 8, 64, 56, 56 }); // Large batch for stress testing

        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);

        // Act - Alternate between MaxPool and AvgPool
        for (int i = 0; i < MediumRunIterations; i++)
        {
            if (i % 2 == 0)
            {
                var maxResult = (Tensor<float>)engine.MaxPool2D(input, poolSize: 2, stride: 2, padding: 0);
                Assert.NotNull(maxResult);
            }
            else
            {
                var avgResult = (Tensor<float>)engine.AvgPool2D(input, poolSize: 2, stride: 2, padding: 0);
                Assert.NotNull(avgResult);
            }
        }

        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert
        Assert.True(memoryGrowth < 15_000_000,
            $"Memory leaked: {memoryGrowth / 1_000_000.0:F2}MB growth over {MediumRunIterations} iterations");
    }

    #endregion

    #region Layer Forward Pass Stress Tests

    [Fact(DisplayName = "GPU Stress: ConvolutionalLayer 1K Forward Passes")]
    public void ConvolutionalLayer_LongRun_1KForwardPasses_Stable()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = engine; // Wire GPU context for layers

            var layer = new ConvolutionalLayer<float>(
                inputDepth: 32, outputDepth: 64, kernelSize: 3,
                inputHeight: 28, inputWidth: 28, stride: 1, padding: 1,
                activationFunction: null);

            var input = CreateRandomTensor(new[] { 4, 32, 28, 28 });

            var initialMemory = GC.GetTotalMemory(forceFullCollection: true);

            // Act
            for (int i = 0; i < MediumRunIterations; i++)
            {
                layer.ResetState();
                var output = layer.Forward(input);
                Assert.NotNull(output);
            }

            var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
            var memoryGrowth = finalMemory - initialMemory;

            // Assert
            Assert.True(memoryGrowth < 10_000_000,
                $"Memory leaked: {memoryGrowth / 1_000_000.0:F2}MB growth");
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine; // Restore previous engine
            engine?.Dispose();
        }
    }

    [Fact(DisplayName = "GPU Stress: Full CNN Pipeline 100 Iterations")]
    public void FullCNNPipeline_100Iterations_NoMemoryLeaks()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = engine; // Wire GPU context for layers

            // Build a small CNN: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool
            var conv1 = new ConvolutionalLayer<float>(3, 16, 3, 32, 32, 1, 1, (AiDotNet.Interfaces.IActivationFunction<float>?)null);
            var pool1 = new PoolingLayer<float>(16, 32, 32, 2, 2, PoolingType.Max);
            var conv2 = new ConvolutionalLayer<float>(16, 32, 3, 16, 16, 1, 1, (AiDotNet.Interfaces.IActivationFunction<float>?)null);
            var pool2 = new PoolingLayer<float>(32, 16, 16, 2, 2, PoolingType.Max);

            var input = CreateRandomTensor(new[] { 2, 3, 32, 32 }); // RGB images

            var initialMemory = GC.GetTotalMemory(forceFullCollection: true);
            var stopwatch = Stopwatch.StartNew();

            // Act - Full forward pass pipeline
            for (int i = 0; i < ShortRunIterations; i++)
            {
                conv1.ResetState();
                pool1.ResetState();
                conv2.ResetState();
                pool2.ResetState();

                var out1 = conv1.Forward(input);
                var out2 = pool1.Forward(out1);
                var out3 = conv2.Forward(out2);
                var out4 = pool2.Forward(out3);

                Assert.NotNull(out4);
            }

            stopwatch.Stop();
            var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
            var memoryGrowth = finalMemory - initialMemory;

            // Assert
            Assert.True(memoryGrowth < 30_000_000,
                $"Memory leaked: {memoryGrowth / 1_000_000.0:F2}MB growth over {ShortRunIterations} full pipeline iterations");

            var avgTime = stopwatch.ElapsedMilliseconds / (double)ShortRunIterations;
            Assert.True(avgTime < 100.0,
                $"Pipeline too slow: {avgTime:F2}ms per iteration (expected < 100ms)");
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine; // Restore previous engine
            engine?.Dispose();
        }
    }

    #endregion

    #region GPU Memory Pool Stress Tests

    [Fact(DisplayName = "GPU Stress: Variable Size Allocations")]
    public void MemoryPool_VariableSizeAllocations_ReuseBuffers()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var sizes = new[] { 64, 128, 256, 512, 256, 128, 64 }; // Varied sizes
        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);

        // Act - Repeatedly allocate different sizes
        for (int iteration = 0; iteration < 100; iteration++)
        {
            foreach (var size in sizes)
            {
                var matrixA = CreateRandomMatrix(size, size);
                var matrixB = CreateRandomMatrix(size, size);
                var result = (Matrix<float>)engine.MatrixMultiply(matrixA, matrixB);
                Assert.NotNull(result);
            }
        }

        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert - Memory pool should prevent excessive growth
        Assert.True(memoryGrowth < 5_000_000,
            $"Memory pool not reusing buffers: {memoryGrowth / 1_000_000.0:F2}MB growth");
    }

    [Fact(DisplayName = "GPU Stress: Rapid Allocation/Deallocation Cycles")]
    public void MemoryPool_RapidAllocDealloc_1KCycles_Stable()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var initialMemory = GC.GetTotalMemory(forceFullCollection: true);

        // Act - Rapid allocation and deallocation
        for (int i = 0; i < MediumRunIterations; i++)
        {
            // Allocate
            var matrix = CreateRandomMatrix(128, 128);
            var vector = CreateRandomVector(128);

            // Use
            var result = (Vector<float>)engine.MatrixVectorMultiply(matrix, vector);
            Assert.NotNull(result);

            // Matrices/vectors go out of scope - should be collected
        }

        var finalMemory = GC.GetTotalMemory(forceFullCollection: true);
        var memoryGrowth = finalMemory - initialMemory;

        // Assert
        Assert.True(memoryGrowth < 5_000_000,
            $"Memory not being released: {memoryGrowth / 1_000_000.0:F2}MB growth");
    }

    #endregion

    #region GPU Recovery Stress Tests

    [Fact(DisplayName = "GPU Stress: Error Recovery - Invalid Operations")]
    public void GPU_InvalidOperations_GracefulErrorHandling()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var validMatrix = CreateRandomMatrix(64, 64);
        var incompatibleMatrix = CreateRandomMatrix(32, 32); // Wrong size

        // Act & Assert - Should handle errors gracefully
        Assert.Throws<ArgumentException>(() =>
        {
            engine.MatrixMultiply(validMatrix, incompatibleMatrix);
        });

        // Engine should still work after error
        var result = (Matrix<float>)engine.MatrixMultiply(validMatrix, validMatrix);
        Assert.NotNull(result);
    }

    [Fact(DisplayName = "GPU Stress: Recovery After Multiple Errors")]
    public void GPU_MultipleErrors_ContinuesOperating()
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

        if (!engine.SupportsGpu)
        {
            return; // GPU not available (CPU accelerator selected)
        }

        var validMatrix = CreateRandomMatrix(64, 64);
        var invalidMatrix = CreateRandomMatrix(32, 32);

        // Act - Cause multiple errors
        for (int i = 0; i < 10; i++)
        {
            try
            {
                engine.MatrixMultiply(validMatrix, invalidMatrix);
                Assert.Fail("Should have thrown ArgumentException");
            }
            catch (ArgumentException)
            {
                // Expected
            }

            // Should still work after error
            var result = (Matrix<float>)engine.MatrixMultiply(validMatrix, validMatrix);
            Assert.NotNull(result);
        }

        // Assert - Final operation should work
        var finalResult = (Matrix<float>)engine.MatrixMultiply(validMatrix, validMatrix);
        Assert.NotNull(finalResult);
    }

    #endregion

    #region Helper Methods

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
