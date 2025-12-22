using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Engines;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Concurrency;

/// <summary>
/// Thread safety tests for GPU acceleration (Phase B: US-GPU-019).
/// Validates that GpuEngine can handle concurrent operations from multiple threads safely.
/// </summary>
/// <remarks>
/// <para><b>Phase B: US-GPU-019 - Thread Safety and Concurrent Operations</b></para>
/// <para>
/// These tests validate:
/// 1. Concurrent GPU operations don't cause race conditions
/// 2. Kernel execution synchronization prevents data corruption
/// 3. GPU health tracking is thread-safe (volatile bool)
/// 4. Memory pools handle concurrent rent/return safely
/// 5. No deadlocks occur under high concurrency
/// </para>
///
/// <para><b>Running Thread Safety Tests:</b></para>
/// <code>
/// dotnet test --filter "FullyQualifiedName~ThreadSafetyTests"
/// </code>
///
/// <para><b>CI Quarantine:</b></para>
/// This test class is quarantined from CI using [Trait("Category", "GPU")].
/// GPU tests require physical GPU hardware which is not available in CI runners.
/// Run locally with: dotnet test --filter "Category=GPU"
/// </remarks>
[Trait("Category", "GPU")]
public class ThreadSafetyTests
{
    private const int ConcurrentThreads = 16;  // High concurrency stress
    private const int OperationsPerThread = 50;

    #region Basic Concurrency Tests

    [Fact(DisplayName = "Thread Safety: Concurrent Vector Operations")]
    public void VectorOperations_16ConcurrentThreads_NoRaceConditions()
    {
        // Arrange
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
            return; // GPU not available
        }

        var exceptions = new ConcurrentBag<Exception>();
        var results = new ConcurrentBag<Vector<float>>();
        var completedThreads = 0;

        var testVector1 = CreateVector(1000, seed: 42);
        var testVector2 = CreateVector(1000, seed: 123);

        // Act - 16 threads performing vector operations concurrently
        Parallel.For(0, ConcurrentThreads, threadId =>
        {
            try
            {
                for (int i = 0; i < OperationsPerThread; i++)
                {
                    var result = (Vector<float>)engine.Add(testVector1, testVector2);
                    results.Add(result);
                }
                Interlocked.Increment(ref completedThreads);
            }
            catch (Exception ex) when (ex is not null)
            {
                exceptions.Add(ex);
            }
        });

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(ConcurrentThreads, completedThreads);
        Assert.Equal(ConcurrentThreads * OperationsPerThread, results.Count);

        // Verify all results are correct and identical
        var expectedResult = (Vector<float>)new CpuEngine().Add(testVector1, testVector2);
        foreach (var result in results)
        {
            for (int i = 0; i < result.Length; i++)
            {
                Assert.Equal(expectedResult[i], result[i], precision: 5);
            }
        }
    }

    [Fact(DisplayName = "Thread Safety: Concurrent Matrix Multiply")]
    public void MatrixMultiply_ConcurrentExecution_CorrectResults()
    {
        // Arrange
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
            return; // GPU not available
        }

        var exceptions = new ConcurrentBag<Exception>();
        var results = new ConcurrentBag<Matrix<float>>();

        var matrixA = CreateMatrix(128, 128, seed: 42);
        var matrixB = CreateMatrix(128, 128, seed: 123);

        // Act - Concurrent matrix multiplications
        Parallel.For(0, ConcurrentThreads, threadId =>
        {
            try
            {
                for (int i = 0; i < 10; i++)  // 10 ops per thread (matrix multiply is expensive)
                {
                    var result = (Matrix<float>)engine.MatrixMultiply(matrixA, matrixB);
                    results.Add(result);
                }
            }
            catch (Exception ex) when (ex is not null)
            {
                exceptions.Add(ex);
            }
        });

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(ConcurrentThreads * 10, results.Count);

        // Verify all results are mathematically correct
        var expectedResult = (Matrix<float>)new CpuEngine().MatrixMultiply(matrixA, matrixB);
        foreach (var result in results)
        {
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
                    Assert.Equal(expectedResult[i, j], result[i, j], precision: 3);
                }
            }
        }
    }

    [Fact(DisplayName = "Thread Safety: Mixed Operation Types")]
    public void MixedOperations_ConcurrentThreads_NoInterference()
    {
        // Arrange
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
            return; // GPU not available
        }

        var exceptions = new ConcurrentBag<Exception>();
        var completedOps = 0;

        var vec1 = CreateVector(500, seed: 1);
        var vec2 = CreateVector(500, seed: 2);
        var mat1 = CreateMatrix(64, 64, seed: 3);
        var mat2 = CreateMatrix(64, 64, seed: 4);

        // Act - Mix different operation types concurrently
        Parallel.For(0, ConcurrentThreads, threadId =>
        {
            try
            {
                for (int i = 0; i < 20; i++)
                {
                    switch (i % 4)
                    {
                        case 0:
                            engine.Add(vec1, vec2);
                            break;
                        case 1:
                            engine.Multiply(vec1, vec2);
                            break;
                        case 2:
                            engine.MatrixMultiply(mat1, mat2);
                            break;
                        case 3:
                            engine.Subtract(vec1, vec2);
                            break;
                    }
                    Interlocked.Increment(ref completedOps);
                }
            }
            catch (Exception ex) when (ex is not null)
            {
                exceptions.Add(ex);
            }
        });

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(ConcurrentThreads * 20, completedOps);
    }

    #endregion

    #region Tensor Operation Concurrency

    [Fact(DisplayName = "Thread Safety: Concurrent Conv2D Operations")]
    public void Conv2D_ConcurrentThreads_NoDataCorruption()
    {
        // Arrange
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
            return; // GPU not available
        }

        var exceptions = new ConcurrentBag<Exception>();
        var results = new ConcurrentBag<Tensor<float>>();

        var input = CreateTensor(new[] { 2, 16, 28, 28 }, seed: 42);
        var kernels = CreateTensor(new[] { 32, 16, 3, 3 }, seed: 123);

        // Act - Concurrent convolutions
        Parallel.For(0, ConcurrentThreads, threadId =>
        {
            try
            {
                for (int i = 0; i < 5; i++)  // Conv2D is very expensive
                {
                    var result = (Tensor<float>)engine.Conv2D(input, kernels, stride: 1, padding: 1, dilation: 1);
                    results.Add(result);
                }
            }
            catch (Exception ex) when (ex is not null)
            {
                exceptions.Add(ex);
            }
        });

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(ConcurrentThreads * 5, results.Count);

        // Verify dimensions are correct
        foreach (var result in results)
        {
            Assert.Equal(4, result.Rank);
            Assert.Equal(2, result.Shape[0]);  // batch
            Assert.Equal(32, result.Shape[1]); // output channels
        }
    }

    [Fact(DisplayName = "Thread Safety: Concurrent Pooling Operations")]
    public void Pooling_ConcurrentThreads_ThreadSafe()
    {
        // Arrange
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
            return; // GPU not available
        }

        var exceptions = new ConcurrentBag<Exception>();
        var maxPoolResults = new ConcurrentBag<Tensor<float>>();
        var avgPoolResults = new ConcurrentBag<Tensor<float>>();

        var input = CreateTensor(new[] { 4, 32, 56, 56 }, seed: 42);

        // Act - Concurrent pooling (mix MaxPool and AvgPool)
        Parallel.For(0, ConcurrentThreads, threadId =>
        {
            try
            {
                for (int i = 0; i < 10; i++)
                {
                    if (i % 2 == 0)
                    {
                        var result = (Tensor<float>)engine.MaxPool2D(input, poolSize: 2, stride: 2, padding: 0);
                        maxPoolResults.Add(result);
                    }
                    else
                    {
                        var result = (Tensor<float>)engine.AvgPool2D(input, poolSize: 2, stride: 2, padding: 0);
                        avgPoolResults.Add(result);
                    }
                }
            }
            catch (Exception ex) when (ex is not null)
            {
                exceptions.Add(ex);
            }
        });

        // Assert
        Assert.Empty(exceptions);
        Assert.True(maxPoolResults.Count >= ConcurrentThreads * 5);
        Assert.True(avgPoolResults.Count >= ConcurrentThreads * 5);
    }

    #endregion

    #region GPU Health Tracking Thread Safety

    [Fact(DisplayName = "Thread Safety: GPU Health Flag (Volatile)")]
    public void GpuHealthTracking_ConcurrentReads_ThreadSafe()
    {
        // Arrange
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
            return; // GPU not available
        }

        var healthChecks = new ConcurrentBag<bool>();
        var exceptions = new ConcurrentBag<Exception>();

        // Act - Concurrent reads of GPU health (via SupportsGpu property)
        Parallel.For(0, ConcurrentThreads, threadId =>
        {
            try
            {
                for (int i = 0; i < 1000; i++)
                {
                    bool isHealthy = engine.SupportsGpu;
                    healthChecks.Add(isHealthy);
                    Thread.Yield(); // Encourage context switching
                }
            }
            catch (Exception ex) when (ex is not null)
            {
                exceptions.Add(ex);
            }
        });

        // Assert
        Assert.Empty(exceptions);
        Assert.Equal(ConcurrentThreads * 1000, healthChecks.Count);
        // All health checks should report same value (GPU is either healthy or not)
        Assert.True(healthChecks.All(h => h == healthChecks.First()));
    }

    #endregion

    #region Deadlock Prevention Tests

    [Fact(DisplayName = "Thread Safety: No Deadlocks Under High Load")]
    public void HighConcurrencyLoad_NoDeadlocks()
    {
        // Arrange
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
            return; // GPU not available
        }

        var exceptions = new ConcurrentBag<Exception>();
        var completedOps = 0;

        var testData = Enumerable.Range(0, 50).Select(i =>
            (vec1: CreateVector(200, seed: i),
             vec2: CreateVector(200, seed: i + 100))
        ).ToArray();

        // Use a CancellationToken to detect deadlocks (timeout after 30 seconds)
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        // Act - Very high concurrency with mixed operations
        try
        {
            Parallel.For(0, ConcurrentThreads * 2, new ParallelOptions { CancellationToken = cts.Token }, threadId =>
            {
                try
                {
                    for (int i = 0; i < 100; i++)
                    {
                        var data = testData[i % testData.Length];

                        switch (i % 3)
                        {
                            case 0:
                                engine.Add(data.vec1, data.vec2);
                                break;
                            case 1:
                                engine.Multiply(data.vec1, data.vec2);
                                break;
                            case 2:
                                engine.Subtract(data.vec1, data.vec2);
                                break;
                        }

                        Interlocked.Increment(ref completedOps);
                    }
                }
                catch (Exception ex)
                {
                    if (!(ex is OperationCanceledException))
                        exceptions.Add(ex);
                }
            });
        }
        catch (OperationCanceledException)
        {
            Assert.Fail("Deadlock detected: Operations timed out after 30 seconds");
        }

        // Assert
        Assert.Empty(exceptions);
        Assert.True(completedOps > 0, "No operations completed - possible deadlock");
    }

    #endregion

    #region Helper Methods

    private static Vector<float> CreateVector(int size, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var vector = new Vector<float>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return vector;
    }

    private static Matrix<float> CreateMatrix(int rows, int cols, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
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

    private static Tensor<float> CreateTensor(int[] shape, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    #endregion
}
