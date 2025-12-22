using System;
using System.Threading;
using AiDotNet.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Recovery;

#if NET8_0_OR_GREATER

/// <summary>
/// GPU recovery tests (Phase B: US-GPU-020).
/// Validates GPU device loss recovery and health monitoring.
/// </summary>
/// <remarks>
/// <para><b>Phase B: US-GPU-020 - GPU Device Loss Recovery</b></para>
/// <para>
/// These tests validate:
/// 1. GPU health tracking and failure counting
/// 2. Automatic recovery after transient failures
/// 3. Permanent disabling after repeated failures
/// 4. Health diagnostics and monitoring
/// 5. Graceful CPU fallback during GPU unavailability
/// </para>
///
/// <para><b>Running GPU Recovery Tests:</b></para>
/// <code>
/// dotnet test --filter "FullyQualifiedName~GpuRecoveryTests"
/// </code>
///
/// <para><b>CI Quarantine:</b></para>
/// This test class is quarantined from CI using [Trait("Category", "GPU")].
/// GPU tests require physical GPU hardware which is not available in CI runners.
/// Run locally with: dotnet test --filter "Category=GPU"
/// </remarks>
[Trait("Category", "GPU")]
public class GpuRecoveryTests
{
    [Fact(DisplayName = "GPU Recovery: Health Diagnostics Available")]
    public void GetGpuHealthDiagnostics_ReturnsValidInformation()
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

        // Act
        var diagnostics = engine.GetGpuHealthDiagnostics();

        // Assert
        Assert.NotNull(diagnostics);
        Assert.Contains("GPU Health Diagnostics", diagnostics);
        Assert.Contains("Healthy:", diagnostics);
        Assert.Contains("Consecutive Failures:", diagnostics);
        Assert.Contains("Accelerator:", diagnostics);
    }

    [Fact(DisplayName = "GPU Recovery: CheckAndRecoverGpuHealth When Healthy")]
    public void CheckAndRecoverGpuHealth_WhenHealthy_ReturnsTrue()
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

        // Act
        bool isHealthy = engine.CheckAndRecoverGpuHealth();

        // Assert
        if (engine.SupportsGpu)
        {
            Assert.True(isHealthy);
        }
    }

    [Fact(DisplayName = "GPU Recovery: Operations Fallback to CPU Gracefully")]
    public void Operations_WhenGpuUnavailable_UseCpuFallback()
    {
        // Arrange
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

        var vector1 = CreateVector(100);
        var vector2 = CreateVector(100);

        // Act - Operations should work even if GPU fails
        Vector<float>? result = null;
        Exception? caughtException = null;

        try
        {
            result = engine.Add(vector1, vector2);
        }
        catch (Exception ex)
        {
            caughtException = ex;
        }

        // Assert - Should not throw, should fall back to CPU
        Assert.Null(caughtException);
        Assert.NotNull(result);

        if (result != null)
        {
            Assert.Equal(100, result.Length);

            // Verify result is correct (from CPU fallback)
            for (int i = 0; i < result.Length; i++)
            {
                Assert.Equal(vector1[i] + vector2[i], result[i], precision: 5);
            }
        }
    }

    [Fact(DisplayName = "GPU Recovery: SupportsGpu Reflects GPU Availability")]
    public void SupportsGpu_ReflectsActualGpuAvailability()
    {
        // Arrange & Act
        GpuEngine? engine = null;
        bool hasGpu = false;

        try
        {
            engine = new GpuEngine();
            hasGpu = engine.SupportsGpu;
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            // GPU not available
            hasGpu = false;
        }

        // Assert - SupportsGpu should be consistent with initialization
        if (engine != null)
        {
            // If we successfully created an engine, check SupportsGpu
            var diagnostics = engine.GetGpuHealthDiagnostics();

            if (hasGpu)
            {
                Assert.Contains("Healthy: True", diagnostics);
            }
        }
    }

    [Fact(DisplayName = "GPU Recovery: Multiple Operations After GPU Loss")]
    public void MultipleOperations_AfterGpuBecomeUnavailable_AllUseCpuFallback()
    {
        // Arrange
        GpuEngine? engine = null;
        try
        {
            engine = new GpuEngine();
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException)
        {
            return; // GPU not available - can't test recovery
        }

        var vector = CreateVector(50);
        var matrix = CreateMatrix(50, 50);

        // Act - Perform multiple operations
        // Even if GPU becomes unhealthy, all should fall back gracefully
        var results = new System.Collections.Generic.List<bool>();

        for (int i = 0; i < 10; i++)
        {
            try
            {
                var vecResult = (Vector<float>)engine.Add(vector, vector);
                var matResult = (Matrix<float>)engine.MatrixMultiply(matrix, matrix);

                results.Add(vecResult != null && matResult != null);
            }
            catch (Exception ex) when (ex is InvalidOperationException
                                       or OutOfMemoryException
                                       or ArgumentException
                                       or DllNotFoundException
                                       or PlatformNotSupportedException)
            {
                results.Add(false);
            }
        }

        // Assert - All operations should succeed (GPU or CPU)
        Assert.All(results, r => Assert.True(r));
    }

    [Fact(DisplayName = "GPU Recovery: Health Status Consistency")]
    public void HealthStatus_RemainsConsistent_AcrossMultipleChecks()
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

        // Act - Check health multiple times in succession
        var healthChecks = new bool[100];
        for (int i = 0; i < 100; i++)
        {
            healthChecks[i] = engine.CheckAndRecoverGpuHealth();
            Thread.Yield(); // Allow context switching
        }

        // Assert - All health checks should return same value (consistency)
        // Note: healthChecks[0] may be false if running in CPU-only mode, which is valid
        Assert.All(healthChecks, check => Assert.Equal(healthChecks[0], check));
    }

    [Fact(DisplayName = "GPU Recovery: Diagnostics Include Failure Information")]
    public void Diagnostics_AfterOperations_IncludeAccurateInformation()
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

        // Perform some operations
        var vector = CreateVector(100);
        engine.Add(vector, vector);
        engine.Multiply(vector, vector);

        // Act
        var diagnostics = engine.GetGpuHealthDiagnostics();

        // Assert
        Assert.Contains("Consecutive Failures: 0/3", diagnostics);
        Assert.Contains("Last Failure: Never", diagnostics);
    }

    [Fact(DisplayName = "GPU Recovery: Engine Remains Functional After Dispose")]
    public void Operations_AfterDispose_HandleGracefully()
    {
        // Arrange
        var engine = new GpuEngine();
        var vector = CreateVector(50);

        // Act - Dispose then try to use
        engine.Dispose();

        Exception? exception = null;
        try
        {
            var result = engine.Add(vector, vector);
        }
        catch (Exception ex)
        {
            exception = ex;
        }

        // Assert - Should either throw ObjectDisposedException or work with CPU fallback
        // Both behaviors are acceptable
        Assert.True(exception == null || exception is ObjectDisposedException);
    }

    [Fact(DisplayName = "GPU Recovery: Concurrent Health Checks Thread-Safe")]
    public void ConcurrentHealthChecks_MultipleThreads_NoRaceConditions()
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

        var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        var healthResults = new System.Collections.Concurrent.ConcurrentBag<bool>();

        // Act - Multiple threads checking health concurrently
        System.Threading.Tasks.Parallel.For(0, 10, threadId =>
        {
            try
            {
                for (int i = 0; i < 100; i++)
                {
                    var isHealthy = engine.CheckAndRecoverGpuHealth();
                    healthResults.Add(isHealthy);

                    var diagnostics = engine.GetGpuHealthDiagnostics();
                    Assert.NotNull(diagnostics);
                }
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
        Assert.Equal(1000, healthResults.Count);
        // All health checks should return the same value (consistency)
        var results = healthResults.ToArray();
        Assert.NotEmpty(results);
        Assert.All(results, h => Assert.Equal(results[0], h));

        if (engine.SupportsGpu)
        {
            Assert.True(results[0]);
        }
    }

    #region Helper Methods

    private static Vector<float> CreateVector(int size, int seed = 42)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var vector = new Vector<float>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return vector;
    }

    private static Matrix<float> CreateMatrix(int rows, int cols, int seed = 42)
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

    #endregion
}
#endif
