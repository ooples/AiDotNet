using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Engines;

/// <summary>
/// Integration tests for GPU execution options.
/// </summary>
/// <remarks>
/// <para>
/// These tests verify that GPU execution options:
/// 1. Have sensible default values
/// 2. Validate correctly
/// 3. Clone properly
/// 4. Handle environment variable configuration
/// </para>
/// <para><b>For Beginners:</b> GPU execution options control low-level GPU behavior
/// like stream counts, memory limits, and execution modes. These tests ensure
/// the options work correctly for different configurations.
/// </para>
/// </remarks>
public class GpuExecutionOptionsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Default Values Tests

    [Fact]
    public void GpuExecutionOptions_DefaultConstructor_HasExpectedDefaults()
    {
        // Act
        var options = new GpuExecutionOptions();

        // Assert
        Assert.Equal(4096, options.MinGpuElements);
        Assert.Equal(3, options.MaxComputeStreams);
        Assert.False(options.ForceGpu);
        Assert.False(options.ForceCpu);
        Assert.True(options.EnableGraphCompilation);
        Assert.True(options.EnableAutoFusion);
        Assert.Equal(0.8, options.MaxMemoryUsage, Tolerance);
        Assert.True(options.EnablePrefetch);
        Assert.True(options.EnableComputeTransferOverlap);
        Assert.Equal(GpuExecutionMode.Auto, options.ExecutionMode);
        Assert.True(options.EnableGpuResidency);
        Assert.Equal(2, options.TransferStreams);
        Assert.False(options.EnableProfiling);
        Assert.Equal(32, options.GraphBatchSize);
        Assert.True(options.CacheCompiledGraphs);
    }

    [Fact]
    public void GpuExecutionOptions_AllPropertiesAreSettable()
    {
        // Arrange
        var options = new GpuExecutionOptions();

        // Act - set all properties to non-default values
        options.MinGpuElements = 1024;
        options.MaxComputeStreams = 8;
        options.ForceGpu = true;
        options.ForceCpu = false;
        options.EnableGraphCompilation = false;
        options.EnableAutoFusion = false;
        options.MaxMemoryUsage = 0.5;
        options.EnablePrefetch = false;
        options.EnableComputeTransferOverlap = false;
        options.ExecutionMode = GpuExecutionMode.Deferred;
        options.EnableGpuResidency = false;
        options.TransferStreams = 4;
        options.EnableProfiling = true;
        options.GraphBatchSize = 64;
        options.CacheCompiledGraphs = false;

        // Assert
        Assert.Equal(1024, options.MinGpuElements);
        Assert.Equal(8, options.MaxComputeStreams);
        Assert.True(options.ForceGpu);
        Assert.False(options.ForceCpu);
        Assert.False(options.EnableGraphCompilation);
        Assert.False(options.EnableAutoFusion);
        Assert.Equal(0.5, options.MaxMemoryUsage, Tolerance);
        Assert.False(options.EnablePrefetch);
        Assert.False(options.EnableComputeTransferOverlap);
        Assert.Equal(GpuExecutionMode.Deferred, options.ExecutionMode);
        Assert.False(options.EnableGpuResidency);
        Assert.Equal(4, options.TransferStreams);
        Assert.True(options.EnableProfiling);
        Assert.Equal(64, options.GraphBatchSize);
        Assert.False(options.CacheCompiledGraphs);
    }

    #endregion

    #region GpuExecutionMode Enum Tests

    [Theory]
    [InlineData(GpuExecutionMode.Eager)]
    [InlineData(GpuExecutionMode.Deferred)]
    [InlineData(GpuExecutionMode.ScopedDeferred)]
    [InlineData(GpuExecutionMode.Auto)]
    public void GpuExecutionOptions_ExecutionMode_AcceptsAllEnumValues(GpuExecutionMode mode)
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { ExecutionMode = mode };

        // Assert
        Assert.Equal(mode, options.ExecutionMode);
    }

    [Fact]
    public void GpuExecutionMode_EnumValuesAreDistinct()
    {
        var values = (GpuExecutionMode[])Enum.GetValues(typeof(GpuExecutionMode));
        var uniqueValues = values.Distinct().ToArray();

        Assert.Equal(values.Length, uniqueValues.Length);
    }

    [Fact]
    public void GpuExecutionMode_HasExpectedCount()
    {
        // 4 modes: Eager, Deferred, ScopedDeferred, Auto
        var values = (GpuExecutionMode[])Enum.GetValues(typeof(GpuExecutionMode));
        Assert.Equal(4, values.Length);
    }

    [Fact]
    public void GpuExecutionMode_EagerIsZero()
    {
        // Eager should be 0 for default fallback
        Assert.Equal(0, (int)GpuExecutionMode.Eager);
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void Validate_DefaultOptions_DoesNotThrow()
    {
        // Arrange
        var options = new GpuExecutionOptions();

        // Act & Assert - should not throw
        options.Validate();
    }

    [Fact]
    public void Validate_NegativeMinGpuElements_Throws()
    {
        // Arrange
        var options = new GpuExecutionOptions { MinGpuElements = -1 };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("MinGpuElements", ex.Message);
    }

    [Fact]
    public void Validate_ZeroMinGpuElements_DoesNotThrow()
    {
        // Arrange
        var options = new GpuExecutionOptions { MinGpuElements = 0 };

        // Act & Assert - zero is valid (always use GPU)
        options.Validate();
    }

    [Fact]
    public void Validate_ZeroMaxComputeStreams_Throws()
    {
        // Arrange
        var options = new GpuExecutionOptions { MaxComputeStreams = 0 };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("MaxComputeStreams", ex.Message);
    }

    [Fact]
    public void Validate_NegativeMaxMemoryUsage_Throws()
    {
        // Arrange
        var options = new GpuExecutionOptions { MaxMemoryUsage = -0.1 };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("MaxMemoryUsage", ex.Message);
    }

    [Fact]
    public void Validate_MaxMemoryUsageOverOne_Throws()
    {
        // Arrange
        var options = new GpuExecutionOptions { MaxMemoryUsage = 1.1 };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("MaxMemoryUsage", ex.Message);
    }

    [Fact]
    public void Validate_MaxMemoryUsageAtZero_DoesNotThrow()
    {
        // Arrange
        var options = new GpuExecutionOptions { MaxMemoryUsage = 0.0 };

        // Act & Assert
        options.Validate();
    }

    [Fact]
    public void Validate_MaxMemoryUsageAtOne_DoesNotThrow()
    {
        // Arrange
        var options = new GpuExecutionOptions { MaxMemoryUsage = 1.0 };

        // Act & Assert
        options.Validate();
    }

    [Fact]
    public void Validate_ZeroTransferStreams_Throws()
    {
        // Arrange
        var options = new GpuExecutionOptions { TransferStreams = 0 };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("TransferStreams", ex.Message);
    }

    [Fact]
    public void Validate_ZeroGraphBatchSize_Throws()
    {
        // Arrange
        var options = new GpuExecutionOptions { GraphBatchSize = 0 };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("GraphBatchSize", ex.Message);
    }

    [Fact]
    public void Validate_BothForceGpuAndForceCpu_Throws()
    {
        // Arrange
        var options = new GpuExecutionOptions { ForceGpu = true, ForceCpu = true };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("ForceGpu", ex.Message);
    }

    [Fact]
    public void Validate_ForceGpuOnly_DoesNotThrow()
    {
        // Arrange
        var options = new GpuExecutionOptions { ForceGpu = true, ForceCpu = false };

        // Act & Assert
        options.Validate();
    }

    [Fact]
    public void Validate_ForceCpuOnly_DoesNotThrow()
    {
        // Arrange
        var options = new GpuExecutionOptions { ForceGpu = false, ForceCpu = true };

        // Act & Assert
        options.Validate();
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void Clone_DefaultOptions_CopiesAllValues()
    {
        // Arrange
        var original = new GpuExecutionOptions();

        // Act
        var clone = original.Clone();

        // Assert - all values should match
        Assert.Equal(original.MinGpuElements, clone.MinGpuElements);
        Assert.Equal(original.MaxComputeStreams, clone.MaxComputeStreams);
        Assert.Equal(original.ForceGpu, clone.ForceGpu);
        Assert.Equal(original.ForceCpu, clone.ForceCpu);
        Assert.Equal(original.EnableGraphCompilation, clone.EnableGraphCompilation);
        Assert.Equal(original.EnableAutoFusion, clone.EnableAutoFusion);
        Assert.Equal(original.MaxMemoryUsage, clone.MaxMemoryUsage, Tolerance);
        Assert.Equal(original.EnablePrefetch, clone.EnablePrefetch);
        Assert.Equal(original.EnableComputeTransferOverlap, clone.EnableComputeTransferOverlap);
        Assert.Equal(original.ExecutionMode, clone.ExecutionMode);
        Assert.Equal(original.EnableGpuResidency, clone.EnableGpuResidency);
        Assert.Equal(original.TransferStreams, clone.TransferStreams);
        Assert.Equal(original.EnableProfiling, clone.EnableProfiling);
        Assert.Equal(original.GraphBatchSize, clone.GraphBatchSize);
        Assert.Equal(original.CacheCompiledGraphs, clone.CacheCompiledGraphs);
    }

    [Fact]
    public void Clone_CustomOptions_CopiesAllValues()
    {
        // Arrange
        var original = new GpuExecutionOptions
        {
            MinGpuElements = 2048,
            MaxComputeStreams = 6,
            ForceGpu = true,
            EnableGraphCompilation = false,
            EnableAutoFusion = false,
            MaxMemoryUsage = 0.6,
            EnablePrefetch = false,
            EnableComputeTransferOverlap = false,
            ExecutionMode = GpuExecutionMode.Deferred,
            EnableGpuResidency = false,
            TransferStreams = 3,
            EnableProfiling = true,
            GraphBatchSize = 16,
            CacheCompiledGraphs = false
        };

        // Act
        var clone = original.Clone();

        // Assert
        Assert.Equal(2048, clone.MinGpuElements);
        Assert.Equal(6, clone.MaxComputeStreams);
        Assert.True(clone.ForceGpu);
        Assert.False(clone.EnableGraphCompilation);
        Assert.False(clone.EnableAutoFusion);
        Assert.Equal(0.6, clone.MaxMemoryUsage, Tolerance);
        Assert.False(clone.EnablePrefetch);
        Assert.False(clone.EnableComputeTransferOverlap);
        Assert.Equal(GpuExecutionMode.Deferred, clone.ExecutionMode);
        Assert.False(clone.EnableGpuResidency);
        Assert.Equal(3, clone.TransferStreams);
        Assert.True(clone.EnableProfiling);
        Assert.Equal(16, clone.GraphBatchSize);
        Assert.False(clone.CacheCompiledGraphs);
    }

    [Fact]
    public void Clone_ReturnsDifferentInstance()
    {
        // Arrange
        var original = new GpuExecutionOptions();

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
    }

    [Fact]
    public void Clone_ChangingCloneDoesNotAffectOriginal()
    {
        // Arrange
        var original = new GpuExecutionOptions { MinGpuElements = 1000 };
        var clone = original.Clone();

        // Act
        clone.MinGpuElements = 2000;

        // Assert
        Assert.Equal(1000, original.MinGpuElements);
        Assert.Equal(2000, clone.MinGpuElements);
    }

    [Fact]
    public void Clone_ChangingOriginalDoesNotAffectClone()
    {
        // Arrange
        var original = new GpuExecutionOptions { MinGpuElements = 1000 };
        var clone = original.Clone();

        // Act
        original.MinGpuElements = 2000;

        // Assert
        Assert.Equal(2000, original.MinGpuElements);
        Assert.Equal(1000, clone.MinGpuElements);
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void GpuExecutionOptions_MinGpuElementsAtZero_IsValid()
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { MinGpuElements = 0 };
        options.Validate();

        // Assert
        Assert.Equal(0, options.MinGpuElements);
    }

    [Fact]
    public void GpuExecutionOptions_LargeMinGpuElements_IsValid()
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { MinGpuElements = int.MaxValue };
        options.Validate();

        // Assert
        Assert.Equal(int.MaxValue, options.MinGpuElements);
    }

    [Fact]
    public void GpuExecutionOptions_SingleStream_IsValid()
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { MaxComputeStreams = 1 };
        options.Validate();

        // Assert
        Assert.Equal(1, options.MaxComputeStreams);
    }

    [Fact]
    public void GpuExecutionOptions_ManyStreams_IsValid()
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { MaxComputeStreams = 100 };
        options.Validate();

        // Assert
        Assert.Equal(100, options.MaxComputeStreams);
    }

    [Fact]
    public void GpuExecutionOptions_VerySmallMaxMemoryUsage_IsValid()
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { MaxMemoryUsage = 0.01 };
        options.Validate();

        // Assert
        Assert.Equal(0.01, options.MaxMemoryUsage, Tolerance);
    }

    [Fact]
    public void GpuExecutionOptions_SmallGraphBatchSize_IsValid()
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { GraphBatchSize = 1 };
        options.Validate();

        // Assert
        Assert.Equal(1, options.GraphBatchSize);
    }

    [Fact]
    public void GpuExecutionOptions_LargeGraphBatchSize_IsValid()
    {
        // Arrange & Act
        var options = new GpuExecutionOptions { GraphBatchSize = 1024 };
        options.Validate();

        // Assert
        Assert.Equal(1024, options.GraphBatchSize);
    }

    #endregion

    #region Configuration Scenario Tests

    [Fact]
    public void GpuExecutionOptions_HighPerformanceConfig_Validates()
    {
        // Arrange - optimized for high-end GPUs
        var options = new GpuExecutionOptions
        {
            MinGpuElements = 512,
            MaxComputeStreams = 16,
            EnableGraphCompilation = true,
            EnableAutoFusion = true,
            MaxMemoryUsage = 0.95,
            EnablePrefetch = true,
            EnableComputeTransferOverlap = true,
            ExecutionMode = GpuExecutionMode.Deferred,
            EnableGpuResidency = true,
            TransferStreams = 4,
            GraphBatchSize = 128,
            CacheCompiledGraphs = true
        };

        // Act & Assert
        options.Validate();
    }

    [Fact]
    public void GpuExecutionOptions_DebugConfig_Validates()
    {
        // Arrange - optimized for debugging
        var options = new GpuExecutionOptions
        {
            MinGpuElements = 100000,
            MaxComputeStreams = 1,
            EnableGraphCompilation = false,
            EnableAutoFusion = false,
            EnablePrefetch = false,
            EnableComputeTransferOverlap = false,
            ExecutionMode = GpuExecutionMode.Eager,
            EnableGpuResidency = false,
            TransferStreams = 1,
            EnableProfiling = true,
            GraphBatchSize = 1,
            CacheCompiledGraphs = false
        };

        // Act & Assert
        options.Validate();
    }

    [Fact]
    public void GpuExecutionOptions_MemoryConstrainedConfig_Validates()
    {
        // Arrange - low memory configuration
        var options = new GpuExecutionOptions
        {
            MaxMemoryUsage = 0.2,
            EnableGpuResidency = false,
            EnablePrefetch = false,
            CacheCompiledGraphs = false,
            GraphBatchSize = 8
        };

        // Act & Assert
        options.Validate();
    }

    #endregion

    #region FromEnvironment Tests

    [Fact]
    public void FromEnvironment_NoVariablesSet_ReturnsDefaults()
    {
        // Act
        var options = GpuExecutionOptions.FromEnvironment();

        // Assert - should have default values
        Assert.Equal(4096, options.MinGpuElements);
        Assert.Equal(3, options.MaxComputeStreams);
        Assert.False(options.ForceGpu);
        Assert.False(options.ForceCpu);
        Assert.True(options.EnableGraphCompilation);
        Assert.Equal(GpuExecutionMode.Auto, options.ExecutionMode);
    }

    [Fact]
    public void FromEnvironment_ReturnsValidOptions()
    {
        // Act
        var options = GpuExecutionOptions.FromEnvironment();

        // Assert
        options.Validate();
    }

    #endregion
}
