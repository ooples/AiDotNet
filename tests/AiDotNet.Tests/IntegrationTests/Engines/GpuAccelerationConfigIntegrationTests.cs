using AiDotNet.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Engines;

/// <summary>
/// Integration tests for GPU acceleration configuration.
/// </summary>
/// <remarks>
/// <para>
/// These tests verify that GPU acceleration configuration:
/// 1. Initializes with sensible defaults
/// 2. Correctly converts to internal GpuExecutionOptions
/// 3. Handles all enum values correctly
/// 4. Properly maps usage levels to force flags
/// </para>
/// <para><b>For Beginners:</b> GPU acceleration configuration controls how your GPU
/// is used for machine learning operations. These tests ensure the configuration
/// works correctly even when no GPU is present.
/// </para>
/// </remarks>
public class GpuAccelerationConfigIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Default Configuration Tests

    [Fact]
    public void GpuAccelerationConfig_DefaultConstructor_HasExpectedDefaults()
    {
        // Act
        var config = new GpuAccelerationConfig();

        // Assert - verify all default values
        Assert.Equal(GpuDeviceType.Auto, config.DeviceType);
        Assert.Equal(GpuUsageLevel.Default, config.UsageLevel);
        Assert.Equal(0, config.DeviceIndex);
        Assert.False(config.VerboseLogging);
        Assert.True(config.EnableForInference);
        Assert.True(config.EnableGpuPersistence);
        Assert.Equal(GpuExecutionModeConfig.Auto, config.ExecutionMode);
        Assert.True(config.EnableGraphCompilation);
        Assert.True(config.EnableAutoFusion);
        Assert.True(config.EnableComputeTransferOverlap);
        Assert.Equal(3, config.MaxComputeStreams);
        Assert.Equal(4096, config.MinGpuElements);
        Assert.Equal(0.8, config.MaxGpuMemoryUsage, Tolerance);
        Assert.True(config.EnablePrefetch);
        Assert.True(config.CacheCompiledGraphs);
        Assert.False(config.EnableProfiling);
        Assert.Equal(2, config.TransferStreams);
    }

    [Fact]
    public void GpuAccelerationConfig_AllPropertiesAreSettable()
    {
        // Arrange
        var config = new GpuAccelerationConfig();

        // Act - set all properties to non-default values
        config.DeviceType = GpuDeviceType.CUDA;
        config.UsageLevel = GpuUsageLevel.Aggressive;
        config.DeviceIndex = 2;
        config.VerboseLogging = true;
        config.EnableForInference = false;
        config.EnableGpuPersistence = false;
        config.ExecutionMode = GpuExecutionModeConfig.Deferred;
        config.EnableGraphCompilation = false;
        config.EnableAutoFusion = false;
        config.EnableComputeTransferOverlap = false;
        config.MaxComputeStreams = 8;
        config.MinGpuElements = 1024;
        config.MaxGpuMemoryUsage = 0.5;
        config.EnablePrefetch = false;
        config.CacheCompiledGraphs = false;
        config.EnableProfiling = true;
        config.TransferStreams = 4;

        // Assert - verify all values were set
        Assert.Equal(GpuDeviceType.CUDA, config.DeviceType);
        Assert.Equal(GpuUsageLevel.Aggressive, config.UsageLevel);
        Assert.Equal(2, config.DeviceIndex);
        Assert.True(config.VerboseLogging);
        Assert.False(config.EnableForInference);
        Assert.False(config.EnableGpuPersistence);
        Assert.Equal(GpuExecutionModeConfig.Deferred, config.ExecutionMode);
        Assert.False(config.EnableGraphCompilation);
        Assert.False(config.EnableAutoFusion);
        Assert.False(config.EnableComputeTransferOverlap);
        Assert.Equal(8, config.MaxComputeStreams);
        Assert.Equal(1024, config.MinGpuElements);
        Assert.Equal(0.5, config.MaxGpuMemoryUsage, Tolerance);
        Assert.False(config.EnablePrefetch);
        Assert.False(config.CacheCompiledGraphs);
        Assert.True(config.EnableProfiling);
        Assert.Equal(4, config.TransferStreams);
    }

    #endregion

    #region GpuDeviceType Enum Tests

    [Theory]
    [InlineData(GpuDeviceType.Auto)]
    [InlineData(GpuDeviceType.CUDA)]
    [InlineData(GpuDeviceType.OpenCL)]
    [InlineData(GpuDeviceType.CPU)]
    public void GpuAccelerationConfig_DeviceType_AcceptsAllEnumValues(GpuDeviceType deviceType)
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { DeviceType = deviceType };

        // Assert
        Assert.Equal(deviceType, config.DeviceType);
    }

    [Fact]
    public void GpuDeviceType_EnumValuesAreDistinct()
    {
        // Verify all enum values are unique
        var values = (GpuDeviceType[])Enum.GetValues(typeof(GpuDeviceType));
        var uniqueValues = values.Distinct().ToArray();

        Assert.Equal(values.Length, uniqueValues.Length);
    }

    [Fact]
    public void GpuDeviceType_HasExpectedCount()
    {
        // 4 device types: Auto, CUDA, OpenCL, CPU
        var values = (GpuDeviceType[])Enum.GetValues(typeof(GpuDeviceType));
        Assert.Equal(4, values.Length);
    }

    #endregion

    #region GpuUsageLevel Enum Tests

    [Theory]
    [InlineData(GpuUsageLevel.Conservative)]
    [InlineData(GpuUsageLevel.Default)]
    [InlineData(GpuUsageLevel.Aggressive)]
    [InlineData(GpuUsageLevel.AlwaysGpu)]
    [InlineData(GpuUsageLevel.AlwaysCpu)]
    public void GpuAccelerationConfig_UsageLevel_AcceptsAllEnumValues(GpuUsageLevel usageLevel)
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { UsageLevel = usageLevel };

        // Assert
        Assert.Equal(usageLevel, config.UsageLevel);
    }

    [Fact]
    public void GpuUsageLevel_EnumValuesAreDistinct()
    {
        var values = (GpuUsageLevel[])Enum.GetValues(typeof(GpuUsageLevel));
        var uniqueValues = values.Distinct().ToArray();

        Assert.Equal(values.Length, uniqueValues.Length);
    }

    [Fact]
    public void GpuUsageLevel_HasExpectedCount()
    {
        // 5 usage levels: Conservative, Default, Aggressive, AlwaysGpu, AlwaysCpu
        var values = (GpuUsageLevel[])Enum.GetValues(typeof(GpuUsageLevel));
        Assert.Equal(5, values.Length);
    }

    #endregion

    #region GpuExecutionModeConfig Enum Tests

    [Theory]
    [InlineData(GpuExecutionModeConfig.Auto)]
    [InlineData(GpuExecutionModeConfig.Eager)]
    [InlineData(GpuExecutionModeConfig.Deferred)]
    [InlineData(GpuExecutionModeConfig.ScopedDeferred)]
    public void GpuAccelerationConfig_ExecutionMode_AcceptsAllEnumValues(GpuExecutionModeConfig mode)
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { ExecutionMode = mode };

        // Assert
        Assert.Equal(mode, config.ExecutionMode);
    }

    [Fact]
    public void GpuExecutionModeConfig_EnumValuesAreDistinct()
    {
        var values = (GpuExecutionModeConfig[])Enum.GetValues(typeof(GpuExecutionModeConfig));
        var uniqueValues = values.Distinct().ToArray();

        Assert.Equal(values.Length, uniqueValues.Length);
    }

    [Fact]
    public void GpuExecutionModeConfig_HasExpectedCount()
    {
        // 4 modes: Auto, Eager, Deferred, ScopedDeferred
        var values = (GpuExecutionModeConfig[])Enum.GetValues(typeof(GpuExecutionModeConfig));
        Assert.Equal(4, values.Length);
    }

    #endregion

    #region ToExecutionOptions Conversion Tests

    [Fact]
    public void ToExecutionOptions_DefaultConfig_MapsCorrectly()
    {
        // Arrange
        var config = new GpuAccelerationConfig();

        // Act
        var options = config.ToExecutionOptions();

        // Assert - verify mapping of all properties
        Assert.Equal(config.MinGpuElements, options.MinGpuElements);
        Assert.Equal(config.MaxComputeStreams, options.MaxComputeStreams);
        Assert.Equal(config.TransferStreams, options.TransferStreams);
        Assert.Equal(config.EnableGraphCompilation, options.EnableGraphCompilation);
        Assert.Equal(config.EnableAutoFusion, options.EnableAutoFusion);
        Assert.Equal(config.MaxGpuMemoryUsage, options.MaxMemoryUsage, Tolerance);
        Assert.Equal(config.EnablePrefetch, options.EnablePrefetch);
        Assert.Equal(config.EnableComputeTransferOverlap, options.EnableComputeTransferOverlap);
        Assert.Equal(config.EnableGpuPersistence, options.EnableGpuResidency);
        Assert.Equal(config.EnableProfiling, options.EnableProfiling);
        Assert.Equal(config.CacheCompiledGraphs, options.CacheCompiledGraphs);
        Assert.False(options.ForceGpu);
        Assert.False(options.ForceCpu);
    }

    [Fact]
    public void ToExecutionOptions_AlwaysGpuUsageLevel_SetsForceGpu()
    {
        // Arrange
        var config = new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysGpu };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.True(options.ForceGpu);
        Assert.False(options.ForceCpu);
    }

    [Fact]
    public void ToExecutionOptions_AlwaysCpuUsageLevel_SetsForceCpu()
    {
        // Arrange
        var config = new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.False(options.ForceGpu);
        Assert.True(options.ForceCpu);
    }

    [Theory]
    [InlineData(GpuUsageLevel.Conservative)]
    [InlineData(GpuUsageLevel.Default)]
    [InlineData(GpuUsageLevel.Aggressive)]
    public void ToExecutionOptions_NonForceUsageLevels_DoNotSetForceFlags(GpuUsageLevel usageLevel)
    {
        // Arrange
        var config = new GpuAccelerationConfig { UsageLevel = usageLevel };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.False(options.ForceGpu);
        Assert.False(options.ForceCpu);
    }

    [Theory]
    [InlineData(GpuExecutionModeConfig.Auto, GpuExecutionMode.Auto)]
    [InlineData(GpuExecutionModeConfig.Eager, GpuExecutionMode.Eager)]
    [InlineData(GpuExecutionModeConfig.Deferred, GpuExecutionMode.Deferred)]
    [InlineData(GpuExecutionModeConfig.ScopedDeferred, GpuExecutionMode.ScopedDeferred)]
    public void ToExecutionOptions_ExecutionMode_MapsCorrectly(
        GpuExecutionModeConfig configMode,
        GpuExecutionMode expectedMode)
    {
        // Arrange
        var config = new GpuAccelerationConfig { ExecutionMode = configMode };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.Equal(expectedMode, options.ExecutionMode);
    }

    [Fact]
    public void ToExecutionOptions_CustomValues_MapsCorrectly()
    {
        // Arrange
        var config = new GpuAccelerationConfig
        {
            MinGpuElements = 2048,
            MaxComputeStreams = 6,
            TransferStreams = 4,
            EnableGraphCompilation = false,
            EnableAutoFusion = false,
            MaxGpuMemoryUsage = 0.6,
            EnablePrefetch = false,
            EnableComputeTransferOverlap = false,
            EnableGpuPersistence = false,
            EnableProfiling = true,
            CacheCompiledGraphs = false,
            ExecutionMode = GpuExecutionModeConfig.Deferred
        };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.Equal(2048, options.MinGpuElements);
        Assert.Equal(6, options.MaxComputeStreams);
        Assert.Equal(4, options.TransferStreams);
        Assert.False(options.EnableGraphCompilation);
        Assert.False(options.EnableAutoFusion);
        Assert.Equal(0.6, options.MaxMemoryUsage, Tolerance);
        Assert.False(options.EnablePrefetch);
        Assert.False(options.EnableComputeTransferOverlap);
        Assert.False(options.EnableGpuResidency);
        Assert.True(options.EnableProfiling);
        Assert.False(options.CacheCompiledGraphs);
        Assert.Equal(GpuExecutionMode.Deferred, options.ExecutionMode);
    }

    #endregion

    #region ToString Tests

    [Fact]
    public void ToString_DefaultConfig_ContainsAllKeyProperties()
    {
        // Arrange
        var config = new GpuAccelerationConfig();

        // Act
        var result = config.ToString();

        // Assert - verify key properties are present
        Assert.Contains("DeviceType=Auto", result);
        Assert.Contains("UsageLevel=Default", result);
        Assert.Contains("DeviceIndex=0", result);
        Assert.Contains("ExecutionMode=Auto", result);
        Assert.Contains("GraphCompilation=True", result);
        Assert.Contains("AutoFusion=True", result);
        Assert.Contains("ComputeTransferOverlap=True", result);
        Assert.Contains("MaxStreams=3", result);
        Assert.Contains("MinElements=4096", result);
        Assert.Contains("MaxMemory=", result);
        Assert.Contains("Prefetch=True", result);
        Assert.Contains("CacheGraphs=True", result);
        Assert.Contains("Profiling=False", result);
    }

    [Fact]
    public void ToString_CustomConfig_ReflectsSettings()
    {
        // Arrange
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CUDA,
            UsageLevel = GpuUsageLevel.Aggressive,
            DeviceIndex = 1,
            ExecutionMode = GpuExecutionModeConfig.Deferred,
            EnableGraphCompilation = false,
            MaxComputeStreams = 8,
            MinGpuElements = 1024,
            EnableProfiling = true
        };

        // Act
        var result = config.ToString();

        // Assert
        Assert.Contains("DeviceType=CUDA", result);
        Assert.Contains("UsageLevel=Aggressive", result);
        Assert.Contains("DeviceIndex=1", result);
        Assert.Contains("ExecutionMode=Deferred", result);
        Assert.Contains("GraphCompilation=False", result);
        Assert.Contains("MaxStreams=8", result);
        Assert.Contains("MinElements=1024", result);
        Assert.Contains("Profiling=True", result);
    }

    [Fact]
    public void ToString_ReturnsNonEmptyString()
    {
        // Arrange
        var config = new GpuAccelerationConfig();

        // Act
        var result = config.ToString();

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result);
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void GpuAccelerationConfig_ZeroDeviceIndex_IsValid()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { DeviceIndex = 0 };

        // Assert
        Assert.Equal(0, config.DeviceIndex);
    }

    [Fact]
    public void GpuAccelerationConfig_HighDeviceIndex_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { DeviceIndex = 15 };

        // Assert
        Assert.Equal(15, config.DeviceIndex);
    }

    [Fact]
    public void GpuAccelerationConfig_ZeroMinGpuElements_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { MinGpuElements = 0 };

        // Assert
        Assert.Equal(0, config.MinGpuElements);
    }

    [Fact]
    public void GpuAccelerationConfig_LargeMinGpuElements_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { MinGpuElements = 1_000_000 };

        // Assert
        Assert.Equal(1_000_000, config.MinGpuElements);
    }

    [Fact]
    public void GpuAccelerationConfig_ZeroMaxMemoryUsage_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { MaxGpuMemoryUsage = 0.0 };

        // Assert
        Assert.Equal(0.0, config.MaxGpuMemoryUsage, Tolerance);
    }

    [Fact]
    public void GpuAccelerationConfig_FullMaxMemoryUsage_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { MaxGpuMemoryUsage = 1.0 };

        // Assert
        Assert.Equal(1.0, config.MaxGpuMemoryUsage, Tolerance);
    }

    [Fact]
    public void GpuAccelerationConfig_SingleComputeStream_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { MaxComputeStreams = 1 };

        // Assert
        Assert.Equal(1, config.MaxComputeStreams);
    }

    [Fact]
    public void GpuAccelerationConfig_ManyComputeStreams_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { MaxComputeStreams = 32 };

        // Assert
        Assert.Equal(32, config.MaxComputeStreams);
    }

    [Fact]
    public void GpuAccelerationConfig_SingleTransferStream_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { TransferStreams = 1 };

        // Assert
        Assert.Equal(1, config.TransferStreams);
    }

    [Fact]
    public void GpuAccelerationConfig_ManyTransferStreams_IsAccepted()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig { TransferStreams = 8 };

        // Assert
        Assert.Equal(8, config.TransferStreams);
    }

    #endregion

    #region Configuration Scenario Tests

    [Fact]
    public void GpuAccelerationConfig_HighPerformanceScenario_ConfiguresCorrectly()
    {
        // Arrange - high-end GPU setup
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CUDA,
            UsageLevel = GpuUsageLevel.Aggressive,
            ExecutionMode = GpuExecutionModeConfig.Deferred,
            EnableGraphCompilation = true,
            EnableAutoFusion = true,
            EnableComputeTransferOverlap = true,
            MaxComputeStreams = 8,
            TransferStreams = 4,
            MinGpuElements = 1024,
            MaxGpuMemoryUsage = 0.9,
            EnablePrefetch = true,
            CacheCompiledGraphs = true,
            EnableGpuPersistence = true,
            EnableProfiling = false
        };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.Equal(GpuExecutionMode.Deferred, options.ExecutionMode);
        Assert.True(options.EnableGraphCompilation);
        Assert.True(options.EnableAutoFusion);
        Assert.Equal(8, options.MaxComputeStreams);
        Assert.Equal(1024, options.MinGpuElements);
        Assert.Equal(0.9, options.MaxMemoryUsage, Tolerance);
    }

    [Fact]
    public void GpuAccelerationConfig_DebuggingScenario_ConfiguresCorrectly()
    {
        // Arrange - debugging setup with verbose output and eager execution
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.Auto,
            UsageLevel = GpuUsageLevel.Conservative,
            VerboseLogging = true,
            ExecutionMode = GpuExecutionModeConfig.Eager,
            EnableGraphCompilation = false,
            EnableAutoFusion = false,
            EnableProfiling = true,
            MaxComputeStreams = 1,
            MinGpuElements = 10000
        };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.Equal(GpuExecutionMode.Eager, options.ExecutionMode);
        Assert.False(options.EnableGraphCompilation);
        Assert.False(options.EnableAutoFusion);
        Assert.True(options.EnableProfiling);
        Assert.Equal(1, options.MaxComputeStreams);
    }

    [Fact]
    public void GpuAccelerationConfig_CpuOnlyScenario_ConfiguresCorrectly()
    {
        // Arrange - CPU-only execution
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CPU,
            UsageLevel = GpuUsageLevel.AlwaysCpu,
            EnableGpuPersistence = false
        };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.True(options.ForceCpu);
        Assert.False(options.ForceGpu);
        Assert.False(options.EnableGpuResidency);
    }

    [Fact]
    public void GpuAccelerationConfig_InferenceOnlyGpuScenario_ConfiguresCorrectly()
    {
        // Arrange - GPU for inference only
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.Auto,
            UsageLevel = GpuUsageLevel.Default,
            EnableForInference = true,
            EnableGpuPersistence = true,
            ExecutionMode = GpuExecutionModeConfig.ScopedDeferred
        };

        // Assert
        Assert.True(config.EnableForInference);
        Assert.True(config.EnableGpuPersistence);
        Assert.Equal(GpuExecutionModeConfig.ScopedDeferred, config.ExecutionMode);
    }

    [Fact]
    public void GpuAccelerationConfig_MemoryConstrainedScenario_ConfiguresCorrectly()
    {
        // Arrange - low memory environment
        var config = new GpuAccelerationConfig
        {
            MaxGpuMemoryUsage = 0.3,
            EnableGpuPersistence = false,
            EnablePrefetch = false,
            CacheCompiledGraphs = false,
            MinGpuElements = 8192
        };

        // Act
        var options = config.ToExecutionOptions();

        // Assert
        Assert.Equal(0.3, options.MaxMemoryUsage, Tolerance);
        Assert.False(options.EnableGpuResidency);
        Assert.False(options.EnablePrefetch);
        Assert.False(options.CacheCompiledGraphs);
        Assert.Equal(8192, options.MinGpuElements);
    }

    #endregion

    #region Multi-GPU Scenario Tests

    [Fact]
    public void GpuAccelerationConfig_MultiGpuSetup_FirstDevice()
    {
        // Arrange - first GPU
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CUDA,
            DeviceIndex = 0
        };

        // Assert
        Assert.Equal(0, config.DeviceIndex);
        Assert.Equal(GpuDeviceType.CUDA, config.DeviceType);
    }

    [Fact]
    public void GpuAccelerationConfig_MultiGpuSetup_SecondDevice()
    {
        // Arrange - second GPU
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CUDA,
            DeviceIndex = 1
        };

        // Assert
        Assert.Equal(1, config.DeviceIndex);
    }

    [Fact]
    public void GpuAccelerationConfig_MultiGpuSetup_OpenCLDevice()
    {
        // Arrange - OpenCL device
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.OpenCL,
            DeviceIndex = 2
        };

        // Assert
        Assert.Equal(GpuDeviceType.OpenCL, config.DeviceType);
        Assert.Equal(2, config.DeviceIndex);
    }

    #endregion

    #region Execution Options Clone Independence Tests

    [Fact]
    public void ToExecutionOptions_ReturnsNewInstanceEachCall()
    {
        // Arrange
        var config = new GpuAccelerationConfig();

        // Act
        var options1 = config.ToExecutionOptions();
        var options2 = config.ToExecutionOptions();

        // Assert - should be different instances
        Assert.NotSame(options1, options2);
    }

    [Fact]
    public void ToExecutionOptions_ChangingConfigDoesNotAffectPreviousOptions()
    {
        // Arrange
        var config = new GpuAccelerationConfig { MinGpuElements = 1000 };
        var optionsBefore = config.ToExecutionOptions();

        // Act - change config after getting options
        config.MinGpuElements = 2000;
        var optionsAfter = config.ToExecutionOptions();

        // Assert - previous options should be unchanged
        Assert.Equal(1000, optionsBefore.MinGpuElements);
        Assert.Equal(2000, optionsAfter.MinGpuElements);
    }

    #endregion
}
