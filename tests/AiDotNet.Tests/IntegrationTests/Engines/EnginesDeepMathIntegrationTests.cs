using AiDotNet.Engines;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Engines;

/// <summary>
/// Deep integration tests for Engines:
/// GpuAccelerationConfig (defaults, property ranges, ToString, ToExecutionOptions conversion),
/// GpuDeviceType, GpuUsageLevel, GpuExecutionModeConfig enums.
/// </summary>
public class EnginesDeepMathIntegrationTests
{
    // ============================
    // GpuAccelerationConfig: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_DeviceType_Auto()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(GpuDeviceType.Auto, config.DeviceType);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_UsageLevel_Default()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(GpuUsageLevel.Default, config.UsageLevel);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_DeviceIndex_Zero()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(0, config.DeviceIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_VerboseLogging_False()
    {
        var config = new GpuAccelerationConfig();
        Assert.False(config.VerboseLogging);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_EnableForInference_True()
    {
        var config = new GpuAccelerationConfig();
        Assert.True(config.EnableForInference);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_EnableGpuPersistence_True()
    {
        var config = new GpuAccelerationConfig();
        Assert.True(config.EnableGpuPersistence);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_ExecutionMode_Auto()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(GpuExecutionModeConfig.Auto, config.ExecutionMode);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_EnableGraphCompilation_True()
    {
        var config = new GpuAccelerationConfig();
        Assert.True(config.EnableGraphCompilation);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_EnableAutoFusion_True()
    {
        var config = new GpuAccelerationConfig();
        Assert.True(config.EnableAutoFusion);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_EnableComputeTransferOverlap_True()
    {
        var config = new GpuAccelerationConfig();
        Assert.True(config.EnableComputeTransferOverlap);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_MaxComputeStreams_Three()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(3, config.MaxComputeStreams);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_MinGpuElements_4096()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(4096, config.MinGpuElements);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_MaxGpuMemoryUsage_80Percent()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(0.8, config.MaxGpuMemoryUsage, 0.001);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_EnablePrefetch_True()
    {
        var config = new GpuAccelerationConfig();
        Assert.True(config.EnablePrefetch);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_CacheCompiledGraphs_True()
    {
        var config = new GpuAccelerationConfig();
        Assert.True(config.CacheCompiledGraphs);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_EnableProfiling_False()
    {
        var config = new GpuAccelerationConfig();
        Assert.False(config.EnableProfiling);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_Defaults_TransferStreams_Two()
    {
        var config = new GpuAccelerationConfig();
        Assert.Equal(2, config.TransferStreams);
    }

    // ============================
    // GpuAccelerationConfig: ToString
    // ============================

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_ToString_ContainsDeviceType()
    {
        var config = new GpuAccelerationConfig();
        var str = config.ToString();
        Assert.Contains("DeviceType=Auto", str);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_ToString_ContainsUsageLevel()
    {
        var config = new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.Aggressive };
        var str = config.ToString();
        Assert.Contains("UsageLevel=Aggressive", str);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_ToString_ContainsExecutionMode()
    {
        var config = new GpuAccelerationConfig { ExecutionMode = GpuExecutionModeConfig.Deferred };
        var str = config.ToString();
        Assert.Contains("ExecutionMode=Deferred", str);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_ToString_ContainsMaxMemory()
    {
        var config = new GpuAccelerationConfig();
        var str = config.ToString();
        Assert.Contains("MaxMemory=", str);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_ToString_ContainsMinElements()
    {
        var config = new GpuAccelerationConfig();
        var str = config.ToString();
        Assert.Contains("MinElements=4096", str);
    }

    // ============================
    // GpuAccelerationConfig: Property Setting
    // ============================

    [Theory]
    [InlineData(GpuDeviceType.CUDA)]
    [InlineData(GpuDeviceType.OpenCL)]
    [InlineData(GpuDeviceType.CPU)]
    [InlineData(GpuDeviceType.Auto)]
    public void GpuConfig_DeviceType_AllValues(GpuDeviceType deviceType)
    {
        var config = new GpuAccelerationConfig { DeviceType = deviceType };
        Assert.Equal(deviceType, config.DeviceType);
    }

    [Theory]
    [InlineData(GpuUsageLevel.Conservative)]
    [InlineData(GpuUsageLevel.Default)]
    [InlineData(GpuUsageLevel.Aggressive)]
    [InlineData(GpuUsageLevel.AlwaysGpu)]
    [InlineData(GpuUsageLevel.AlwaysCpu)]
    public void GpuConfig_UsageLevel_AllValues(GpuUsageLevel level)
    {
        var config = new GpuAccelerationConfig { UsageLevel = level };
        Assert.Equal(level, config.UsageLevel);
    }

    [Theory]
    [InlineData(GpuExecutionModeConfig.Auto)]
    [InlineData(GpuExecutionModeConfig.Eager)]
    [InlineData(GpuExecutionModeConfig.Deferred)]
    [InlineData(GpuExecutionModeConfig.ScopedDeferred)]
    public void GpuConfig_ExecutionMode_AllValues(GpuExecutionModeConfig mode)
    {
        var config = new GpuAccelerationConfig { ExecutionMode = mode };
        Assert.Equal(mode, config.ExecutionMode);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuConfig_CustomValues_AllSet()
    {
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CUDA,
            UsageLevel = GpuUsageLevel.Aggressive,
            DeviceIndex = 1,
            VerboseLogging = true,
            EnableForInference = false,
            EnableGpuPersistence = false,
            ExecutionMode = GpuExecutionModeConfig.Deferred,
            EnableGraphCompilation = false,
            EnableAutoFusion = false,
            EnableComputeTransferOverlap = false,
            MaxComputeStreams = 8,
            MinGpuElements = 1024,
            MaxGpuMemoryUsage = 0.95,
            EnablePrefetch = false,
            CacheCompiledGraphs = false,
            EnableProfiling = true,
            TransferStreams = 4
        };

        Assert.Equal(GpuDeviceType.CUDA, config.DeviceType);
        Assert.Equal(GpuUsageLevel.Aggressive, config.UsageLevel);
        Assert.Equal(1, config.DeviceIndex);
        Assert.True(config.VerboseLogging);
        Assert.False(config.EnableForInference);
        Assert.False(config.EnableGpuPersistence);
        Assert.Equal(GpuExecutionModeConfig.Deferred, config.ExecutionMode);
        Assert.False(config.EnableGraphCompilation);
        Assert.False(config.EnableAutoFusion);
        Assert.False(config.EnableComputeTransferOverlap);
        Assert.Equal(8, config.MaxComputeStreams);
        Assert.Equal(1024, config.MinGpuElements);
        Assert.Equal(0.95, config.MaxGpuMemoryUsage, 0.001);
        Assert.False(config.EnablePrefetch);
        Assert.False(config.CacheCompiledGraphs);
        Assert.True(config.EnableProfiling);
        Assert.Equal(4, config.TransferStreams);
    }

    // ============================
    // GpuDeviceType Enum: Count
    // ============================

    [Fact(Timeout = 120000)]
    public async Task GpuDeviceType_HasFourValues()
    {
        var values = (((GpuDeviceType[])Enum.GetValues(typeof(GpuDeviceType))));
        Assert.Equal(4, values.Length);
    }

    // ============================
    // GpuUsageLevel Enum: Count
    // ============================

    [Fact(Timeout = 120000)]
    public async Task GpuUsageLevel_HasFiveValues()
    {
        var values = (((GpuUsageLevel[])Enum.GetValues(typeof(GpuUsageLevel))));
        Assert.Equal(5, values.Length);
    }

    // ============================
    // GpuExecutionModeConfig Enum: Count
    // ============================

    [Fact(Timeout = 120000)]
    public async Task GpuExecutionModeConfig_HasFourValues()
    {
        var values = (((GpuExecutionModeConfig[])Enum.GetValues(typeof(GpuExecutionModeConfig))));
        Assert.Equal(4, values.Length);
    }

    // ============================
    // GpuAccelerationConfig: MaxGpuMemoryUsage Range
    // ============================

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void GpuConfig_MaxGpuMemoryUsage_AcceptsRange(double fraction)
    {
        var config = new GpuAccelerationConfig { MaxGpuMemoryUsage = fraction };
        Assert.Equal(fraction, config.MaxGpuMemoryUsage, 0.001);
    }

    // ============================
    // GpuAccelerationConfig: MinGpuElements Thresholds
    // ============================

    [Theory]
    [InlineData(1024)]
    [InlineData(2048)]
    [InlineData(4096)]
    [InlineData(10000)]
    public void GpuConfig_MinGpuElements_AcceptsThresholds(int threshold)
    {
        var config = new GpuAccelerationConfig { MinGpuElements = threshold };
        Assert.Equal(threshold, config.MinGpuElements);
    }
}
