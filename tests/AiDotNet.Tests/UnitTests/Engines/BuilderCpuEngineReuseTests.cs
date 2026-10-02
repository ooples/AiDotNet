using System.Reflection;
using AiDotNet.Engines;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Engines;

/// <summary>
/// Building on a CPU configuration must keep the CPU engine the process already runs on. <c>AiDotNetEngine.Current</c>
/// is process-wide and every model reads it on each operation; the builder used to install a new <see cref="CpuEngine"/>
/// on every such build, and a model training on another thread lost that step's update when the engine was swapped
/// under it.
/// </summary>
[Collection(AiDotNet.Tests.Fixtures.EngineCurrentGlobalStateCollection.Name)]
public class BuilderCpuEngineReuseTests
{
    private static void ApplyGpuConfiguration(GpuUsageLevel usage, GpuDeviceType device)
    {
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        builder.ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = usage, DeviceType = device });
        var apply = typeof(AiModelBuilder<double, Tensor<double>, Tensor<double>>).GetMethod(
            "ApplyGpuConfigurationCore", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new System.InvalidOperationException("AiModelBuilder has no ApplyGpuConfigurationCore");
        apply.Invoke(builder, null);
    }

    [Theory]
    [InlineData(GpuUsageLevel.AlwaysCpu, GpuDeviceType.Auto)]
    [InlineData(GpuUsageLevel.Default, GpuDeviceType.CPU)]
    public void A_cpu_configuration_keeps_the_current_cpu_engine(GpuUsageLevel usage, GpuDeviceType device)
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            ApplyGpuConfiguration(usage, device);
            Assert.Same(cpu, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }
}
