using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// An engine the caller assigned is an explicit choice. The builder's default GPU path (no ConfigureGpuAcceleration)
/// auto-detected a GPU and reinstalled it over a caller's <c>AiDotNetEngine.Current = new CpuEngine()</c>, so a run the
/// caller set to the CPU held the whole GPU. On a machine without a GPU the auto-detect finds nothing either way, so this
/// test bites only where a GPU is present.
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class CallerAssignedEngineTests
{
    [Fact]
    public void Default_gpu_path_keeps_a_caller_assigned_cpu_engine()
    {
        var previous = AiDotNetEngine.Current;
        var gpuVar = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", null);
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            Assert.True(AiDotNetEngine.IsCallerAssigned);

            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
            typeof(AiModelBuilder<float, Tensor<float>, Tensor<float>>)
                .GetMethod("ApplyGpuConfigurationCore", BindingFlags.NonPublic | BindingFlags.Instance)!
                .Invoke(builder, null);

            Assert.Same(cpu, AiDotNetEngine.Current);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", gpuVar);
            AiDotNetEngine.Current = previous;
        }
    }
}
