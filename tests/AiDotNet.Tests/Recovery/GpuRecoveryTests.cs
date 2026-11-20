using AiDotNet.Tests.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Recovery;

/// <summary>Supported diagnostics, rejected-input isolation, and GPU resource lifecycle contracts.</summary>
[Collection("EngineCurrentGlobalState")]
[Trait("Category", "GPU")]
[Trait("GpuContract", "Required")]
public class GpuRecoveryTests
{
    private readonly ITestOutputHelper _output;
    public GpuRecoveryTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void AvailableBackend_ReportsCudaDeviceAndExecutes()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        var backend = gpu.Engine.GetBackend();
        Assert.NotNull(backend);
        Assert.True(backend.IsAvailable);
        Assert.False(string.IsNullOrWhiteSpace(backend.DeviceName));
        RequiredGpuTestContract.CheckAddition(gpu.Dispatch, 1);
        gpu.AssertExecuted();
    }

    [Fact]
    public void DiagnosticReport_IdentifiesActiveBackendAndExecutes()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        string report = GpuEngine.GetDiagnosticReport();
        _output.WriteLine(report);
        Assert.Contains("CUDA", report, System.StringComparison.OrdinalIgnoreCase);
        RequiredGpuTestContract.CheckAddition(gpu.Dispatch, 2);
        gpu.AssertExecuted();
    }

    [Fact]
    public void InvalidMatrixDimensions_DoNotPoisonFollowingOperation()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        Assert.ThrowsAny<System.ArgumentException>(() => gpu.Dispatch.MatrixMultiply(new Matrix<float>(2, 3), new Matrix<float>(4, 2)));
        RequiredGpuTestContract.CheckAddition(gpu.Dispatch, 3);
        gpu.AssertExecuted();
    }

    [Fact]
    public void RepeatedOperations_KeepInputsAndResultsIndependent()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        for (int step = 0; step < 12; step++) RequiredGpuTestContract.CheckAddition(gpu.Dispatch, step);
        gpu.AssertExecuted();
    }

    [Fact]
    public void Dispose_DoesNotDamageFreshEngine()
    {
        using (var previous = new RequiredGpuTestContract(_output))
        {
            RequiredGpuTestContract.CheckAddition(previous.Dispatch, 5);
            previous.AssertExecuted();
        }
        using var fresh = new RequiredGpuTestContract(_output);
        RequiredGpuTestContract.CheckAddition(fresh.Dispatch, 6);
        fresh.AssertExecuted();
    }
}
