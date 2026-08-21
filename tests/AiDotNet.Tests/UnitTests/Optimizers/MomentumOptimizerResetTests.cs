using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
#if !NET462
using AiDotNet.Tensors.Engines.DirectGpu;
#endif

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Verifies that restarting a momentum optimizer starts from a genuinely fresh state.
/// </summary>
public class MomentumOptimizerResetTests
{
    /// <summary>
    /// Reset must clear the vector velocity used by the public update API.
    /// </summary>
    [Fact]
    public void Reset_ClearsClassicalVelocity()
    {
        var optimizer = CreateOptimizer();
        var parameter = new Vector<double>(new[] { 1.0 });
        var gradient = new Vector<double>(new[] { 1.0 });

        Assert.Equal(0.9, optimizer.UpdateParameters(parameter, gradient)[0], 12);
        Assert.Equal(0.81, optimizer.UpdateParameters(parameter, gradient)[0], 12);

        optimizer.Reset();

        Assert.Equal(0.9, optimizer.UpdateParameters(parameter, gradient)[0], 12);
    }

    /// <summary>
    /// Reset must clear tape-side velocity for the same parameter object, not merely reset shared counters.
    /// </summary>
    [Fact]
    public void Reset_ClearsTapeVelocityForReusedParameters()
    {
        var optimizer = CreateOptimizer();
        var parameter = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var gradient = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var context = CreateContext(parameter, gradient);

        optimizer.Step(context);
        optimizer.Step(context);
        Assert.Equal(0.71, parameter[0], 12);

        parameter[0] = 1.0;
        optimizer.Reset();
        optimizer.Step(context);

        Assert.Equal(0.9, parameter[0], 12);
    }

#if !NET462
    /// <summary>
    /// Reset must dispose device velocity so the first GPU update after a restart cannot inherit
    /// momentum from the previous run, even when the caller reuses the same parameter buffer.
    /// </summary>
    [SkippableFact(Timeout = 60000)]
    [Trait("Category", "GPU")]
    public async Task Reset_ClearsGpuVelocityForReusedParameterBuffer()
    {
        await Task.Yield();
        using var engine = new DirectGpuEngine();
        var backend = engine.Backend;
        if (!engine.IsAvailable || backend is null)
        {
            Skip.If(true, "DirectGpu backend is unavailable on this runner.");
            return;
        }

        var optimizer = CreateOptimizer();
        using var parameter = backend.AllocateBuffer(new[] { 1.0f });
        using var gradient = backend.AllocateBuffer(new[] { 1.0f });
        using var initialParameter = backend.AllocateBuffer(new[] { 1.0f });

        optimizer.UpdateParametersGpu(parameter, gradient, 1, backend);
        optimizer.UpdateParametersGpu(parameter, gradient, 1, backend);
        backend.Synchronize();
        Assert.InRange(backend.DownloadBuffer(parameter)[0], 0.7099f, 0.7101f);

        backend.Copy(initialParameter, parameter, 1);
        optimizer.Reset();
        optimizer.UpdateParametersGpu(parameter, gradient, 1, backend);
        backend.Synchronize();

        Assert.InRange(backend.DownloadBuffer(parameter)[0], 0.8999f, 0.9001f);
        optimizer.DisposeGpuState();
    }
#endif

    private static MomentumOptimizer<double, Tensor<double>, Tensor<double>> CreateOptimizer()
    {
        return new MomentumOptimizer<double, Tensor<double>, Tensor<double>>(
            null!,
            new MomentumOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.1,
                InitialMomentum = 0.9,
                UseAdaptiveLearningRate = false,
            });
    }

    private static TapeStepContext<double> CreateContext(
        Tensor<double> parameter,
        Tensor<double> gradient)
    {
        Tensor<double> Forward(Tensor<double> _, Tensor<double> __) => parameter;
        Tensor<double> Loss(Tensor<double> prediction, Tensor<double> _) => prediction;

        return new TapeStepContext<double>(
            new[] { parameter },
            new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
            {
                [parameter] = gradient,
            },
            1.0,
            parameter,
            parameter,
            Forward,
            Loss);
    }
}
