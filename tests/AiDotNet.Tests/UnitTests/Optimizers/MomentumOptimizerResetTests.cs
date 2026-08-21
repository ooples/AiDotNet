using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

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
