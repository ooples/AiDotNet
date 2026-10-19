using System.Collections.Generic;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// The AdamW tape step skips entirely when any gradient element is NaN or infinite, so the parameter and the m/v
/// moments are left untouched. On a host engine the probe is a parallel sum of squares
/// (FusedOptimizer.TrySumOfSquaresHost). The bad element sits in the last chunk of a multi-chunk gradient, and a
/// finite-gradient control proves the same step does move the parameter.
/// </summary>
public class AdamWTapeAnomalySkipTests
{
    private const int Length = 150_000;

    private static float[] RunStep(float badValue, bool inject)
    {
        var optimizer = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(
            null, new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>> { InitialLearningRate = 0.01 });
        var param = new Tensor<float>(new[] { Length });
        var grad = new Tensor<float>(new[] { Length });
        for (int i = 0; i < Length; i++) { param[i] = 1f; grad[i] = 0.1f; }
        if (inject) grad[Length - 1] = badValue;

        optimizer.Step(new TapeStepContext<float>(
            parameters: new[] { param },
            gradients: new Dictionary<Tensor<float>, Tensor<float>> { [param] = grad },
            loss: 0f));
        return param.ToArray();
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonFiniteGradient_SkipsTheStep(float bad)
    {
        var after = RunStep(bad, inject: true);
        for (int i = 0; i < Length; i++)
            Assert.True(after[i] == 1f, $"param[{i}] = {after[i]} changed although the gradient held {bad}");
    }

    [Fact]
    public void FiniteGradient_TakesTheStep()
    {
        var after = RunStep(0f, inject: false);
        Assert.True(after[0] < 1f && after[Length - 1] < 1f, $"finite step did not move the parameter: {after[0]}, {after[Length - 1]}");
    }
}
