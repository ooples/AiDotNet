using System.Collections.Generic;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// LAMB's <c>ExcludeBiasFromWeightDecay</c> in tape training: rank-1 parameters (biases, normalization scales) take
/// no weight decay.
/// </summary>
public class LambWeightDecayExclusionTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Step_WithOnlyWeightDecayActing_DecaysBiasesOnlyWhenNotExcluded(bool exclude)
    {
        // With a zero gradient the Adam part is 0/(0 + eps) = 0, so the update is the decay term alone and the trust
        // ratio ||w|| / ||wd w|| turns it into exactly lr * w for any decayed tensor.
        const double lr = 0.1;
        var optimizer = new LAMBOptimizer<double, Tensor<double>, Tensor<double>>(
            null!, new LAMBOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = lr,
                WeightDecay = 0.5,
                ExcludeBiasFromWeightDecay = exclude,
            });
        var weight = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.4, -0.2, 0.3, 0.1 }));
        var bias = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.5, -0.5 }));

        optimizer.Step(new TapeStepContext<double>(
            new[] { weight, bias },
            new Dictionary<Tensor<double>, Tensor<double>>
            {
                [weight] = new Tensor<double>(new[] { 2, 2 }),
                [bias] = new Tensor<double>(new[] { 2 }),
            },
            0.0));

        Assert.Equal(0.4 * (1 - lr), weight[0], 12);
        Assert.Equal(0.1 * (1 - lr), weight[3], 12);
        Assert.Equal(exclude ? 0.5 : 0.5 * (1 - lr), bias[0], 12);
        Assert.Equal(exclude ? -0.5 : -0.5 * (1 - lr), bias[1], 12);
    }

    [Fact]
    public void Step_WithATinyButNonZeroWeightNorm_ScalesByTheTrustRatio()
    {
        // ||w|| = 5e-8 is below Adam's epsilon but not zero. The reference implementations (and the fused kernel) fall
        // back to a ratio of 1 only for a zero norm, so the step is lr * ||w|| / ||r|| * r, about 5e-9 here; treating
        // the small norm as zero would move the weight by the full lr.
        var optimizer = new LAMBOptimizer<double, Tensor<double>, Tensor<double>>(
            null!, new LAMBOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.1,
                WeightDecay = 0.0,
                ClipTrustRatio = false,
            });
        var weight = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 3e-8, 4e-8, 0.0, 0.0 }));

        optimizer.Step(new TapeStepContext<double>(
            new[] { weight },
            new Dictionary<Tensor<double>, Tensor<double>>
            {
                [weight] = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 })),
            },
            0.0));

        for (int i = 0; i < 4; i++)
            Assert.True(System.Math.Abs(weight[i] - new[] { 3e-8, 4e-8, 0.0, 0.0 }[i]) < 1e-7,
                $"weight[{i}] moved to {weight[i]:R}; a sub-epsilon norm was treated as zero");
    }
}
