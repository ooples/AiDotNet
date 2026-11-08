using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// <see cref="RootMeanSquarePropagationOptimizerOptions{T,TInput,TOutput}.InitialMeanSquare"/>: TensorFlow 1's
/// tf.train.RMSPropOptimizer starts the mean of squared gradients at 1, PyTorch's RMSprop at 0. The expected values
/// are the update rule computed by hand: ms = decay * ms + (1 - decay) * g^2, p -= lr * g / (sqrt(ms) + eps).
/// </summary>
public class RmsPropInitialMeanSquareTests
{
    private const double Lr = 0.01, Decay = 0.9, Eps = 1e-10;
    private static readonly double[] Gradient = { 0.2, -0.5, 0.05 };
    private static readonly double[] Start = { 1.0, -1.0, 0.5 };

    private static RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>> Create(double initial) => new(null,
        new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = Lr, Decay = Decay, Epsilon = Eps, InitialMeanSquare = initial, EnableGradientClipping = false,
        });

    private static double[] Expected(double initial, int steps)
    {
        var p = (double[])Start.Clone();
        var ms = new double[p.Length];
        for (int i = 0; i < ms.Length; i++) ms[i] = initial;
        for (int step = 0; step < steps; step++)
            for (int i = 0; i < p.Length; i++)
            {
                ms[i] = Decay * ms[i] + (1 - Decay) * Gradient[i] * Gradient[i];
                p[i] -= Lr * Gradient[i] / (Math.Sqrt(ms[i]) + Eps);
            }

        return p;
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.0)]
    public void TapeStep_StartsTheMeanSquareAtTheInitialValue(double initial)
    {
        const int steps = 3;
        var optimizer = Create(initial);
        var parameter = new Tensor<double>((double[])Start.Clone(), new[] { Start.Length });
        for (int step = 0; step < steps; step++)
        {
            var gradients = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
            {
                [parameter] = new Tensor<double>((double[])Gradient.Clone(), new[] { Gradient.Length }),
            };
            optimizer.Step(new TapeStepContext<double>(new[] { parameter }, gradients, loss: 0.0));
        }

        var expected = Expected(initial, steps);
        var actual = parameter.ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-12, $"element {i}: {actual[i]:R}, expected {expected[i]:R}");
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.0)]
    public void VectorUpdate_StartsTheMeanSquareAtTheInitialValue(double initial)
    {
        const int steps = 3;
        var optimizer = Create(initial);
        var p = new Vector<double>((double[])Start.Clone());
        for (int step = 0; step < steps; step++) p = optimizer.UpdateParameters(p, new Vector<double>((double[])Gradient.Clone()));
        var expected = Expected(initial, steps);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(p[i] - expected[i]) <= 1e-12, $"element {i}: {p[i]:R}, expected {expected[i]:R}");
    }

    /// <summary>The fused kernel's mean square starts at zero, so a non-zero start stays on the eager step.</summary>
    [Fact]
    public void FusedConfig_IsDeclined_ForANonZeroStart()
    {
        Assert.False(((IFusedOptimizerSpec)Create(1.0)).TryGetFusedOptimizerConfig(out _));
        Assert.True(((IFusedOptimizerSpec)Create(0.0)).TryGetFusedOptimizerConfig(out _));
    }

    /// <summary>Rejected when the options are configured, not at the first state allocation mid-training.</summary>
    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InvalidStart_IsRejected_ByTheOptions(double initial)
    {
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => options.InitialMeanSquare = initial);
        Assert.Equal(nameof(options.InitialMeanSquare), error.ParamName);
        Assert.Equal(0.0, options.InitialMeanSquare);
    }

    /// <summary>Streaming RMSProp is uncentered and starts at zero, so it refuses both variants rather than train another rule.</summary>
    [Theory]
    [InlineData(1.0, false)]
    [InlineData(0.0, true)]
    public void Streaming_Refuses_ANonZeroStartOrTheCenteredForm(double initial, bool centered)
    {
        var optimizer = new RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new RootMeanSquarePropagationOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = Lr, InitialMeanSquare = initial, Centered = centered,
            });
        var error = Assert.Throws<NotSupportedException>(() => StreamingOptimizerResolver<float>.Create(
            optimizer, useStreamingDefaults: false, fallbackLearningRate: 0.01, fallbackWeightDecay: 0.0));
        Assert.Contains("InitialMeanSquare", error.Message);
    }
}
