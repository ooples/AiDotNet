using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The clipping options act on the tape step of every optimizer, as they do on the vector path. Before, only Adam and
/// AdamW clipped there (Adam by norm only), so the same options trained clipped through Optimize and unclipped through
/// Train, and a by-value clip was silently ignored. Each case steps one optimizer on a raw gradient with clipping on
/// and a twin with clipping off on the gradient clipped by hand; the parameters must land in the same place.
/// </summary>
public class TapeStepGradientClippingTests
{
    private static readonly double[] RawGradient = { 3.0, -0.1, 0.5, -2.0 };
    private static readonly double[] Start = { 0.4, -0.3, 0.2, 0.1 };

    public static TheoryData<string> Optimizers => new() { "SGD", "Momentum", "RMSProp", "Adagrad", "Adam", "AdamW" };

    private static GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>> Create(
        string name, Action<GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>> configure)
    {
        T Configured<T>(T options) where T : GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            options.InitialLearningRate = 0.05;
            configure(options);
            return options;
        }

        return name switch
        {
            "SGD" => new StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "Momentum" => new MomentumOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new MomentumOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "RMSProp" => new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "Adagrad" => new AdagradOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new AdagradOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "Adam" => new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "AdamW" => new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
    }

    // Two steps on the same gradient, so a clip that only took effect on the first would show.
    private static double[] TwoSteps(GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>> optimizer, double[] gradient)
    {
        var parameter = new Tensor<double>((double[])Start.Clone(), new[] { Start.Length });
        for (int step = 0; step < 2; step++)
        {
            var gradients = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
            {
                [parameter] = new Tensor<double>((double[])gradient.Clone(), new[] { gradient.Length }),
            };
            optimizer.Step(new TapeStepContext<double>(new[] { parameter }, gradients, loss: 0.0));
        }

        return parameter.ToArray();
    }

    private static void AssertSameStep(string name, double[] clipped, double[] byHand)
    {
        for (int i = 0; i < clipped.Length; i++)
            Assert.True(Math.Abs(clipped[i] - byHand[i]) <= 1e-12,
                $"{name}, element {i}: clipped by the options {clipped[i]:R}, by hand {byHand[i]:R}");
    }

    [Theory]
    [MemberData(nameof(Optimizers))]
    public void ValueClip_ClampsEachGradientElement(string name)
    {
        const double bound = 0.25;
        var clipped = TwoSteps(Create(name, o =>
        {
            o.EnableGradientClipping = true;
            o.GradientClippingMethod = GradientClippingMethod.ByValue;
            o.MaxGradientValue = bound;
        }), RawGradient);
        var byHand = TwoSteps(Create(name, o => o.EnableGradientClipping = false),
            Array.ConvertAll(RawGradient, g => Math.Max(-bound, Math.Min(bound, g))));
        AssertSameStep(name, clipped, byHand);

        var unclipped = TwoSteps(Create(name, o => o.EnableGradientClipping = false), RawGradient);
        Assert.NotEqual(unclipped, clipped);
    }

    [Theory]
    [MemberData(nameof(Optimizers))]
    public void NormClip_ScalesTheGradientToTheMaximumNorm(string name)
    {
        const double maxNorm = 1.0;
        double norm = 0;
        foreach (var g in RawGradient) norm += g * g;
        norm = Math.Sqrt(norm);
        var clipped = TwoSteps(Create(name, o =>
        {
            o.EnableGradientClipping = true;
            o.GradientClippingMethod = GradientClippingMethod.ByNorm;
            o.MaxGradientNorm = maxNorm;
        }), RawGradient);
        // torch.nn.utils.clip_grad_norm_'s coefficient: max_norm / (total_norm + 1e-6).
        var byHand = TwoSteps(Create(name, o => o.EnableGradientClipping = false),
            Array.ConvertAll(RawGradient, g => g * maxNorm / (norm + 1e-6)));
        AssertSameStep(name, clipped, byHand);
    }

    /// <summary>
    /// The fused plan clips by global norm only, so it takes the optimizer's norm and refuses a by-value clip (NaN),
    /// for every optimizer rather than only Adam and AdamW.
    /// </summary>
    [Theory]
    [MemberData(nameof(Optimizers))]
    public void FusedClipThreshold_FollowsTheOptions(string name)
    {
        var byNorm = Create(name, o => { o.EnableGradientClipping = true; o.GradientClippingMethod = GradientClippingMethod.ByNorm; o.MaxGradientNorm = 0.7; });
        var byValue = Create(name, o => { o.EnableGradientClipping = true; o.GradientClippingMethod = GradientClippingMethod.ByValue; o.MaxGradientValue = 0.3; });
        var off = Create(name, o => o.EnableGradientClipping = false);
        Assert.Equal(0.7, byNorm.TapeStepGradientClipNorm);
        Assert.Equal(0.0, byNorm.TapeStepGradientClipValue);
        Assert.True(double.IsNaN(byValue.TapeStepGradientClipNorm));
        Assert.Equal(0.3, byValue.TapeStepGradientClipValue);
        Assert.Equal(0.0, off.TapeStepGradientClipNorm);
        Assert.Equal(0.0, off.TapeStepGradientClipValue);
    }
}
