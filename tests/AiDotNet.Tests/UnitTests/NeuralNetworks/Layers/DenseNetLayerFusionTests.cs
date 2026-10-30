using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// In inference a DenseNet layer runs its pre-activation BN1 + ReLU as one pass and folds BN2 + ReLU into the 1x1
/// convolution before it, and a transition layer runs its BN + ReLU as one pass. Both must equal the unfused layers up
/// to rounding.
/// </summary>
[Collection("InferenceFusion")]
public class DenseNetLayerFusionTests
{
    private static Tensor<double> Random(Random rng, int[] shape, double scale)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static T Trained<T>(T layer, int channels, int seed) where T : ILayer<double>
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        layer.SetTrainingMode(true);
        for (int step = 0; step < 4; step++) layer.Forward(Random(rng, new[] { 3, channels, 8, 8 }, 2.0));
        // Every parameter (weights, gamma/beta, running statistics) away from its initial value; all positive keeps each
        // running variance valid.
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) p[i] = 0.05 + rng.NextDouble() * 0.6;
        layer.SetParameters(p);
        layer.SetTrainingMode(false);
        return layer;
    }

    private static void AssertFusedEqualsUnfused(ILayer<double> layer, Tensor<double> x)
    {
        // Force each mode explicitly: under AIDOTNET_FOLD_BATCHNORM=0 the "fused" forward would otherwise run unfused too.
        bool previous = InferenceFusionSettings.FoldBatchNormIntoConvolution;
        double[] fused, unfused;
        try
        {
            InferenceFusionSettings.FoldBatchNormIntoConvolution = true;
            fused = layer.Forward(x).ToArray();
            InferenceFusionSettings.FoldBatchNormIntoConvolution = false;
            unfused = layer.Forward(x).ToArray();
        }
        finally { InferenceFusionSettings.FoldBatchNormIntoConvolution = previous; }
        Assert.Equal(unfused.Length, fused.Length);
        for (int i = 0; i < fused.Length; i++)
            Assert.True(Math.Abs(unfused[i] - fused[i]) <= 1e-10 + 1e-9 * Math.Abs(unfused[i]),
                $"[{i}]: unfused {unfused[i]}, fused {fused[i]}");
        Assert.Contains(fused, v => Math.Abs(v) > 1e-6);
    }

    [Fact]
    public void DenseBlockLayer_FusedInference_EqualsUnfused()
    {
        var layer = Trained(new DenseBlockLayer<double>(growthRate: 12), channels: 16, seed: 3);
        AssertFusedEqualsUnfused(layer, Random(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(4), new[] { 2, 16, 8, 8 }, 30.0));
    }

    /// <summary>Counts the batch-norm inference passes that normalize with one particular gamma.</summary>
    private sealed class CountingEngine : CpuEngine
    {
        internal object? Gamma;
        internal int Passes;

        // The activation overload delegates here, so this sees every inference normalization, fused or not.
        public override Tensor<T> BatchNormInference<T>(Tensor<T> x, Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean,
            Tensor<T> variance, double epsilon)
        {
            if (ReferenceEquals(gamma, Gamma)) Passes++;
            return base.BatchNormInference(x, gamma, beta, mean, variance, epsilon);
        }
    }

    private static TField Field<TField>(object owner, string name)
    {
        var field = owner.GetType().GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new InvalidOperationException($"{owner.GetType().Name} has no field {name}.");
        return Assert.IsType<TField>(field.GetValue(owner));
    }

    [Fact]
    public void DenseBlockLayer_KeepsTheFusedPreActivation_WhenTheBottleneckFoldDeclines()
    {
        var layer = Trained(new DenseBlockLayer<double>(growthRate: 12), channels: 16, seed: 7);
        var bn1 = Field<BatchNormalizationLayer<double>>(layer, "_bn1");
        var bn2 = Field<BatchNormalizationLayer<double>>(layer, "_bn2");
        Assert.True(bn1.TryGetFoldingStatistics(16, out var gamma1, out _, out _, out _, out _));
        // A batch norm in training mode does not fold, so the 1x1 fold declines while BN1 + ReLU still fuses.
        bn2.SetTrainingMode(true);

        var engine = new CountingEngine { Gamma = gamma1 };
        var previousEngine = AiDotNetEngine.Current;
        bool previous = InferenceFusionSettings.FoldBatchNormIntoConvolution;
        try
        {
            AiDotNetEngine.Current = engine;
            InferenceFusionSettings.FoldBatchNormIntoConvolution = true;
            layer.Forward(Random(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(8), new[] { 2, 16, 8, 8 }, 3.0));
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            InferenceFusionSettings.FoldBatchNormIntoConvolution = previous;
        }

        // One BN1 pass: the fused one. Discarding it when the fold declined ran BN1 a second time, unfused.
        Assert.Equal(1, engine.Passes);
    }

    [Fact]
    public void TransitionLayer_FusedInference_EqualsUnfused()
    {
        var layer = Trained(new TransitionLayer<double>(0.5), channels: 16, seed: 5);
        AssertFusedEqualsUnfused(layer, Random(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(6), new[] { 2, 16, 8, 8 }, 30.0));
    }
}
