using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// In inference an inverted residual block folds its expansion and projection batch norms (and the ReLU6) into the
/// convolutions, and runs the depthwise stage as a bias-free depthwise convolution plus one batch-norm-and-ReLU6 pass with
/// the depthwise bias folded into the mean. The result must equal the unfused block up to rounding, with and without
/// expansion and residual, at stride 1 and 2; training never fuses.
/// </summary>
[Collection("InferenceFusion")]
public class InvertedResidualBlockFusionTests
{
    private static Tensor<T> Random<T>(Random rng, int[] shape, double scale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ops.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static InvertedResidualBlock<T> TrainedBlock<T>(int inChannels, int outChannels, int expansion, int stride, int seed)
    {
        var block = new InvertedResidualBlock<T>(outChannels, expansion, stride);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        block.SetTrainingMode(true);
        for (int step = 0; step < 4; step++) block.Forward(Random<T>(rng, new[] { 3, inChannels, 8, 8 }, 2.0));
        // Every parameter (weights, biases, gamma/beta, running statistics) away from its initial value; all positive
        // keeps each running variance valid.
        var ops = MathHelper.GetNumericOperations<T>();
        var p = block.GetParameters();
        for (int i = 0; i < p.Length; i++) p[i] = ops.FromDouble(0.05 + rng.NextDouble() * 0.6);
        block.SetParameters(p);
        block.SetTrainingMode(false);
        return block;
    }

    [Theory]
    [InlineData(16, 16, 6, 1)]   // expansion + residual
    [InlineData(16, 24, 6, 2)]   // expansion, strided, no residual
    [InlineData(16, 16, 1, 1)]   // no expansion stage
    public void FusedInference_EqualsTheUnfusedBlock(int inChannels, int outChannels, int expansion, int stride)
    {
        var block = TrainedBlock<double>(inChannels, outChannels, expansion, stride, seed: 11);
        var x = Random<double>(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5), new[] { 2, inChannels, 8, 8 }, 3.0);
        // Force each mode explicitly: under AIDOTNET_FOLD_BATCHNORM=0 the "fused" forward would otherwise run unfused too.
        bool previous = InferenceFusionSettings.FoldBatchNormIntoConvolution;
        double[] fused, unfused;
        try
        {
            InferenceFusionSettings.FoldBatchNormIntoConvolution = true;
            fused = block.Forward(x).ToArray();
            InferenceFusionSettings.FoldBatchNormIntoConvolution = false;
            unfused = block.Forward(x).ToArray();
        }
        finally { InferenceFusionSettings.FoldBatchNormIntoConvolution = previous; }

        Assert.Equal(unfused.Length, fused.Length);
        int clampedHigh = 0;
        for (int i = 0; i < fused.Length; i++)
        {
            Assert.True(Math.Abs(unfused[i] - fused[i]) <= 1e-10 + 1e-9 * Math.Abs(unfused[i]),
                $"[{i}]: unfused {unfused[i]}, fused {fused[i]}");
            if (unfused[i] >= 6) clampedHigh++;
        }
        Assert.Contains(fused, v => Math.Abs(v) > 1e-6);   // not a degenerate all-zero output
    }
}