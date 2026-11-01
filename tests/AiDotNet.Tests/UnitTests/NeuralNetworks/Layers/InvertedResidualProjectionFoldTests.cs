using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Graph;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// An inverted residual block folds its linear projection's batch norm into the convolution in inference whatever its
/// block activation, including one the fused kernels do not compute (hard-swish, MobileNetV3). The folded forward must
/// equal the unfolded one, which a layer-forward observer forces (folding is suppressed while one records).
/// </summary>
/// <remarks>
/// In the InferenceFusion collection because the fold reads the static
/// <see cref="InferenceFusionSettings.FoldBatchNormIntoConvolution"/>, which the other fusion tests toggle.
/// </remarks>
[Collection("InferenceFusion")]
public class InvertedResidualProjectionFoldTests
{
    /// <summary>Counts the folded convolution-batch-norm calls.</summary>
    private sealed class CountingEngine : CpuEngine
    {
        internal int Folds;

        public override Tensor<T> FusedConv2DBatchNorm<T>(
            Tensor<T> input, Tensor<T> kernel, Tensor<T>? bias,
            Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVariance, double epsilon,
            int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
            FusedActivationType activation)
        {
            Folds++;
            return base.FusedConv2DBatchNorm(input, kernel, bias, gamma, beta, runningMean, runningVariance, epsilon,
                strideH, strideW, padH, padW, dilationH, dilationW, activation);
        }
    }

    [Fact]
    public void HardSwishBlock_FoldedProjection_EqualsUnfolded()
    {
        var block = new InvertedResidualBlock<float>(8, expansionRatio: 4, activationFunction: new HardSwishActivation<float>());
        var rng = new Random(5);
        var x = new Tensor<float>(new[] { 2, 8, 6, 6 });
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        // Train-mode passes give the batch norms non-trivial running statistics to fold.
        block.SetTrainingMode(true);
        for (int i = 0; i < 3; i++) block.Forward(x);
        block.SetTrainingMode(false);

        var engine = new CountingEngine();
        var previousEngine = AiDotNetEngine.Current;
        bool previousFold = InferenceFusionSettings.FoldBatchNormIntoConvolution;
        float[] folded, unfolded;
        int foldsWhileFolding, foldsWhileObserved;
        try
        {
            AiDotNetEngine.Current = engine;
            InferenceFusionSettings.FoldBatchNormIntoConvolution = true;
            folded = block.Forward(x).ToArray();
            foldsWhileFolding = engine.Folds;

            engine.Folds = 0;
            using (new LayerForwardObserver<float>())
                unfolded = block.Forward(x).ToArray();
            foldsWhileObserved = engine.Folds;
        }
        finally
        {
            InferenceFusionSettings.FoldBatchNormIntoConvolution = previousFold;
            AiDotNetEngine.Current = previousEngine;
        }

        // Hard-swish is not a fused activation, so the only stage that can fold is the linear projection: exactly one
        // fold proves it ran, and none under the observer proves the comparison is against the unfolded path.
        Assert.Equal(1, foldsWhileFolding);
        Assert.Equal(0, foldsWhileObserved);
        Assert.Equal(unfolded.Length, folded.Length);
        Assert.Contains(unfolded, v => v != 0f);
        for (int i = 0; i < folded.Length; i++)
            Assert.True(Math.Abs(unfolded[i] - folded[i]) <= 1e-4f * (1f + Math.Abs(unfolded[i])),
                $"[{i}]: unfolded {unfolded[i]}, folded {folded[i]}");
    }
}
