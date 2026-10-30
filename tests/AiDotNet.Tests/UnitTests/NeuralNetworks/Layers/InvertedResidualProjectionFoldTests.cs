using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Graph;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// An inverted residual block folds its linear projection's batch norm into the convolution in inference whatever its
/// block activation, including one the fused kernels do not compute (hard-swish, MobileNetV3). The folded forward must
/// equal the unfolded one, which a layer-forward observer forces (folding is suppressed while one records).
/// </summary>
public class InvertedResidualProjectionFoldTests
{
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

        var folded = block.Forward(x).ToArray();
        float[] unfolded;
        using (new LayerForwardObserver<float>())
            unfolded = block.Forward(x).ToArray();

        Assert.Equal(unfolded.Length, folded.Length);
        Assert.Contains(unfolded, v => v != 0f);
        for (int i = 0; i < folded.Length; i++)
            Assert.True(Math.Abs(unfolded[i] - folded[i]) <= 1e-4f * (1f + Math.Abs(unfolded[i])),
                $"[{i}]: unfolded {unfolded[i]}, folded {folded[i]}");
    }
}
