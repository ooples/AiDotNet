using System;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SelfSupervisedLearning;

/// <summary>
/// Half-precision references for self-supervised reductions where master counted or accumulated
/// outside T. Half sums of ones stop at 2048, so counts must be taken in int.
/// </summary>
public class SelfSupervisedHalfPrecisionTests
{
    [Fact]
    public void MaePerSampleLoss_Half_DividesByTheTrueMaskedPatchCount()
    {
        // One sample, 3000 masked one-dimensional patches; only patch 0 differs (by 2, loss 4).
        // Master: 4 / 3000. A Half count of the 3000 masked patches would stop at 2048.
        const int patches = 3000;
        var reconstructed = new Tensor<Half>(new[] { 1, patches, 1 });
        reconstructed[0, 0, 0] = (Half)2.0f;
        var original = new Tensor<Half>(new[] { 1, patches, 1 });
        var mask = new Tensor<Half>(new[] { 1, patches });
        for (int p = 0; p < patches; p++)
        {
            mask[0, p] = (Half)1.0f;
        }

        var loss = new MAEReconstructionLoss<Half>(normalize: true, perPatchNormalization: false);
        Half[] perSample = loss.ComputePerSampleLoss(reconstructed, original, mask);

        Assert.Single(perSample);
        Assert.Equal(4.0 / patches, (double)perSample[0], 5);
    }
}
