using System;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Regressions found reviewing the vectorized segmentation rewrites: batched shapes of the k-means hard assignment
/// and the text classifier, and a rejected correction mask that stayed queued in the video trackers.
/// </summary>
public class SegmentationSweepReviewRegressionTests
{
    private static Tensor<double> T(int[] shape, params double[] values)
        => new Tensor<double>(shape, new Vector<double>(values));

    private static NeuralNetworkArchitecture<float> Arch()
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Deep, 0, 32, 32, 3, 0);

    private static Tensor<float> Rand(params int[] shape)
    {
        int total = 1;
        foreach (int s in shape) total *= s;
        var data = new float[total];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < total; i++) data[i] = (float)rng.NextDouble();
        return new Tensor<float>(shape, new Vector<float>(data));
    }

    [Fact]
    public void KMaXHardAssignment_KeepsEachBatchItemsOwnAssignment()
    {
        // [B=2, N=2, hw=3]. Item 0: pixels go to clusters 0, 1, 0. Item 1: pixels go to clusters 1, 1, 0.
        var assignment = KMaXTransformerDecoder.HardAssignment(T(new[] { 2, 2, 3 },
            1, 0, 5,
            0, 2, 4,
            0, 1, 9,
            3, 2, 4));

        Assert.Equal(new[] { 2, 2, 3 }, assignment.Shape.ToArray());
        Assert.Equal(new[] { 1.0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0 }, assignment.ToArray());
    }

    [Fact]
    public void Classify_ScoresEveryBatchItemAgainstTheSameText()
    {
        // [B=2, D=2, H=1, W=1]: item 0 is (3, 4) -> (0.6, 0.8); item 1 is (0, 2) -> (0, 1).
        var pixels = T(new[] { 2, 2, 1, 1 }, 3, 4, 0, 2);
        var text = TextEmbeddingClassifier.NormalizeRows(T(new[] { 2, 2 }, 1, 0, 0, 2));

        var logits = TextEmbeddingClassifier.Classify(pixels, text, 10.0);

        Assert.Equal(new[] { 2, 2, 1, 1 }, logits.Shape.ToArray());
        Assert.Equal(6.0, logits[0, 0, 0, 0], 1e-9);
        Assert.Equal(8.0, logits[0, 1, 0, 0], 1e-9);
        Assert.Equal(0.0, logits[1, 0, 0, 0], 1e-9);
        Assert.Equal(10.0, logits[1, 1, 0, 0], 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task Deva_ARejectedCorrectionDoesNotFailLaterFrames()
    {
        await System.Threading.Tasks.Task.Yield();
        using var model = new DEVA<float>(Arch(), options: new DEVAOptions { NumClasses = 5, ModelSize = DEVAModelSize.Base });
        var video = (IVideoSegmentation<float>)model;
        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32), new[] { 7 });

        video.AddCorrection(7, Rand(4, 4));
        Assert.Throws<ArgumentException>(() => video.PropagateToFrame(Rand(3, 32, 32)));

        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.Equal(new[] { 1, 32, 32 }, result.Masks.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task UniVS_ARejectedCorrectionDoesNotFailLaterFrames()
    {
        await System.Threading.Tasks.Task.Yield();
        using var model = new UniVS<float>(Arch(), options: new UniVSOptions { NumClasses = 5, ModelSize = UniVSModelSize.R50 });
        var video = (IVideoSegmentation<float>)model;
        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32), new[] { 7 });

        video.AddCorrection(7, Rand(4, 4));
        Assert.Throws<ArgumentException>(() => video.PropagateToFrame(Rand(3, 32, 32)));

        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.Equal(new[] { 1, 32, 32 }, result.Masks.Shape.ToArray());
    }
}
