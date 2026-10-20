using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// EfficientTAM's streaming tracking surface and the video predictor's output post-processing shared by the SAM 2 family
/// (the network itself is checked against the official code in <c>EfficientTamReferenceTests</c>).
/// </summary>
public class EfficientTamModelTests
{
    [Fact]
    public void FillHoles_FillsSmallEightConnectedBackgroundRegions()
    {
        // A 6x6 foreground plane with a 2x2 hole (filled), and a background region of 9 pixels touching the border (kept).
        var logits = new Tensor<double>(new[] { 1, 1, 8, 8 });
        for (int i = 0; i < logits.Length; i++) logits[i] = 1.0;
        foreach (var (y, x) in new[] { (2, 2), (2, 3), (3, 2), (3, 3) }) logits[0, 0, y, x] = -1.0;
        for (int y = 5; y < 8; y++) for (int x = 5; x < 8; x++) logits[0, 0, y, x] = -2.0;
        // Two diagonal pixels form one 8-connected hole of 2.
        logits[0, 0, 0, 6] = -1.0;
        logits[0, 0, 1, 7] = -1.0;
        var filled = Sam2Runtime<double>.FillHoles(logits, 8);
        Assert.Equal(0.1, filled[0, 0, 2, 2], 12);
        Assert.Equal(0.1, filled[0, 0, 3, 3], 12);
        Assert.Equal(0.1, filled[0, 0, 0, 6], 12);
        Assert.Equal(-2.0, filled[0, 0, 6, 6], 12);
        Assert.Equal(1.0, filled[0, 0, 0, 0], 12);
    }

    private static EfficientTAMOptions Small() => new()
    {
        ModelSize = EfficientTAMModelSize.Tiny, ImageSize = 32, VitEmbeddingDimension = 16, VitDepth = 2, VitHeadCount = 2, VitWindowSize = 2,
        VitGlobalAttentionBlockIndexes = new[] { 1 }, ModelDimension = 32, MemoryDimension = 16, DecoderHeadCount = 2, MaskDecoderMlpDimension = 32,
        MemoryAttentionLayerCount = 1, MemoryAttentionFeedForwardDimension = 32, MemoryFuserDimension = 32, MemoryBankSize = 3, Seed = 5,
        MemoryCrossAttention = MemoryCrossAttention.PooledScores,
    };

    private static Tensor<double> Frame(int seed)
    {
        var frame = new Tensor<double>(new[] { 3, 40, 48 });
        for (int i = 0; i < frame.Length; i++) frame[i] = 0.5 + 0.5 * Math.Sin(0.23 * i + seed);
        return frame;
    }

    [Fact(Timeout = 600000)]
    public async Task Streaming_TracksEachObject_AndTakesCorrections()
    {
        await Task.Yield();
        using var model = new EfficientTAM<double>(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification, inputHeight: 40, inputWidth: 48, inputDepth: 3, outputSize: 1), options: Small());
        var video = (IVideoSegmentation<double>)model;
        var masks = new Tensor<double>(new[] { 2, 40, 48 });
        for (int y = 5; y < 18; y++) for (int x = 6; x < 20; x++) masks[0, y, x] = 1;
        for (int y = 20; y < 35; y++) for (int x = 25; x < 40; x++) masks[1, y, x] = 1;
        video.InitializeTracking(Frame(0), masks, new[] { 7, 9 });
        for (int t = 1; t < 4; t++)
        {
            if (t == 2)
            {
                var correction = new Tensor<double>(new[] { 40, 48 });
                for (int y = 20; y < 35; y++) for (int x = 25; x < 40; x++) correction[y, x] = 1;
                video.AddCorrection(9, correction);
            }
            var result = video.PropagateToFrame(Frame(t));
            Assert.Equal(new[] { 7, 9 }, result.ObjectIds);
            Assert.Equal(new[] { 2, 40, 48 }, result.Masks.Shape.ToArray());
            Assert.All(result.Masks.ToArray(), v => Assert.InRange(v, 0.0, 1.0));
            if (t == 2)
            {
                // A correction is the object's mask on this frame, taken as the output.
                Assert.True(result.Masks[1, 25, 30] > 0.99 && result.Masks[1, 2, 2] < 0.01, $"{result.Masks[1, 25, 30]} {result.Masks[1, 2, 2]}");
            }
        }
        video.ResetTracking();
    }
}
