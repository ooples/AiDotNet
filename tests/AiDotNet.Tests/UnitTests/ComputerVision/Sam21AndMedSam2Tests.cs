using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// The SAM 2.1 and MedSAM2 public surfaces on the shared SAM 2 network (itself checked against the official code in
/// <c>Sam2VideoReferenceTests</c>): the image predictor, the automatic mask generator, tracking, and MedSAM2's
/// box-on-a-key-slice volumes with the backward pass started afresh.
/// </summary>
public class Sam21AndMedSam2Tests
{
    private static NeuralNetworkArchitecture<double> Architecture(int height, int width, int channels) => new(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.BinaryClassification,
        inputHeight: height, inputWidth: width, inputDepth: channels, outputSize: 1);

    private static SAM21Options SmallSam21() => new()
    {
        ModelSize = SAM21ModelSize.Tiny, ImageSize = 32, HieraEmbeddingDimension = 8, HieraStageDepths = new[] { 1, 1, 1, 1 },
        HieraInitialHeadCount = 1, HieraWindowSizes = new[] { 4, 4, 2, 1 }, HieraGlobalAttentionBlockIndexes = new[] { 2 },
        ModelDimension = 32, MemoryDimension = 16, DecoderHeadCount = 2, MaskDecoderMlpDimension = 32, MemoryAttentionLayerCount = 1,
        MemoryAttentionFeedForwardDimension = 32, MemoryFuserDimension = 32, MemoryBankSize = 3, Seed = 5,
    };

    private static MedSAM2Options SmallMedSam2() => new()
    {
        ImageSize = 32, HieraEmbeddingDimension = 8, HieraStageDepths = new[] { 1, 1, 1, 1 }, HieraInitialHeadCount = 1,
        HieraWindowSizes = new[] { 4, 4, 2, 1 }, HieraGlobalAttentionBlockIndexes = new[] { 2 }, ModelDimension = 32, MemoryDimension = 16,
        DecoderHeadCount = 2, MaskDecoderMlpDimension = 32, MemoryAttentionLayerCount = 1, MemoryAttentionFeedForwardDimension = 32,
        MemoryFuserDimension = 32, MemoryBankSize = 3, Seed = 5,
    };

    private static Tensor<double> Image(int channels, int height, int width, int seed)
    {
        var image = new Tensor<double>(new[] { channels, height, width });
        for (int i = 0; i < image.Length; i++) image[i] = 0.5 + 0.5 * Math.Sin(0.31 * i + seed);
        return image;
    }

    [Fact(Timeout = 600000)]
    public async Task Sam21_ImagePredictor_GivesMasksAtTheImageSize()
    {
        await Task.Yield();
        using var model = new SAM21<double>(Architecture(40, 48, 3), options: SmallSam21());
        model.SetImage(Image(3, 40, 48, 1));
        var points = model.SegmentFromPoints(new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 20.0, 15.0 })),
            new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 })));
        Assert.Equal(new[] { 3, 40, 48 }, points.Masks.Shape.ToArray());
        Assert.Equal(3, points.Scores.Length);
        Assert.All(points.Masks.ToArray(), v => Assert.True(v == 0.0 || v == 1.0));
        var box = model.SegmentFromBox(new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 8.0, 6.0, 30.0, 28.0 })));
        Assert.Equal(new[] { 1, 40, 48 }, box.Masks.Shape.ToArray());
        // A previous result's low-resolution logits are the next mask prompt.
        var low = box.LowResLogits!;
        var refined = model.SegmentFromMask(AiDotNetEngine.Current.Reshape(low, new[] { low.Shape[2], low.Shape[3] }));
        Assert.Equal(new[] { 1, 40, 48 }, refined.Masks.Shape.ToArray());
        Assert.NotNull(model.SegmentEverything());
        var video = Enumerable.Range(0, 3).Select(t => Image(3, 40, 48, t)).ToList();
        var tracked = model.TrackObject(video, new float[,] { { 20, 15 } }, new[] { 1 });
        Assert.Equal(3, tracked.Count);
        Assert.All(tracked, m => Assert.Equal(new[] { 40, 48 }, m.Shape.ToArray()));
    }

    [Fact(Timeout = 600000)]
    public async Task MedSam2_Volume_PropagatesTheKeySliceBoxBothWays_EachFromAFreshState()
    {
        await Task.Yield();
        using var model = new MedSAM2<double>(Architecture(32, 32, 1), options: SmallMedSam2());
        var volume = new Tensor<double>(new[] { 1, 5, 32, 32 });
        for (int i = 0; i < volume.Length; i++) volume[i] = 0.5 + 0.5 * Math.Sin(0.13 * i);
        var box = new float[] { 6, 8, 24, 26 };
        var result = model.SegmentVolume(volume, keySlice: 2, box: box);
        Assert.Equal(new[] { 5, 32, 32 }, result.Labels.Shape.ToArray());
        Assert.Equal(new[] { 1, 5, 32, 32 }, result.Probabilities.Shape.ToArray());

        // The backward pass starts again from the key slice: slice 1's mask is what tracking slices 2 then 1 gives, with
        // none of the forward pass's memories.
        var backward = new Tensor<double>(new[] { 1, 2, 32, 32 });
        for (int s = 0; s < 2; s++)
            for (int j = 0; j < 32 * 32; j++) backward[s * 1024 + j] = volume[(2 - s) * 1024 + j];
        var fromKey = model.SegmentVolume(backward, keySlice: 0, box: box);
        for (int j = 0; j < 32 * 32; j++)
            Assert.Equal(fromKey.Probabilities[1024 + j], result.Probabilities[1024 + j], 9);
    }
}

