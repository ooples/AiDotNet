using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// Grounded SAM 2 as the official demo runs it: Grounding DINO's boxes, each prompting SAM 2 for one mask at the image's
/// size; per-class unions for SegmentWithText; and detector-only training.
/// </summary>
public class GroundedSam2ModelTests
{
    private static GroundedSAM2<double> Small() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.ImageSegmentation,
            inputHeight: 48, inputWidth: 64, inputDepth: 3, outputSize: 32),
        new GroundedSAM2Options
        {
            Seed = 5,
            Detector = new GroundingDINOOptions
            {
                VisionDim = 16, SwinDepths = new[] { 1, 1, 2, 1 }, SwinHeads = new[] { 1, 2, 2, 4 }, SwinWindowSize = 4, SwinDropPathRate = 0.0,
                TextHiddenSize = 32, TextLayers = 2, TextHeads = 2, TextIntermediateSize = 64, DecoderDim = 32, NumVisionLayers = 1,
                NumDecoderLayers = 1, NumHeads = 4, FeedForwardDim = 64, NumSamplingPoints = 2, NumQueryPositions = 6, MaxSequenceLength = 16,
                MaxDetections = 6, FusionDropPathRate = 0.0, ConfidenceThreshold = 0.0,
            },
            SamImageSize = 64, SamBlocksPerStage = new[] { 1, 1, 2, 1 }, SamEmbedDims = new[] { 16, 32, 64, 128 },
            SamHeadsPerStage = new[] { 1, 2, 4, 8 }, SamWindowSizes = new[] { 4, 4, 4, 2 }, SamGlobalAttentionBlocks = new[] { 3 },
            SamDim = 32, SamMaskInChannels = 8, SamDecoderHeads = 2, SamDecoderMlpDim = 64, SamIouHiddenDim = 16,
        });

    private static Tensor<double> Image()
    {
        // 48 x 64, not square: SAM 2 resizes it to 64 x 64 and the masks come back at 48 x 64.
        var image = new Tensor<double>(new[] { 3, 48, 64 });
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 48; y++)
                for (int x = 0; x < 64; x++)
                    image[c, y, x] = x >= 10 && x < 34 && y >= 8 && y < 30 ? 0.8 - 0.1 * c : 0.2 + 0.1 * c;
        return image;
    }

    [Fact(Timeout = 300000)]
    public async Task Segment_GivesOneBinaryMaskPerDetection_AtTheImageSize()
    {
        await Task.Yield();
        using var model = Small();
        var image = Image();
        var detections = model.Ground(image, "cat . remote control .");
        var segments = model.Segment(image, "cat . remote control .");
        Assert.Equal(6, detections.Count);   // a zero box threshold keeps every query
        Assert.Equal(detections.Count, segments.Count);
        for (int i = 0; i < segments.Count; i++)
        {
            Assert.Equal(detections[i], segments[i].Detection);
            Assert.Equal(new[] { 48, 64 }, segments[i].Mask.Shape.ToArray());
            Assert.All(segments[i].Mask.ToArray(), v => Assert.True(v == 0 || v == 1));
            Assert.False(double.IsNaN(segments[i].MaskScore));
        }
        var embedding = model.EncodeImage(image);
        Assert.Equal(new[] { 32, 4, 4 }, embedding.Shape.ToArray());

        var result = model.SegmentWithText(image, new[] { "cat", "remote control" });
        Assert.Equal(new[] { 2, 48, 64 }, result.Masks.Shape.ToArray());
        Assert.Equal(new[] { 48, 64 }, result.SemanticMap.Shape.ToArray());
        Assert.Equal(new[] { "cat", "remote control" }, result.ClassNames);
        Assert.All(result.SemanticMap.ToArray(), v => Assert.InRange(v, -1, 1));
        // A pixel is labelled exactly when some class's mask covers it.
        for (int y = 0; y < 48; y++)
            for (int x = 0; x < 64; x++)
                Assert.Equal(result.Masks[0, y, x] == 1 || result.Masks[1, y, x] == 1, result.SemanticMap[y, x] >= 0);
        var prompt = model.SegmentWithPrompt(image, "a cat");
        Assert.Equal(new[] { 1, 48, 64 }, prompt.Masks.Shape.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task TrainGrounding_TrainsTheDetectorOnly()
    {
        await Task.Yield();
        using var model = Small();
        var image = Image();
        var boxes = new[] { new GroundingBox("cat", 10 / 64.0, 8 / 48.0, 34 / 64.0, 30 / 48.0) };
        var segmenterBefore = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(SegmenterLayers(model)).Select(p => p.ToArray()).ToList();
        double before = model.EvaluateGroundingLoss(image, "cat . remote control .", boxes);
        for (int step = 0; step < 20; step++) model.TrainGrounding(image, "cat . remote control .", boxes);
        double after = model.EvaluateGroundingLoss(image, "cat . remote control .", boxes);
        Assert.True(after < 0.6 * before, $"{before} -> {after}");
        var segmenterAfter = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(SegmenterLayers(model)).Select(p => p.ToArray()).ToList();
        Assert.Equal(segmenterBefore.Count, segmenterAfter.Count);
        for (int i = 0; i < segmenterBefore.Count; i++) Assert.Equal(segmenterBefore[i], segmenterAfter[i]);
        Assert.Throws<NotSupportedException>(() => model.Train(image, new Tensor<double>(new[] { 32, 4, 4 })));
    }

    /// <summary>SAM 2's layers: those registered after the detector's (the model registers the detector first).</summary>
    private static System.Collections.Generic.List<AiDotNet.NeuralNetworks.Layers.LayerBase<double>> SegmenterLayers(GroundedSAM2<double> model)
    {
        var probe = new System.Collections.Generic.List<AiDotNet.NeuralNetworks.Layers.LayerBase<double>>();
        _ = new GroundingDinoNetwork<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, probe,
            GroundingDinoDetector<double>.Configuration(((GroundedSAM2Options)model.GetOptions()).Detector));
        return model.Layers.Skip(probe.Count).OfType<AiDotNet.NeuralNetworks.Layers.LayerBase<double>>().ToList();
    }
}
