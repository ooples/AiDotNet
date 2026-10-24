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
/// Grounding DINO's public surface: the caption as the official code reads it, its phrases and their decoding, the
/// detection formats, the typed training entry point and the refusal of a tensor target.
/// </summary>
public class GroundingDinoModelTests
{
    private static GroundingDINO<double> Small(int seed = 3) => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.ObjectDetection,
            inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 32),
        new GroundingDINOOptions
        {
            Seed = seed, ImageSize = 64, VisionDim = 16, SwinDepths = new[] { 1, 1, 2, 1 }, SwinHeads = new[] { 1, 2, 2, 4 }, SwinWindowSize = 4,
            SwinDropPathRate = 0.0, TextHiddenSize = 32, TextLayers = 2, TextHeads = 2, TextIntermediateSize = 64, DecoderDim = 32,
            NumVisionLayers = 1, NumDecoderLayers = 1, NumHeads = 4, FeedForwardDim = 64, NumSamplingPoints = 2, NumQueryPositions = 10,
            MaxSequenceLength = 16, MaxDetections = 10, FusionDropPathRate = 0.0,
        });

    private static Tensor<double> Image()
    {
        var image = new Tensor<double>(new[] { 3, 64, 64 });
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    image[c, y, x] = x >= 16 && x < 40 && y >= 20 && y < 48 ? 0.9 - 0.2 * c : 0.1 + 0.05 * c;
        return image;
    }

    [Fact]
    public void Caption_IsReadAsTheOfficialCodeReadsIt()
    {
        var prompt = GroundingDinoPrompt.Create("  Cat . Remote Control");
        Assert.Equal("cat . remote control.", prompt.Caption);
        // [CLS] cat . remote control . [SEP] in the BERT-base-uncased vocabulary.
        Assert.Equal(new[] { 101, 4937, 1012, 6556, 2491, 1012, 102 }, prompt.Ids);
        Assert.Equal(new[] { new[] { 1 }, new[] { 3, 4 } }, prompt.Phrases());
        Assert.Equal(1, prompt.PhraseIndex("Remote Control"));
        Assert.Throws<ArgumentException>(() => prompt.PhraseIndex("remote"));
        var probabilities = new double[16];
        probabilities[0] = probabilities[3] = probabilities[4] = 0.9;
        Assert.Equal("remote control", prompt.PhraseAbove(probabilities, 0.25));
        Assert.Equal(1, GroundingDinoPrompt.ClassOf("remote control", new[] { "cat", "remote" }));
        Assert.Equal(-1, GroundingDinoPrompt.ClassOf("dog", new[] { "cat", "remote" }));
    }

    [Fact(Timeout = 300000)]
    public async Task Detections_HaveTheInterfaceFormats()
    {
        await Task.Yield();
        using var model = Small();
        var options = (GroundingDINOOptions)model.GetOptions();
        options.ConfidenceThreshold = 0.0;   // keep every query, so the formats are exercised
        var image = Image();
        var detections = model.Ground(image, "cat . remote control .");
        Assert.Equal(10, detections.Count);
        Assert.All(detections, d => Assert.True(d.X2 > d.X1 && d.Y2 > d.Y1 && d.Score > 0 && d.Score < 1));
        var grounded = model.GroundText(image, "cat . remote control .");
        Assert.Equal(new[] { 10, 5 }, grounded.Shape.ToArray());
        Assert.Equal(detections[0].Score, grounded[0, 4], 12);
        var objects = model.DetectObjects(image, new[] { "cat", "remote control" });
        Assert.Equal(new[] { 10, 6 }, objects.Shape.ToArray());
        Assert.All(Enumerable.Range(0, 10), i => Assert.InRange(objects[i, 4], -1, 1));
        options.ConfidenceThreshold = 1.0;
        Assert.Equal(new[] { 0, 5 }, model.GroundText(image, "cat .").Shape.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task TrainGrounding_FitsOneImage_AndTrainRefusesATensorTarget()
    {
        await Task.Yield();
        using var model = Small();
        var image = Image();
        var boxes = new[] { new GroundingBox("cat", 0.25, 0.3125, 0.625, 0.75) };
        double before = model.EvaluateGroundingLoss(image, "cat . remote control .", boxes);
        for (int step = 0; step < 30; step++) model.TrainGrounding(image, "cat . remote control .", boxes);
        double after = model.EvaluateGroundingLoss(image, "cat . remote control .", boxes);
        Assert.True(after < 0.5 * before, $"{before} -> {after}");
        Assert.Throws<ArgumentException>(() => model.TrainGrounding(image, "cat .", new[] { new GroundingBox("dog", 0.1, 0.1, 0.5, 0.5) }));
        var ex = Assert.Throws<NotSupportedException>(() => model.Train(image, new Tensor<double>(new[] { 85, 32 })));
        Assert.Contains("TrainGrounding", ex.Message);
    }
}
