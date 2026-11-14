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
    public void Detections_AreTheOfficialPredictInTheModelsNumericType()
    {
        var prompt = GroundingDinoPrompt.Create("cat . remote control");
        // Query 0 scores "remote control" (tokens 3 and 4) at sigmoid(2); query 1 scores nothing above the box threshold.
        var logits = new Tensor<float>(new[] { 1, 2, 7 });
        logits.Fill(-5f);
        logits[0, 0, 3] = 2f;
        logits[0, 0, 4] = 2f;
        var boxes = new Tensor<float>(new[] { 1, 2, 4 }, new Vector<float>(new[] { 0.5f, 0.5f, 0.2f, 0.4f, 0.1f, 0.1f, 0.1f, 0.1f }));
        var output = new GroundingDinoOutput<float>(new[] { logits }, new[] { boxes }, logits, boxes, new bool[1, 7]);

        var detections = prompt.Detections(output, 0, 0.35, 0.25);

        var d = Assert.Single(detections);
        Assert.Equal("remote control", d.Phrase);
        Assert.Equal(0.4f, d.X1, 6);
        Assert.Equal(0.3f, d.Y1, 6);
        Assert.Equal(0.6f, d.X2, 6);
        Assert.Equal(0.7f, d.Y2, 6);
        Assert.Equal((float)(1 / (1 + Math.Exp(-2))), d.Score, 6);
    }

    [Fact]
    public void Detections_DecideBothThresholdsOnTheLogits()
    {
        // sigmoid(1e-17) is above 0.5, but it is exactly 0.5 once rounded to double: a threshold on the rounded
        // probability dropped the query and its token. The official rule p > t is logit > ln(t / (1 - t)) = 0 here.
        var prompt = GroundingDinoPrompt.Create("cat . remote control");
        var logits = new Tensor<double>(new[] { 1, 2, 7 });
        logits.Fill(-5.0);
        logits[0, 0, 3] = 1e-17;
        var boxes = new Tensor<double>(new[] { 1, 2, 4 }, new Vector<double>(new[] { 0.5, 0.5, 0.2, 0.4, 0.1, 0.1, 0.1, 0.1 }));
        var output = new GroundingDinoOutput<double>(new[] { logits }, new[] { boxes }, logits, boxes, new bool[1, 7]);

        var detections = prompt.Detections(output, 0, 0.5, 0.5);

        var d = Assert.Single(detections);
        Assert.Equal("remote", d.Phrase);
        Assert.Equal(0.5, d.Score, 15);
    }

    private static GroundingDinoOutput<Half> HalfOutput(params (int Query, int Token, float Logit)[] peaks)
    {
        var logits = new Tensor<Half>(new[] { 1, 2, 7 });
        logits.Fill((Half)(-5f));
        foreach (var (query, token, logit) in peaks) logits[0, query, token] = (Half)logit;
        var boxes = new Tensor<Half>(new[] { 1, 2, 4 });
        boxes.Fill((Half)0.25f);
        return new GroundingDinoOutput<Half>(new[] { logits }, new[] { boxes }, logits, boxes, new bool[1, 7]);
    }

    [Fact]
    public void Strongest_RanksScoresThatTieInHalfByTheirLogits()
    {
        // In Half sigmoid(8.5) and sigmoid(9) both round to 1, so ranking by score alone kept the earlier query.
        // Query 1 ("cat", logit 9) is the stronger detection and is the one a cap of one must keep.
        var prompt = GroundingDinoPrompt.Create("cat . remote control");
        var detections = prompt.Detections(HalfOutput((0, 3, 8.5f), (1, 1, 9f)), 0, 0.35, 0.25);
        Assert.Equal(2, detections.Count);
        Assert.Equal(detections[0].Score, detections[1].Score);

        var strongest = GroundingDinoDetector<Half>.Strongest(detections, 1);

        Assert.Equal("cat", Assert.Single(strongest).Phrase);
    }

    [Fact]
    public void GroundedSam2BoxCorners_AreScaledInDoubleAndRoundedOnce()
    {
        // Half box cx = 0.300048828125, w = 0.04998779296875: x1 = 0.275054931640625, and x1 * 1000 = 275.05 rounds to
        // 275 in Half. Rounding x1 to Half first (0.27514648) and then scaling gave 275.25.
        var output = HalfOutput((0, 1, 9f));
        var boxes = output.Boxes[0];
        boxes[0, 0, 0] = (Half)0.300048828125f;
        boxes[0, 0, 2] = (Half)0.04998779296875f;
        var detections = GroundingDinoPrompt.Create("cat . remote control").Detections(output, 0, 0.35, 0.25);

        var corners = GroundedSAM2<Half>.ScaledCorners(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, detections, 1000);

        Assert.Equal(275f, (float)corners[0, 0]);
        Assert.Equal(325f, (float)corners[0, 2]); // x2 = 0.325042724609375 * 1000 = 325.04 -> 325
    }

    [Fact]
    public void GroundedSam2Combine_GivesAPixelToTheStrongerOfTwoTiedHalfScores()
    {
        // Both detections score 1 in Half; the second has the larger logit, so the pixel both masks cover is its class.
        var one = (Half)1f;
        var weak = new GroundingDetection<Half>(default, default, one, one, one, "a") { Logit = (Half)8.5f };
        var strong = new GroundingDetection<Half>(default, default, one, one, one, "b") { Logit = (Half)9f };
        var both = new Tensor<Half>(new[] { 1, 2 }, new Vector<Half>(new[] { one, one }));
        var right = new Tensor<Half>(new[] { 1, 2 }, new Vector<Half>(new[] { default(Half), one }));
        var segments = new[] { new GroundedSegment<Half>(weak, both, one), new GroundedSegment<Half>(strong, right, one) };
        var image = new Tensor<Half>(new[] { 3, 1, 2 });

        var result = GroundedSAM2<Half>.Combine(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, image, segments,
            new[] { "a", "b" }, s => s.Detection.Phrase == "a" ? 0 : 1);

        Assert.Equal(0f, (float)result.SemanticMap[0, 0]);
        Assert.Equal(1f, (float)result.SemanticMap[0, 1]);
    }

    [Fact]
    public void GroundingRows_KeepTheModelsNumericType_WithClassesByPhrase()
    {
        var detections = new[]
        {
            new GroundingDetection<float>(0.1f, 0.2f, 0.3f, 0.4f, 0.9f, "remote control"),
            new GroundingDetection<float>(0.5f, 0.5f, 0.6f, 0.7f, 0.4f, "dog"),
        };

        var boxes = GroundingRows<float>.Boxes(detections);
        var classified = GroundingRows<float>.Classified(detections, new[] { "cat", "remote" });

        Assert.Equal(new[] { 2, 5 }, boxes.Shape.ToArray());
        Assert.Equal(new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.9f, 0.5f, 0.5f, 0.6f, 0.7f, 0.4f }, boxes.ToArray());
        Assert.Equal(new[] { 0.1f, 0.2f, 0.3f, 0.4f, 1f, 0.9f, 0.5f, 0.5f, 0.6f, 0.7f, -1f, 0.4f }, classified.ToArray());
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
