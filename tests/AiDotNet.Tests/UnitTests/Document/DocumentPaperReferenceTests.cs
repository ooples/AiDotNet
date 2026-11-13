using System;
using System.Linq;
using AiDotNet.Document.LayoutAware;
using AiDotNet.Document.OCR.TextDetection;
using AiDotNet.Document.OCR.TextRecognition;
using AiDotNet.Document.VisionLanguage;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Document;

/// <summary>
/// Document models checked against their papers' reference implementations, with expected values worked out
/// here from the published algorithm or image processor rather than read off the code.
/// </summary>
public sealed class DocumentPaperReferenceTests
{
    public DocumentPaperReferenceTests() => TestModuleInitializer.EnsureInitialized();

    private static PSENet<double> SmallPseNet(int kernels) => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 2),
        options: new PSENetOptions { ImageSize = 64, NumKernels = kernels });

    /// <summary>Kernel logits <c>[1, kernels, 8, 8]</c>: +10 where a kernel map is on, -10 elsewhere.</summary>
    private static Tensor<double> KernelLogits(params bool[][,] maps)
    {
        var t = new Tensor<double>(new[] { 1, maps.Length, 8, 8 });
        for (int k = 0; k < maps.Length; k++)
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    t[0, k, y, x] = maps[k][y, x] ? 10.0 : -10.0;
        return t;
    }

    private static bool[,] Map(params (int y, int x)[] on)
    {
        var m = new bool[8, 8];
        foreach (var (y, x) in on) m[y, x] = true;
        return m;
    }

    private static (int y, int x)[] UShape() =>
        Enumerable.Range(1, 6).Select(y => (y, 1))
            .Concat(Enumerable.Range(1, 6).Select(y => (y, 5)))
            .Concat(Enumerable.Range(2, 3).Select(x => (6, x)))
            .ToArray();

    [Fact]
    public void Pse_ConnectedComponentsFollowTheWholeShape()
    {
        // A U-shaped kernel is ONE 4-connected component (15 pixels). A raster pass that only looks up and left,
        // without merging, splits it into two arms of 9 and 6 pixels, both below the 10-pixel minimum.
        using var model = SmallPseNet(2);
        var u = Map(UShape());
        var regions = model.ParsePSENetOutput(KernelLogits(u, u), 0.5);

        var region = Assert.Single(regions);
        Assert.Equal(new[] { 1.0, 1.0, 5.0, 6.0 }, region.BoundingBox.ToArray());
        // The score is the mean complete-map probability over the instance: sigmoid(10) everywhere.
        Assert.Equal(1.0 / (1.0 + Math.Exp(-10.0)), region.Confidence, 12);
    }

    [Fact]
    public void Pse_ExpandsBreadthFirstSoTouchingInstancesSplitTheGap()
    {
        // Two seeds in the smallest kernel, columns 1 and 6 of rows 0-7, and a complete map covering columns 1-6.
        // Breadth-first growth from both seeds at once gives each instance the columns nearer to it (1-3 and
        // 4-6). Growing one raster sweep at a time instead lets the first seed reach further than the second.
        var seeds = Map(Enumerable.Range(0, 8).SelectMany(y => new[] { (y, 1), (y, 6) }).ToArray());
        var text = Map(Enumerable.Range(0, 8).SelectMany(y => Enumerable.Range(1, 6).Select(x => (y, x))).ToArray());
        using var model = SmallPseNet(2);
        var regions = model.ParsePSENetOutput(KernelLogits(seeds, text), 0.5);

        Assert.Equal(2, regions.Count);
        Assert.Equal(new[] { 1.0, 0.0, 3.0, 7.0 }, regions[0].BoundingBox.ToArray());
        Assert.Equal(new[] { 4.0, 0.0, 6.0, 7.0 }, regions[1].BoundingBox.ToArray());
    }

    [Fact]
    public void Pse_KeepsAnInstanceOnlyWhenItsMeanTextProbabilityReachesTheThreshold()
    {
        // A 4x4 square instance whose complete-map logit is 10: its mean probability is sigmoid(10) ~ 1, so a
        // threshold of 1.5 rejects it. Comparing the raw logits (10) with the threshold would keep it.
        using var model = SmallPseNet(2);
        var square = Map(Enumerable.Range(2, 4).SelectMany(y => Enumerable.Range(2, 4).Select(x => (y, x))).ToArray());
        Assert.Single(model.ParsePSENetOutput(KernelLogits(square, square), 0.5));
        Assert.Empty(model.ParsePSENetOutput(KernelLogits(square, square), 1.5));
    }

    private static Tensor<double> ConstantPage(int size, double value) =>
        Tensor<double>.CreateDefault(new[] { 3, size, size }, value);

    [Fact]
    public void LayoutLMv3_NormalizesWithHalfMeanAndHalfStd()
    {
        // LayoutLMv3's image processor: (x - 0.5) / 0.5 per channel, so 0.75 maps to 0.5 (ImageNet's
        // statistics would give (0.75 - 0.485) / 0.229 = 1.157 on the red channel).
        using var model = new LayoutLMv3<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 16, outputSize: 4),
            options: new LayoutLMv3Options { NumClasses = 4, ImageSize = 32, PatchSize = 16, MaxSequenceLength = 64, HiddenDim = 64, NumLayers = 2, NumHeads = 4, VocabSize = 100 });
        var processed = model.PreprocessForInspection(ConstantPage(32, 0.75));
        Assert.All(processed.ToArray(), v => Assert.Equal(0.5, v, 12));
    }

    [Fact]
    public void DiT_NormalizesWithHalfMeanAndHalfStd()
    {
        // DiT's released image processor (BEiT-style): (x - 0.5) / 0.5 on every channel.
        using var model = new DiT<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 4),
            options: new DiTOptions { NumClasses = 4, ImageSize = 32, PatchSize = 16, HiddenDim = 32, NumLayers = 1, NumHeads = 2 });
        var processed = model.PreprocessForInspection(ConstantPage(32, 0.25));
        Assert.All(processed.ToArray(), v => Assert.Equal(-0.5, v, 12));
    }

    [Fact]
    public void Udop_NormalizesWithHalfMeanAndHalfStd()
    {
        // UDOP's processor reuses the LayoutLMv3 image processor: (x - 0.5) / 0.5 on every channel.
        using var model = new UDOP<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 64),
            options: new UDOPOptions { ImageSize = 32, PatchSize = 16, MaxSequenceLength = 16, HiddenDim = 32, NumEncoderLayers = 2, NumDecoderLayers = 2, NumHeads = 4, KeyValueDim = 8, FeedForwardDim = 64, VocabSize = 64, RelativeAttentionBuckets = 8, RelativeAttentionMaxDistance = 16, RelativeAttentionMaxDistance2D = 10, Max2DPositions = 64, NumLocationBins = 16, MaxGenerationLength = 8 });
        var processed = model.PreprocessForInspection(ConstantPage(32, 1.0));
        Assert.All(processed.ToArray(), v => Assert.Equal(1.0, v, 12));
    }

    [Fact]
    public void Svtr_CtcConfidenceIsTheMeanSoftmaxProbabilityOfTheKeptCharacters()
    {
        // PaddleOCR CTCLabelDecode over the softmax of the head's logits: steps 0-1 both pick 'a' (collapsed to
        // one), step 2 picks 'b'. Averaging the raw logits instead would give (2 + 3) / 2 = 2.5.
        using var model = new SVTR<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 100, outputSize: 3),
            options: new SVTROptions { Charset = "ab" });
        var logits = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new[] { 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0 }));
        var (text, confidence, perCharacter) = model.CTCDecode(logits);

        double pa = Math.Exp(2) / (Math.Exp(2) + 2), pb = Math.Exp(3) / (Math.Exp(3) + 2);
        Assert.Equal("ab", text);
        Assert.Equal(new[] { pa, pb }, perCharacter, new Tolerance(1e-12));
        Assert.Equal((pa + pb) / 2, confidence, 12);
    }

    private sealed class Tolerance : System.Collections.Generic.IEqualityComparer<double>
    {
        private readonly double _epsilon;
        public Tolerance(double epsilon) => _epsilon = epsilon;
        public bool Equals(double x, double y) => Math.Abs(x - y) <= _epsilon;
        public int GetHashCode(double obj) => 0;
    }
}
