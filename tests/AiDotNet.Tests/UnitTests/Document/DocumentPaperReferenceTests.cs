using System;
using System.Linq;
using AiDotNet.Document;
using AiDotNet.Document.Analysis.PageSegmentation;
using AiDotNet.Document.Analysis.TableDetection;
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

    [Fact]
    public void DocBank_RegionConfidenceIsTheMeanSoftmaxProbabilityOfItsClass()
    {
        // The segmentation head emits logits. A 10x10 block whose class-1 logit is 2 (all others 0) among 13
        // classes has probability e^2 / (e^2 + 12) per pixel; averaging the raw logits instead gives 2.
        using var model = new DocBank<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 13),
            options: new DocBankOptions { ImageSize = 64 });
        var logits = new Tensor<double>(new[] { 1, 13, 12, 12 });
        for (int y = 1; y <= 10; y++)
            for (int x = 1; x <= 10; x++)
                logits[0, 1, y, x] = 2.0;
        var result = model.ParseSegmentationOutput(logits, 0.3);

        var region = Assert.Single(result.Regions);
        Assert.Equal(DocumentRegionType.Paragraph, region.RegionType);
        Assert.Equal(new[] { 1.0, 1.0, 10.0, 10.0 }, region.BoundingBox.ToArray());
        Assert.Equal(Math.Exp(2) / (Math.Exp(2) + 12), region.Confidence, 12);
        Assert.Equal(1.0, Enumerable.Range(0, 13).Sum(c => result.ClassProbabilities[0, c, 0, 0]), 12);
    }

    private static TableTransformer<double> SmallTableTransformer() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 13),
        options: new TableTransformerOptions { ImageSize = 64 });

    /// <summary>One native DETR query: box logits (the inverse sigmoid of cx, cy, w, h) then class logits.</summary>
    private static double[] Query(double cx, double cy, double w, double h, int classes, int label, double logit)
    {
        static double Logit(double p) => Math.Log(p / (1 - p));
        var row = new double[4 + classes];
        row[0] = Logit(cx); row[1] = Logit(cy); row[2] = Logit(w); row[3] = Logit(h);
        row[4 + label] = logit;
        return row;
    }

    private static Tensor<double> Queries(params double[][] rows) =>
        new(new[] { rows.Length, rows[0].Length }, new Vector<double>(rows.SelectMany(r => r).ToArray()));

    [Fact]
    public void TableTransformer_DetectionScoresAreTheSoftmaxWithNoObjectLast()
    {
        // DETR: softmax over [table, no object]; the score is the best class except the trailing no-object.
        // Reading the first logit as background (the old layout) rejects the real table.
        using var model = SmallTableTransformer();
        var output = Queries(Query(0.5, 0.5, 0.6, 0.4, 2, 0, 3.0), Query(0.5, 0.5, 0.6, 0.4, 2, 1, 3.0));
        var table = Assert.Single(model.ParseTableDetections(output, Tensor<double>.CreateDefault(new[] { 3, 100, 100 }, 0.5), 0.5));
        Assert.Equal(Math.Exp(3) / (Math.Exp(3) + 1), table.Confidence, 12);
        Assert.Equal(new[] { 20.0, 30.0, 80.0, 70.0 }, table.BoundingBox.ToArray(), new Tolerance(1e-9));
        Assert.NotNull(table.Image);
    }

    [Fact]
    public void TableTransformer_CellsAreRowColumnIntersectionsWithSoftmaxScores()
    {
        // Two rows, two columns and a column header over the first row, each with its class logit 5 among the
        // 7 structure logits (probability e^5 / (e^5 + 6)), plus a no-object query that must be ignored. The
        // cells are the 2 x 2 intersections; the first row is the header row.
        using var model = SmallTableTransformer();
        const int classes = 7, column = 1, row = 2, header = 3, noObject = 6;
        var output = Queries(
            Query(0.5, 0.75, 0.8, 0.4, classes, row, 5.0),
            Query(0.5, 0.25, 0.8, 0.4, classes, row, 5.0),
            Query(0.75, 0.5, 0.4, 0.8, classes, column, 5.0),
            Query(0.25, 0.5, 0.4, 0.8, classes, column, 5.0),
            Query(0.5, 0.25, 0.8, 0.4, classes, header, 5.0),
            Query(0.5, 0.5, 0.5, 0.5, classes, noObject, 5.0));
        var result = model.ParseStructureOutput(output, Tensor<double>.CreateDefault(new[] { 3, 100, 100 }, 0.5), 0.5);

        double p = Math.Exp(5) / (Math.Exp(5) + 6);
        Assert.Equal(2, result.NumRows);
        Assert.Equal(2, result.NumColumns);
        Assert.Equal(new[] { 0 }, result.HeaderRows);
        Assert.Equal(4, result.Cells.Count);
        var first = result.Cells[0];
        Assert.Equal((0, 0, true), (first.Row, first.Column, first.IsHeader));
        Assert.Equal(new[] { 5.0, 5.0, 45.0, 45.0 }, first.BoundingBox.ToArray(), new Tolerance(1e-9));
        Assert.False(result.Cells[3].IsHeader);
        Assert.Equal(new[] { 55.0, 55.0, 95.0, 95.0 }, result.Cells[3].BoundingBox.ToArray(), new Tolerance(1e-9));
        Assert.All(result.Cells, c => Assert.Equal(p, c.Confidence, 12));
        Assert.Equal(p, result.Confidence, 12);
    }

    [Fact]
    public void DocGcn_ClassifiesEveryNodeBySoftmaxIncludingText()
    {
        // Doc-GCN classifies each node with a softmax over its layout classes; class 0 is Text, a real class
        // (PubLayNet has no background). The old parser reserved four of the four class columns for a box and
        // treated class 0 as background, so it returned nothing.
        using var model = new AiDotNet.Document.GraphBased.DocGCN<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 16, outputSize: 4),
            options: new DocGCNOptions { NodeDim = 16, EdgeDim = 8, GcnLayers = 1, NumClasses = 4, MaxNodes = 16 });
        var logits = new Tensor<double>(new[] { 3, 4 }, new Vector<double>(new[] { 3.0, 0, 0, 0, 0, 3.0, 0, 0, 0.1, 0, 0, 0 }));
        var regions = model.ParseLayoutOutput(logits, 0.5);

        Assert.Equal(new[] { 0, 1 }, regions.Select(r => r.Index).ToArray());
        Assert.Equal(new[] { LayoutElementType.Text, LayoutElementType.Title }, regions.Select(r => r.ElementType).ToArray());
        Assert.All(regions, r => Assert.Equal(Math.Exp(3) / (Math.Exp(3) + 3), r.Confidence, 12));
    }

    [Fact]
    public void DiT_ModelSizeIsTheEnumOfThePublishedVariants()
    {
        // DiT ships Base and Large (Li et al. 2022); the size is an enum, not a free-form string.
        var options = new DiTOptions();
        Assert.Equal(DiTModelSize.Base, options.ModelSize);
        using var model = new DiT<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 4),
            options: new DiTOptions { NumClasses = 4, ImageSize = 32, PatchSize = 16, HiddenDim = 32, NumLayers = 1, NumHeads = 2, ModelSize = DiTModelSize.Large });
        Assert.Equal(DiTModelSize.Large, model.ModelSize);
        Assert.Contains("DiT-Large", model.GetModelSummary());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pix2StructAndMatcha_StandardizeEachPageByItsOwnMeanAndStd(bool matcha)
    {
        // The Pix2Struct image processor (MatCha reuses it): (x - mean) / max(std, 1/sqrt(N)) over the page. A
        // page alternating 0 and 1 has mean 0.5 and std 0.5, so it maps to -1 and +1; ImageNet statistics give
        // (0 - 0.485) / 0.229 on the red channel. A flat page maps to zeros, finite thanks to the std floor.
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 64);
        using AiDotNet.Document.DocumentNeuralNetworkBase<double> model = matcha
            ? new AiDotNet.Document.PixelToSequence.MATCHA<double>(arch, options: new MATCHAOptions { ImageSize = 32, MaxSequenceLength = 8, EncoderDim = 32, DecoderDim = 32, EncoderLayers = 1, DecoderLayers = 1, NumHeads = 2, VocabSize = 64, MaxPatchesPerImage = 4 })
            : new AiDotNet.Document.PixelToSequence.Pix2Struct<double>(arch, options: new Pix2StructOptions { ImageSize = 32, PatchSize = 16, MaxPatches = 4, MaxSequenceLength = 8, HiddenDim = 32, NumEncoderLayers = 1, NumDecoderLayers = 1, NumHeads = 2, VocabSize = 64 });
        var page = new Tensor<double>(new[] { 3, 32, 32 }, new Vector<double>(Enumerable.Range(0, 3 * 32 * 32).Select(i => (double)(i % 2)).ToArray()));
        var processed = model.PreprocessForInspection(page).ToArray();
        Assert.All(processed.Select((v, i) => (v, i)), p => Assert.Equal(p.i % 2 == 0 ? -1.0 : 1.0, p.v, 12));

        Assert.All(model.PreprocessForInspection(ConstantPage(32, 0.7)).ToArray(), v => Assert.Equal(0.0, v, 12));
    }

    [Fact]
    public void Trie_DecodesFieldTextByArgmaxOverTheCharacterLogits()
    {
        // TRIE's text reading emits per-step logits over its character vocabulary (PAD 0, BOS 1, EOS 2, then
        // the characters): the key "Ab" (tokens 35, 62) and value "7" (token 26) each end at EOS. The old
        // parser read one column per character and multiplied it by 255.
        const int vocabulary = 215, types = 4, keySteps = 32, valueSteps = 64;
        using var model = new AiDotNet.Document.GraphBased.TRIE<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: types),
            options: new TRIEOptions { ImageSize = 64, VisualDim = 16, TextDim = 16, GraphDim = 16, NumEntityTypes = types, MaxEntities = 8 });
        int width = 1 + types + ((keySteps + valueSteps) * vocabulary) + 4;
        var row = new double[width];
        row[0] = 0.9;
        row[1 + 1] = 2.0; // entity type 1 ("name")
        void Step(int start, int step, int token) => row[start + (step * vocabulary) + token] = 5.0;
        int key = 1 + types, value = key + (keySteps * vocabulary);
        Step(key, 0, 35); Step(key, 1, 62); Step(key, 2, 2); Step(key, 3, 40);
        Step(value, 0, 26); Step(value, 1, 2);
        var fields = model.ParseFormFields(new Tensor<double>(new[] { 1, width }, new Vector<double>(row)), 0.5);

        var field = Assert.Single(fields);
        Assert.Equal("Ab", field.FieldName);
        Assert.Equal("7", field.FieldValue);
        Assert.Equal("name", field.FieldType);
        Assert.Equal(0.9, field.Confidence, 12);
    }

    [Fact]
    public void LayoutGraph_ReadingOrderConfidenceIsTheNodeSoftmaxProbability()
    {
        // GLAM classifies each node with a softmax; the confidence is that probability, not a constant 0.9.
        using var model = new AiDotNet.Document.GraphBased.LayoutGraph<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 4, inputWidth: 8, outputSize: 4),
            new LayoutGraphOptions { NodeDim = 8, EdgeDim = 8, GraphLayers = 1, NumClasses = 4, MaxNodes = 8 });
        var logits = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[] { 2.0, 0, 0, 0, 0, 0, 0, 1.0 }));
        var order = model.PredictReadingOrder(logits);

        Assert.Equal(new[] { 0, 1 }, order.Select(e => e.ElementIndex).ToArray());
        Assert.Equal(Math.Exp(2) / (Math.Exp(2) + 3), order[0].Confidence, 12);
        Assert.Equal(Math.E / (Math.E + 3), order[1].Confidence, 12);
    }

    [Fact]
    public void LayoutGraph_PageGraphJoinsNearestNeighboursSymmetricallyAndNormalizes()
    {
        // Three boxes on a line, centres at x = 0.1, 0.2 and 0.9, one neighbour each: 0-1, 1-0 and 2-1. Made
        // symmetric with self-loops the degrees are 2, 3 and 2, so the normalized 0-1 weight is 1/sqrt(2 * 3).
        var boxes = new Tensor<double>(new[] { 3, 4 }, new Vector<double>(new[]
        {
            0.05, 0.4, 0.15, 0.6,
            0.15, 0.4, 0.25, 0.6,
            0.85, 0.4, 0.95, 0.6
        }));
        var a = AiDotNet.Document.GraphBased.LayoutGraph<double>.SpatialAdjacency(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, boxes, 1);
        double s23 = 1 / Math.Sqrt(6);
        var expected = new[] { 0.5, s23, 0, s23, 1.0 / 3, s23, 0, s23, 0.5 };
        Assert.Equal(expected, a.ToArray(), new Tolerance(1e-12));
    }

    [Fact]
    public void DocBank_TokensAreLabelledWithTheTwelveDocBankClassesBySoftmax()
    {
        // DocBank's baselines are LayoutLM token classifiers over its 12 labels (no background class).
        using var model = new DocBank<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 13),
            options: new DocBankOptions
            {
                ImageSize = 64,
                TokenClassifierOptions = new LayoutLMOptions { MaxSequenceLength = 16, HiddenDim = 16, NumLayers = 1, NumHeads = 2, VocabSize = 50 }
            });
        Assert.Equal(12, DocBank<double>.TokenLabels.Count);
        var logits = new double[2 * 12];
        logits[7] = 3.0;       // token 0: Paragraph
        logits[12 + 11] = 3.0; // token 1: Title
        var labels = model.TokenPredictions(new Tensor<double>(new[] { 2, 12 }, new Vector<double>(logits)));

        Assert.Equal(new[] { DocumentRegionType.Paragraph, DocumentRegionType.Title }, labels.Select(l => l.Label).ToArray());
        Assert.All(labels, l => Assert.Equal(Math.Exp(3) / (Math.Exp(3) + 11), l.Confidence, 12));

        var packed = new Tensor<double>(new[] { 3, 5 }, new Vector<double>(new[] { 5.0, 10, 10, 200, 40, 6.0, 210, 10, 400, 40, 7.0, 10, 60, 300, 90 }));
        Assert.Equal(3, model.ClassifyTokens(packed).Count);
    }

    private sealed class Tolerance : System.Collections.Generic.IEqualityComparer<double>
    {
        private readonly double _epsilon;
        public Tolerance(double epsilon) => _epsilon = epsilon;
        public bool Equals(double x, double y) => Math.Abs(x - y) <= _epsilon;
        public int GetHashCode(double obj) => 0;
    }
}
