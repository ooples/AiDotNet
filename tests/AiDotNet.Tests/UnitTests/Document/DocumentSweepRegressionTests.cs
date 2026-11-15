using System;
using System.Linq;
using AiDotNet.Document;
using AiDotNet.Document.Analysis.PageSegmentation;
using AiDotNet.Document.GraphBased;
using AiDotNet.Document.OCR.TextDetection;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Document;

/// <summary>
/// Regressions found reviewing the document element-wise sweep: inputs the removed loops handled and the
/// engine-op rewrites did not.
/// </summary>
public sealed class DocumentSweepRegressionTests
{
    public DocumentSweepRegressionTests() => TestModuleInitializer.EnsureInitialized();

    private static IEngine Cpu => AiDotNetEngine.Current;

    [Fact]
    public void GraphLayoutRegions_APageWithNoNodesHasNoRegions()
    {
        // A page with no elements gives a [0, classes] head. The loop ran zero times and returned no regions;
        // dividing the (zero) length by the (zero) row count to find the row width threw DivideByZeroException.
        var empty = new Tensor<double>(new[] { 0, 4 });
        Assert.Empty(DocumentTensorOps.GraphLayoutRegions(Cpu, empty, maxNodes: 16, classes: 4, imageSize: 64, threshold: 0.5));
    }

    [Fact]
    public void DecodeRows_DropsTheSinglePageBatchAxis()
    {
        // [Batch = 1, Time = 3, Classes = 4] with step argmaxes 2, 0, 3: flattened as one row it gave a single token
        // indexing the flattened tensor.
        var output = new Tensor<double>(new[] { 1, 3, 4 }, new Vector<double>(new[] { 0, 0, 5.0, 0, 5.0, 0, 0, 0, 0, 0, 0, 5.0 }));
        Assert.Equal(new[] { 2, 0, 3 }, DocumentTensorOps.DecodeRows(Cpu, output, maxLength: 10, temperature: 0.0, seed: 0).Tokens);
    }

    [Fact]
    public void GraphLayoutRegions_TakeTheBoxHead_OrTheNodeBoxes_AndInventNoGeometry()
    {
        // Two classes plus a box head: node 0's box [0.1, 0.1, 0.5, 0.5] is [10, 10, 50, 50] on a 100-pixel page;
        // node 1's [1.2, 0.2, 1.4, 0.4] clamps to x1 = x2 = 100, no area, so it is not a region.
        var withHead = new Tensor<double>(new[] { 2, 6 }, new Vector<double>(new[] { 3.0, 0, 0.1, 0.1, 0.5, 0.5, 3.0, 0, 1.2, 0.2, 1.4, 0.4 }));
        var region = Assert.Single(DocumentTensorOps.GraphLayoutRegions(Cpu, withHead, maxNodes: 8, classes: 2, imageSize: 100, threshold: 0.5));
        Assert.Equal(0, region.Index);
        Assert.Equal(new[] { 10.0, 10.0, 50.0, 50.0 }, region.BoundingBox.ToArray().Select(v => Math.Round(v, 9)).ToArray());
        Assert.True(region.HasBoundingBox);

        var logits = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 3.0, 0, 0, 3.0 }));
        var boxes = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[] { 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1 }));
        var located = DocumentTensorOps.GraphLayoutRegions(Cpu, logits, maxNodes: 8, classes: 2, imageSize: 100, threshold: 0.5, nodeBoxes: boxes);
        Assert.Equal(new[] { 0.0, 0.0, 50.0, 50.0, 50.0, 50.0, 100.0, 100.0 }, located.SelectMany(r => r.BoundingBox.ToArray()).ToArray());

        // Five nodes without either source used to get grid cells, node 4 at [0, 100, 50, 150] off a 100-pixel page.
        var noGeometry = new Tensor<double>(new[] { 5, 2 }, new Vector<double>(Enumerable.Repeat(new[] { 3.0, 0 }, 5).SelectMany(r => r).ToArray()));
        Assert.Throws<InvalidOperationException>(() => DocumentTensorOps.GraphLayoutRegions(Cpu, noGeometry, maxNodes: 8, classes: 2, imageSize: 100, threshold: 0.5));
    }

    [Fact]
    public void PageBoxes_ClampEveryCoordinateToThePage_AndFlagBoxesWithoutArea()
    {
        // At 224 pixels [1.2, 0.2, 1.4, 0.4] is [268.8, 44.8, 313.6, 89.6]: clamped to [224, 44.8, 224, 89.6], no width.
        // [-0.1, 0.1, 0.5, 0.6] keeps its area once x1 is clamped to 0.
        var normalized = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[] { 1.2, 0.2, 1.4, 0.4, -0.1, 0.1, 0.5, 0.6 }));
        var (corners, hasArea) = DocumentTensorOps.PageBoxes(Cpu, normalized, 224);
        Assert.Equal(new[] { 224.0, 44.8, 224.0, 89.6, 0.0, 22.4, 112.0, 134.4 }, corners.Select(v => Math.Round(v, 9)).ToArray());
        Assert.Equal(new[] { false, true }, hasArea);
    }

    [Fact]
    public void SampleWithProbabilities_InHalf_ShiftsTheLogitsBeforeTheTemperature()
    {
        // [20, 10] / 0.0001 overflows Half to [+inf, +inf]; inf - inf made the row NaN and it returned token 0 with
        // probability 0. Shifted first, the best token is drawn with probability 1.
        var logits = new Tensor<Half>(new[] { 1, 2 }, new Vector<Half>(new[] { (Half)20.0, (Half)10.0 }));
        var (tokens, probabilities) = DocumentTensorOps.SampleWithProbabilities(Cpu, logits, (Half)0.0001, seed: 7);
        Assert.Equal(new[] { 0 }, tokens);
        Assert.Equal(1.0, (double)probabilities.ToArray()[0], 12);
    }

    [Fact]
    public void PseNetLoss_InHalf_ReducesInDouble()
    {
        // 40,000 confident positives: the intersection (~40,000) and the denominator (~80,000) pass Half's 65,504.
        const int n = 40000;
        var logits = new Vector<Half>(Enumerable.Repeat((Half)10.0, n).ToArray());
        var targets = new Vector<Half>(Enumerable.Repeat((Half)1.0, n).ToArray());
        double loss = (double)new PSENetLoss<Half>().CalculateLoss(logits, targets);
        Assert.False(double.IsNaN(loss));
        Assert.InRange(loss, 0.0, 1e-3);
    }

    [Fact]
    public void LayoutGraph_MaxNodesCapsTheParsedReadingOrder()
    {
        // MaxNodes no longer sizes a positional table (the graph convolutions have none); what it still does is
        // cap the node rows parsed out of the head: three rows with MaxNodes = 2 give two ordered elements.
        using var model = new LayoutGraph<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 4, inputWidth: 8, outputSize: 4),
            new LayoutGraphOptions { NodeDim = 8, EdgeDim = 8, GraphLayers = 1, NumClasses = 4, MaxNodes = 2 });
        var logits = new Tensor<double>(new[] { 3, 4 }, new Vector<double>(new[] { 2.0, 0, 0, 0, 0, 0, 0, 1.0, 0, 3.0, 0, 0 }));
        Assert.Equal(new[] { 0, 1 }, model.PredictReadingOrder(logits).Select(e => e.ElementIndex).ToArray());
    }

    [Fact]
    public void DocBank_CloneCarriesTheTokenClassifierOnceItHasBeenBuilt()
    {
        // The token classifier is built on first use and is part of the model's saved state. LayoutLM has no
        // parameterless constructor, so cloning a DocBank after ClassifyTokens threw "carries a LayoutLM child
        // that is absent from the model being restored". Once it could be built, the restored classifier's dropout
        // layers were still in training mode, so the clone labelled the same tokens differently on every call.
        // The clone must build it and label tokens identically.
        using var model = new DocBank<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 13),
            options: new DocBankOptions
            {
                ImageSize = 64,
                TokenClassifierOptions = new LayoutLMOptions { MaxSequenceLength = 16, HiddenDim = 16, NumLayers = 1, NumHeads = 2, VocabSize = 50 }
            });
        var packed = new Tensor<double>(new[] { 3, 5 }, new Vector<double>(new[] { 5.0, 10, 10, 200, 40, 6.0, 210, 10, 400, 40, 7.0, 10, 60, 300, 90 }));
        var before = model.ClassifyTokens(packed);

        using var clone = (DocBank<double>)model.Clone();
        var after = clone.ClassifyTokens(packed);

        Assert.Equal(before.Select(p => p.Label).ToArray(), after.Select(p => p.Label).ToArray());
        Assert.All(before.Zip(after, (b, a) => (b, a)), p => Assert.Equal(p.b.Confidence, p.a.Confidence, 10));
    }

    private static PICK<double> SmallPick() => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 3, inputWidth: 9, outputSize: 5),
        new PICKOptions
        {
            NumEntityTypes = 5,
            ImageSize = 64,
            MaxSequenceLength = 32,
            HiddenDim = 32,
            NumHeads = 2,
            NumEncoderLayers = 1,
            FeedForwardDim = 64,
            VocabSize = 50,
            ImageFeatureDim = 16,
            ImageEncoderDepths = new[] { 1, 1, 1, 1 },
            GraphLearningDim = 8,
            NumGcnLayers = 2,
            LstmHiddenDim = 16,
            LstmLayers = 1
        });

    [Fact]
    public void Pick_ReadsANarrowedViewOfItsSegmentsFromTheViewsOwnRows()
    {
        // Rows 1-3 of a four-row buffer are the page (5, 3 and 2 tokens); row 0 is another page's full five-token
        // segment. Cast on a narrowed view reads from the start of the buffer, so the segments were parsed as
        // rows 0-2 (5, 5 and 3 tokens) and the wrong slots were tagged.
        double[][] rows =
        {
            new double[] { 1, 1, 9, 9, 21, 22, 23, 24, 25 },
            new double[] { 4, 4, 30, 12, 3, 7, 11, 2, 9 },
            new double[] { 34, 4, 60, 12, 5, 5, 8, 0, 0 },
            new double[] { 4, 40, 44, 52, 12, 6, 0, 0, 0 },
        };
        var buffer = new Tensor<double>(new[] { 4, 9 }, new Vector<double>(rows.SelectMany(r => r).ToArray()));
        using var model = SmallPick();
        using var output = model.PredictDocument(Cpu.TensorNarrow(buffer, 0, 1, 3));

        int[] lengths = { 5, 3, 2 };
        int[] tagged = Enumerable.Range(0, 3)
            .Select(i => Enumerable.Range(0, 5).Count(t => Enumerable.Range(0, 5).Sum(c => output[(i * 5) + t, c]) == 1.0))
            .ToArray();
        Assert.Equal(lengths, tagged);
    }
}
