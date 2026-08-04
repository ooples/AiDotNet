using System;
using AiDotNet.Finance.Trading.Factors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance.Factors;

/// <summary>
/// End-to-end coverage for <see cref="Stockformer{T}"/> and its dual-frequency encoder
/// (Ma et al., arXiv:2401.06139).
/// </summary>
public class StockformerModelTests
{
    private const int Stocks = 6;
    private const int Window = 16;

    private static StockformerOptions<double> SmallOptions(int classes = 2) => new()
    {
        NumAssets = Stocks,
        NumFeatures = 8,
        HiddenDimension = 8,
        SpatialSamples = 1,
        SequenceLength = Window,
        NumDirectionClasses = classes,
    };

    private static Matrix<double> Returns(int seed = 7)
    {
        var rng = new Random(seed);
        var m = new Matrix<double>(Stocks, Window);
        for (int s = 0; s < Stocks; s++)
            for (int t = 0; t < Window; t++)
                m[s, t] = rng.NextDouble() - 0.5;
        return m;
    }

    [Fact]
    public void PredictProducesFourOutputsWithTheRightShapes()
    {
        // Four, not two: both heads are applied to BOTH the fused and low-frequency representations,
        // which is what makes the paper's loss a four-term sum.
        var model = new Stockformer<double>(SmallOptions());
        var p = model.PredictBands(Returns());

        Assert.Equal(Stocks, p.Returns.Length);
        Assert.Equal(Stocks, p.LowReturns.Length);
        Assert.Equal(Stocks * 2, p.DirectionLogits.Length);
        Assert.Equal(Stocks * 2, p.LowDirectionLogits.Length);
    }

    [Fact]
    public void DirectionHeadWidthFollowsTheConfiguredClassCount()
    {
        var model = new Stockformer<double>(SmallOptions(classes: 3));
        var p = model.PredictBands(Returns());
        Assert.Equal(Stocks * 3, p.DirectionLogits.Length);
    }

    [Fact]
    public void OutputsAreFinite()
    {
        var model = new Stockformer<double>(SmallOptions());
        var p = model.PredictBands(Returns());

        foreach (var v in new[] { p.Returns, p.LowReturns, p.DirectionLogits, p.LowDirectionLogits })
        {
            for (int i = 0; i < v.Length; i++)
                Assert.False(double.IsNaN(v[i]) || double.IsInfinity(v[i]), $"Non-finite output at {i}.");
        }
    }

    [Fact]
    public void SeededConstructionIsReproducible()
    {
        var a = new Stockformer<double>(SmallOptions()).PredictBands(Returns());
        var b = new Stockformer<double>(SmallOptions()).PredictBands(Returns());
        for (int i = 0; i < a.Returns.Length; i++) Assert.Equal(a.Returns[i], b.Returns[i], 10);
    }

    [Fact]
    public void TheGraphActuallyChangesTheCrossSection()
    {
        // The adjacency is a real input, not decoration: connecting stocks must alter their
        // representations relative to the identity (isolated) graph. If this passes trivially, the
        // spatial attention is not consuming the graph.
        var model = new Stockformer<double>(SmallOptions());
        var isolated = model.PredictBands(Returns());

        var dense = new Matrix<double>(Stocks, Stocks);
        for (int i = 0; i < Stocks; i++)
            for (int j = 0; j < Stocks; j++) dense[i, j] = 1.0;
        model.Adjacency = dense;
        var connected = model.PredictBands(Returns());

        bool differs = false;
        for (int i = 0; i < isolated.Returns.Length; i++)
            if (Math.Abs(isolated.Returns[i] - connected.Returns[i]) > 1e-9) { differs = true; break; }

        Assert.True(differs, "A fully connected graph produced identical output to an isolated one.");
    }

    [Fact]
    public void FusedAndLowRepresentationsDiffer()
    {
        // If they matched, the adaptive fusion of the high band would be contributing nothing and the
        // model would be single-frequency with extra steps.
        var model = new Stockformer<double>(SmallOptions());
        var p = model.PredictBands(Returns());

        bool differs = false;
        for (int i = 0; i < p.Returns.Length; i++)
            if (Math.Abs(p.Returns[i] - p.LowReturns[i]) > 1e-9) { differs = true; break; }

        Assert.True(differs, "Fused output equals the low-band output; the high band never reached the head.");
    }

    [Fact]
    public void LossIsFiniteAndSumsItsTwoTasks()
    {
        var model = new Stockformer<double>(SmallOptions());
        var target = new Vector<double>(Stocks);
        var direction = new Vector<double>(Stocks);
        for (int s = 0; s < Stocks; s++)
        {
            target[s] = 0.01 * (s + 1);          // non-sentinel so the mask keeps every entry
            direction[s] = s % 2;
        }

        var (regression, classification, total) = model.ComputeLoss(Returns(), target, direction);

        Assert.False(double.IsNaN(total) || double.IsInfinity(total), $"Non-finite loss: {total}");
        Assert.True(regression > 0.0, "Regression term collapsed to zero on non-sentinel targets.");
        Assert.True(classification > 0.0, "Classification term collapsed to zero.");
        Assert.Equal(regression + classification, total, 10);   // unweighted 1:1
    }

    [Fact]
    public void ConstructorRejectsDegenerateConfiguration()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new Stockformer<double>(new StockformerOptions<double> { HiddenDimension = 0 }));
        // One class makes the classification task vacuous.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new Stockformer<double>(new StockformerOptions<double> { NumDirectionClasses = 1 }));
    }

    [Fact]
    public void MetadataNamesTheRealPaperNotTheFabricatedOne()
    {
        var metadata = new Stockformer<double>(SmallOptions()).GetModelMetadata();
        Assert.Equal("Stockformer", metadata.Name);
        // The class this replaced described itself as a plain factor transformer and cited a
        // general-relativity paper; the wavelet and multi-task properties are the substance.
        Assert.Equal("sym2", metadata.Properties["wavelet"]);
        Assert.Equal(true, metadata.Properties["multi_task"]);
    }

    [Fact]
    public void EncoderRejectsAnAdjacencyThatIsNotStockByStock()
    {
        var encoder = new StockformerDualEncoder<double>(features: 4, spatialSamples: 1, kernelWidth: 3, new Random(1));
        var band = new Tensor<double>(new[] { 3, 5, 4 });
        var te = new Matrix<double>(5, 4);
        var wrong = new Matrix<double>(2, 2);

        Assert.Throws<ArgumentException>(() => encoder.Encode(band, band, te, wrong));
    }

    [Fact]
    public void EncoderRejectsMismatchedBands()
    {
        var encoder = new StockformerDualEncoder<double>(features: 4, spatialSamples: 1, kernelWidth: 3, new Random(1));
        var low = new Tensor<double>(new[] { 3, 5, 4 });
        var high = new Tensor<double>(new[] { 3, 6, 4 });   // the two bands come from ONE split
        var te = new Matrix<double>(5, 4);
        var adjacency = new Matrix<double>(3, 3);

        Assert.Throws<ArgumentException>(() => encoder.Encode(low, high, te, adjacency));
    }
}
