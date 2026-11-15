using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Adversarial checks of the neural-network vectorization sweep: inputs the scalar loops handled that a
/// whole-tensor rewrite can get wrong (out-of-range indices, partial exclusion lists, strided views, large
/// pair grids).
/// </summary>
public class NeuralNetworkSweepAdversarialTests
{
    private static Tensor<double> Doubles(int[] shape, params double[] values) => new Tensor<double>(values, shape);

    private static TabROptions<double> SmallTabROptions(int neighbors) => new TabROptions<double>
    {
        EmbeddingDimension = 2,
        NumAttentionHeads = 1,
        NumLayers = 1,
        NumContextLayers = 1,
        NumNeighbors = neighbors,
        UseLayerNorm = false,
        NormalizeEmbeddings = false,
        DropoutRate = 0
    };

    private static TabPFNOptions<double> SmallTabPFNOptions() => new TabPFNOptions<double>
    {
        EmbeddingDimension = 4,
        NumHeads = 1,
        NumLayers = 1,
        FeedForwardMultiplier = 2,
        OutputHeadDimensions = new[] { 4 },
        DropoutRate = 0,
        UsePositionalEncoding = false
    };

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(1 << 20)]
    public void LinkPredictionAuc_BlockedCountMatchesThePairwiseLoop(int maxPairsPerBlock)
    {
        // Ties, NaN and infinities included: a pair counts only when the positive is strictly greater.
        var pos = new[] { 0.5, -1.0, 2.0, double.NaN, 0.5, double.PositiveInfinity, -3.0 };
        var neg = new[] { 0.5, 0.25, double.NaN, -1.0, 2.0, double.NegativeInfinity, 0.0 };
        long expected = 0;
        foreach (double p in pos)
            foreach (double n in neg)
                if (p > n) expected++;

        long actual = LinkPredictionModel<double>.CountCorrectRankings(
            Doubles(new[] { pos.Length }, pos), Doubles(new[] { neg.Length }, neg), maxPairsPerBlock);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void LinkPredictionAuc_BlockedCountCoversEveryPositiveWhenBlocksDoNotDivideEvenly()
    {
        // 1000 x 700 with blocks of 4096 pairs: 5 positives per block, 200 blocks. Positive i beats min(i, 700)
        // negatives.
        const int numPos = 1000;
        const int numNeg = 700;
        var pos = Enumerable.Range(0, numPos).Select(i => (double)i).ToArray();
        var neg = Enumerable.Range(0, numNeg).Select(i => (double)i).ToArray();

        long actual = LinkPredictionModel<double>.CountCorrectRankings(
            Doubles(new[] { numPos }, pos), Doubles(new[] { numNeg }, neg), 4096);

        Assert.Equal(Enumerable.Range(0, numPos).Sum(i => (long)Math.Min(i, numNeg)), actual);
    }

    [Fact]
    public void TabPFNClassifier_OutOfRangeContextLabelsAreAllZeroRows()
    {
        // The scalar loop skipped labels outside [0, numClasses); the one-hot rewrite must not throw on them.
        var model = new TabPFNClassifier<double>(2, 2, SmallTabPFNOptions());
        model.SetContext(Doubles(new[] { 3, 2 }, 0.4, 0.7, 1.1, 0.2, 0.6, 0.9), new Vector<int>(new[] { 0, -1, 5 }));

        var probabilities = model.PredictProbabilities(Doubles(new[] { 1, 2 }, 0.5, 0.5)).ToArray();

        Assert.All(probabilities, p => Assert.False(double.IsNaN(p) || double.IsInfinity(p)));
    }

    [Fact]
    public void TabR_ExclusionListShorterThanTheBatch_ExcludesOnlyTheCoveredRows()
    {
        // The scalar loop tested "b < excludeIndices.Length", so rows past the list excluded nothing.
        var model = new TabRRegression<double>(2, 1, SmallTabROptions(neighbors: 3));
        var features = Doubles(new[] { 3, 2 }, 0.4, 0.7, 1.1, 0.2, 0.6, 0.9);
        model.BuildIndex(features, Doubles(new[] { 3, 1 }, 0.1, 0.5, 0.9));

        _ = model.Forward(features, new Vector<int>(new[] { 0 }));

        var neighbors = model.GetRetrievedNeighborIndices();
        Assert.NotNull(neighbors);
        Assert.Equal(2, neighbors.Columns);
        Assert.NotEqual(0, neighbors[0, 0]);
        Assert.NotEqual(0, neighbors[0, 1]);
        // Rows 1 and 2 exclude nothing, so each retrieves itself first (distance 0).
        Assert.Equal(1, neighbors[1, 0]);
        Assert.Equal(2, neighbors[2, 0]);
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(false, false)]
    public void TabR_OptionalComponentsLeftOut_ParametersStillReadAndRestore(bool useLayerNorm, bool includeNeighborTargets)
    {
        // A component the options leave out was still registered as a null entry, which the registry treats
        // as a shape-deferred slot: every parameter count and read threw ParameterLayoutNotReadyException.
        var options = SmallTabROptions(neighbors: 2);
        options.UseLayerNorm = useLayerNorm;
        options.IncludeNeighborTargets = includeNeighborTargets;
        var model = new TabRClassifier<double>(2, 2, options);

        var parameters = model.GetParameters();

        Assert.True(parameters.Length > 0);
        Assert.Equal((long)parameters.Length, model.ParameterCount);
        for (int i = 0; i < parameters.Length; i++) parameters[i] = 0.3 + 0.04 * (i % 7);
        model.SetParameters(parameters);
        Assert.Equal(parameters.ToArray(), model.GetParameters().ToArray());
    }

    [Fact]
    public void NonFiniteMask_Half_FlagsNaNAndBothInfinities()
    {
        // The engine's TensorIsFinite throws InvalidCastException on Half (Half is not IConvertible).
        var values = new Tensor<Half>(
            new[] { (Half)1.0, Half.NaN, Half.PositiveInfinity, Half.NegativeInfinity, Half.MaxValue, Half.MinValue, (Half)0.0 },
            new[] { 7 });

        var mask = TensorFiniteness.NonFiniteMask(values).ToArray().Select(v => (double)v).ToArray();

        Assert.Equal(new[] { 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 }, mask);
    }

    [Fact]
    public void NonFiniteMask_Double_MatchesIsFinite()
    {
        var raw = new[] { 0.0, -1.5, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.MaxValue, double.Epsilon };

        var mask = TensorFiniteness.NonFiniteMask(Doubles(new[] { raw.Length }, raw)).ToArray();

        Assert.Equal(raw.Select(v => double.IsNaN(v) || double.IsInfinity(v) ? 1.0 : 0.0).ToArray(), mask);
    }

    [Fact]
    public void CriticScoreMetrics_Half_InfiniteScoreDoesNotThrow()
    {
        // One critic output overflowed: the scale is the largest FINITE magnitude, and the maximum is the infinity.
        var metrics = new Dictionary<string, double>();
        var scores = new Tensor<Half>(new[] { (Half)2.0, Half.PositiveInfinity, (Half)(-4.0) }, new[] { 3, 1 });

        CriticScoreMetrics.Add(metrics, scores, 3);

        Assert.Equal(-4.0, metrics["MinCriticScore"]);
        Assert.True(double.IsPositiveInfinity(metrics["MaxCriticScore"]));
        Assert.True(double.IsPositiveInfinity(metrics["AverageCriticScore"]));
    }

    [Fact]
    public void CriticScoreMetrics_ReadTheFirstColumnOfAMultiColumnCritic()
    {
        // Column 0 of a [3, 2] tensor is a strided view; the statistics must not read column 1.
        var metrics = new Dictionary<string, double>();

        CriticScoreMetrics.Add(metrics, Doubles(new[] { 3, 2 }, 1, 10, 2, 20, 3, 30), 3);

        Assert.Equal(2.0, metrics["AverageCriticScore"], 10);
        Assert.Equal(1.0, metrics["MinCriticScore"], 10);
        Assert.Equal(3.0, metrics["MaxCriticScore"], 10);
        Assert.Equal(1.0, metrics["CriticScoreStdDev"], 10);
    }

    [Fact]
    public void CriticScoreMetrics_UseOnlyTheFirstSampleSizeRows()
    {
        var metrics = new Dictionary<string, double>();

        CriticScoreMetrics.Add(metrics, Doubles(new[] { 4, 2 }, 1, 10, 3, 20, 100, 30, 200, 40), 2);

        Assert.Equal(2.0, metrics["AverageCriticScore"], 10);
        Assert.Equal(3.0, metrics["MaxCriticScore"], 10);
    }

    [Fact]
    public void NarrowedViews_KeepTheirOffsetThroughTheOpsTheSweepUses()
    {
        var engine = AiDotNetEngine.Current;
        var grid = Doubles(new[] { 3, 3 }, 1, 0, 0, 0, 1, 0, 0, 0, 1);

        // GraphClassificationModel.GetTrueClass: row 2 of the labels.
        var row = engine.TensorNarrow(engine.TensorNarrow(grid, 0, 2, 1), 1, 0, 3);
        Assert.Equal(2, TieBreakArgMax.FirstMaxIndex(engine.TensorNotEquals(row, 0.0)));

        // LinkPredictionModel.EdgeEndpoints: column 1 of the edges.
        var edges = Doubles(new[] { 3, 2 }, 0, 5, 1, 6, 2, 7);
        var targets = engine.TensorRound(engine.TensorNarrow(edges, 1, 1, 1)).Reshape(3).Cast<int>().ToArray();
        Assert.Equal(new[] { 5, 6, 7 }, targets);

        // TabNetBase.SplitTensor + ApplyReLU: the trailing columns.
        var tail = engine.TensorNarrow(Doubles(new[] { 2, 3 }, 1, -2, 3, 4, 5, -6), 1, 1, 2);
        var relu = engine.TensorWhere(engine.TensorGreaterThan(tail, 0.0), tail, engine.TensorZerosLike(tail));
        Assert.Equal(new[] { 0.0, 3.0, 5.0, 0.0 }, relu.ToArray());

        // TabularMetrics.ClassIndices over a prefix, and Flat over a strided view.
        var classes = TabularMetrics.ClassIndices(Doubles(new[] { 4 }, 2.9, 1.2, 0.7, 3.0), 3).ToArray();
        Assert.Equal(new[] { 2, 1, 0 }, classes);
        Assert.Equal(new[] { -2.0, 3.0, 5.0 }, TabularMetrics.Flat(tail, 3).ToArray());

        // SAINTBase.ApplyRowAttention: [batch, features, dim] -> [features, batch, dim] -> flat.
        var cube = Doubles(new[] { 2, 3, 1 }, 1, 2, 3, 4, 5, 6);
        var byFeature = engine.TensorPermute(cube, new[] { 1, 0, 2 }).Reshape(6).ToArray();
        Assert.Equal(new[] { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 }, byFeature);

        // CriticScoreMetrics / VideoCLIP: a strided column reshaped to a vector, then reduced.
        var column = engine.TensorNarrow(Doubles(new[] { 3, 2 }, 1, 10, 2, 20, 3, 30), 1, 1, 1).Reshape(3);
        Assert.Equal(60.0, engine.TensorSum(column), 10);
        Assert.Equal(30.0, engine.TensorMaxValue(column), 10);
    }

    [Fact]
    public void IndexAdd_AccumulatesRepeatedRows_AsTheEdgeGradientLoopDid()
    {
        // LinkPredictionModel.ComputeBCEGradients adds one update per edge endpoint; a node on several edges
        // appears several times in the index list and must receive the sum, not the last update.
        var engine = AiDotNetEngine.Current;
        var indices = new Tensor<int>(new[] { 0, 2, 0, 0 }, new[] { 4 });
        var updates = Doubles(new[] { 4, 2 }, 1, 10, 2, 20, 3, 30, 4, 40);

        var sum = engine.TensorIndexAdd(new Tensor<double>(new[] { 3, 2 }), 0, indices, updates).ToArray();

        Assert.Equal(new[] { 8.0, 80.0, 0.0, 0.0, 2.0, 20.0 }, sum);
    }

    [Fact]
    public void GhostBatchNormalization_ForwardThatIsNotATrainingStep_LeavesRunningStatisticsUnchanged()
    {
        // The fused session's lazy-shape pass and replay checks run training-mode forwards that are not steps;
        // like BatchNormalizationLayer, the ghost batch norm must not advance its EMA on them.
        var layer = new GhostBatchNormalization<double>(2);
        layer.SetTrainingMode(true);
        var batch = Doubles(new[] { 4, 2 }, 1, 10, 3, 30, 5, 50, 7, 70);
        var meanBefore = layer.RunningMean.ToArray();
        var varBefore = layer.RunningVar.ToArray();

        using (AiDotNet.Training.CompiledStepRandom<double>.SuppressDraws())
        {
            _ = layer.Forward(batch);
        }

        Assert.Equal(meanBefore, layer.RunningMean.ToArray());
        Assert.Equal(varBefore, layer.RunningVar.ToArray());

        // A real step does advance it: momentum 0.02 toward the batch mean (4, 40) and the unbiased variance
        // (20/3, 2000/3), starting from mean 0 and variance 1.
        _ = layer.Forward(batch);
        var mean = layer.RunningMean.ToArray();
        var variance = layer.RunningVar.ToArray();
        Assert.Equal(0.02 * 4, mean[0], 12);
        Assert.Equal(0.02 * 40, mean[1], 12);
        Assert.Equal(0.98 + 0.02 * 20.0 / 3.0, variance[0], 12);
        Assert.Equal(0.98 + 0.02 * 2000.0 / 3.0, variance[1], 12);
    }

    [Fact]
    public void SquaredDistances_KeysCloseRelativeToTheirNorm_AreExact()
    {
        // ||q||^2 + ||c||^2 - 2 q.c at norm 1e8 cancels 1e16-sized terms whose spacing is 2, so distances of 1
        // and 0.25 come out as 0 or 2 and the two candidates swap rank. Direct differences are exact.
        var queries = Doubles(new[] { 1, 2 }, 1e8, 0);
        var candidates = Doubles(new[] { 2, 2 }, 1e8 + 1, 0, 1e8, 0.5);

        var distances = TabularMetrics.SquaredDistances(queries, candidates);

        Assert.Equal(new[] { 1, 2 }, distances.Shape.ToArray());
        Assert.Equal(new[] { 1.0, 0.25 }, distances.ToArray());
    }

    [Fact]
    public void TabR_WritesToTheCallersFeaturesAfterBuildIndex_DoNotChangePredictions()
    {
        // Every forward re-encodes the indexed features, so the index must own them, as it owns the labels.
        var model = new TabRRegression<double>(2, 1, SmallTabROptions(neighbors: 2));
        var features = Doubles(new[] { 3, 2 }, 0.4, 0.7, 1.1, 0.2, 0.6, 0.9);
        model.BuildIndex(features, Doubles(new[] { 3, 1 }, 0.1, 0.5, 0.9));
        var query = Doubles(new[] { 1, 2 }, 0.5, 0.6);
        var before = model.Predict(query).ToArray();

        features[0, 0] = 40.0;
        features[1, 1] = -25.0;
        var after = model.Predict(query).ToArray();

        Assert.Equal(before, after);
    }

    [Fact]
    public void TabPFN_WritesToTheCallersFeaturesAfterSetContext_DoNotChangePredictions()
    {
        var model = new TabPFNClassifier<double>(2, 2, SmallTabPFNOptions());
        var features = Doubles(new[] { 3, 2 }, 0.4, 0.7, 1.1, 0.2, 0.6, 0.9);
        model.SetContext(features, new Vector<int>(new[] { 0, 1, 0 }));
        var query = Doubles(new[] { 1, 2 }, 0.5, 0.5);
        var before = model.PredictProbabilities(query).ToArray();

        features[0, 0] = 40.0;
        features[2, 1] = -25.0;
        var after = model.PredictProbabilities(query).ToArray();

        Assert.Equal(before, after);
    }

    [Fact]
    public void NODELeafPathBits_Half_DeepTreesRouteEveryLeafByItsBinaryId()
    {
        // At depth 12 there are 4096 leaves; Half holds integers exactly only up to 2048, so leaf ids computed in
        // T collapse (2049 -> 2048) and route leaves down the wrong branch.
        const int depth = 12;
        var bits = NODEBase<Half>.LeafPathBits(depth);

        Assert.Equal(new[] { depth, 1 << depth }, bits.Shape.ToArray());
        for (int leaf = 0; leaf < 1 << depth; leaf++)
        {
            for (int d = 0; d < depth; d++)
            {
                double expected = (leaf >> (depth - 1 - d)) & 1;
                Assert.Equal(expected, (double)bits[d, leaf]);
            }
        }
    }
}
