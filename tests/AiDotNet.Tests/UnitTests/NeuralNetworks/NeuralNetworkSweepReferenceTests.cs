using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Reference values for the neural-network vectorization review: integer counts that Half cannot hold, Half
/// overflow and underflow, PyTorch batch-norm running statistics, and fields that alias a caller's tensor.
/// </summary>
public class NeuralNetworkSweepReferenceTests
{
    private static Tensor<Half> HalfTensor(int[] shape, params double[] values) =>
        new Tensor<Half>(values.Select(v => (Half)v).ToArray(), shape);

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    [Fact]
    public void FirstMaxIndex_Half_FindsTheMaximumPastTheHalfIntegerRange()
    {
        // rank * cols + column overflows Half (65504) for 40000 columns, so a Half key ties column 0 with the max.
        var values = new Half[40000];
        values[20000] = (Half)1.0;

        Assert.Equal(20000, TieBreakArgMax.FirstMaxIndex(new Tensor<Half>(values, new[] { 40000 })));
    }

    [Fact]
    public void StableDescendingOrder_Half_RanksMoreThan2048Columns()
    {
        // Distinct exact integers, increasing: the descending order is the reversed columns. A Half count of
        // the larger elements stops at 2048 and ties the first 52 columns.
        const int cols = 2100;
        var values = new Half[cols];
        for (int j = 0; j < cols; j++) values[j] = (Half)(j - 1050);

        var order = TieBreakArgMax.StableDescendingOrder(new Tensor<Half>(values, new[] { 1, cols })).ToArray();

        Assert.Equal(Enumerable.Range(0, cols).Select(i => cols - 1 - i).ToArray(), order);
    }

    [Fact]
    public void GhostBatchNorm_RunningVarianceIsUnbiased()
    {
        // torch.nn.BatchNorm1d(1, momentum=0.02) on [0, 2]: biased var 1, unbiased 2, running 0.98 * 1 + 0.02 * 2.
        using var layer = new GhostBatchNormalization<double>(1);
        layer.SetTrainingMode(true);

        _ = layer.Forward(new Tensor<double>(new[] { 0.0, 2.0 }, new[] { 2, 1 }));

        Assert.Equal(1.02, layer.RunningVar[0], 12);
        Assert.Equal(0.02, layer.RunningMean[0], 12);
    }

    [Fact]
    public void CriticScoreMetrics_Half_StandardDeviationDoesNotOverflow()
    {
        // Sample std of [4e4, 4e4, -4e4, -4e4] is 4e4 * sqrt(4 / 3) = 46188; squaring 4e4 in Half is infinity.
        var metrics = new Dictionary<string, double>();

        CriticScoreMetrics.Add(metrics, HalfTensor(new[] { 4, 1 }, 40000, 40000, -40000, -40000), 4);

        Assert.Equal(0.0, metrics["AverageCriticScore"], 6);
        Assert.Equal(-40000.0, metrics["MinCriticScore"]);
        Assert.Equal(40000.0, metrics["MaxCriticScore"]);
        Assert.InRange(metrics["CriticScoreStdDev"], 46140.0, 46240.0);
    }

    [Fact]
    public void GraphLabelLogLikelihood_Half_ZeroProbabilityOnAZeroLabelIsZero()
    {
        // 1e-10 rounds to 0 in Half: log(0) on the zero label gave 0 * -inf = NaN.
        Half value = GraphClassificationModel<Half>.LabelLogLikelihood(
            HalfTensor(new[] { 1, 2 }, 1, 0), HalfTensor(new[] { 1, 2 }, 1, 0));

        Assert.Equal(0.0, (double)value, 12);
    }

    [Fact]
    public void NodeCrossEntropy_Half_MeanDoesNotOverflow()
    {
        // 3000 nodes each with loss 30: the total 90000 overflows Half, the mean 30 does not.
        const int rows = 3000;
        var logits = new double[rows * 2];
        var labels = new double[rows * 2];
        for (int r = 0; r < rows; r++)
        {
            logits[2 * r + 1] = 30;
            labels[2 * r] = 1;
        }

        double loss = NodeClassificationModel<Half>.MeanLogitCrossEntropy(
            HalfTensor(new[] { rows, 2 }, logits), HalfTensor(new[] { rows, 2 }, labels));

        Assert.InRange(loss, 29.9, 30.1);
    }

    [Fact]
    public void CrossEntropyObjective_Half_ZeroProbabilityOnAZeroLabelIsZero()
    {
        // epsilon 1e-15 is 0 in Half, so log(p + eps) on the zero label was log(0) and 0 * log(0) NaN.
        var loss = TabularTraining.CrossEntropyObjective(
            HalfTensor(new[] { 1, 2 }, 1, 0), HalfTensor(new[] { 1, 2 }, 1, 0), (Half)1e-15);

        Assert.Equal(0.0, (double)loss.GetFlat(0), 12);
    }

    [Fact]
    public void Entmax15_Half_SupportCountedPast2048()
    {
        // X = z / 2 = [0, -2, ...]: tau = -1, so the output is one-hot. 2099 entries lie outside the support,
        // which a Half count cannot hold.
        const int n = 2100;
        var logits = new double[n];
        logits[0] = 4;

        var output = NODEBase<Half>.Entmax15(HalfTensor(new[] { 1, n }, logits)).ToArray();

        Assert.Equal(1.0, (double)output[0], 6);
        for (int i = 1; i < n; i++) Assert.Equal(0.0, (double)output[i]);
    }

    private static TabNetOptions<T> SmallTabNetOptions<T>() => new TabNetOptions<T>
    {
        NumDecisionSteps = 2,
        FeatureDimension = 8,
        OutputDimension = 4,
        SparsityCoefficient = 0.5
    };

    [Fact]
    public void TabNet_TrainStepTrainsTheInitialBatchNorm()
    {
        // Arik and Pfister (2021) batch-normalize the raw input with a learned gamma and beta; GetParameters
        // starts with them (gamma then beta, 4 each).
        var model = new TabNetRegression<double>(4, SmallTabNetOptions<double>());
        var input = new Tensor<double>(new[] { 4, 4 }, new Vector<double>(new[]
        {
            0.4, -0.7, 1.1, 0.2, 0.6, 0.9, -0.3, 0.5, -1.2, 0.3, 0.8, -0.4, 0.1, 1.4, -0.6, 0.7
        }));
        var targets = new Tensor<double>(new[] { 4, 1 }, new Vector<double>(new[] { 0.3, -0.8, 0.5, 1.1 }));
        var before = model.GetParameters().ToArray();

        _ = model.TrainStep(input, targets, 0.05);

        var after = model.GetParameters().ToArray();
        Assert.False(before.Take(8).SequenceEqual(after.Take(8)), "The initial batch norm's gamma and beta must train.");
    }

    [Fact]
    public void TabNet_Half_SparsityObjectiveIsFiniteWithZeroMaskEntries()
    {
        var model = new TabNetRegression<Half>(8, SmallTabNetOptions<Half>());
        var inputValues = new double[32];
        for (int i = 0; i < inputValues.Length; i++) inputValues[i] = 3.0 * Math.Sin(1.3 * i + 0.2) * (1 + i % 4);
        var input = HalfTensor(new[] { 4, 8 }, inputValues);
        var targets = HalfTensor(new[] { 4, 1 }, 0.3, -0.8, 0.5, 1.1);

        Half loss = model.TrainStep(input, targets, (Half)0.01);

        // Sparsemax masks hold exact zeros, whose 0 * log(0 + eps) with eps = 0 in Half was NaN.
        Assert.Contains(model.GetStepAttentionMasks().Skip(1).SelectMany(m => m.ToArray()), m => (double)m == 0.0);
        Assert.True(IsFinite((double)loss), $"Loss {(double)loss} must be finite.");
        Assert.All(model.GetParameters().ToArray(), p => Assert.True(IsFinite((double)p)));
    }

    [Fact]
    public void TabMF1_Half_CountsPast2048()
    {
        var options = new TabMOptions<Half>
        {
            NumEnsembleMembers = 1,
            HiddenDimensions = new[] { 4 },
            DropoutRate = 0,
            UseLayerNorm = false,
            UseFeatureEmbeddings = false
        };
        var model = new TabMClassifier<Half>(2, 2, options);

        // Identical rows share one prediction; 2051 targets agree with it and 2049 do not.
        const int n = 4100;
        var values = new double[n * 2];
        for (int i = 0; i < n; i++)
        {
            values[2 * i] = 0.5;
            values[2 * i + 1] = -0.25;
        }

        var features = HalfTensor(new[] { n, 2 }, values);
        var predictions = model.Predict(features);
        var targets = predictions.Select((p, i) => i < 2051 ? p : 1 - p).ToArray();

        double expected = 0;
        for (int c = 0; c < 2; c++)
        {
            int tp = Enumerable.Range(0, n).Count(i => predictions[i] == c && targets[i] == c);
            int predicted = predictions.Count(p => p == c);
            int actual = targets.Count(t => t == c);
            double precision = predicted > 0 ? (double)tp / predicted : 0;
            double recall = actual > 0 ? (double)tp / actual : 0;
            expected += precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        }

        expected /= 2;
        double f1 = (double)model.ComputeF1Score(features, targets);
        Assert.InRange(f1, expected - 2e-3, expected + 2e-3);
    }

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

    private static Tensor<double> ThreeSamples() =>
        new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2, 0.6, 0.9 }));

    private static void SetDistinctParameters(Vector<double> parameters)
    {
        for (int i = 0; i < parameters.Length; i++) parameters[i] = 0.3 + 0.04 * (i % 7);
    }

    [Fact]
    public void TabRExplanations_UseTheNeighboursActuallyRetrieved()
    {
        // NumNeighbors 5 over an index of 3 retrieves 3 neighbours per sample.
        var model = new TabRClassifier<double>(2, 2, SmallTabROptions(neighbors: 5));
        model.BuildIndex(ThreeSamples(), new[] { 0, 1, 1 });

        var explanations = model.GetPredictionExplanations(ThreeSamples());

        Assert.Equal(3, explanations.Count);
        Assert.All(explanations, sample => Assert.Equal(3, sample.Count));
    }

    [Fact]
    public void TabRIndex_KeepsItsOwnCopyOfTheTargets()
    {
        var model = new TabRRegression<double>(2, 1, SmallTabROptions(neighbors: 2));
        var parameters = model.GetParameters();
        SetDistinctParameters(parameters);
        model.SetParameters(parameters);
        var targets = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 }));
        model.BuildIndex(ThreeSamples(), targets);
        var before = model.Predict(ThreeSamples()).ToArray();

        for (int i = 0; i < targets.Length; i++) targets[i] = 100.0;
        var after = model.Predict(ThreeSamples()).ToArray();

        Assert.Equal(before, after);
    }

    private static TabPFNOptions<T> SmallTabPFNOptions<T>() => new TabPFNOptions<T>
    {
        EmbeddingDimension = 4,
        NumHeads = 1,
        NumLayers = 1,
        FeedForwardMultiplier = 2,
        OutputHeadDimensions = new[] { 4 },
        DropoutRate = 0,
        UsePositionalEncoding = false
    };

    [Fact]
    public void TabPFNContext_KeepsItsOwnCopyOfTheLabels()
    {
        var model = new TabPFNRegression<double>(2, 1, SmallTabPFNOptions<double>());
        var labels = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 }));
        model.SetContext(ThreeSamples(), labels);
        var query = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.5, 0.5 }));
        var before = model.Predict(query).ToArray();

        for (int i = 0; i < labels.Length; i++) labels[i] = 100.0;
        var after = model.Predict(query).ToArray();

        Assert.Equal(before, after);
    }

    [Fact]
    public void TabPFN_Half_ContextMaskKeepsPredictionsFinite()
    {
        // An additive -1e9 mask is -infinity in Half, and 0 * -infinity on the context columns made every score NaN.
        var model = new TabPFNClassifier<Half>(2, 2, SmallTabPFNOptions<Half>());
        model.SetContext(HalfTensor(new[] { 3, 2 }, 0.4, 0.7, 1.1, 0.2, 0.6, 0.9), new Vector<int>(new[] { 0, 1, 1 }));

        var probabilities = model.PredictProbabilities(HalfTensor(new[] { 2, 2 }, 0.5, 0.5, 1.0, 0.3)).ToArray();

        Assert.All(probabilities, p => Assert.True(IsFinite((double)p), $"Probability {(double)p} must be finite."));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NetworkImportance_KeepsItsOwnCopyOfThePredictedBatch(bool saint)
    {
        TabularNeuralNetworkBase<double> network = saint
            ? new SAINTNetwork<double>()
            : new TabTransformerNetwork<double>();
        int features = saint ? 10 : 16;
        var batch = new Tensor<double>(new[] { 4, features });
        for (int i = 0; i < batch.Length; i++) batch[i] = Math.Sin(0.7 * i + 0.3) * (1 + i % 3);
        _ = network.Predict(batch);

        // Overwriting the caller's batch must not change what is attributed: a constant batch would be uniform.
        for (int i = 0; i < batch.Length; i++) batch[i] = 0.0;
        var importance = network.GetFeatureImportance();

        Assert.Contains(importance.Values, value => Math.Abs(value - 1.0 / features) > 1e-9);
    }

    [Fact]
    public void VideoClipNucleus_Half_ReachesTopPOverManySmallProbabilities()
    {
        // Three tokens hold 0.618 of the mass and 1900 tokens 2e-4 each. A Half running total stops at 0.618
        // (half its spacing there exceeds 2e-4), so the nucleus never reached 0.9 and only the three were drawn.
        const int small = 1900;
        var logits = new Half[small + 3];
        for (int i = 0; i < small; i++) logits[i] = (Half)Math.Log(1.0 / 1024);
        for (int i = small; i < logits.Length; i++) logits[i] = (Half)0.0;

        var draws = Enumerable.Range(0, 200)
            .Select(_ => VideoCLIPNeuralNetwork<Half>.SampleNextToken(logits, temperature: 1.0, topP: 0.9))
            .ToArray();

        // The nucleus is the three large tokens and about the first 1400 small ones, which carry 0.28 / 0.9 of it.
        Assert.True(draws.Count(t => t < small) >= 5, "Small tokens inside the nucleus must be drawn.");
        Assert.All(draws, t => Assert.True(t >= small || t < 1500, $"Token {t} lies outside the nucleus."));
    }
}
