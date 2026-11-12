using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tabular;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// TabR's retrieval module (Gorishniy et al. 2024): L2 key similarity, label-conditioned values, leave-one-out
/// candidates, the tape training step, and the neighbour-spread confidence interval.
/// </summary>
public class TabRRetrievalTests
{
    private static TabROptions<double> SmallOptions(int neighbors) => new TabROptions<double>
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

    private static Tensor<double> Features() =>
        new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2, 0.6, 0.9 }));

    private static void SetDistinctParameters(TabRBase<double> model)
    {
        // Positive, distinct values keep the ReLUs away from their kink and break attention symmetry.
        var initial = model.GetParameters();
        for (int i = 0; i < initial.Length; i++) initial[i] = 0.3 + 0.04 * (i % 7);
        model.SetParameters(initial);
    }

    [Fact]
    public void LeaveOneOut_NeverRetrievesTheTargetItself()
    {
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 3));
        var features = Features();
        model.BuildIndex(features, new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 })));

        _ = model.Forward(features, new Vector<int>(new[] { 0, 1, 2 }));

        // Three candidates, one excluded per row: k = 2, and row b never holds b.
        var neighbors = model.GetRetrievedNeighborIndices();
        Assert.NotNull(neighbors);
        Assert.Equal(2, neighbors.Columns);
        for (int b = 0; b < 3; b++)
        {
            Assert.NotEqual(b, neighbors[b, 0]);
            Assert.NotEqual(b, neighbors[b, 1]);
            Assert.NotEqual(neighbors[b, 0], neighbors[b, 1]);
        }
    }

    [Fact]
    public void NeighbourLabels_ChangeThePrediction()
    {
        var features = Features();
        var model = new TabRClassifier<double>(2, 2, SmallOptions(neighbors: 2));
        SetDistinctParameters(model);

        model.BuildIndex(features, new[] { 0, 1, 1 });
        var withLabels = model.PredictProbabilities(features).ToArray();
        model.BuildIndex(features, new[] { 1, 0, 0 });
        var withFlippedLabels = model.PredictProbabilities(features).ToArray();
        model.BuildIndex(features);
        var withoutLabels = model.PredictProbabilities(features).ToArray();

        Assert.False(withLabels.SequenceEqual(withFlippedLabels), "The label embedding W_Y(y_i) must reach the output.");
        Assert.False(withLabels.SequenceEqual(withoutLabels), "Indexing labels must add the label term.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TrainStep_WholeModelUpdateMatchesFiniteDifferences(bool classification)
    {
        var features = Features();
        var sampleIndices = new Vector<int>(new[] { 0, 1, 2 });
        var targets = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.3, 0.8, 0.5 }));
        var labels = new[] { 0, 1, 1 };

        // Every row retrieves both other samples, so the neighbour set cannot change under a perturbation.
        TabRBase<double> model = classification
            ? new TabRClassifier<double>(2, 2, SmallOptions(neighbors: 2))
            : new TabRRegression<double>(2, 1, SmallOptions(neighbors: 2));
        if (model is TabRClassifier<double> indexedClassifier) indexedClassifier.BuildIndex(features, labels);
        else ((TabRRegression<double>)model).BuildIndex(features, targets);
        SetDistinctParameters(model);
        var initial = model.GetParameters();

        double Loss() => model is TabRClassifier<double> classifier
            ? classifier.ComputeCrossEntropyLoss(
                Softmax(classifier.Forward(features, sampleIndices)), labels)
            : ((TabRRegression<double>)model).ComputeMSELoss(
                ((TabRRegression<double>)model).Forward(features, sampleIndices), targets);

        double beforeLoss = Loss();
        const double epsilon = 1e-5;
        var numerical = new double[initial.Length];
        for (int i = 0; i < initial.Length; i++)
        {
            var perturbed = new Vector<double>(initial.ToArray());
            perturbed[i] += epsilon;
            model.SetParameters(perturbed);
            double plus = Loss();
            perturbed[i] -= 2 * epsilon;
            model.SetParameters(perturbed);
            numerical[i] = (plus - Loss()) / (2 * epsilon);
        }

        model.SetParameters(initial);
        const double rate = 0.01;
        double reported = model is TabRClassifier<double> c
            ? c.TrainStep(features, labels, rate, sampleIndices)
            : ((TabRRegression<double>)model).TrainStep(features, targets, rate, sampleIndices);
        Assert.Equal(beforeLoss, reported, 10);

        var updated = model.GetParameters();
        for (int i = 0; i < initial.Length; i++)
        {
            double actual = (initial[i] - updated[i]) / rate;
            Assert.True(Math.Abs(numerical[i] - actual) < 1e-6 + 1e-4 * Math.Abs(numerical[i]),
                $"Parameter {i}: finite difference {numerical[i]:R}, training gradient {actual:R}.");
        }

        Assert.True(Loss() < beforeLoss, "One SGD step should lower this smooth objective.");
    }

    [Fact]
    public void PredictWithConfidence_IsTheRetrievalWeightedSpreadOfNeighbourTargets()
    {
        // Two candidates with targets 0 and 2, both retrieved: var = w0 (0 - m)^2 + w1 (2 - m)^2 with
        // m = 2 w1, which is 4 w0 w1; the 95% half-width is z * 2 sqrt(w0 w1) with z = 1.959963984540054.
        var index = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2 }));
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 2));
        model.BuildIndex(index, new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new[] { 0.0, 2.0 })));
        SetDistinctParameters(model);

        var query = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.6, 0.9 }));
        var (predictions, lower, upper) = model.PredictWithConfidence(query, 0.95);

        var weights = model.GetAttentionWeights();
        Assert.NotNull(weights);
        double w0 = weights[0];
        double w1 = weights[1];
        Assert.Equal(1.0, w0 + w1, 12);
        double halfWidth = 1.959963984540054 * 2.0 * Math.Sqrt(w0 * w1);
        Assert.Equal(predictions[0] - halfWidth, lower[0], 9);
        Assert.Equal(predictions[0] + halfWidth, upper[0], 9);
    }

    [Fact]
    public void PredictWithConfidence_IsZeroWidthWhenNeighboursAgree()
    {
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 3));
        model.BuildIndex(Features(), new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 1.5, 1.5, 1.5 })));

        var query = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.6, 0.9 }));
        var (predictions, lower, upper) = model.PredictWithConfidence(query);

        Assert.Equal(predictions[0], lower[0], 12);
        Assert.Equal(predictions[0], upper[0], 12);
    }

    [Fact]
    public void PredictWithConfidence_WithoutIndexedTargets_Throws()
    {
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 2));
        model.BuildIndex(Features());

        Assert.Throws<InvalidOperationException>(() => model.PredictWithConfidence(Features()));
    }

    private static Tensor<double> Softmax(Tensor<double> logits)
    {
        int rows = logits.Shape[0];
        int cols = logits.Shape[1];
        var result = new Tensor<double>(new[] { rows, cols });
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity;
            for (int c = 0; c < cols; c++) max = Math.Max(max, logits[r, c]);
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += Math.Exp(logits[r, c] - max);
            for (int c = 0; c < cols; c++) result[r, c] = Math.Exp(logits[r, c] - max) / sum;
        }

        return result;
    }
}
