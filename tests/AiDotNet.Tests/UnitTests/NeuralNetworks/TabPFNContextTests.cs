using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tabular;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// TabPFN's in-context conditioning (Hollmann et al. 2023): context tokens carry their labels, and the
/// attention mask lets every token see only the context.
/// </summary>
public class TabPFNContextTests
{
    private static TabPFNOptions<double> SmallOptions() => new TabPFNOptions<double>
    {
        EmbeddingDimension = 4,
        NumHeads = 1,
        NumLayers = 1,
        FeedForwardMultiplier = 2,
        OutputHeadDimensions = new[] { 4 },
        DropoutRate = 0,
        UsePositionalEncoding = false
    };

    private static Tensor<double> ContextFeatures() =>
        new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2, 0.6, 0.9 }));

    [Fact]
    public void ContextLabels_ChangeThePrediction()
    {
        var model = new TabPFNClassifier<double>(2, 2, SmallOptions());
        var query = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.5, 0.5, 1.0, 0.3 }));

        model.SetContext(ContextFeatures(), new Vector<int>(new[] { 0, 1, 1 }));
        var withLabels = model.PredictProbabilities(query).ToArray();
        model.SetContext(ContextFeatures(), new Vector<int>(new[] { 1, 0, 0 }));
        var withFlippedLabels = model.PredictProbabilities(query).ToArray();

        Assert.False(withLabels.SequenceEqual(withFlippedLabels),
            "The label encoding of the context tokens must reach the query predictions.");
    }

    [Fact]
    public void Queries_DoNotAttendToEachOther()
    {
        // Under the TabPFN mask a query reads only the context, so its prediction cannot depend on which other
        // queries share the batch.
        var model = new TabPFNRegression<double>(2, 1, SmallOptions());
        model.SetContext(ContextFeatures(), new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 })));

        var alone = model.Predict(new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.5, 0.5 })));
        var together = model.Predict(new Tensor<double>(new[] { 3, 2 },
            new Vector<double>(new[] { 0.5, 0.5, 3.0, -1.0, -2.0, 4.0 })));

        Assert.Equal(alone[0], together[0], 12);
    }

    [Fact]
    public void SetContext_RejectsLabelsOfTheWrongWidth()
    {
        var model = new TabPFNRegression<double>(2, 1, SmallOptions());
        Assert.Throws<ArgumentException>(() =>
            model.SetContext(ContextFeatures(), new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.1, 0.5 }))));
    }

    [Fact]
    public void TrainStep_MovesTheLabelEncoderAndStaysFinite()
    {
        var model = new TabPFNClassifier<double>(2, 2, SmallOptions());
        model.SetContext(ContextFeatures(), new Vector<int>(new[] { 0, 1, 1 }));
        var query = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.5, 0.5, 1.0, 0.3 }));
        var before = model.GetParameters().ToArray();

        double loss = model.TrainStep(query, new Vector<int>(new[] { 0, 1 }), 0.05);

        var after = model.GetParameters().ToArray();
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
        Assert.All(after, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
        Assert.False(before.SequenceEqual(after), "A tape training step must move the weights.");
    }
}
