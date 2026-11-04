using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.ReinforcementLearning.Agents.DuelingDQN;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class PredictionCacheBoundaryTests
{
    [Fact]
    public void DuelingPrediction_ReleasesCaches_ThenTrainingUpdatesWeights()
    {
        using var network = new DuelingNetwork<double>(2, 3, new[] { 4 }, new[] { 4 },
            new[] { 4 }, MathHelper.GetNumericOperations<double>());
        var input = new Vector<double>(new[] { 0.3, -0.2 });
        var expected = network.Forward(input);
        Assert.NotNull(ReadField(network, "_lastSharedOutput"));

        var actual = network.Predict(input);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        foreach (var name in new[] { "_lastSharedOutput", "_lastValueOutput", "_lastAdvantageOutput" })
            Assert.Null(ReadField(network, name));
        foreach (var name in new[] { "_lastSharedInputs", "_lastSharedOutputs", "_lastValueInputs",
            "_lastValueOutputs", "_lastAdvantageInputs", "_lastAdvantageOutputs" })
            Assert.Empty((System.Collections.IEnumerable)ReadField(network, name)!);
        Assert.False(InferenceMode.IsActive);

        network.Forward(input);
        var before = network.GetFlattenedParameters().ToArray();
        network.UpdateWeights(new Vector<double>(new[] { 1.0, -0.5, 0.25 }), 0.01);
        Assert.False(before.SequenceEqual(network.GetFlattenedParameters().ToArray()));
    }

    [Fact]
    public void AutoIntRegression_PredictionReleasesEmbeddingCache_ForwardRestoresIt()
    {
        var model = new AutoIntRegression<double>(2, options: Options());
        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.3, -0.2 }));
        var expected = model.Forward(input);
        Assert.NotNull(EmbeddingCache(model));
        Assert.Equal(expected.ToArray(), model.Predict(input).ToArray());
        Assert.Null(EmbeddingCache(model));
        model.Forward(input);
        Assert.NotNull(EmbeddingCache(model));
        Assert.False(InferenceMode.IsActive);
    }

    [Fact]
    public void AutoIntClassifier_TrainingAfterPredictionUpdatesParameters()
    {
        var model = new AutoIntClassifier<double>(2, 2, Options());
        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.3, -0.2 }));
        model.Forward(input);
        Assert.NotNull(EmbeddingCache(model));
        var probabilities = model.PredictProbabilities(input);
        Assert.Equal(1.0, probabilities.ToArray().Sum(), 10);
        Assert.Null(EmbeddingCache(model));
        var before = model.GetParameters().ToArray();
        model.TrainStep(input, new Vector<int>(new[] { 1 }), 0.01);
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
        Assert.Null(EmbeddingCache(model));
        Assert.False(InferenceMode.IsActive);
    }

    [Fact]
    public void DuelingPrediction_OnInvalidInput_RestoresInferenceScope()
    {
        using var network = new DuelingNetwork<double>(2, 3, new[] { 4 }, new[] { 4 },
            new[] { 4 }, MathHelper.GetNumericOperations<double>());
        network.EnsureInitialized();
        Assert.ThrowsAny<Exception>(() => network.Predict(null!));
        Assert.False(InferenceMode.IsActive);
    }

    private static AutoIntOptions<double> Options() => new()
    {
        EmbeddingDimension = 4, NumLayers = 1, NumHeads = 1,
        AttentionDimension = 4, MLPHiddenDimensions = new[] { 4 }, DropoutRate = 0
    };

    private static object? EmbeddingCache(AutoIntBase<double> model) =>
        typeof(AutoIntBase<double>).GetField("_numericalFeaturesCache", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model);

    private static object? ReadField(object instance, string name) =>
        instance.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(instance);

}
