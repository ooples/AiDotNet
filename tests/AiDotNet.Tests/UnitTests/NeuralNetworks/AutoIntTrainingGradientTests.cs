using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Tabular;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class AutoIntTrainingGradientTests
{
    [Fact]
    public void Prediction_PreservesPublicFeatureInteractionWeights()
    {
        var model = new AutoIntRegression<double>(2, options: new AutoIntOptions<double>
        {
            EmbeddingDimension = 4, NumHeads = 2, AttentionDimension = 2,
            NumLayers = 1, MLPHiddenDimensions = new[] { 2 }
        });
        model.Predict(new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.3, 0.7 })));
        var weights = model.GetInteractionWeights();
        Assert.NotNull(weights);
        Assert.Equal(new[] { 1, 2, 2 }, weights.Shape.ToArray());
        Assert.Equal(1.0, weights[0] + weights[1], 10);
        Assert.Equal(1.0, weights[2] + weights[3], 10);
    }

    [Fact]
    public void TrainStep_CategoricalOnlyModelUpdatesParameters()
    {
        var options = new AutoIntOptions<double>
        {
            EmbeddingDimension = 2, NumHeads = 1, AttentionDimension = 2, NumLayers = 1,
            MLPHiddenDimensions = new[] { 2 }, CategoricalCardinalities = new[] { 3 }
        };
        var model = new AutoIntRegression<double>(0, options: options);
        var input = new Tensor<double>(new[] { 2, 0 });
        var categories = new Matrix<int>(2, 1);
        categories[1, 0] = 2;
        var before = model.GetParameters().ToArray();
        double loss = model.TrainStep(input,
            new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new[] { 0.3, 0.8 })), 0.01, categories);
        Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, true, true)]
    public void TrainStep_WholeModelUpdateMatchesFiniteDifferences(bool classification, bool categorical, bool repeatedCategory)
    {
        var options = new AutoIntOptions<double>
        {
            EmbeddingDimension = 2, NumHeads = 2, AttentionDimension = 1, NumLayers = 1,
            MLPHiddenDimensions = new[] { 2 }, DropoutRate = 0,
            CategoricalCardinalities = categorical ? new[] { 3 } : null
        };
        AutoIntBase<double> model = classification
            ? new AutoIntClassifier<double>(2, 2, options)
            : new AutoIntRegression<double>(2, options: options);
        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2 }));
        var targets = new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new[] { 0.3, 0.8 }));
        var labels = new Vector<int>(new[] { 0, 1 });
        Matrix<int>? categories = categorical ? new Matrix<int>(2, 1) : null;
        if (categories != null) categories[1, 0] = repeatedCategory ? 0 : 2;
        // Keep ReLUs away from their kink; use distinct values to avoid symmetric attention.
        var initial = model.GetParameters();
        for (int i = 0; i < initial.Length; i++) initial[i] = 0.3 + 0.04 * (i % 7);
        model.SetParameters(initial);
        double Loss() => model is AutoIntClassifier<double> classifier
            ? classifier.ComputeCrossEntropyLoss(classifier.PredictProbabilities(input, categories), labels)
            : ((AutoIntRegression<double>)model).ComputeMSELoss(((AutoIntRegression<double>)model).Predict(input, categories), targets);
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
        double reported = model is AutoIntClassifier<double> c
            ? c.TrainStep(input, labels, rate, categories)
            : ((AutoIntRegression<double>)model).TrainStep(input, targets, rate, categories);
        Assert.Equal(beforeLoss, reported, 10);
        var updated = model.GetParameters();
        for (int i = 0; i < initial.Length; i++)
        {
            double actual = (initial[i] - updated[i]) / rate;
            Assert.True(Math.Abs(numerical[i] - actual) < 1e-6 + 1e-4 * Math.Abs(numerical[i]),
                $"Parameter {i}: finite difference {numerical[i]:R}, training gradient {actual:R}.");
        }
        Assert.True(Loss() < beforeLoss, "One SGD step should lower this smooth objective.");
        Assert.True(Enumerable.Range(0, 4).Any(i => Math.Abs(initial[i] - updated[i]) > 1e-10),
            "Numerical embeddings must receive gradients through the whole backbone.");
        if (categorical)
            Assert.True(Enumerable.Range(4, 6).Any(i => Math.Abs(initial[i] - updated[i]) > 1e-10),
                "Categorical embeddings must receive gradients through the lookup.");
    }

    [Fact]
    public void InteractingLayer_HeadsAttendIndependently()
    {
        var layer = new InteractingLayer<double>(2, 2, 2, useResidual: false);
        foreach (string name in new[] { "_queryWeights", "_keyWeights", "_valueWeights", "_outputWeights" })
        {
            var weights = (Tensor<double>)typeof(InteractingLayer<double>)
                .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(layer)!;
            for (int i = 0; i < 4; i++) weights[i] = i is 0 or 3 ? 1 : 0;
        }
        var output = layer.Forward(new Tensor<double>(new[] { 1, 2, 2 },
            new Vector<double>(new[] { 1.0, 3.0, 2.0, 1.0 })));
        double Expected(double query, double a, double b)
        {
            double p = 1 / (1 + Math.Exp(query * (b - a)));
            return p * a + (1 - p) * b;
        }
        Assert.Equal(Expected(1, 1, 2), output[0], 10);
        Assert.Equal(Expected(3, 3, 1), output[1], 10);
        Assert.Equal(Expected(2, 1, 2), output[2], 10);
        Assert.Equal(Expected(1, 3, 1), output[3], 10);
    }
}
