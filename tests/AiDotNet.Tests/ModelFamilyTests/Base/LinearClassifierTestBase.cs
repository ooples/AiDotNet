using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for linear classifiers (Perceptron, SGD, Ridge, etc.).
/// Inherits probabilistic classifier invariants and adds linear-specific:
/// linearly separable data should achieve high accuracy.
/// </summary>
public abstract class LinearClassifierTestBase : ProbabilisticClassifierTestBase
{
    [Fact]
    public void LinearSeparable_HighAccuracy()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        // Very well-separated linear data
        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Vector<double>(TrainSamples);
        int half = TrainSamples / 2;
        for (int i = 0; i < half; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = -5.0 + ModelTestHelpers.NextGaussian(rng) * 0.1;
            y[i] = 0;
        }
        for (int i = half; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = 5.0 + ModelTestHelpers.NextGaussian(rng) * 0.1;
            y[i] = 1;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double accuracy = ModelTestHelpers.CalculateAccuracy(y, predictions);
            Assert.True(accuracy > 0.8,
                $"Linear classifier accuracy = {accuracy:F4} on linearly separable data — should be high.");
        }
    }

    [Fact]
    public void Predictions_ShouldBeValidLabels()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = Math.Round(predictions[i]);
            Assert.True(pred >= 0 && pred < NumClasses,
                $"Linear classifier prediction[{i}] = {predictions[i]:F2} outside valid range.");
        }
    }
}
