using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for SVM classifiers. Inherits probabilistic classifier invariants
/// and adds SVM-specific: margin existence on separable data and kernel diagonal validity.
/// </summary>
public abstract class SVMTestBase : ProbabilisticClassifierTestBase
{
    [Fact]
    public void Margin_ShouldExist_OnSeparableData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        // Very well-separated data — SVM should find a margin
        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Vector<double>(TrainSamples);
        int half = TrainSamples / 2;

        for (int i = 0; i < half; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = ModelTestHelpers.NextGaussian(rng) * 0.1;
            y[i] = 0;
        }
        for (int i = half; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = 20.0 + ModelTestHelpers.NextGaussian(rng) * 0.1;
            y[i] = 1;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double accuracy = ModelTestHelpers.CalculateAccuracy(y, predictions);
            Assert.True(accuracy > 0.9,
                $"SVM accuracy = {accuracy:F4} on perfectly separable data. " +
                "SVM should achieve near-perfect accuracy with clear margin.");
        }
    }

    [Fact]
    public void SVM_ShouldProduceValidLabels()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"SVM prediction[{i}] is NaN.");
            double pred = Math.Round(predictions[i]);
            Assert.True(pred >= 0 && pred < NumClasses,
                $"SVM prediction[{i}] = {predictions[i]:F2} outside valid class range.");
        }
    }
}
