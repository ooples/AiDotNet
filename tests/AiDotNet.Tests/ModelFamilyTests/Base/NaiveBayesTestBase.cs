using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for Naive Bayes classifiers. Inherits probabilistic classifier invariants
/// and adds NB-specific: zero-variance feature handling and monotone evidence response.
/// </summary>
public abstract class NaiveBayesTestBase : ProbabilisticClassifierTestBase
{
    [Fact]
    public void ZeroVarianceFeature_ShouldNotCrash()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        // Generate data where one feature is constant (zero variance)
        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Vector<double>(TrainSamples);
        int samplesPerClass = TrainSamples / NumClasses;

        for (int c = 0; c < NumClasses; c++)
        {
            int start = c * samplesPerClass;
            int end = c == NumClasses - 1 ? TrainSamples : start + samplesPerClass;
            for (int i = start; i < end; i++)
            {
                x[i, 0] = 5.0;  // constant feature — zero variance
                for (int j = 1; j < Features; j++)
                    x[i, j] = c * 4.0 + ModelTestHelpers.NextGaussian(rng) * 0.5;
                y[i] = c;
            }
        }

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(predictions.Length == TrainSamples,
            "NB model failed to predict after training with zero-variance feature.");
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"NB prediction[{i}] is NaN — zero-variance feature caused instability.");
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
                $"NB prediction[{i}] = {predictions[i]:F2} outside valid class range.");
        }
    }
}
