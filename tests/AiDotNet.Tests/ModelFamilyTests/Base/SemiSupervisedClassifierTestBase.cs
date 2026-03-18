using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for semi-supervised classifiers (self-training, label propagation).
/// Inherits classification invariants and adds semi-supervised-specific: valid labels
/// and finite predictions.
/// </summary>
public abstract class SemiSupervisedClassifierTestBase : ClassificationModelTestBase
{
    [Fact]
    public void SemiSupervisedPredictions_ShouldBeValid()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Semi-supervised prediction[{i}] is NaN.");
            double pred = Math.Round(predictions[i]);
            Assert.True(pred >= 0 && pred < NumClasses,
                $"Semi-supervised prediction[{i}] = {predictions[i]:F2} outside valid range.");
        }
    }

    [Fact]
    public void SemiSupervised_OutputDimMatchesInput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        model.Train(trainX, trainY);
        Assert.Equal(TrainSamples, model.Predict(trainX).Length);
    }
}
