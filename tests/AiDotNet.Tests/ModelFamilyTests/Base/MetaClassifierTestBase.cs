using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for meta-classifiers (Voting, Stacking, OneVsRest, etc.).
/// Inherits classification invariants and adds meta-specific: valid labels
/// and output consistency.
/// </summary>
public abstract class MetaClassifierTestBase : ClassificationModelTestBase
{
    [Fact]
    public void MetaPredictions_ShouldBeValidLabels()
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
                $"Meta-classifier prediction[{i}] = {predictions[i]:F2} outside valid range.");
        }
    }

    [Fact]
    public void MetaClassifier_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        model.Train(trainX, trainY);
        var pred1 = model.Predict(trainX);
        var pred2 = model.Predict(trainX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }
}
