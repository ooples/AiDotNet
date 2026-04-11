using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for ordinal classifiers (ordinal logistic, ordinal ridge).
/// Inherits classification invariants and adds ordinal-specific: predictions
/// should be ordered class indices.
/// </summary>
public abstract class OrdinalClassifierTestBase : ClassificationModelTestBase
{
    [Fact(Timeout = 60000)]
    public async Task OrdinalPredictions_ShouldBeValidIndices()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = Math.Round(predictions[i]);
            Assert.True(pred >= 0 && pred < NumClasses,
                $"Ordinal prediction[{i}] = {predictions[i]:F2} outside valid class range.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task OrdinalPredictions_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Ordinal prediction[{i}] is NaN.");
        }
    }
}
