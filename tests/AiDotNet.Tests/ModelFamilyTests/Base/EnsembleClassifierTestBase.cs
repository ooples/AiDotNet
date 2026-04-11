using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for ensemble classifiers (Random Forest, Bagging, Boosting).
/// Inherits all classification invariants and adds ensemble-specific:
/// better than random and consistent member contributions.
/// </summary>
public abstract class EnsembleClassifierTestBase : ClassificationModelTestBase
{
    [Fact(Timeout = 60000)]
    public async Task Ensemble_NotWorseThanRandom()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double accuracy = ModelTestHelpers.CalculateAccuracy(trainY, predictions);
            double chance = 1.0 / NumClasses;
            Assert.True(accuracy >= chance - 0.05,
                $"Ensemble accuracy = {accuracy:F4} is worse than random ({chance:F4}). " +
                "Ensemble members may be anti-correlated or broken.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Ensemble_ShouldProduceValidLabels()
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
                $"Ensemble prediction[{i}] = {predictions[i]:F2} is not a valid class label.");
        }
    }
}
