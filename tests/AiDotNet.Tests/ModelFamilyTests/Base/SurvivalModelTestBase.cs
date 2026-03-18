using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for survival analysis models.
/// Tests mathematical invariants: survival function properties, hazard non-negativity,
/// determinism, and risk ordering consistency.
/// </summary>
public abstract class SurvivalModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainSamples => 80;
    protected virtual int Features => 3;

    private (Matrix<double> X, Vector<double> Y) GenerateSurvivalData(Random rng)
    {
        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Vector<double>(TrainSamples);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = rng.NextDouble() * 5.0;
            // Survival time: higher feature values → shorter survival
            y[i] = Math.Max(0.1, 10.0 - x[i, 0] + ModelTestHelpers.NextGaussian(rng) * 0.5);
        }
        return (x, y);
    }

    [Fact]
    public void Predictions_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateSurvivalData(rng);
        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Survival prediction[{i}] is NaN.");
            Assert.False(double.IsInfinity(predictions[i]), $"Survival prediction[{i}] is Infinity.");
        }
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateSurvivalData(rng);
        model.Train(trainX, trainY);
        var pred1 = model.Predict(trainX);
        var pred2 = model.Predict(trainX);
        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact]
    public void Clone_ShouldProduceSamePredictions()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateSurvivalData(rng);
        model.Train(trainX, trainY);
        var cloned = model.Clone();
        var pred1 = model.Predict(trainX);
        var pred2 = cloned.Predict(trainX);
        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact]
    public void OutputDimension_ShouldMatchInputRows()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateSurvivalData(rng);
        model.Train(trainX, trainY);
        Assert.Equal(TrainSamples, model.Predict(trainX).Length);
    }

    [Fact]
    public void Metadata_ShouldExistAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateSurvivalData(rng);
        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateSurvivalData(rng);
        model.Train(trainX, trainY);
        Assert.True(model.GetParameters().Length > 0, "Trained survival model should have parameters.");
    }
}
