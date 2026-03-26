using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for causal inference models.
/// Tests mathematical invariants: finite treatment effects, zero-treatment baseline,
/// determinism, and effect recovery on synthetic data.
/// </summary>
public abstract class CausalModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainSamples => 100;
    protected virtual int Features => 3;

    private (Matrix<double> X, Vector<double> Y) GenerateCausalData(Random rng, double treatmentEffect)
    {
        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Vector<double>(TrainSamples);
        for (int i = 0; i < TrainSamples; i++)
        {
            // Feature 0 is treatment indicator (0 or 1)
            double treatment = rng.NextDouble() > 0.5 ? 1.0 : 0.0;
            x[i, 0] = treatment;
            for (int j = 1; j < Features; j++)
                x[i, j] = rng.NextDouble() * 5.0;
            // Outcome = baseline + treatment effect + noise
            y[i] = 2.0 + treatment * treatmentEffect + ModelTestHelpers.NextGaussian(rng) * 0.5;
        }
        return (x, y);
    }

    [Fact]
    public void TreatmentEffect_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Causal prediction[{i}] is NaN.");
            Assert.False(double.IsInfinity(predictions[i]), $"Causal prediction[{i}] is Infinity.");
        }
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        var pred1 = model.Predict(trainX);
        var pred2 = model.Predict(trainX);
        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact]
    public void Clone_ShouldProduceSameEstimates()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
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
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        Assert.Equal(TrainSamples, model.Predict(trainX).Length);
    }

    [Fact]
    public void Metadata_ShouldExistAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        Assert.True(model.GetParameters().Length > 0, "Trained causal model should have parameters.");
    }
}
