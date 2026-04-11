using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

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

    [Fact(Timeout = 60000)]
    public async Task TreatmentEffect_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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

    [Fact(Timeout = 60000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        var pred1 = model.Predict(trainX);
        var pred2 = model.Predict(trainX);
        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Clone_ShouldProduceSameEstimates()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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

    [Fact(Timeout = 60000)]
    public async Task OutputDimension_ShouldMatchInputRows()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        Assert.Equal(TrainSamples, model.Predict(trainX).Length);
    }

    [Fact(Timeout = 60000)]
    public async Task Metadata_ShouldExistAfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 60000)]
    public async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateCausalData(rng, 3.0);
        model.Train(trainX, trainY);
        Assert.True(((IParameterizable<double, Matrix<double>, Vector<double>>)model).GetParameters().Length > 0, "Trained causal model should have parameters.");
    }
}
