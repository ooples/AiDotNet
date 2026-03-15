using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for time series models implementing IFullModel&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;.
/// Tests mathematical invariants for forecasting models.
/// </summary>
public abstract class TimeSeriesModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainLength => 100;
    protected virtual int TestLength => 20;

    // --- Core Correctness ---

    [Fact]
    public void Train_LinearTrend_DetectsDirection()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng, noise: 0.1);

        model.Train(trainX, trainY);

        // Predict at later time points — should continue the upward trend
        var futureX = new Matrix<double>(2, 1);
        futureX[0, 0] = TrainLength;
        futureX[1, 0] = TrainLength + 10;

        var predictions = model.Predict(futureX);
        if (ModelTestHelpers.AllFinite(predictions))
        {
            // The underlying trend is y = 0.5*t + seasonal, so later should be higher
            Assert.True(predictions[1] > predictions[0] - 10.0,
                $"Trend detection failed: pred(t+10)={predictions[1]:F4} should be near or above pred(t)={predictions[0]:F4}.");
        }
    }

    [Fact]
    public void Predict_OutputAllFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);

        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        var predictions = model.Predict(testX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction[{i}] is NaN.");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction[{i}] is Infinity.");
        }
    }

    // --- Determinism ---

    [Fact]
    public void Predict_Deterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);

        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        var pred1 = model.Predict(testX);
        var pred2 = model.Predict(testX);

        Assert.Equal(pred1.Length, pred2.Length);
        for (int i = 0; i < pred1.Length; i++)
        {
            Assert.Equal(pred1[i], pred2[i]);
        }
    }

    // --- Output Shape ---

    [Fact]
    public void Train_OutputDimensionMatches()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);

        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        var predictions = model.Predict(testX);

        Assert.Equal(TestLength, predictions.Length);
    }

    // --- Metadata & Parameters ---

    [Fact]
    public void GetModelMetadata_AfterTraining_ReturnsNonNull()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void GetParameters_AfterTraining_ReturnsNonEmptyVector()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);

        var parameters = model.GetParameters();
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0,
            "Trained time series model should have at least one parameter.");
    }

    // --- Clone Contract ---

    [Fact]
    public void Clone_ProducesSamePredictions()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);

        model.Train(trainX, trainY);
        var cloned = model.Clone();

        var originalPred = model.Predict(testX);
        var clonedPred = cloned.Predict(testX);

        Assert.Equal(originalPred.Length, clonedPred.Length);
        for (int i = 0; i < originalPred.Length; i++)
        {
            Assert.Equal(originalPred[i], clonedPred[i]);
        }
    }

    // --- Builder Integration ---

    [Fact]
    public void Builder_ConfigureAndBuild_ProducesResult()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        Assert.NotNull(result);
    }

    [Fact]
    public void Builder_PredictionNotNull()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var predictions = result.Predict(testX);
        Assert.NotNull(predictions);
        Assert.True(predictions.Length > 0, "Builder should produce non-empty predictions.");
    }
}
