using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for time series / forecasting models.
/// Tests mathematical invariants: trend recovery, translation equivariance,
/// training-vs-test error, residual analysis, and extrapolation consistency.
/// </summary>
public abstract class TimeSeriesModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainLength => 100;
    protected virtual int TestLength => 20;

    // =====================================================
    // MATHEMATICAL INVARIANT: Trend Direction Recovery
    // Data has a positive linear trend y = 0.5t + seasonal + noise.
    // The model should predict higher values for later time points.
    // =====================================================

    [Fact]
    public void TrendRecovery_LaterTimeShouldHaveHigherPrediction()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng, noise: 0.1);

        model.Train(trainX, trainY);

        // Compare predictions at early vs late time points within training range
        var earlyX = new Matrix<double>(1, 1);
        earlyX[0, 0] = 10.0;
        var lateX = new Matrix<double>(1, 1);
        lateX[0, 0] = 80.0;

        var earlyPred = model.Predict(earlyX);
        var latePred = model.Predict(lateX);

        if (ModelTestHelpers.AllFinite(earlyPred) && ModelTestHelpers.AllFinite(latePred))
        {
            // y(80) - y(10) ≈ 0.5*(80-10) = 35 — prediction at t=80 should be substantially higher
            Assert.True(latePred[0] > earlyPred[0],
                $"Trend recovery failed: pred(t=80)={latePred[0]:F4} should be > pred(t=10)={earlyPred[0]:F4}. " +
                "Model failed to capture positive linear trend.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Translation Equivariance of Targets
    // Shifting all y-values by constant C should shift predictions by C.
    // =====================================================

    [Fact]
    public void TranslationEquivariance_ShiftingTargets_ShiftsPredictions()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng1, noise: 0.01);
        var (trainX2, trainY2) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng2, noise: 0.01);

        const double shift = 500.0;
        var shiftedY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            shiftedY[i] = trainY2[i] + shift;

        model1.Train(trainX1, trainY1);
        model2.Train(trainX2, shiftedY);

        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 50.0;
        var pred1 = model1.Predict(testX);
        var pred2 = model2.Predict(testX);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            double actualShift = pred2[0] - pred1[0];
            Assert.True(Math.Abs(actualShift - shift) < shift * 0.3,
                $"Translation equivariance violated: actual shift = {actualShift:F2}, expected ~{shift}.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: R² > 0 on Known Data
    // On data with clear trend + seasonality, model should outperform mean.
    // =====================================================

    [Fact]
    public void R2_ShouldBePositive_OnTrendData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng, noise: 0.5);

        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double r2 = ModelTestHelpers.CalculateR2(trainY, predictions);
            Assert.True(r2 > 0.0,
                $"R² = {r2:F4} on time series with clear trend — model is worse than mean baseline.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Error ≤ Test Error
    // =====================================================

    [Fact]
    public void TrainingError_ShouldNotExceedTestError()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng, noise: 0.5);

        model.Train(trainX, trainY);
        var trainPred = model.Predict(trainX);

        // Use adjacent time points as "test" (in-distribution)
        var (testX, testY) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, ModelTestHelpers.CreateSeededRandom(99), noise: 0.5);
        var testPred = model.Predict(testX);

        if (ModelTestHelpers.AllFinite(trainPred) && ModelTestHelpers.AllFinite(testPred))
        {
            double trainMSE = ModelTestHelpers.CalculateMSE(trainY, trainPred);
            double testMSE = ModelTestHelpers.CalculateMSE(testY, testPred);

            Assert.True(trainMSE <= testMSE * 3.0 + 1.0,
                $"Training MSE ({trainMSE:F4}) is much higher than test MSE ({testMSE:F4}). " +
                "Model is not fitting training data properly.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Residual Mean ≈ 0
    // =====================================================

    [Fact]
    public void ResidualMean_ShouldBeNearZero()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng, noise: 0.5);

        model.Train(trainX, trainY);
        var predictions = model.Predict(trainX);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double residualSum = 0;
            for (int i = 0; i < trainY.Length; i++)
                residualSum += trainY[i] - predictions[i];
            double meanResidual = residualSum / trainY.Length;

            double targetRange = ModelTestHelpers.ComputeRange(trainY);
            Assert.True(Math.Abs(meanResidual) < targetRange * 0.15,
                $"Mean residual = {meanResidual:F4} is large relative to target range {targetRange:F4}.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaling Equivariance
    // =====================================================

    [Fact]
    public void ScalingEquivariance_ScalingTargets_ScalesPredictions()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng1, noise: 0.01);
        var (trainX2, trainY2) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng2, noise: 0.01);

        const double scale = 50.0;
        var scaledY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            scaledY[i] = trainY2[i] * scale;

        model1.Train(trainX1, trainY1);
        model2.Train(trainX2, scaledY);

        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 50.0;
        var pred1 = model1.Predict(testX);
        var pred2 = model2.Predict(testX);

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2) && Math.Abs(pred1[0]) > 0.01)
        {
            double ratio = pred2[0] / pred1[0];
            Assert.True(ratio > scale * 0.3 && ratio < scale * 3.0,
                $"Scaling equivariance violated: ratio = {ratio:F2}, expected ~{scale}.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Finite Predictions, Determinism, Output Shape, Clone, Metadata
    // =====================================================

    [Fact]
    public void Predictions_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        var predictions = model.Predict(testX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Prediction[{i}] is NaN.");
            Assert.False(double.IsInfinity(predictions[i]), $"Prediction[{i}] is Infinity.");
        }
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        var pred1 = model.Predict(testX);
        var pred2 = model.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact]
    public void OutputDimension_ShouldMatchInputRows()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        Assert.Equal(TestLength, model.Predict(testX).Length);
    }

    [Fact]
    public void Clone_ShouldProduceIdenticalPredictions()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);

        model.Train(trainX, trainY);
        var cloned = model.Clone();
        var pred1 = model.Predict(testX);
        var pred2 = cloned.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact]
    public void Metadata_ShouldExistAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        Assert.True(model.GetParameters().Length > 0, "Trained model should have parameters.");
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline
    // =====================================================

    [Fact]
    public void Builder_ShouldProduceResult()
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
    public void Builder_R2ShouldBePositive()
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

        var predictions = result.Predict(trainX);
        double r2 = ModelTestHelpers.CalculateR2(trainY, predictions);
        Assert.True(r2 > 0.0, $"Builder pipeline R² = {r2:F4} — should be positive.");
    }
}
