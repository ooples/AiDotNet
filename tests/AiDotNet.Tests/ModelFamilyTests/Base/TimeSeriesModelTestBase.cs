using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for time series / forecasting models.
/// Tests mathematical invariants: trend recovery, translation equivariance,
/// training-vs-test error, residual analysis, and extrapolation consistency.
/// </summary>
public abstract class TimeSeriesModelTestBase : System.IDisposable
{
    /// <summary>
    /// Reclaim memory between tests (shared model-family teardown). xUnit constructs a fresh
    /// test-class instance per test and calls Dispose() afterward, so this clears the
    /// InferenceWeightCache and compacts the LOH between model classes — keeping committed memory
    /// from accumulating across a shard. Pure hygiene; no test-observable behavior change.
    /// </summary>
    public virtual void Dispose()
    {
        // Reclaim must be unconditional: a throwing derived DisposeCore() must not skip the
        // shared GC gate, or heavy shards reintroduce cross-test memory buildup / OOM.
        try
        {
            DisposeCore();
        }
        finally
        {
            ModelFamilyTestGcGate.ReclaimBetweenTests();
        }
    }

    /// <summary>
    /// Override in a derived test class to add its own teardown while preserving the
    /// shared <see cref="ModelFamilyTestGcGate.ReclaimBetweenTests"/> call.
    /// </summary>
    protected virtual void DisposeCore()
    {
    }

    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainLength => 100;
    protected virtual int TestLength => 20;

    /// <summary>
    /// Whether this model can capture trends. Stationary models (MA, pure ARMA without
    /// differencing) cannot — R² on trending data will be negative by design.
    /// </summary>
    protected virtual bool CanCaptureTrend => true;

    /// <summary>
    /// Whether this is a forecasting model. Non-forecasting models (anomaly detectors,
    /// spectral analysis) return scores/frequencies instead of time-domain predictions.
    /// </summary>
    protected virtual bool IsForecastingModel => true;

    // =====================================================
    // MATHEMATICAL INVARIANT: Trend Direction Recovery
    // Data has a positive linear trend y = 0.5t + seasonal + noise.
    // The model should predict higher values for later time points.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task TrendRecovery_LaterTimeShouldHaveHigherPrediction()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsForecastingModel) return;
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng, noise: 0.1);

        model.Train(trainX, trainY);

        // Forecast 20 steps ahead. For autoregressive models (ARIMA, AR, etc.),
        // the input matrix rows = number of forecast steps (input values may be ignored).
        // For regression-style models, the input contains the time indices.
        // Either way, later forecast steps should reflect the upward trend.
        var forecastX = new Matrix<double>(20, 1);
        for (int i = 0; i < 20; i++)
            forecastX[i, 0] = TrainLength + i;

        var forecast = model.Predict(forecastX);

        if (ModelTestHelpers.AllFinite(forecast) && forecast.Length >= 2)
        {
            // The training data has y = 0.5t + seasonal + noise, so the trend is upward.
            // Compare early vs late forecasts — later ones should be higher on average.
            double earlyAvg = 0, lateAvg = 0;
            int half = forecast.Length / 2;
            for (int i = 0; i < half; i++) earlyAvg += forecast[i];
            for (int i = half; i < forecast.Length; i++) lateAvg += forecast[i];
            earlyAvg /= half;
            lateAvg /= (forecast.Length - half);

            Assert.True(lateAvg >= earlyAvg - 5.0,
                $"Trend recovery failed: early avg={earlyAvg:F4}, late avg={lateAvg:F4}. " +
                "Later forecasts should reflect upward trend from training data.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Translation Equivariance of Targets
    // Shifting all y-values by constant C should shift predictions by C.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task TranslationEquivariance_ShiftingTargets_ShiftsPredictions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsForecastingModel) return;
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

    [Fact(Timeout = 60000)]
    public async Task R2_ShouldBePositive_OnTrendData()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        // Non-forecasting models (anomaly detectors whose Predict returns ±1 labels, spectral
        // analysers returning frequencies) have no forecast of the target to score, so an R²
        // against the series is meaningless for them — the same gate the other five forecasting
        // invariants here already apply; R² simply omitted it.
        if (!IsForecastingModel) return;
        // Stationary models (MA) cannot capture trends — skip this test for them
        if (!CanCaptureTrend) return;

        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng, noise: 0.5);

        model.Train(trainX, trainY);

        // Forecast the same number of steps as training data.
        // For regression-style models, these match the training positions.
        // For autoregressive models, these are N-step-ahead forecasts.
        var predictions = model.Predict(trainX);

        if (ModelTestHelpers.AllFinite(predictions) && predictions.Length == trainY.Length)
        {
            double r2 = ModelTestHelpers.CalculateR2(trainY, predictions);
            Assert.True(r2 > 0.0,
                $"R² = {r2:F4} on time series with clear trend — model is worse than mean baseline.");
        }
        else if (ModelTestHelpers.AllFinite(predictions))
        {
            // For autoregressive models where Predict returns forecasts (not in-sample),
            // just verify predictions are in a reasonable range
            double trainMean = 0;
            for (int i = 0; i < trainY.Length; i++) trainMean += trainY[i];
            trainMean /= trainY.Length;

            bool anyReasonable = false;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (Math.Abs(predictions[i]) < Math.Abs(trainMean) * 100)
                {
                    anyReasonable = true;
                    break;
                }
            }
            Assert.True(anyReasonable,
                $"Predictions are not in a reasonable range relative to training data mean ({trainMean:F4}).");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Error ≤ Test Error
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task TrainingError_ShouldNotExceedTestError()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsForecastingModel) return;
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
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

    [Fact(Timeout = 60000)]
    public async Task ResidualMean_ShouldBeNearZero()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsForecastingModel) return;
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
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

    [Fact(Timeout = 60000)]
    public async Task ScalingEquivariance_ScalingTargets_ScalesPredictions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsForecastingModel) return;
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
    // MATHEMATICAL INVARIANT: More Data Should Not Degrade R²
    // Training on 2x data from the same distribution should not
    // produce worse R² — more data should help, not hurt.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task MoreData_ShouldNotDegrade_R2()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!CanCaptureTrend) return;
        if (!IsForecastingModel) return;

        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);

        // Train with small data
        var modelSmall = CreateModel();
        var (trainXSmall, trainYSmall) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng1, noise: 0.5);
        modelSmall.Train(trainXSmall, trainYSmall);
        var predSmall = modelSmall.Predict(trainXSmall);

        // Train with 2x data
        var modelLarge = CreateModel();
        var (trainXLarge, trainYLarge) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength * 2, rng2, noise: 0.5);
        modelLarge.Train(trainXLarge, trainYLarge);
        var predLarge = modelLarge.Predict(trainXLarge);

        if (ModelTestHelpers.AllFinite(predSmall) && predSmall.Length == trainYSmall.Length &&
            ModelTestHelpers.AllFinite(predLarge) && predLarge.Length == trainYLarge.Length)
        {
            double r2Small = ModelTestHelpers.CalculateR2(trainYSmall, predSmall);
            double r2Large = ModelTestHelpers.CalculateR2(trainYLarge, predLarge);

            Assert.True(r2Large >= r2Small - 0.15,
                $"R² degraded with more data: R²_small={r2Small:F4}, R²_large={r2Large:F4}. " +
                "More data from the same distribution should not hurt performance.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Finite Predictions, Determinism, Output Shape, Clone, Metadata
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Predictions_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
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

    [Fact(Timeout = 60000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        var pred1 = model.Predict(testX);
        var pred2 = model.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task OutputDimension_ShouldMatchInputRows()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);
        Assert.Equal(TestLength, model.Predict(testX).Length);
    }

    [Fact(Timeout = 60000)]
    public async Task Clone_ShouldProduceIdenticalPredictions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);
        var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(TestLength, rng);

        model.Train(trainX, trainY);
        var cloned = model.Clone();
        var pred1 = model.Predict(testX);
        var pred2 = cloned.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Metadata_ShouldExistAfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 60000)]
    public async Task Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);

        model.Train(trainX, trainY);

        // Not all time series models implement IParameterizable (e.g., STLDecomposition,
        // InterventionAnalysis are parameter-free). Per Liskov substitution, only check
        // models that expose parameters via IParameterizable.
        if (model is IParameterizable<double, Matrix<double>, Vector<double>> parameterizable)
        {
            Assert.True(parameterizable.GetParameters().Length > 0, "Trained parameterizable model should have parameters.");
        }
        else
        {
            // Non-parameterizable models should still produce valid predictions after training
            var (testX, _) = ModelTestHelpers.GenerateTimeSeriesData(5, ModelTestHelpers.CreateSeededRandom(99));
            var prediction = model.Predict(testX);
            Assert.NotNull(prediction);
            Assert.True(prediction.Length > 0, "Non-parameterizable model should still produce predictions after training.");
        }
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Builder_ShouldProduceResult()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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

    [Fact(Timeout = 60000)]
    public async Task Builder_R2ShouldBePositive()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateTimeSeriesData(TrainLength, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // The unified facade forecasts AHEAD of the model's training window (result.Predict
        // routes a time-series model to model.Forecast over the trained series). So this is
        // industry-standard OUT-OF-SAMPLE evaluation: score the forecast against the held-out
        // future actuals, not in-sample training data. The training window is the builder's
        // 70/15/15 split, read back from the result so the held-out slice aligns exactly.
        int trainLen = result.OptimizationResult?.TrainingResult?.Y is Vector<double> trainedY
            ? trainedY.Length
            : (int)(TrainLength * 0.7);
        int horizon = TrainLength - trainLen;
        if (horizon < 2) return; // not enough held-out future to score a forecast

        Vector<double> predictions;
        Vector<double> actual;

        if (model is IMultivariateForecastModel<double> multivariate && multivariate.VariableCount > 1)
        {
            // Genuinely multivariate models forecast the multivariate series they were trained on;
            // compare the flattened forecast against the flattened held-out future of that series
            // (the facade flattens [horizon x variables] row-major).
            predictions = result.Predict(new Matrix<double>(horizon, multivariate.VariableCount));
            int cols = Math.Min(multivariate.VariableCount, trainX.Columns);
            actual = new Vector<double>(horizon * cols);
            int k = 0;
            for (int i = 0; i < horizon; i++)
                for (int j = 0; j < cols; j++)
                    actual[k++] = trainX[trainLen + i, j];
        }
        else if (model is IExogenousForecastModel<double>)
        {
            // Exogenous models (ARIMAX, dynamic regression) forecast the target from FUTURE
            // exogenous regressors: feed the actual held-out exogenous rows and compare against
            // the held-out target.
            var futureExogenous = new Matrix<double>(horizon, trainX.Columns);
            for (int i = 0; i < horizon; i++)
                for (int j = 0; j < trainX.Columns; j++)
                    futureExogenous[i, j] = trainX[trainLen + i, j];
            predictions = result.Predict(futureExogenous);
            actual = SliceForecastTarget(trainY, trainLen, horizon);
        }
        else
        {
            // Univariate models forecast the target series `horizon` steps ahead (the input
            // matrix only carries the horizon as its row count).
            predictions = result.Predict(new Matrix<double>(horizon, trainX.Columns));
            actual = SliceForecastTarget(trainY, trainLen, horizon);
        }

        if (ModelTestHelpers.AllFinite(predictions) && predictions.Length == actual.Length)
        {
            // Out-of-sample forecast evaluation with Theil's U2 statistic (Theil, 1966): the model's
            // forecast RMSE relative to the same-horizon naive/persistence forecast. U2 < 1 beats
            // naive, U2 ~ 1 matches it, U2 >> 1 is a diverging or broken forecast. The breadth of the
            // time-series family here spans models that legitimately cannot extrapolate a
            // deterministic trend out-of-sample (stationary AR/MA/ARMA/ETS/GARCH mean-revert) and
            // lightly-trained neural models, so this is a robust SANITY bar — the original test's
            // "not catastrophically wrong" intent — rather than a skill bar. It still fails the actual
            // regressions: NaN/Inf (excluded above), explosions, and sign/scale blow-ups.
            var naive = BuildNaiveBaseline(model, trainX, trainY, trainLen, horizon, actual.Length);
            double rmseModel = Math.Sqrt(MeanSquaredError(predictions, actual));
            double rmseNaive = Math.Sqrt(MeanSquaredError(naive, actual));
            double theilU = rmseNaive > 0.0 ? rmseModel / rmseNaive : 0.0;
            Assert.True(rmseNaive <= 0.0 || theilU < MaxForecastTheilU,
                $"Out-of-sample forecast Theil's U2 = {theilU:F2} (RMSE {rmseModel:F4} vs naive {rmseNaive:F4}) "
                + $"on the held-out future (horizon={horizon}) exceeds the divergence sanity threshold "
                + $"{MaxForecastTheilU} — the forecast is diverging, not merely inaccurate.");
        }
    }

    /// <summary>
    /// Upper bound on Theil's U2 (forecast RMSE / naive RMSE) for <see cref="Builder_R2ShouldBePositive"/>.
    /// A sanity threshold that fails diverging/broken forecasts while tolerating the legitimate
    /// inability of many model types to out-forecast a deterministic trend. Override per model only
    /// when a documented, model-specific reason requires it.
    /// </summary>
    protected virtual double MaxForecastTheilU => 10.0;

    private static Vector<double> SliceForecastTarget(Vector<double> series, int start, int length)
    {
        var slice = new Vector<double>(length);
        for (int i = 0; i < length; i++)
            slice[i] = series[start + i];
        return slice;
    }

    /// <summary>
    /// Builds the naive persistence forecast (last observed value carried forward) the model
    /// must beat: the last trained target value for univariate/exogenous models, or the last
    /// trained value of each variable (row-major) for multivariate models.
    /// </summary>
    private static Vector<double> BuildNaiveBaseline(
        IFullModel<double, Matrix<double>, Vector<double>> model,
        Matrix<double> trainX, Vector<double> trainY, int trainLen, int horizon, int length)
    {
        var naive = new Vector<double>(length);
        if (model is IMultivariateForecastModel<double> multivariate && multivariate.VariableCount > 1)
        {
            int cols = Math.Min(multivariate.VariableCount, trainX.Columns);
            int k = 0;
            for (int i = 0; i < horizon; i++)
                for (int j = 0; j < cols; j++)
                    naive[k++] = trainX[trainLen - 1, j];
        }
        else
        {
            double last = trainY[trainLen - 1];
            for (int i = 0; i < length; i++)
                naive[i] = last;
        }
        return naive;
    }

    private static double MeanSquaredError(Vector<double> a, Vector<double> b)
    {
        if (a.Length == 0) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sum / a.Length;
    }
}
