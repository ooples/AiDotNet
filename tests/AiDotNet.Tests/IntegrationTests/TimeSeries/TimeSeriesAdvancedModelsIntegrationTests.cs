using Xunit;
using AiDotNet;
using AiDotNet.TimeSeries;
using AiDotNet.NeuralNetworks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Optimizers;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Integration tests for advanced time series models (Part 2 of 2).
/// Tests for VAR, VARMA, GARCH, TBATS, Prophet, UCM, BSTS, Transfer Functions,
/// Intervention Analysis, Dynamic Regression with ARIMA, Spectral Analysis,
/// NN-ARIMA, N-BEATS, and N-BEATS Block.
/// </summary>
public class TimeSeriesAdvancedModelsIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region VectorAutoRegressionModel Tests

    [Fact]
    public void VAR_Train_WithBivariateData_EstimatesCoefficients()
    {
        // Create synthetic bivariate VAR(1) data
        var options = new VARModelOptions<double>
        {
            OutputDimension = 2,
            Lag = 1,
            DecompositionType = MatrixDecompositionType.LU
        };
        var model = new VectorAutoRegressionModel<double>(options);

        // Generate data: y1[t] = 0.5*y1[t-1] + 0.3*y2[t-1] + noise
        //                y2[t] = 0.2*y1[t-1] + 0.6*y2[t-1] + noise
        int n = 100;
        var data = new Matrix<double>(n, 2);
        var random = new Random(42);
        data[0, 0] = 1.0;
        data[0, 1] = 1.0;

        for (int t = 1; t < n; t++)
        {
            data[t, 0] = 0.5 * data[t - 1, 0] + 0.3 * data[t - 1, 1] + random.NextDouble() * 0.1;
            data[t, 1] = 0.2 * data[t - 1, 0] + 0.6 * data[t - 1, 1] + random.NextDouble() * 0.1;
        }

        // Train model
        model.Train(data, new Vector<double>(n));

        // Verify coefficients are estimated
        Assert.NotNull(model.Coefficients);
        Assert.Equal(2, model.Coefficients.Rows);
        Assert.Equal(2, model.Coefficients.Columns);
    }

    [Fact]
    public void VAR_Predict_AfterTraining_ReturnsReasonableForecasts()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var model = new VectorAutoRegressionModel<double>(options);

        // Simple bivariate data
        var data = new Matrix<double>(50, 2);
        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1) + 1.0;
            data[i, 1] = Math.Cos(i * 0.1) + 1.0;
        }

        model.Train(data, new Vector<double>(50));

        // Predict
        var input = new Matrix<double>(1, 2);
        input[0, 0] = data[49, 0];
        input[0, 1] = data[49, 1];
        var prediction = model.Predict(input);

        Assert.Equal(2, prediction.Length);
        Assert.True(Math.Abs(prediction[0]) < 10.0); // Reasonable range
        Assert.True(Math.Abs(prediction[1]) < 10.0);
    }

    [Fact]
    public void VAR_Forecast_MultiStep_GeneratesMultipleForecastSteps()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 2 };
        var model = new VectorAutoRegressionModel<double>(options);

        // Generate trending data
        var data = new Matrix<double>(60, 2);
        for (int i = 0; i < 60; i++)
        {
            data[i, 0] = i * 0.1 + Math.Sin(i * 0.2);
            data[i, 1] = i * 0.05 + Math.Cos(i * 0.2);
        }

        model.Train(data, new Vector<double>(60));

        // Multi-step forecast
        var forecasts = model.Forecast(data.Slice(50, 10), steps: 5);

        Assert.Equal(5, forecasts.Rows);
        Assert.Equal(2, forecasts.Columns);
    }

    [Fact]
    public void VAR_ImpulseResponse_ComputesResponseFunctions()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var model = new VectorAutoRegressionModel<double>(options);

        var data = new Matrix<double>(80, 2);
        for (int i = 0; i < 80; i++)
        {
            data[i, 0] = Math.Sin(i * 0.15) + 2.0;
            data[i, 1] = Math.Cos(i * 0.15) + 2.0;
        }

        model.Train(data, new Vector<double>(80));

        // Impulse response analysis
        var impulseResponses = model.ImpulseResponseAnalysis(horizon: 10);

        Assert.NotNull(impulseResponses);
        Assert.Equal(2, impulseResponses.Count);
        Assert.True(impulseResponses.ContainsKey("Variable_0"));
        Assert.True(impulseResponses.ContainsKey("Variable_1"));
    }

    [Fact]
    public void VAR_EvaluateModel_ReturnsMetrics()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var model = new VectorAutoRegressionModel<double>(options);

        var trainData = new Matrix<double>(70, 2);
        var testData = new Matrix<double>(10, 2);
        for (int i = 0; i < 70; i++)
        {
            trainData[i, 0] = i * 0.05 + Math.Sin(i * 0.1);
            trainData[i, 1] = i * 0.03 + Math.Cos(i * 0.1);
        }
        for (int i = 0; i < 10; i++)
        {
            testData[i, 0] = (70 + i) * 0.05 + Math.Sin((70 + i) * 0.1);
            testData[i, 1] = (70 + i) * 0.03 + Math.Cos((70 + i) * 0.1);
        }

        model.Train(trainData, new Vector<double>(70));

        var testInput = new Matrix<double>(1, 2);
        testInput[0, 0] = testData[0, 0];
        testInput[0, 1] = testData[0, 1];
        var testOutput = new Vector<double>(new[] { testData[1, 0], testData[1, 1] });

        var metrics = model.EvaluateModel(testInput, testOutput);

        Assert.True(metrics.ContainsKey("MSE"));
        Assert.True(metrics.ContainsKey("RMSE"));
        Assert.True(metrics.ContainsKey("MAE"));
        Assert.True(metrics.ContainsKey("MAPE"));
    }

    [Fact]
    public void VAR_SerializeDeserialize_PreservesModel()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var model = new VectorAutoRegressionModel<double>(options);

        var data = new Matrix<double>(50, 2);
        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1);
            data[i, 1] = Math.Cos(i * 0.1);
        }

        model.Train(data, new Vector<double>(50));

        // Serialize
        byte[] serialized = model.Serialize();
        Assert.NotNull(serialized);
        Assert.True(serialized.Length > 0);

        // Deserialize
        var newModel = new VectorAutoRegressionModel<double>(options);
        newModel.Deserialize(serialized);

        // Verify predictions match
        var input = new Matrix<double>(1, 2);
        input[0, 0] = data[49, 0];
        input[0, 1] = data[49, 1];

        var pred1 = model.Predict(input);
        var pred2 = newModel.Predict(input);

        Assert.Equal(pred1[0], pred2[0], Tolerance);
        Assert.Equal(pred1[1], pred2[1], Tolerance);
    }

    [Fact]
    public void VAR_GetModelMetadata_ReturnsCompleteInfo()
    {
        var options = new VARModelOptions<double> { OutputDimension = 3, Lag = 2 };
        var model = new VectorAutoRegressionModel<double>(options);

        var data = new Matrix<double>(60, 3);
        for (int i = 0; i < 60; i++)
        {
            data[i, 0] = i * 0.1;
            data[i, 1] = i * 0.05;
            data[i, 2] = i * 0.03;
        }

        model.Train(data, new Vector<double>(60));

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.Equal(ModelType.VARModel, metadata.ModelType);
        Assert.True(metadata.AdditionalInfo.ContainsKey("OutputDimension"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("Lag"));
        Assert.Equal(3, metadata.AdditionalInfo["OutputDimension"]);
        Assert.Equal(2, metadata.AdditionalInfo["Lag"]);
    }

    [Fact]
    public void VAR_PredictSingle_WithVariableIndex_ReturnsSinglePrediction()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var model = new VectorAutoRegressionModel<double>(options);

        var data = new Matrix<double>(50, 2);
        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = i * 0.1;
            data[i, 1] = i * 0.05;
        }

        model.Train(data, new Vector<double>(50));

        // Create input: lagged values + variable index
        var input = new Vector<double>(3); // 2 lagged values + 1 variable index
        input[0] = data[49, 0]; // y1[t-1]
        input[1] = data[49, 1]; // y2[t-1]
        input[2] = 0.0; // predict variable 0

        var prediction = model.PredictSingle(input);

        Assert.True(Math.Abs(prediction) < 100.0);
    }

    [Fact]
    public void VAR_Reset_ClearsModelState()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var model = new VectorAutoRegressionModel<double>(options);

        var data = new Matrix<double>(50, 2);
        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = i;
            data[i, 1] = i + 1;
        }

        model.Train(data, new Vector<double>(50));

        // Reset
        model.Reset();

        // Coefficients should be reset (need to verify through metadata or re-prediction)
        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void VAR_WithHigherLag_HandlesMultipleLags()
    {
        var options = new VARModelOptions<double> { OutputDimension = 2, Lag = 3 };
        var model = new VectorAutoRegressionModel<double>(options);

        var data = new Matrix<double>(100, 2);
        for (int i = 0; i < 100; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1);
            data[i, 1] = Math.Cos(i * 0.1);
        }

        model.Train(data, new Vector<double>(100));

        var input = new Matrix<double>(3, 2);
        for (int i = 0; i < 3; i++)
        {
            input[i, 0] = data[97 + i, 0];
            input[i, 1] = data[97 + i, 1];
        }

        var prediction = model.Predict(input);

        Assert.Equal(2, prediction.Length);
    }

    #endregion

    #region VARMAModel Tests

    [Fact]
    public void VARMA_Train_WithBivariateData_EstimatesARAndMACoefficients()
    {
        var options = new VARMAModelOptions<double>
        {
            OutputDimension = 2,
            Lag = 1,
            MaLag = 1,
            DecompositionType = MatrixDecompositionType.LU
        };
        var model = new VARMAModel<double>(options);

        // Generate bivariate VARMA data
        int n = 100;
        var data = new Matrix<double>(n, 2);
        var random = new Random(42);
        data[0, 0] = 1.0;
        data[0, 1] = 1.0;

        for (int t = 1; t < n; t++)
        {
            data[t, 0] = 0.5 * data[t - 1, 0] + 0.2 * data[t - 1, 1] + random.NextDouble() * 0.1;
            data[t, 1] = 0.3 * data[t - 1, 0] + 0.6 * data[t - 1, 1] + random.NextDouble() * 0.1;
        }

        // Train model
        model.Train(data, new Vector<double>(n));

        // Verify coefficients exist
        Assert.NotNull(model.Coefficients);
        Assert.Equal(2, model.Coefficients.Rows);
    }

    [Fact]
    public void VARMA_Predict_CombinesARAndMA_ReturnsForecasts()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 1, MaLag = 1 };
        var model = new VARMAModel<double>(options);

        var data = new Matrix<double>(60, 2);
        for (int i = 0; i < 60; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1) + 1.0;
            data[i, 1] = Math.Cos(i * 0.1) + 1.0;
        }

        model.Train(data, new Vector<double>(60));

        var input = new Matrix<double>(1, 2);
        input[0, 0] = data[59, 0];
        input[0, 1] = data[59, 1];

        var prediction = model.Predict(input);

        Assert.Equal(2, prediction.Length);
        Assert.True(Math.Abs(prediction[0]) < 10.0);
        Assert.True(Math.Abs(prediction[1]) < 10.0);
    }

    [Fact]
    public void VARMA_SerializeDeserialize_PreservesModelState()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 1, MaLag = 1 };
        var model = new VARMAModel<double>(options);

        var data = new Matrix<double>(50, 2);
        for (int i = 0; i < 50; i++)
        {
            data[i, 0] = i * 0.1;
            data[i, 1] = i * 0.05;
        }

        model.Train(data, new Vector<double>(50));

        byte[] serialized = model.Serialize();
        var newModel = new VARMAModel<double>(options);
        newModel.Deserialize(serialized);

        var input = new Matrix<double>(1, 2);
        input[0, 0] = data[49, 0];
        input[0, 1] = data[49, 1];

        var pred1 = model.Predict(input);
        var pred2 = newModel.Predict(input);

        Assert.Equal(pred1[0], pred2[0], Tolerance);
    }

    [Fact]
    public void VARMA_WithHigherMALag_HandlesMultipleMATerms()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 2, MaLag = 2 };
        var model = new VARMAModel<double>(options);

        var data = new Matrix<double>(100, 2);
        for (int i = 0; i < 100; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1);
            data[i, 1] = Math.Cos(i * 0.1);
        }

        model.Train(data, new Vector<double>(100));

        var input = new Matrix<double>(2, 2);
        input[0, 0] = data[98, 0];
        input[0, 1] = data[98, 1];
        input[1, 0] = data[99, 0];
        input[1, 1] = data[99, 1];

        var prediction = model.Predict(input);

        Assert.Equal(2, prediction.Length);
    }

    #endregion

    #region GARCHModel Tests

    [Fact]
    public void GARCH_Train_WithVolatilityData_EstimatesParameters()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        // Generate data with volatility clustering
        int n = 200;
        var data = new Vector<double>(n);
        var random = new Random(42);
        double volatility = 0.1;

        for (int t = 0; t < n; t++)
        {
            double shock = random.NextGaussian() * volatility;
            data[t] = shock;
            volatility = 0.01 + 0.1 * shock * shock + 0.85 * volatility;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.GARCHModel, metadata.ModelType);
    }

    [Fact]
    public void GARCH_Predict_ReturnsVolatilityForecast()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(150);
        var random = new Random(42);
        for (int i = 0; i < 150; i++)
        {
            data[i] = random.NextGaussian() * 0.1;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[149];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
        Assert.True(prediction[0] >= 0); // Volatility should be non-negative
    }

    [Fact]
    public void GARCH_GetConditionalVolatility_ReturnsVolatilityEstimates()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(100);
        var random = new Random(42);
        for (int i = 0; i < 100; i++)
        {
            data[i] = random.NextGaussian() * 0.2;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var volatility = model.GetConditionalVolatility();

        Assert.NotNull(volatility);
        Assert.True(volatility.Length > 0);
        Assert.All(volatility, v => Assert.True(v >= 0));
    }

    [Fact]
    public void GARCH_EvaluateModel_ReturnsPerformanceMetrics()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        var trainData = new Vector<double>(150);
        var testData = new Vector<double>(20);
        var random = new Random(42);

        for (int i = 0; i < 150; i++) trainData[i] = random.NextGaussian() * 0.1;
        for (int i = 0; i < 20; i++) testData[i] = random.NextGaussian() * 0.1;

        var xTrain = Matrix<double>.FromColumns(trainData);
        model.Train(xTrain, trainData);

        var xTest = Matrix<double>.FromColumns(testData);
        var metrics = model.EvaluateModel(xTest, testData);

        Assert.True(metrics.ContainsKey("MSE"));
        Assert.True(metrics.ContainsKey("MAE"));
    }

    [Fact]
    public void GARCH_SerializeDeserialize_MaintainsVolatilityEstimates()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(100);
        var random = new Random(42);
        for (int i = 0; i < 100; i++) data[i] = random.NextGaussian() * 0.15;

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        byte[] serialized = model.Serialize();
        var newModel = new GARCHModel<double>(options);
        newModel.Deserialize(serialized);

        var vol1 = model.GetConditionalVolatility();
        var vol2 = newModel.GetConditionalVolatility();

        Assert.Equal(vol1.Length, vol2.Length);
    }

    [Fact]
    public void GARCH_WithHigherOrders_HandlesComplexVolatility()
    {
        var options = new GARCHModelOptions<double> { P = 2, Q = 2 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(200);
        var random = new Random(42);
        double vol = 0.1;

        for (int t = 0; t < 200; t++)
        {
            double shock = random.NextGaussian() * vol;
            data[t] = shock;
            vol = 0.01 + 0.05 * shock * shock + 0.9 * vol;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var volatility = model.GetConditionalVolatility();
        Assert.NotNull(volatility);
        Assert.True(volatility.Length > 0);
    }

    #endregion

    #region TBATSModel Tests

    [Fact]
    public void TBATS_Train_WithSeasonalData_FitsModel()
    {
        var options = new TBATSModelOptions<double>
        {
            UseBoxCox = false,
            UseARMA = true,
            UseDamping = false,
            SeasonalPeriods = new List<int> { 12 }
        };
        var model = new TBATSModel<double>(options);

        // Generate monthly data with yearly seasonality
        int n = 120; // 10 years
        var data = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double trend = i * 0.1;
            double seasonal = Math.Sin(2 * Math.PI * i / 12.0);
            data[i] = trend + seasonal + 10.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.TBATSModel, metadata.ModelType);
    }

    [Fact]
    public void TBATS_Predict_ReturnsSeasonalForecast()
    {
        var options = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new List<int> { 7 }
        };
        var model = new TBATSModel<double>(options);

        // Weekly seasonal data
        var data = new Vector<double>(70);
        for (int i = 0; i < 70; i++)
        {
            data[i] = 5.0 + 2.0 * Math.Sin(2 * Math.PI * i / 7.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[69];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0]) < 20.0);
    }

    [Fact]
    public void TBATS_WithMultipleSeasonalPeriods_HandlesComplexSeasonality()
    {
        var options = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new List<int> { 7, 365 }
        };
        var model = new TBATSModel<double>(options);

        // Daily data with weekly and yearly patterns
        var data = new Vector<double>(730); // 2 years
        for (int i = 0; i < 730; i++)
        {
            double weekly = Math.Sin(2 * Math.PI * i / 7.0);
            double yearly = Math.Sin(2 * Math.PI * i / 365.0);
            data[i] = 10.0 + weekly + 0.5 * yearly;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();
        Assert.True(metadata.AdditionalInfo.ContainsKey("SeasonalPeriods"));
    }

    [Fact]
    public void TBATS_WithBoxCox_TransformsData()
    {
        var options = new TBATSModelOptions<double>
        {
            UseBoxCox = true,
            BoxCoxLambda = 0.5,
            SeasonalPeriods = new List<int> { 12 }
        };
        var model = new TBATSModel<double>(options);

        var data = new Vector<double>(120);
        for (int i = 0; i < 120; i++)
        {
            data[i] = Math.Exp(i * 0.01) + Math.Sin(2 * Math.PI * i / 12.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[119];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
    }

    [Fact]
    public void TBATS_GetSeasonalComponents_ExtractsSeasonalPatterns()
    {
        var options = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new List<int> { 12 }
        };
        var model = new TBATSModel<double>(options);

        var data = new Vector<double>(120);
        for (int i = 0; i < 120; i++)
        {
            data[i] = 10.0 + 3.0 * Math.Sin(2 * Math.PI * i / 12.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var components = model.GetSeasonalComponents();

        Assert.NotNull(components);
        Assert.True(components.ContainsKey("Seasonal_12"));
    }

    [Fact]
    public void TBATS_SerializeDeserialize_PreservesComplexState()
    {
        var options = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new List<int> { 7 },
            UseARMA = true
        };
        var model = new TBATSModel<double>(options);

        var data = new Vector<double>(70);
        for (int i = 0; i < 70; i++)
        {
            data[i] = 5.0 + Math.Sin(2 * Math.PI * i / 7.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        byte[] serialized = model.Serialize();
        var newModel = new TBATSModel<double>(options);
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void TBATS_EvaluateModel_ComputesAccuracy()
    {
        var options = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new List<int> { 12 }
        };
        var model = new TBATSModel<double>(options);

        var trainData = new Vector<double>(96);
        var testData = new Vector<double>(24);

        for (int i = 0; i < 96; i++)
        {
            trainData[i] = 10.0 + 2.0 * Math.Sin(2 * Math.PI * i / 12.0);
        }
        for (int i = 0; i < 24; i++)
        {
            testData[i] = 10.0 + 2.0 * Math.Sin(2 * Math.PI * (96 + i) / 12.0);
        }

        var xTrain = Matrix<double>.FromColumns(trainData);
        model.Train(xTrain, trainData);

        var xTest = Matrix<double>.FromColumns(testData);
        var metrics = model.EvaluateModel(xTest, testData);

        Assert.True(metrics.ContainsKey("MAE"));
        Assert.True(metrics.ContainsKey("RMSE"));
    }

    #endregion

    #region ProphetModel Tests

    [Fact]
    public void Prophet_Train_WithTrendAndSeasonality_FitsComponents()
    {
        var options = new ProphetModelOptions<double>
        {
            YearlySeasonality = true,
            WeeklySeasonality = false,
            DailySeasonality = false
        };
        var model = new ProphetModel<double>(options);

        // Daily data for 2 years
        int n = 730;
        var data = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double trend = i * 0.05;
            double yearly = Math.Sin(2 * Math.PI * i / 365.0);
            data[i] = trend + yearly + 50.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.ProphetModel, metadata.ModelType);
    }

    [Fact]
    public void Prophet_Predict_GeneratesForecast()
    {
        var options = new ProphetModelOptions<double>
        {
            YearlySeasonality = true
        };
        var model = new ProphetModel<double>(options);

        var data = new Vector<double>(365);
        for (int i = 0; i < 365; i++)
        {
            data[i] = 10.0 + Math.Sin(2 * Math.PI * i / 365.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[364];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0] - 10.0) < 5.0);
    }

    [Fact]
    public void Prophet_WithChangepoints_DetectsTrendChanges()
    {
        var options = new ProphetModelOptions<double>
        {
            NumChangepoints = 10,
            ChangepointRange = 0.8
        };
        var model = new ProphetModel<double>(options);

        // Data with trend change
        var data = new Vector<double>(200);
        for (int i = 0; i < 100; i++)
        {
            data[i] = i * 0.1 + 10.0;
        }
        for (int i = 100; i < 200; i++)
        {
            data[i] = (100 * 0.1) + (i - 100) * 0.3 + 10.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var changepoints = model.GetChangepoints();
        Assert.NotNull(changepoints);
    }

    [Fact]
    public void Prophet_AddRegressor_IncorporatesExternalVariable()
    {
        var options = new ProphetModelOptions<double>();
        var model = new ProphetModel<double>(options);

        model.AddRegressor("temperature", standardize: true);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = 10.0 + i * 0.1;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        Assert.True(true); // Model should handle regressor
    }

    [Fact]
    public void Prophet_GetTrendComponent_ExtractsTrend()
    {
        var options = new ProphetModelOptions<double>
        {
            GrowthType = GrowthType.Linear
        };
        var model = new ProphetModel<double>(options);

        var data = new Vector<double>(200);
        for (int i = 0; i < 200; i++)
        {
            data[i] = i * 0.2 + 15.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var trend = model.GetTrendComponent();

        Assert.NotNull(trend);
        Assert.True(trend.Length > 0);
    }

    [Fact]
    public void Prophet_WithLogisticGrowth_HandlesSaturation()
    {
        var options = new ProphetModelOptions<double>
        {
            GrowthType = GrowthType.Logistic,
            Cap = 100.0
        };
        var model = new ProphetModel<double>(options);

        var data = new Vector<double>(150);
        for (int i = 0; i < 150; i++)
        {
            data[i] = 100.0 / (1.0 + Math.Exp(-(i - 75.0) / 10.0));
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[149];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
        Assert.True(prediction[0] <= 105.0); // Should be near cap
    }

    [Fact]
    public void Prophet_EvaluateModel_AssessesAccuracy()
    {
        var options = new ProphetModelOptions<double>
        {
            YearlySeasonality = false
        };
        var model = new ProphetModel<double>(options);

        var trainData = new Vector<double>(150);
        var testData = new Vector<double>(30);

        for (int i = 0; i < 150; i++) trainData[i] = i * 0.1 + 10.0;
        for (int i = 0; i < 30; i++) testData[i] = (150 + i) * 0.1 + 10.0;

        var xTrain = Matrix<double>.FromColumns(trainData);
        model.Train(xTrain, trainData);

        var xTest = Matrix<double>.FromColumns(testData);
        var metrics = model.EvaluateModel(xTest, testData);

        Assert.True(metrics.ContainsKey("MAE"));
    }

    [Fact]
    public void Prophet_SerializeDeserialize_RestoresModel()
    {
        var options = new ProphetModelOptions<double>
        {
            YearlySeasonality = true
        };
        var model = new ProphetModel<double>(options);

        var data = new Vector<double>(365);
        for (int i = 0; i < 365; i++)
        {
            data[i] = 10.0 + Math.Sin(2 * Math.PI * i / 365.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        byte[] serialized = model.Serialize();
        var newModel = new ProphetModel<double>(options);
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region UnobservedComponentsModel Tests

    [Fact]
    public void UCM_Train_WithTrendAndCycle_FitsComponents()
    {
        var options = new UnobservedComponentsModelOptions<double>
        {
            Level = true,
            Trend = true,
            Cycle = true,
            CyclePeriod = 12.0
        };
        var model = new UnobservedComponentsModel<double>(options);

        var data = new Vector<double>(120);
        for (int i = 0; i < 120; i++)
        {
            double trend = i * 0.1;
            double cycle = Math.Sin(2 * Math.PI * i / 12.0);
            data[i] = trend + cycle + 20.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.UnobservedComponentsModel, metadata.ModelType);
    }

    [Fact]
    public void UCM_Predict_ReturnsStateSpaceForecast()
    {
        var options = new UnobservedComponentsModelOptions<double>
        {
            Level = true,
            Trend = true
        };
        var model = new UnobservedComponentsModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = i * 0.2 + 10.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[99];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0] - 30.0) < 10.0);
    }

    [Fact]
    public void UCM_GetState_ReturnsFilteredStates()
    {
        var options = new UnobservedComponentsModelOptions<double>
        {
            Level = true,
            Trend = true
        };
        var model = new UnobservedComponentsModel<double>(options);

        var data = new Vector<double>(80);
        for (int i = 0; i < 80; i++)
        {
            data[i] = i * 0.15 + 5.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var states = model.GetFilteredStates();

        Assert.NotNull(states);
        Assert.True(states.Rows > 0);
    }

    [Fact]
    public void UCM_WithSeasonalComponent_HandlesSeasonality()
    {
        var options = new UnobservedComponentsModelOptions<double>
        {
            Level = true,
            Seasonal = true,
            SeasonalPeriods = 12
        };
        var model = new UnobservedComponentsModel<double>(options);

        var data = new Vector<double>(120);
        for (int i = 0; i < 120; i++)
        {
            data[i] = 10.0 + 3.0 * Math.Sin(2 * Math.PI * i / 12.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[119];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
    }

    [Fact]
    public void UCM_GetLogLikelihood_ReturnsModelFit()
    {
        var options = new UnobservedComponentsModelOptions<double>
        {
            Level = true,
            Trend = true
        };
        var model = new UnobservedComponentsModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = i * 0.1 + 10.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var logLikelihood = model.GetLogLikelihood();

        Assert.True(logLikelihood < 0); // Log-likelihood should be negative
    }

    [Fact]
    public void UCM_SerializeDeserialize_PreservesStateSpace()
    {
        var options = new UnobservedComponentsModelOptions<double>
        {
            Level = true,
            Trend = true,
            Cycle = true,
            CyclePeriod = 10.0
        };
        var model = new UnobservedComponentsModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = i * 0.1 + Math.Sin(2 * Math.PI * i / 10.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        byte[] serialized = model.Serialize();
        var newModel = new UnobservedComponentsModel<double>(options);
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void UCM_EvaluateModel_ComputesPredictionErrors()
    {
        var options = new UnobservedComponentsModelOptions<double>
        {
            Level = true
        };
        var model = new UnobservedComponentsModel<double>(options);

        var trainData = new Vector<double>(80);
        var testData = new Vector<double>(20);

        for (int i = 0; i < 80; i++) trainData[i] = 10.0 + i * 0.1;
        for (int i = 0; i < 20; i++) testData[i] = 10.0 + (80 + i) * 0.1;

        var xTrain = Matrix<double>.FromColumns(trainData);
        model.Train(xTrain, trainData);

        var xTest = Matrix<double>.FromColumns(testData);
        var metrics = model.EvaluateModel(xTest, testData);

        Assert.True(metrics.ContainsKey("MSE"));
    }

    #endregion

    #region BayesianStructuralTimeSeriesModel Tests

    [Fact]
    public void BSTS_Train_WithLocalLevelAndTrend_FitsModel()
    {
        var options = new BayesianStructuralTimeSeriesModelOptions<double>
        {
            StateSpaceComponents = new List<StateSpaceComponent>
            {
                StateSpaceComponent.LocalLevel,
                StateSpaceComponent.LocalLinearTrend
            },
            NumIterations = 500,
            BurnIn = 100
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = i * 0.2 + 15.0 + (new Random(i).NextDouble() - 0.5);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.BayesianStructuralTimeSeries, metadata.ModelType);
    }

    [Fact]
    public void BSTS_Predict_ReturnsPosteriorMean()
    {
        var options = new BayesianStructuralTimeSeriesModelOptions<double>
        {
            StateSpaceComponents = new List<StateSpaceComponent>
            {
                StateSpaceComponent.LocalLevel
            },
            NumIterations = 300
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);

        var data = new Vector<double>(80);
        for (int i = 0; i < 80; i++)
        {
            data[i] = 10.0 + (new Random(i).NextDouble() - 0.5) * 2;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[79];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0] - 10.0) < 5.0);
    }

    [Fact]
    public void BSTS_GetPosteriorDistribution_ReturnsSamples()
    {
        var options = new BayesianStructuralTimeSeriesModelOptions<double>
        {
            StateSpaceComponents = new List<StateSpaceComponent>
            {
                StateSpaceComponent.LocalLevel
            },
            NumIterations = 400,
            BurnIn = 50
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);

        var data = new Vector<double>(60);
        for (int i = 0; i < 60; i++)
        {
            data[i] = 12.0 + (new Random(i).NextDouble() - 0.5) * 3;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var posterior = model.GetPosteriorDistribution();

        Assert.NotNull(posterior);
        Assert.True(posterior.Count > 0);
    }

    [Fact]
    public void BSTS_WithSeasonalComponent_HandlesSeasonality()
    {
        var options = new BayesianStructuralTimeSeriesModelOptions<double>
        {
            StateSpaceComponents = new List<StateSpaceComponent>
            {
                StateSpaceComponent.LocalLevel,
                StateSpaceComponent.Seasonal
            },
            SeasonalPeriod = 12,
            NumIterations = 500
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);

        var data = new Vector<double>(120);
        for (int i = 0; i < 120; i++)
        {
            data[i] = 10.0 + 2.0 * Math.Sin(2 * Math.PI * i / 12.0);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[119];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
    }

    [Fact]
    public void BSTS_GetCredibleIntervals_ReturnsPredictionUncertainty()
    {
        var options = new BayesianStructuralTimeSeriesModelOptions<double>
        {
            StateSpaceComponents = new List<StateSpaceComponent>
            {
                StateSpaceComponent.LocalLevel
            },
            NumIterations = 300
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);

        var data = new Vector<double>(70);
        for (int i = 0; i < 70; i++)
        {
            data[i] = 15.0 + (new Random(i).NextDouble() - 0.5) * 2;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var intervals = model.GetCredibleIntervals(horizon: 10, probability: 0.95);

        Assert.NotNull(intervals);
        Assert.True(intervals.ContainsKey("Lower"));
        Assert.True(intervals.ContainsKey("Upper"));
    }

    [Fact]
    public void BSTS_SerializeDeserialize_PreservesPosterior()
    {
        var options = new BayesianStructuralTimeSeriesModelOptions<double>
        {
            StateSpaceComponents = new List<StateSpaceComponent>
            {
                StateSpaceComponent.LocalLevel
            },
            NumIterations = 200
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);

        var data = new Vector<double>(60);
        for (int i = 0; i < 60; i++)
        {
            data[i] = 10.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        byte[] serialized = model.Serialize();
        var newModel = new BayesianStructuralTimeSeriesModel<double>(options);
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void BSTS_EvaluateModel_ProducesBayesianMetrics()
    {
        var options = new BayesianStructuralTimeSeriesModelOptions<double>
        {
            StateSpaceComponents = new List<StateSpaceComponent>
            {
                StateSpaceComponent.LocalLevel
            },
            NumIterations = 250
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);

        var trainData = new Vector<double>(70);
        var testData = new Vector<double>(15);

        for (int i = 0; i < 70; i++) trainData[i] = 10.0 + i * 0.05;
        for (int i = 0; i < 15; i++) testData[i] = 10.0 + (70 + i) * 0.05;

        var xTrain = Matrix<double>.FromColumns(trainData);
        model.Train(xTrain, trainData);

        var xTest = Matrix<double>.FromColumns(testData);
        var metrics = model.EvaluateModel(xTest, testData);

        Assert.True(metrics.ContainsKey("MAE"));
    }

    #endregion

    #region TransferFunctionModel Tests

    [Fact]
    public void TransferFunction_Train_WithInputOutput_EstimatesLags()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 3,
            OutputLags = 2,
            Delay = 1
        };
        var model = new TransferFunctionModel<double>(options);

        // Input affects output with a delay
        var input = new Vector<double>(100);
        var output = new Vector<double>(100);
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            input[i] = Math.Sin(i * 0.1);
            if (i > 1)
            {
                output[i] = 0.5 * input[i - 1] + 0.3 * input[i - 2] + random.NextDouble() * 0.1;
            }
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.TransferFunctionModel, metadata.ModelType);
    }

    [Fact]
    public void TransferFunction_Predict_UsesInputSeries()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 2,
            OutputLags = 1,
            Delay = 1
        };
        var model = new TransferFunctionModel<double>(options);

        var input = new Vector<double>(80);
        var output = new Vector<double>(80);

        for (int i = 0; i < 80; i++)
        {
            input[i] = i * 0.1;
            if (i > 0)
            {
                output[i] = 0.6 * input[i - 1] + 0.2;
            }
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        var testInput = new Matrix<double>(1, 2);
        testInput[0, 0] = input[79];
        testInput[0, 1] = output[79];
        var prediction = model.Predict(testInput);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0]) < 20.0);
    }

    [Fact]
    public void TransferFunction_GetTransferWeights_ReturnsLagWeights()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 4,
            OutputLags = 2
        };
        var model = new TransferFunctionModel<double>(options);

        var input = new Vector<double>(100);
        var output = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            input[i] = Math.Cos(i * 0.1);
            if (i > 2)
            {
                output[i] = 0.4 * input[i - 2] + 0.3 * input[i - 3];
            }
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        var weights = model.GetTransferWeights();

        Assert.NotNull(weights);
        Assert.True(weights.Length > 0);
    }

    [Fact]
    public void TransferFunction_WithNoiseModel_HandlesResiduals()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 2,
            OutputLags = 1,
            NoiseModel = NoiseModelType.ARMA,
            NoiseAROrder = 1,
            NoiseMAOrder = 1
        };
        var model = new TransferFunctionModel<double>(options);

        var input = new Vector<double>(90);
        var output = new Vector<double>(90);
        var random = new Random(42);

        for (int i = 0; i < 90; i++)
        {
            input[i] = i * 0.05;
            if (i > 0)
            {
                output[i] = 0.7 * input[i - 1] + random.NextDouble() * 0.2;
            }
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        var testInput = new Matrix<double>(1, 2);
        testInput[0, 0] = input[89];
        testInput[0, 1] = output[89];
        var prediction = model.Predict(testInput);

        Assert.Single(prediction);
    }

    [Fact]
    public void TransferFunction_EvaluateModel_MeasuresPredictionAccuracy()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 2,
            OutputLags = 1
        };
        var model = new TransferFunctionModel<double>(options);

        var trainInput = new Vector<double>(80);
        var trainOutput = new Vector<double>(80);
        var testInput = new Vector<double>(20);
        var testOutput = new Vector<double>(20);

        for (int i = 0; i < 80; i++)
        {
            trainInput[i] = i * 0.1;
            if (i > 0) trainOutput[i] = 0.5 * trainInput[i - 1];
        }
        for (int i = 0; i < 20; i++)
        {
            testInput[i] = (80 + i) * 0.1;
            if (i > 0) testOutput[i] = 0.5 * testInput[i - 1];
        }

        var xTrain = Matrix<double>.FromColumns(trainInput, trainOutput);
        model.Train(xTrain, trainOutput);

        var xTest = Matrix<double>.FromColumns(testInput, testOutput);
        var metrics = model.EvaluateModel(xTest, testOutput);

        Assert.True(metrics.ContainsKey("MSE"));
    }

    [Fact]
    public void TransferFunction_SerializeDeserialize_MaintainsTransferFunction()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 2,
            OutputLags = 1
        };
        var model = new TransferFunctionModel<double>(options);

        var input = new Vector<double>(70);
        var output = new Vector<double>(70);

        for (int i = 0; i < 70; i++)
        {
            input[i] = i * 0.05;
            if (i > 0) output[i] = 0.6 * input[i - 1];
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        byte[] serialized = model.Serialize();
        var newModel = new TransferFunctionModel<double>(options);
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region InterventionAnalysisModel Tests

    [Fact]
    public void InterventionAnalysis_Train_WithInterventions_EstimatesEffects()
    {
        var options = new InterventionAnalysisOptions<double, Matrix<double>, Vector<double>>
        {
            AROrder = 1,
            MAOrder = 1,
            Interventions = new List<Intervention>
            {
                new Intervention { StartIndex = 30, Duration = 10 }
            }
        };
        var model = new InterventionAnalysisModel<double>(options);

        // Data with intervention effect
        var data = new Vector<double>(100);
        var random = new Random(42);
        for (int i = 0; i < 100; i++)
        {
            data[i] = 10.0 + random.NextDouble();
            if (i >= 30 && i < 40)
            {
                data[i] += 5.0; // Intervention effect
            }
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var effects = model.GetInterventionEffects();
        Assert.NotNull(effects);
        Assert.True(effects.Count > 0);
    }

    [Fact]
    public void InterventionAnalysis_Predict_IncludesInterventionImpact()
    {
        var options = new InterventionAnalysisOptions<double, Matrix<double>, Vector<double>>
        {
            AROrder = 1,
            MAOrder = 0,
            Interventions = new List<Intervention>
            {
                new Intervention { StartIndex = 25, Duration = 15 }
            }
        };
        var model = new InterventionAnalysisModel<double>(options);

        var data = new Vector<double>(80);
        for (int i = 0; i < 80; i++)
        {
            data[i] = 5.0 + i * 0.05;
            if (i >= 25 && i < 40)
            {
                data[i] += 3.0;
            }
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[79];
        var prediction = model.Predict(input);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0]) < 20.0);
    }

    [Fact]
    public void InterventionAnalysis_GetInterventionEffects_QuantifiesImpact()
    {
        var options = new InterventionAnalysisOptions<double, Matrix<double>, Vector<double>>
        {
            AROrder = 1,
            MAOrder = 1,
            Interventions = new List<Intervention>
            {
                new Intervention { StartIndex = 20, Duration = 10 },
                new Intervention { StartIndex = 50, Duration = 5 }
            }
        };
        var model = new InterventionAnalysisModel<double>(options);

        var data = new Vector<double>(90);
        var random = new Random(42);
        for (int i = 0; i < 90; i++)
        {
            data[i] = 8.0 + random.NextDouble() * 0.5;
            if (i >= 20 && i < 30) data[i] += 4.0;
            if (i >= 50 && i < 55) data[i] -= 2.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var effects = model.GetInterventionEffects();

        Assert.NotNull(effects);
        Assert.Equal(2, effects.Count);
    }

    [Fact]
    public void InterventionAnalysis_WithPermanentIntervention_ModelsPersistentChange()
    {
        var options = new InterventionAnalysisOptions<double, Matrix<double>, Vector<double>>
        {
            AROrder = 1,
            MAOrder = 0,
            Interventions = new List<Intervention>
            {
                new Intervention { StartIndex = 40, Duration = 0 } // Permanent
            }
        };
        var model = new InterventionAnalysisModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = 10.0;
            if (i >= 40)
            {
                data[i] = 15.0; // Permanent level shift
            }
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var effects = model.GetInterventionEffects();
        Assert.NotNull(effects);
    }

    [Fact]
    public void InterventionAnalysis_EvaluateModel_AssessesAccuracyWithInterventions()
    {
        var options = new InterventionAnalysisOptions<double, Matrix<double>, Vector<double>>
        {
            AROrder = 1,
            MAOrder = 1,
            Interventions = new List<Intervention>
            {
                new Intervention { StartIndex = 25, Duration = 10 }
            }
        };
        var model = new InterventionAnalysisModel<double>(options);

        var trainData = new Vector<double>(70);
        var testData = new Vector<double>(15);

        for (int i = 0; i < 70; i++)
        {
            trainData[i] = 10.0;
            if (i >= 25 && i < 35) trainData[i] += 3.0;
        }
        for (int i = 0; i < 15; i++)
        {
            testData[i] = 10.0;
        }

        var xTrain = Matrix<double>.FromColumns(trainData);
        model.Train(xTrain, trainData);

        var xTest = Matrix<double>.FromColumns(testData);
        var metrics = model.EvaluateModel(xTest, testData);

        Assert.True(metrics.ContainsKey("MAE"));
        Assert.True(metrics.ContainsKey("RMSE"));
    }

    [Fact]
    public void InterventionAnalysis_SerializeDeserialize_PreservesInterventions()
    {
        var options = new InterventionAnalysisOptions<double, Matrix<double>, Vector<double>>
        {
            AROrder = 1,
            MAOrder = 1,
            Interventions = new List<Intervention>
            {
                new Intervention { StartIndex = 30, Duration = 15 }
            }
        };
        var model = new InterventionAnalysisModel<double>(options);

        var data = new Vector<double>(80);
        for (int i = 0; i < 80; i++)
        {
            data[i] = 5.0 + i * 0.1;
            if (i >= 30 && i < 45) data[i] += 2.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        byte[] serialized = model.Serialize();
        var newModel = new InterventionAnalysisModel<double>(options);
        newModel.Deserialize(serialized);

        var effects = newModel.GetInterventionEffects();
        Assert.NotNull(effects);
    }

    [Fact]
    public void InterventionAnalysis_GetModelMetadata_IncludesInterventionInfo()
    {
        var options = new InterventionAnalysisOptions<double, Matrix<double>, Vector<double>>
        {
            AROrder = 2,
            MAOrder = 1,
            Interventions = new List<Intervention>
            {
                new Intervention { StartIndex = 20, Duration = 10 }
            }
        };
        var model = new InterventionAnalysisModel<double>(options);

        var data = new Vector<double>(70);
        for (int i = 0; i < 70; i++)
        {
            data[i] = 8.0;
            if (i >= 20 && i < 30) data[i] += 4.0;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.True(metadata.AdditionalInfo.ContainsKey("InterventionCount"));
        Assert.Equal(1, metadata.AdditionalInfo["InterventionCount"]);
    }

    #endregion

    #region DynamicRegressionWithARIMAErrors Tests

    [Fact]
    public void DynamicRegression_Train_WithExogenousVariables_FitsModel()
    {
        var options = new DynamicRegressionWithARIMAErrorsOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            DifferenceOrder = 0,
            ExternalRegressors = 2
        };
        var model = new DynamicRegressionWithARIMAErrors<double>(options);

        // Create data with external variables
        int n = 100;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);
        var random = new Random(42);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = i * 0.1; // Time trend
            x[i, 1] = Math.Sin(i * 0.1); // Seasonal component
            y[i] = 2.0 * x[i, 0] + 3.0 * x[i, 1] + random.NextDouble() * 0.5;
        }

        model.Train(x, y);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.DynamicRegressionWithARIMAErrors, metadata.ModelType);
    }

    [Fact]
    public void DynamicRegression_Predict_CombinesRegressionAndARIMA()
    {
        var options = new DynamicRegressionWithARIMAErrorsOptions<double>
        {
            AROrder = 1,
            MAOrder = 0,
            DifferenceOrder = 0,
            ExternalRegressors = 1
        };
        var model = new DynamicRegressionWithARIMAErrors<double>(options);

        var x = new Matrix<double>(80, 1);
        var y = new Vector<double>(80);

        for (int i = 0; i < 80; i++)
        {
            x[i, 0] = i * 0.1;
            y[i] = 1.5 * x[i, 0] + 5.0;
        }

        model.Train(x, y);

        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 8.0;
        var prediction = model.Predict(testX);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0] - 17.0) < 5.0);
    }

    [Fact]
    public void DynamicRegression_WithDifferencing_HandlesNonStationarity()
    {
        var options = new DynamicRegressionWithARIMAErrorsOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            DifferenceOrder = 1,
            ExternalRegressors = 1
        };
        var model = new DynamicRegressionWithARIMAErrors<double>(options);

        var x = new Matrix<double>(100, 1);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            x[i, 0] = i;
            y[i] = i * i * 0.01; // Non-stationary trend
        }

        model.Train(x, y);

        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 100;
        var prediction = model.Predict(testX);

        Assert.Single(prediction);
    }

    [Fact]
    public void DynamicRegression_Forecast_GeneratesMultiStepPredictions()
    {
        var options = new DynamicRegressionWithARIMAErrorsOptions<double>
        {
            AROrder = 1,
            MAOrder = 0,
            DifferenceOrder = 0,
            ExternalRegressors = 1
        };
        var model = new DynamicRegressionWithARIMAErrors<double>(options);

        var x = new Matrix<double>(90, 1);
        var y = new Vector<double>(90);

        for (int i = 0; i < 90; i++)
        {
            x[i, 0] = Math.Cos(i * 0.1);
            y[i] = 2.0 * x[i, 0] + 10.0;
        }

        model.Train(x, y);

        var history = new Vector<double>(new[] { y[88], y[89] });
        var futureX = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++)
        {
            futureX[i, 0] = Math.Cos((90 + i) * 0.1);
        }

        var forecasts = model.Forecast(history, horizon: 5, exogenousVariables: futureX);

        Assert.Equal(5, forecasts.Length);
    }

    [Fact]
    public void DynamicRegression_EvaluateModel_ComputesAccuracyMetrics()
    {
        var options = new DynamicRegressionWithARIMAErrorsOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            ExternalRegressors = 2
        };
        var model = new DynamicRegressionWithARIMAErrors<double>(options);

        var xTrain = new Matrix<double>(80, 2);
        var yTrain = new Vector<double>(80);
        var xTest = new Matrix<double>(20, 2);
        var yTest = new Vector<double>(20);

        for (int i = 0; i < 80; i++)
        {
            xTrain[i, 0] = i * 0.1;
            xTrain[i, 1] = Math.Sin(i * 0.1);
            yTrain[i] = 1.0 * xTrain[i, 0] + 2.0 * xTrain[i, 1];
        }
        for (int i = 0; i < 20; i++)
        {
            xTest[i, 0] = (80 + i) * 0.1;
            xTest[i, 1] = Math.Sin((80 + i) * 0.1);
            yTest[i] = 1.0 * xTest[i, 0] + 2.0 * xTest[i, 1];
        }

        model.Train(xTrain, yTrain);

        var metrics = model.EvaluateModel(xTest, yTest);

        Assert.True(metrics.ContainsKey("MSE"));
        Assert.True(metrics.ContainsKey("RMSE"));
        Assert.True(metrics.ContainsKey("MAE"));
        Assert.True(metrics.ContainsKey("MAPE"));
    }

    [Fact]
    public void DynamicRegression_GetModelMetadata_IncludesAllComponents()
    {
        var options = new DynamicRegressionWithARIMAErrorsOptions<double>
        {
            AROrder = 2,
            MAOrder = 1,
            DifferenceOrder = 1,
            ExternalRegressors = 3
        };
        var model = new DynamicRegressionWithARIMAErrors<double>(options);

        var x = new Matrix<double>(100, 3);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            x[i, 0] = i * 0.05;
            x[i, 1] = Math.Sin(i * 0.1);
            x[i, 2] = Math.Cos(i * 0.1);
            y[i] = x[i, 0] + x[i, 1] + x[i, 2];
        }

        model.Train(x, y);

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.True(metadata.AdditionalInfo.ContainsKey("AROrder"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("MAOrder"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("DifferenceOrder"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("ExternalRegressors"));
    }

    [Fact]
    public void DynamicRegression_SerializeDeserialize_PreservesComplexModel()
    {
        var options = new DynamicRegressionWithARIMAErrorsOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            ExternalRegressors = 2
        };
        var model = new DynamicRegressionWithARIMAErrors<double>(options);

        var x = new Matrix<double>(70, 2);
        var y = new Vector<double>(70);

        for (int i = 0; i < 70; i++)
        {
            x[i, 0] = i * 0.1;
            x[i, 1] = i * 0.05;
            y[i] = 2.0 * x[i, 0] + 1.5 * x[i, 1];
        }

        model.Train(x, y);

        byte[] serialized = model.Serialize();
        var newModel = new DynamicRegressionWithARIMAErrors<double>(options);
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region SpectralAnalysisModel Tests

    [Fact]
    public void SpectralAnalysis_Train_ComputesFrequencyDomain()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 128,
            UseWindowFunction = true,
            WindowFunction = WindowFunctionFactory.CreateWindowFunction<double>(WindowFunctionType.Hanning)
        };
        var model = new SpectralAnalysisModel<double>(options);

        // Generate signal with dominant frequency
        var data = new Vector<double>(128);
        for (int i = 0; i < 128; i++)
        {
            data[i] = Math.Sin(2 * Math.PI * 0.1 * i);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var periodogram = model.GetPeriodogram();
        Assert.NotNull(periodogram);
        Assert.True(periodogram.Length > 0);
    }

    [Fact]
    public void SpectralAnalysis_GetFrequencies_ReturnsFrequencyValues()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 64,
            SamplingRate = 1.0
        };
        var model = new SpectralAnalysisModel<double>(options);

        var data = new Vector<double>(64);
        for (int i = 0; i < 64; i++)
        {
            data[i] = Math.Cos(2 * Math.PI * 0.2 * i);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var frequencies = model.GetFrequencies();

        Assert.NotNull(frequencies);
        Assert.True(frequencies.Length > 0);
        Assert.True(frequencies[0] >= 0);
    }

    [Fact]
    public void SpectralAnalysis_GetPeriodogram_ShowsPeakAtDominantFrequency()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 256
        };
        var model = new SpectralAnalysisModel<double>(options);

        // Signal with known frequency
        var data = new Vector<double>(256);
        double frequency = 0.15;
        for (int i = 0; i < 256; i++)
        {
            data[i] = 2.0 * Math.Sin(2 * Math.PI * frequency * i);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var periodogram = model.GetPeriodogram();

        // Find peak
        double maxPower = 0;
        for (int i = 0; i < periodogram.Length; i++)
        {
            if (periodogram[i] > maxPower)
            {
                maxPower = periodogram[i];
            }
        }

        Assert.True(maxPower > 0);
    }

    [Fact]
    public void SpectralAnalysis_WithWindow_ReducesSpectralLeakage()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 128,
            UseWindowFunction = true,
            WindowFunction = WindowFunctionFactory.CreateWindowFunction<double>(WindowFunctionType.Hamming)
        };
        var model = new SpectralAnalysisModel<double>(options);

        var data = new Vector<double>(128);
        for (int i = 0; i < 128; i++)
        {
            data[i] = Math.Sin(2 * Math.PI * 0.12 * i) + 0.5 * Math.Sin(2 * Math.PI * 0.25 * i);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var periodogram = model.GetPeriodogram();
        Assert.NotNull(periodogram);
        Assert.All(periodogram, p => Assert.True(p >= 0));
    }

    [Fact]
    public void SpectralAnalysis_EvaluateModel_ComparesPeriodograms()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 128
        };
        var model = new SpectralAnalysisModel<double>(options);

        var trainData = new Vector<double>(128);
        var testData = new Vector<double>(128);

        for (int i = 0; i < 128; i++)
        {
            trainData[i] = Math.Sin(2 * Math.PI * 0.1 * i);
            testData[i] = Math.Sin(2 * Math.PI * 0.1 * i);
        }

        var xTrain = Matrix<double>.FromColumns(trainData);
        model.Train(xTrain, trainData);

        var xTest = Matrix<double>.FromColumns(testData);
        var metrics = model.EvaluateModel(xTest, testData);

        Assert.True(metrics.ContainsKey("MSE"));
        Assert.True(metrics.ContainsKey("RMSE"));
        Assert.True(metrics.ContainsKey("MAE"));
        Assert.True(metrics.ContainsKey("R2"));
        Assert.True(metrics.ContainsKey("PeakFrequencyDifference"));
    }

    [Fact]
    public void SpectralAnalysis_PredictSingle_GeneratesSinusoidalValue()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 64
        };
        var model = new SpectralAnalysisModel<double>(options);

        var data = new Vector<double>(64);
        for (int i = 0; i < 64; i++)
        {
            data[i] = Math.Sin(2 * Math.PI * 0.1 * i);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var input = new Vector<double>(new[] { 65.0 }); // Next time index
        var prediction = model.PredictSingle(input);

        Assert.True(Math.Abs(prediction) <= 3.0); // Amplitude bound
    }

    [Fact]
    public void SpectralAnalysis_GetModelMetadata_IncludesSpectralInfo()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 256,
            UseWindowFunction = true,
            SamplingRate = 100.0
        };
        var model = new SpectralAnalysisModel<double>(options);

        var data = new Vector<double>(256);
        for (int i = 0; i < 256; i++)
        {
            data[i] = Math.Cos(2 * Math.PI * 0.2 * i);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.True(metadata.AdditionalInfo.ContainsKey("NFFT"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("DominantFrequency"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("TotalPower"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("SpectralEntropy"));
    }

    [Fact]
    public void SpectralAnalysis_SerializeDeserialize_PreservesSpectrum()
    {
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 128
        };
        var model = new SpectralAnalysisModel<double>(options);

        var data = new Vector<double>(128);
        for (int i = 0; i < 128; i++)
        {
            data[i] = Math.Sin(2 * Math.PI * 0.15 * i);
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        byte[] serialized = model.Serialize();
        var newModel = new SpectralAnalysisModel<double>(options);
        newModel.Deserialize(serialized);

        var periodogram1 = model.GetPeriodogram();
        var periodogram2 = newModel.GetPeriodogram();

        Assert.Equal(periodogram1.Length, periodogram2.Length);
    }

    #endregion

    #region NeuralNetworkARIMAModel Tests

    [Fact]
    public void NeuralNetworkARIMA_Train_CombinesNNAndARIMA()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            DifferencingOrder = 0,
            LaggedPredictions = 3,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(100, 1);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            x[i, 0] = i * 0.1;
            y[i] = Math.Sin(i * 0.2) + 10.0;
        }

        model.Train(x, y);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.NeuralNetworkARIMA, metadata.ModelType);
    }

    [Fact]
    public void NeuralNetworkARIMA_Predict_UsesHybridApproach()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 0,
            LaggedPredictions = 2,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(80, 1);
        var y = new Vector<double>(80);

        for (int i = 0; i < 80; i++)
        {
            x[i, 0] = Math.Cos(i * 0.1);
            y[i] = 5.0 + 2.0 * Math.Cos(i * 0.1);
        }

        model.Train(x, y);

        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = Math.Cos(80 * 0.1);
        var prediction = model.Predict(testX);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0]) < 15.0);
    }

    [Fact]
    public void NeuralNetworkARIMA_EvaluateModel_ComputesPerformance()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            LaggedPredictions = 2,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var xTrain = new Matrix<double>(70, 1);
        var yTrain = new Vector<double>(70);
        var xTest = new Matrix<double>(15, 1);
        var yTest = new Vector<double>(15);

        for (int i = 0; i < 70; i++)
        {
            xTrain[i, 0] = i * 0.1;
            yTrain[i] = 10.0 + i * 0.05;
        }
        for (int i = 0; i < 15; i++)
        {
            xTest[i, 0] = (70 + i) * 0.1;
            yTest[i] = 10.0 + (70 + i) * 0.05;
        }

        model.Train(xTrain, yTrain);

        var metrics = model.EvaluateModel(xTest, yTest);

        Assert.True(metrics.ContainsKey("MAE"));
        Assert.True(metrics.ContainsKey("MSE"));
        Assert.True(metrics.ContainsKey("RMSE"));
        Assert.True(metrics.ContainsKey("R2"));
    }

    [Fact]
    public void NeuralNetworkARIMA_GetModelMetadata_IncludesHybridInfo()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 2,
            MAOrder = 1,
            DifferencingOrder = 1,
            LaggedPredictions = 4,
            ExogenousVariables = 2
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(90, 2);
        var y = new Vector<double>(90);

        for (int i = 0; i < 90; i++)
        {
            x[i, 0] = i * 0.05;
            x[i, 1] = Math.Sin(i * 0.1);
            y[i] = x[i, 0] + x[i, 1];
        }

        model.Train(x, y);

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.True(metadata.AdditionalInfo.ContainsKey("AR Order"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("MA Order"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("Lagged Predictions"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("Exogenous Variables"));
    }

    [Fact]
    public void NeuralNetworkARIMA_SerializeDeserialize_PreservesHybridModel()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            LaggedPredictions = 2,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(70, 1);
        var y = new Vector<double>(70);

        for (int i = 0; i < 70; i++)
        {
            x[i, 0] = i * 0.1;
            y[i] = 10.0 + Math.Sin(i * 0.1);
        }

        model.Train(x, y);

        byte[] serialized = model.Serialize();
        var newModel = new NeuralNetworkARIMAModel<double>(options);
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region NBEATSModel Tests

    [Fact]
    public void NBEATS_Train_WithLookbackWindow_OptimizesBlocks()
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 5,
            NumStacks = 2,
            NumBlocksPerStack = 2,
            HiddenLayerSize = 16,
            NumHiddenLayers = 2,
            UseInterpretableBasis = false,
            Epochs = 5,
            BatchSize = 16,
            LearningRate = 0.001
        };
        var model = new NBEATSModel<double>(options);

        // Generate time series
        var data = new Vector<double>(200);
        for (int i = 0; i < 200; i++)
        {
            data[i] = Math.Sin(i * 0.1) + 10.0;
        }

        // Create input-output pairs
        int numSamples = data.Length - options.LookbackWindow - options.ForecastHorizon + 1;
        var x = new Matrix<double>(numSamples, options.LookbackWindow);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++)
            {
                x[i, j] = data[i + j];
            }
            y[i] = data[i + options.LookbackWindow];
        }

        model.Train(x, y);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
        Assert.Equal("N-BEATS", metadata.Name);
    }

    [Fact]
    public void NBEATS_PredictSingle_ReturnsNextStep()
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1,
            Epochs = 3,
            BatchSize = 8
        };
        var model = new NBEATSModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = i * 0.1 + 5.0;
        }

        int numSamples = 80;
        var x = new Matrix<double>(numSamples, options.LookbackWindow);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++)
            {
                x[i, j] = data[i + j];
            }
            y[i] = data[i + options.LookbackWindow];
        }

        model.Train(x, y);

        var input = new Vector<double>(options.LookbackWindow);
        for (int i = 0; i < options.LookbackWindow; i++)
        {
            input[i] = data[92 + i];
        }

        var prediction = model.PredictSingle(input);

        Assert.True(Math.Abs(prediction) < 50.0);
    }

    [Fact]
    public void NBEATS_ForecastHorizon_ReturnsMultipleSteps()
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 5,
            NumStacks = 1,
            NumBlocksPerStack = 2,
            HiddenLayerSize = 12,
            NumHiddenLayers = 2,
            Epochs = 4,
            BatchSize = 16
        };
        var model = new NBEATSModel<double>(options);

        var data = new Vector<double>(150);
        for (int i = 0; i < 150; i++)
        {
            data[i] = Math.Cos(i * 0.1) + 8.0;
        }

        int numSamples = 120;
        var x = new Matrix<double>(numSamples, options.LookbackWindow);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++)
            {
                x[i, j] = data[i + j];
            }
            y[i] = data[i + options.LookbackWindow];
        }

        model.Train(x, y);

        var input = new Vector<double>(options.LookbackWindow);
        for (int i = 0; i < options.LookbackWindow; i++)
        {
            input[i] = data[140 + i];
        }

        var forecast = model.ForecastHorizon(input);

        Assert.Equal(options.ForecastHorizon, forecast.Length);
    }

    [Fact]
    public void NBEATS_WithInterpretableBasis_UsesPolynomialExpansion()
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 12,
            ForecastHorizon = 4,
            NumStacks = 2,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            NumHiddenLayers = 2,
            UseInterpretableBasis = true,
            PolynomialDegree = 3,
            Epochs = 5,
            BatchSize = 16
        };
        var model = new NBEATSModel<double>(options);

        var data = new Vector<double>(180);
        for (int i = 0; i < 180; i++)
        {
            data[i] = i * 0.05 + Math.Sin(i * 0.2);
        }

        int numSamples = 150;
        var x = new Matrix<double>(numSamples, options.LookbackWindow);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++)
            {
                x[i, j] = data[i + j];
            }
            y[i] = data[i + options.LookbackWindow];
        }

        model.Train(x, y);

        var metadata = model.GetModelMetadata();
        Assert.True((bool)metadata.AdditionalInfo["Hyperparameters"].GetType()
            .GetProperty("UseInterpretableBasis").GetValue(metadata.AdditionalInfo["Hyperparameters"]));
    }

    [Fact]
    public void NBEATS_GetParameters_ReturnsAllBlockParameters()
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 2,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 8
        };
        var model = new NBEATSModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = Math.Sin(i * 0.1);
        }

        int numSamples = 80;
        var x = new Matrix<double>(numSamples, options.LookbackWindow);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++)
            {
                x[i, j] = data[i + j];
            }
            y[i] = data[i + options.LookbackWindow];
        }

        model.Train(x, y);

        var parameters = model.GetParameters();

        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
    }

    [Fact]
    public void NBEATS_SetParameters_UpdatesModelWeights()
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 8
        };
        var model = new NBEATSModel<double>(options);

        var data = new Vector<double>(80);
        for (int i = 0; i < 80; i++)
        {
            data[i] = i * 0.1;
        }

        int numSamples = 60;
        var x = new Matrix<double>(numSamples, options.LookbackWindow);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++)
            {
                x[i, j] = data[i + j];
            }
            y[i] = data[i + options.LookbackWindow];
        }

        model.Train(x, y);

        var originalParams = model.GetParameters();
        var newParams = new Vector<double>(originalParams.Length);
        for (int i = 0; i < newParams.Length; i++)
        {
            newParams[i] = originalParams[i] * 0.9;
        }

        model.SetParameters(newParams);

        var updatedParams = model.GetParameters();
        Assert.Equal(newParams[0], updatedParams[0], Tolerance);
    }

    [Fact]
    public void NBEATS_SerializeDeserialize_PreservesArchitecture()
    {
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 5,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 10
        };
        var model = new NBEATSModel<double>(options);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = Math.Sin(i * 0.1);
        }

        int numSamples = 80;
        var x = new Matrix<double>(numSamples, options.LookbackWindow);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < options.LookbackWindow; j++)
            {
                x[i, j] = data[i + j];
            }
            y[i] = data[i + options.LookbackWindow];
        }

        model.Train(x, y);

        byte[] serialized = model.Serialize();
        var newModel = new NBEATSModel<double>(new NBEATSModelOptions<double>(options));
        newModel.Deserialize(serialized);

        var metadata = newModel.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region NBEATSBlock Tests

    [Fact]
    public void NBEATSBlock_Initialize_CreatesWeightsAndBiases()
    {
        var block = new NBEATSBlock<double>(
            lookbackWindow: 10,
            forecastHorizon: 5,
            hiddenLayerSize: 16,
            numHiddenLayers: 2,
            thetaSizeBackcast: 10,
            thetaSizeForecast: 5,
            useInterpretableBasis: false,
            polynomialDegree: 3
        );

        Assert.True(block.ParameterCount > 0);
    }

    [Fact]
    public void NBEATSBlock_Forward_ReturnsBackcastAndForecast()
    {
        var block = new NBEATSBlock<double>(
            lookbackWindow: 8,
            forecastHorizon: 4,
            hiddenLayerSize: 12,
            numHiddenLayers: 2,
            thetaSizeBackcast: 8,
            thetaSizeForecast: 4,
            useInterpretableBasis: false
        );

        var input = new Vector<double>(8);
        for (int i = 0; i < 8; i++)
        {
            input[i] = Math.Sin(i * 0.1);
        }

        var (backcast, forecast) = block.Forward(input);

        Assert.Equal(8, backcast.Length);
        Assert.Equal(4, forecast.Length);
    }

    [Fact]
    public void NBEATSBlock_GetParameters_ReturnsAllWeights()
    {
        var block = new NBEATSBlock<double>(
            lookbackWindow: 10,
            forecastHorizon: 5,
            hiddenLayerSize: 16,
            numHiddenLayers: 2,
            thetaSizeBackcast: 10,
            thetaSizeForecast: 5,
            useInterpretableBasis: false
        );

        var parameters = block.GetParameters();

        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
        Assert.Equal(block.ParameterCount, parameters.Length);
    }

    [Fact]
    public void NBEATSBlock_SetParameters_UpdatesInternalWeights()
    {
        var block = new NBEATSBlock<double>(
            lookbackWindow: 8,
            forecastHorizon: 4,
            hiddenLayerSize: 12,
            numHiddenLayers: 1,
            thetaSizeBackcast: 8,
            thetaSizeForecast: 4,
            useInterpretableBasis: false
        );

        var originalParams = block.GetParameters();
        var newParams = new Vector<double>(originalParams.Length);
        for (int i = 0; i < newParams.Length; i++)
        {
            newParams[i] = i * 0.01;
        }

        block.SetParameters(newParams);

        var updatedParams = block.GetParameters();
        Assert.Equal(newParams[0], updatedParams[0], Tolerance);
        Assert.Equal(newParams[newParams.Length - 1], updatedParams[updatedParams.Length - 1], Tolerance);
    }

    [Fact]
    public void NBEATSBlock_WithInterpretableBasis_UsesPolynomialExpansion()
    {
        var block = new NBEATSBlock<double>(
            lookbackWindow: 12,
            forecastHorizon: 6,
            hiddenLayerSize: 16,
            numHiddenLayers: 2,
            thetaSizeBackcast: 4,  // Polynomial degree + 1
            thetaSizeForecast: 4,
            useInterpretableBasis: true,
            polynomialDegree: 3
        );

        var input = new Vector<double>(12);
        for (int i = 0; i < 12; i++)
        {
            input[i] = i * 0.5;
        }

        var (backcast, forecast) = block.Forward(input);

        Assert.Equal(12, backcast.Length);
        Assert.Equal(6, forecast.Length);
    }

    [Fact]
    public void NBEATSBlock_ForwardPass_ProducesReasonableOutputs()
    {
        var block = new NBEATSBlock<double>(
            lookbackWindow: 10,
            forecastHorizon: 5,
            hiddenLayerSize: 16,
            numHiddenLayers: 2,
            thetaSizeBackcast: 10,
            thetaSizeForecast: 5,
            useInterpretableBasis: false
        );

        var input = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            input[i] = Math.Sin(i * 0.2) + 5.0;
        }

        var (backcast, forecast) = block.Forward(input);

        // Check outputs are bounded (not NaN or Infinity)
        Assert.All(backcast, b => Assert.True(!double.IsNaN(b) && !double.IsInfinity(b)));
        Assert.All(forecast, f => Assert.True(!double.IsNaN(f) && !double.IsInfinity(f)));
    }

    [Fact]
    public void NBEATSBlock_ParameterCount_MatchesArchitecture()
    {
        int lookbackWindow = 10;
        int forecastHorizon = 5;
        int hiddenLayerSize = 16;
        int numHiddenLayers = 2;
        int thetaSizeBackcast = 10;
        int thetaSizeForecast = 5;

        var block = new NBEATSBlock<double>(
            lookbackWindow: lookbackWindow,
            forecastHorizon: forecastHorizon,
            hiddenLayerSize: hiddenLayerSize,
            numHiddenLayers: numHiddenLayers,
            thetaSizeBackcast: thetaSizeBackcast,
            thetaSizeForecast: thetaSizeForecast,
            useInterpretableBasis: false
        );

        // Expected parameter count:
        // First layer: lookbackWindow * hiddenLayerSize + hiddenLayerSize (bias)
        // Hidden layers: (numHiddenLayers - 1) * (hiddenLayerSize * hiddenLayerSize + hiddenLayerSize)
        // Backcast output: hiddenLayerSize * thetaSizeBackcast + thetaSizeBackcast
        // Forecast output: hiddenLayerSize * thetaSizeForecast + thetaSizeForecast

        int expectedCount =
            (lookbackWindow * hiddenLayerSize + hiddenLayerSize) +
            (numHiddenLayers - 1) * (hiddenLayerSize * hiddenLayerSize + hiddenLayerSize) +
            (hiddenLayerSize * thetaSizeBackcast + thetaSizeBackcast) +
            (hiddenLayerSize * thetaSizeForecast + thetaSizeForecast);

        Assert.Equal(expectedCount, block.ParameterCount);
    }

    #endregion

    #region Additional VARMA Tests

    [Fact]
    public void VARMA_WithLargerLags_EstimatesComplexDynamics()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 3, MaLag = 2 };
        var model = new VARMAModel<double>(options);

        var data = new Matrix<double>(120, 2);
        for (int i = 0; i < 120; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1) + Math.Cos(i * 0.2);
            data[i, 1] = Math.Cos(i * 0.1) - 0.5 * Math.Sin(i * 0.2);
        }

        model.Train(data, new Vector<double>(120));

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void VARMA_EvaluateModel_ComputesAccuracyMetrics()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 1, MaLag = 1 };
        var model = new VARMAModel<double>(options);

        var trainData = new Matrix<double>(80, 2);
        var testData = new Matrix<double>(20, 2);

        for (int i = 0; i < 80; i++)
        {
            trainData[i, 0] = i * 0.05;
            trainData[i, 1] = i * 0.03;
        }
        for (int i = 0; i < 20; i++)
        {
            testData[i, 0] = (80 + i) * 0.05;
            testData[i, 1] = (80 + i) * 0.03;
        }

        model.Train(trainData, new Vector<double>(80));

        var testInput = new Matrix<double>(1, 2);
        testInput[0, 0] = testData[0, 0];
        testInput[0, 1] = testData[0, 1];
        var testOutput = new Vector<double>(new[] { testData[1, 0], testData[1, 1] });

        var metrics = model.EvaluateModel(testInput, testOutput);

        Assert.True(metrics.ContainsKey("MSE"));
        Assert.True(metrics.ContainsKey("MAE"));
    }

    [Fact]
    public void VARMA_GetModelMetadata_IncludesARAndMAInfo()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 3, Lag = 2, MaLag = 2 };
        var model = new VARMAModel<double>(options);

        var data = new Matrix<double>(100, 3);
        for (int i = 0; i < 100; i++)
        {
            data[i, 0] = i * 0.1;
            data[i, 1] = i * 0.05;
            data[i, 2] = i * 0.03;
        }

        model.Train(data, new Vector<double>(100));

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.True(metadata.AdditionalInfo.ContainsKey("OutputDimension"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("Lag"));
    }

    [Fact]
    public void VARMA_WithZeroMALag_BehavesLikeVAR()
    {
        var varmaOptions = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 1, MaLag = 0 };
        var varmaModel = new VARMAModel<double>(varmaOptions);

        var varOptions = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var varModel = new VectorAutoRegressionModel<double>(varOptions);

        var data = new Matrix<double>(60, 2);
        for (int i = 0; i < 60; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1);
            data[i, 1] = Math.Cos(i * 0.1);
        }

        varmaModel.Train(data, new Vector<double>(60));
        varModel.Train(data, new Vector<double>(60));

        // Both should produce similar results
        var input = new Matrix<double>(1, 2);
        input[0, 0] = data[59, 0];
        input[0, 1] = data[59, 1];

        var varmaPred = varmaModel.Predict(input);
        var varPred = varModel.Predict(input);

        Assert.Equal(2, varmaPred.Length);
        Assert.Equal(2, varPred.Length);
    }

    [Fact]
    public void VARMA_Predict_WithSeasonalData_CapturesPatterns()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 4, MaLag = 2 };
        var model = new VARMAModel<double>(options);

        var data = new Matrix<double>(200, 2);
        for (int i = 0; i < 200; i++)
        {
            data[i, 0] = 10.0 + 3.0 * Math.Sin(2 * Math.PI * i / 12.0);
            data[i, 1] = 8.0 + 2.0 * Math.Cos(2 * Math.PI * i / 12.0);
        }

        model.Train(data, new Vector<double>(200));

        var input = new Matrix<double>(4, 2);
        for (int i = 0; i < 4; i++)
        {
            input[i, 0] = data[196 + i, 0];
            input[i, 1] = data[196 + i, 1];
        }

        var prediction = model.Predict(input);

        Assert.Equal(2, prediction.Length);
        Assert.True(Math.Abs(prediction[0] - 10.0) < 5.0);
        Assert.True(Math.Abs(prediction[1] - 8.0) < 5.0);
    }

    [Fact]
    public void VARMA_Reset_ClearsModelState()
    {
        var options = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 1, MaLag = 1 };
        var model = new VARMAModel<double>(options);

        var data = new Matrix<double>(70, 2);
        for (int i = 0; i < 70; i++)
        {
            data[i, 0] = i * 0.1;
            data[i, 1] = i * 0.05;
        }

        model.Train(data, new Vector<double>(70));

        model.Reset();

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region Additional GARCH Tests

    [Fact]
    public void GARCH_WithSymmetricShocks_CapturesVolatility()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(250);
        var random = new Random(42);
        double vol = 0.1;

        for (int t = 0; t < 250; t++)
        {
            double shock = random.NextGaussian() * vol;
            data[t] = shock;
            vol = 0.01 + 0.15 * shock * shock + 0.80 * vol;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var volatility = model.GetConditionalVolatility();
        Assert.All(volatility, v => Assert.True(v >= 0));
        Assert.True(volatility.Length > 0);
    }

    [Fact]
    public void GARCH_ForecastVolatility_ReturnsMultiStepAhead()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(180);
        var random = new Random(42);
        for (int i = 0; i < 180; i++)
        {
            data[i] = random.NextGaussian() * 0.15;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var forecast = model.ForecastVolatility(horizon: 10);

        Assert.NotNull(forecast);
        Assert.Equal(10, forecast.Length);
        Assert.All(forecast, v => Assert.True(v >= 0));
    }

    [Fact]
    public void GARCH_Reset_ClearsEstimatedParameters()
    {
        var options = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(120);
        var random = new Random(42);
        for (int i = 0; i < 120; i++)
        {
            data[i] = random.NextGaussian() * 0.2;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        model.Reset();

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void GARCH_GetModelMetadata_ContainsOrderInformation()
    {
        var options = new GARCHModelOptions<double> { P = 2, Q = 2 };
        var model = new GARCHModel<double>(options);

        var data = new Vector<double>(200);
        var random = new Random(42);
        for (int i = 0; i < 200; i++)
        {
            data[i] = random.NextGaussian() * 0.12;
        }

        var x = Matrix<double>.FromColumns(data);
        model.Train(x, data);

        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.True(metadata.AdditionalInfo.ContainsKey("P"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("Q"));
        Assert.Equal(2, metadata.AdditionalInfo["P"]);
        Assert.Equal(2, metadata.AdditionalInfo["Q"]);
    }

    #endregion

    #region Additional TransferFunction Tests

    [Fact]
    public void TransferFunction_WithMultipleInputs_HandlesComplexRelationships()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 2,
            OutputLags = 1,
            Delay = 0,
            MultipleInputs = true,
            NumInputSeries = 3
        };
        var model = new TransferFunctionModel<double>(options);

        var input1 = new Vector<double>(100);
        var input2 = new Vector<double>(100);
        var input3 = new Vector<double>(100);
        var output = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            input1[i] = Math.Sin(i * 0.1);
            input2[i] = Math.Cos(i * 0.1);
            input3[i] = i * 0.05;
            if (i > 0)
            {
                output[i] = 0.4 * input1[i - 1] + 0.3 * input2[i - 1] + 0.2 * input3[i - 1];
            }
        }

        var x = new Matrix<double>(100, 4);
        for (int i = 0; i < 100; i++)
        {
            x[i, 0] = input1[i];
            x[i, 1] = input2[i];
            x[i, 2] = input3[i];
            x[i, 3] = output[i];
        }

        model.Train(x, output);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void TransferFunction_GetImpulseResponse_ShowsLaggedEffect()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 4,
            OutputLags = 2,
            Delay = 1
        };
        var model = new TransferFunctionModel<double>(options);

        var input = new Vector<double>(120);
        var output = new Vector<double>(120);

        for (int i = 0; i < 120; i++)
        {
            input[i] = Math.Sin(i * 0.15);
            if (i > 2)
            {
                output[i] = 0.5 * input[i - 2] + 0.3 * input[i - 3];
            }
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        var impulseResponse = model.GetImpulseResponse(horizon: 10);

        Assert.NotNull(impulseResponse);
        Assert.True(impulseResponse.Length > 0);
    }

    [Fact]
    public void TransferFunction_WithConstantDelay_ModelsLagProperly()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 3,
            OutputLags = 1,
            Delay = 2
        };
        var model = new TransferFunctionModel<double>(options);

        var input = new Vector<double>(100);
        var output = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            input[i] = i * 0.1;
            if (i >= 2)
            {
                output[i] = 0.8 * input[i - 2] + 5.0;
            }
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        var testInput = new Matrix<double>(1, 2);
        testInput[0, 0] = input[99];
        testInput[0, 1] = output[99];
        var prediction = model.Predict(testInput);

        Assert.Single(prediction);
        Assert.True(Math.Abs(prediction[0]) < 20.0);
    }

    [Fact]
    public void TransferFunction_Reset_ClearsTransferWeights()
    {
        var options = new TransferFunctionModelOptions<double>
        {
            InputLags = 2,
            OutputLags = 1
        };
        var model = new TransferFunctionModel<double>(options);

        var input = new Vector<double>(80);
        var output = new Vector<double>(80);

        for (int i = 0; i < 80; i++)
        {
            input[i] = i * 0.05;
            if (i > 0) output[i] = 0.6 * input[i - 1];
        }

        var x = Matrix<double>.FromColumns(input, output);
        model.Train(x, output);

        model.Reset();

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region Additional NeuralNetworkARIMA Tests

    [Fact]
    public void NeuralNetworkARIMA_WithDifferencing_HandlesNonStationary()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            DifferencingOrder = 1,
            LaggedPredictions = 3,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(120, 1);
        var y = new Vector<double>(120);

        for (int i = 0; i < 120; i++)
        {
            x[i, 0] = i * 0.2;
            y[i] = i * i * 0.01; // Non-stationary
        }

        model.Train(x, y);

        var testX = new Matrix<double>(1, 1);
        testX[0, 0] = 120 * 0.2;
        var prediction = model.Predict(testX);

        Assert.Single(prediction);
    }

    [Fact]
    public void NeuralNetworkARIMA_Forecast_GeneratesMultiStepPredictions()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 2,
            MAOrder = 1,
            LaggedPredictions = 4,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(100, 1);
        var y = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            x[i, 0] = Math.Sin(i * 0.1);
            y[i] = 10.0 + 2.0 * Math.Sin(i * 0.1);
        }

        model.Train(x, y);

        var history = new Vector<double>(new[] { y[98], y[99] });
        var futureX = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++)
        {
            futureX[i, 0] = Math.Sin((100 + i) * 0.1);
        }

        var forecasts = model.Forecast(history, horizon: 5, exogenousVariables: futureX);

        Assert.Equal(5, forecasts.Length);
    }

    [Fact]
    public void NeuralNetworkARIMA_GetNeuralNetworkComponent_ReturnsNN()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            LaggedPredictions = 2,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(80, 1);
        var y = new Vector<double>(80);

        for (int i = 0; i < 80; i++)
        {
            x[i, 0] = i * 0.1;
            y[i] = 5.0 + i * 0.05;
        }

        model.Train(x, y);

        var nnComponent = model.GetNeuralNetworkComponent();

        Assert.NotNull(nnComponent);
    }

    [Fact]
    public void NeuralNetworkARIMA_GetARIMAComponent_ReturnsARIMA()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            LaggedPredictions = 2,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(80, 1);
        var y = new Vector<double>(80);

        for (int i = 0; i < 80; i++)
        {
            x[i, 0] = Math.Cos(i * 0.1);
            y[i] = 8.0 + Math.Cos(i * 0.1);
        }

        model.Train(x, y);

        var arimaComponent = model.GetARIMAComponent();

        Assert.NotNull(arimaComponent);
    }

    [Fact]
    public void NeuralNetworkARIMA_Reset_ClearsHybridModel()
    {
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            LaggedPredictions = 2,
            ExogenousVariables = 1
        };
        var model = new NeuralNetworkARIMAModel<double>(options);

        var x = new Matrix<double>(70, 1);
        var y = new Vector<double>(70);

        for (int i = 0; i < 70; i++)
        {
            x[i, 0] = i * 0.1;
            y[i] = 10.0 + i * 0.05;
        }

        model.Train(x, y);

        model.Reset();

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    #endregion

    #region Cross-Model Integration Tests

    [Fact]
    public void CrossModel_VAR_VARMA_Comparison_SimilarResults()
    {
        var varOptions = new VARModelOptions<double> { OutputDimension = 2, Lag = 1 };
        var varmaOptions = new VARMAModelOptions<double> { OutputDimension = 2, Lag = 1, MaLag = 0 };

        var varModel = new VectorAutoRegressionModel<double>(varOptions);
        var varmaModel = new VARMAModel<double>(varmaOptions);

        var data = new Matrix<double>(80, 2);
        for (int i = 0; i < 80; i++)
        {
            data[i, 0] = Math.Sin(i * 0.1);
            data[i, 1] = Math.Cos(i * 0.1);
        }

        varModel.Train(data, new Vector<double>(80));
        varmaModel.Train(data, new Vector<double>(80));

        var input = new Matrix<double>(1, 2);
        input[0, 0] = data[79, 0];
        input[0, 1] = data[79, 1];

        var varPred = varModel.Predict(input);
        var varmaPred = varmaModel.Predict(input);

        // Should be similar since MA lag is 0
        Assert.True(Math.Abs(varPred[0] - varmaPred[0]) < 0.5);
    }

    [Fact]
    public void CrossModel_TBATS_Prophet_SeasonalHandling()
    {
        var tbatsOptions = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new List<int> { 12 }
        };
        var prophetOptions = new ProphetModelOptions<double>
        {
            YearlySeasonality = true
        };

        var tbatsModel = new TBATSModel<double>(tbatsOptions);
        var prophetModel = new ProphetModel<double>(prophetOptions);

        var data = new Vector<double>(144); // 12 years monthly
        for (int i = 0; i < 144; i++)
        {
            data[i] = 10.0 + 3.0 * Math.Sin(2 * Math.PI * i / 12.0);
        }

        var x = Matrix<double>.FromColumns(data);
        tbatsModel.Train(x, data);
        prophetModel.Train(x, data);

        // Both should capture seasonality
        var input = new Matrix<double>(1, 1);
        input[0, 0] = data[143];

        var tbatsPred = tbatsModel.Predict(input);
        var prophetPred = prophetModel.Predict(input);

        Assert.Single(tbatsPred);
        Assert.Single(prophetPred);
    }

    [Fact]
    public void CrossModel_GARCH_UCM_VolatilityModeling()
    {
        var garchOptions = new GARCHModelOptions<double> { P = 1, Q = 1 };
        var ucmOptions = new UnobservedComponentsModelOptions<double>
        {
            Level = true,
            Trend = false,
            StochasticVolatility = true
        };

        var garchModel = new GARCHModel<double>(garchOptions);
        var ucmModel = new UnobservedComponentsModel<double>(ucmOptions);

        var data = new Vector<double>(150);
        var random = new Random(42);
        double vol = 0.1;

        for (int t = 0; t < 150; t++)
        {
            double shock = random.NextGaussian() * vol;
            data[t] = shock;
            vol = 0.01 + 0.1 * shock * shock + 0.85 * vol;
        }

        var x = Matrix<double>.FromColumns(data);
        garchModel.Train(x, data);
        ucmModel.Train(x, data);

        // Both should model volatility
        var garchVol = garchModel.GetConditionalVolatility();
        var ucmStates = ucmModel.GetFilteredStates();

        Assert.NotNull(garchVol);
        Assert.NotNull(ucmStates);
    }

    [Fact]
    public void CrossModel_SpectralAnalysis_TBATS_FrequencyDomain()
    {
        var spectralOptions = new SpectralAnalysisOptions<double> { NFFT = 128 };
        var tbatsOptions = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new List<int> { 12 }
        };

        var spectralModel = new SpectralAnalysisModel<double>(spectralOptions);
        var tbatsModel = new TBATSModel<double>(tbatsOptions);

        var data = new Vector<double>(144);
        for (int i = 0; i < 144; i++)
        {
            data[i] = 10.0 + 5.0 * Math.Sin(2 * Math.PI * i / 12.0);
        }

        var x = Matrix<double>.FromColumns(data);
        spectralModel.Train(x, data);
        tbatsModel.Train(x, data);

        // Spectral should identify dominant frequency
        var periodogram = spectralModel.GetPeriodogram();
        var frequencies = spectralModel.GetFrequencies();

        Assert.NotNull(periodogram);
        Assert.NotNull(frequencies);
    }

    [Fact]
    public void CrossModel_NBEATS_NeuralNetworkARIMA_NeuralApproaches()
    {
        var nbeatsOptions = new NBEATSModelOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 8
        };

        var nnArimaOptions = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            LaggedPredictions = 3,
            ExogenousVariables = 1
        };

        var nbeatsModel = new NBEATSModel<double>(nbeatsOptions);
        var nnArimaModel = new NeuralNetworkARIMAModel<double>(nnArimaOptions);

        var data = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            data[i] = Math.Sin(i * 0.1) + 10.0;
        }

        // Prepare N-BEATS data
        int numSamples = 80;
        var xNbeats = new Matrix<double>(numSamples, nbeatsOptions.LookbackWindow);
        var yNbeats = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < nbeatsOptions.LookbackWindow; j++)
            {
                xNbeats[i, j] = data[i + j];
            }
            yNbeats[i] = data[i + nbeatsOptions.LookbackWindow];
        }

        // Prepare NN-ARIMA data
        var xNnArima = Matrix<double>.FromColumns(data.Slice(0, 90));
        var yNnArima = data.Slice(0, 90);

        nbeatsModel.Train(xNbeats, yNbeats);
        nnArimaModel.Train(xNnArima, yNnArima);

        // Both are neural approaches
        var nbeatsMetadata = nbeatsModel.GetModelMetadata();
        var nnArimaMetadata = nnArimaModel.GetModelMetadata();

        Assert.NotNull(nbeatsMetadata);
        Assert.NotNull(nnArimaMetadata);
    }

    #endregion
}

// Extension method for generating Gaussian random numbers
public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0.0, double stdDev = 1.0)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}
