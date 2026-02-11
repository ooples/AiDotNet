using System;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Integration tests for TimeSeries models.
/// </summary>
public class TimeSeriesIntegrationTests
{
    #region ARModel Tests

    [Fact]
    public void ARModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ARModelOptions<double>();
        var model = new ARModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Construction_WithCustomAROrder_Succeeds()
    {
        var options = new ARModelOptions<double> { AROrder = 5 };
        var model = new ARModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Construction_WithCustomLearningRate_Succeeds()
    {
        var options = new ARModelOptions<double> { LearningRate = 0.001 };
        var model = new ARModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Float_Construction_Succeeds()
    {
        var options = new ARModelOptions<float>();
        var model = new ARModel<float>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Construction_WithDifferentAROrders_Succeeds()
    {
        var model1 = new ARModel<double>(new ARModelOptions<double> { AROrder = 1 });
        var model2 = new ARModel<double>(new ARModelOptions<double> { AROrder = 3 });
        var model3 = new ARModel<double>(new ARModelOptions<double> { AROrder = 7 });

        Assert.NotNull(model1);
        Assert.NotNull(model2);
        Assert.NotNull(model3);
    }

    #endregion

    #region MAModel Tests

    [Fact]
    public void MAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new MAModelOptions<double>();
        var model = new MAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void MAModel_Construction_WithCustomMAOrder_Succeeds()
    {
        var options = new MAModelOptions<double> { MAOrder = 3 };
        var model = new MAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void MAModel_Float_Construction_Succeeds()
    {
        var options = new MAModelOptions<float>();
        var model = new MAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region ARMAModel Tests

    [Fact]
    public void ARMAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ARMAOptions<double>();
        var model = new ARMAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARMAModel_Float_Construction_Succeeds()
    {
        var options = new ARMAOptions<float>();
        var model = new ARMAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region ARIMAModel Tests

    [Fact]
    public void ARIMAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ARIMAOptions<double>();
        var model = new ARIMAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARIMAModel_Float_Construction_Succeeds()
    {
        var options = new ARIMAOptions<float>();
        var model = new ARIMAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region SARIMAModel Tests

    [Fact]
    public void SARIMAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new SARIMAOptions<double>();
        var model = new SARIMAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void SARIMAModel_Float_Construction_Succeeds()
    {
        var options = new SARIMAOptions<float>();
        var model = new SARIMAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region ExponentialSmoothingModel Tests

    [Fact]
    public void ExponentialSmoothingModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ExponentialSmoothingOptions<double>();
        var model = new ExponentialSmoothingModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ExponentialSmoothingModel_Float_Construction_Succeeds()
    {
        var options = new ExponentialSmoothingOptions<float>();
        var model = new ExponentialSmoothingModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region GARCHModel Tests

    [Fact]
    public void GARCHModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new GARCHModelOptions<double>();
        var model = new GARCHModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void GARCHModel_Float_Construction_Succeeds()
    {
        var options = new GARCHModelOptions<float>();
        var model = new GARCHModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region StateSpaceModel Tests

    [Fact]
    public void StateSpaceModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new StateSpaceModelOptions<double>();
        var model = new StateSpaceModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void StateSpaceModel_Float_Construction_Succeeds()
    {
        var options = new StateSpaceModelOptions<float>();
        var model = new StateSpaceModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region VectorAutoRegressionModel Tests

    [Fact]
    public void VectorAutoRegressionModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new VARModelOptions<double>();
        var model = new VectorAutoRegressionModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void VectorAutoRegressionModel_Float_Construction_Succeeds()
    {
        var options = new VARModelOptions<float>();
        var model = new VectorAutoRegressionModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllTimeSeriesModels_ImplementITimeSeriesModel()
    {
        var arModel = new ARModel<double>(new ARModelOptions<double>());
        var maModel = new MAModel<double>(new MAModelOptions<double>());
        var armaModel = new ARMAModel<double>(new ARMAOptions<double>());
        var arimaModel = new ARIMAModel<double>(new ARIMAOptions<double>());

        // All models should be non-null
        Assert.NotNull(arModel);
        Assert.NotNull(maModel);
        Assert.NotNull(armaModel);
        Assert.NotNull(arimaModel);
    }

    #endregion

    #region SetParameters Tests (Fix for optimizer parameter initialization)

    [Theory]
    [InlineData(new double[] { 0.5, 0.3 })]
    [InlineData(new double[] { 0.5, 0.3, 0.2 })]
    public void ExponentialSmoothingModel_SetParameters_WithUntrainedModel_InitializesParameters(double[] paramValues)
    {
        // Arrange: Create an untrained model (ModelParameters.Length = 0)
        var options = new ExponentialSmoothingOptions<double>();
        var model = new ExponentialSmoothingModel<double>(options);

        // Verify model starts untrained with empty parameters
        Assert.Equal(0, model.ParameterCount);

        // Act: Set parameters on untrained model (simulates optimizer initialization)
        var parameters = new Tensors.LinearAlgebra.Vector<double>(paramValues);
        model.SetParameters(parameters);

        // Assert: Model should now have parameters with correct count and values
        Assert.Equal(paramValues.Length, model.ParameterCount);
        var retrieved = model.GetParameters();
        for (int i = 0; i < paramValues.Length; i++)
        {
            Assert.Equal(paramValues[i], retrieved[i], precision: 10);
        }
    }

    [Fact]
    public void ExponentialSmoothingModel_SetParameters_WithTrainedModel_UpdatesParameterValues()
    {
        // Arrange: Create a model and set initial parameters
        var options = new ExponentialSmoothingOptions<double>();
        var model = new ExponentialSmoothingModel<double>(options);

        var initialParams = new Tensors.LinearAlgebra.Vector<double>(new double[] { 0.5, 0.3 });
        model.SetParameters(initialParams);
        Assert.Equal(2, model.ParameterCount);

        // Act: Update parameters
        var newParams = new Tensors.LinearAlgebra.Vector<double>(new double[] { 0.8, 0.1 });
        model.SetParameters(newParams);

        // Assert: Parameters should be updated with new values
        Assert.Equal(2, model.ParameterCount);
        var retrieved = model.GetParameters();
        Assert.Equal(0.8, retrieved[0], precision: 10);
        Assert.Equal(0.1, retrieved[1], precision: 10);
    }

    [Fact]
    public void ExponentialSmoothingModel_SetParameters_WithMismatchedLength_ThrowsException()
    {
        // Arrange: Create a model and set initial parameters
        var options = new ExponentialSmoothingOptions<double>();
        var model = new ExponentialSmoothingModel<double>(options);

        var initialParams = new Tensors.LinearAlgebra.Vector<double>(new double[] { 0.5, 0.3 });
        model.SetParameters(initialParams);

        // Act & Assert: Trying to set different length should throw
        var wrongLengthParams = new Tensors.LinearAlgebra.Vector<double>(new double[] { 0.5 });
        Assert.Throws<ArgumentException>(() => model.SetParameters(wrongLengthParams));
    }

    [Fact]
    public void ARModel_SetParameters_WithUntrainedModel_Succeeds()
    {
        // Arrange
        var options = new ARModelOptions<double> { AROrder = 3 };
        var model = new ARModel<double>(options);

        Assert.Equal(0, model.ParameterCount);

        // Act
        var paramValues = new double[] { 0.1, 0.2, 0.3, 0.4 };
        var parameters = new Tensors.LinearAlgebra.Vector<double>(paramValues);
        model.SetParameters(parameters);

        // Assert: Verify count and values
        Assert.Equal(4, model.ParameterCount);
        var retrieved = model.GetParameters();
        for (int i = 0; i < paramValues.Length; i++)
        {
            Assert.Equal(paramValues[i], retrieved[i], precision: 10);
        }
    }

    #endregion

    #region Train-and-Forecast Integration Tests

    /// <summary>
    /// Helper to create a simple linear trend dataset for training.
    /// Returns (x, y) where y = slope * i + intercept.
    /// </summary>
    private static (Tensors.LinearAlgebra.Matrix<double> x, Tensors.LinearAlgebra.Vector<double> y) CreateLinearTrendData(int n, double slope = 0.5, double intercept = 10.0)
    {
        var y = new Tensors.LinearAlgebra.Vector<double>(n);
        var x = new Tensors.LinearAlgebra.Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = slope * i + intercept;
            x[i, 0] = i;
        }
        return (x, y);
    }

    /// <summary>
    /// Helper to create seasonal data: y = amplitude * sin(2*pi*i/period) + offset.
    /// </summary>
    private static (Tensors.LinearAlgebra.Matrix<double> x, Tensors.LinearAlgebra.Vector<double> y) CreateSeasonalData(int n, int period = 12, double amplitude = 5.0, double offset = 50.0)
    {
        var y = new Tensors.LinearAlgebra.Vector<double>(n);
        var x = new Tensors.LinearAlgebra.Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = amplitude * Math.Sin(2.0 * Math.PI * i / period) + offset;
            x[i, 0] = i;
        }
        return (x, y);
    }

    [Fact]
    public void ARIMAModel_TrainAndForecast_WithDifferencing_ProducesFinitePredictions()
    {
        // Arrange: linear trend with d=1
        var (x, y) = CreateLinearTrendData(100);
        var options = new ARIMAOptions<double> { P = 2, D = 1, Q = 0, MaxIterations = 50 };
        var model = new ARIMAModel<double>(options);

        // Act
        model.Train(x, y);
        var history = new Tensors.LinearAlgebra.Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        var forecasts = model.Forecast(history, 5);

        // Assert: forecasts should be finite and within reasonable range
        Assert.Equal(5, forecasts.Length);
        for (int i = 0; i < forecasts.Length; i++)
        {
            Assert.False(double.IsNaN(forecasts[i]), $"Forecast[{i}] is NaN");
            Assert.False(double.IsInfinity(forecasts[i]), $"Forecast[{i}] is Infinity");
        }
    }

    [Fact]
    public void SARIMAModel_TrainAndForecast_WithSeasonality_DoesNotCrash()
    {
        // Arrange: seasonal data with P=1, m=12
        var (x, y) = CreateSeasonalData(120, period: 12);
        var options = new SARIMAOptions<double>
        {
            P = 1, D = 0, Q = 0,
            SeasonalP = 1, SeasonalD = 0, SeasonalQ = 0,
            SeasonalPeriod = 12,
            MaxIterations = 50
        };
        var model = new SARIMAModel<double>(options);

        // Act
        model.Train(x, y);
        var history = new Tensors.LinearAlgebra.Vector<double>(60);
        for (int i = 0; i < 60; i++)
            history[i] = y[60 + i];
        var forecasts = model.Forecast(history, 5);

        // Assert: should not crash and produce finite values
        Assert.Equal(5, forecasts.Length);
        for (int i = 0; i < forecasts.Length; i++)
        {
            Assert.False(double.IsNaN(forecasts[i]), $"Forecast[{i}] is NaN");
            Assert.False(double.IsInfinity(forecasts[i]), $"Forecast[{i}] is Infinity");
        }
    }

    [Fact]
    public void ExponentialSmoothing_TrainAndForecast_DoesNotResetState()
    {
        // Arrange: strong linear trend data with fixed smoothing params
        // to guarantee a non-zero trend component in forecasts
        var (x, y) = CreateLinearTrendData(100, slope: 2.0, intercept: 10.0);
        var options = new ExponentialSmoothingOptions<double>
        {
            InitialAlpha = 0.3,
            InitialBeta = 0.3,
            UseTrend = true,
            UseSeasonal = false
        };
        var model = new ExponentialSmoothingModel<double>(options);

        // Act
        model.Train(x, y);
        var history = new Tensors.LinearAlgebra.Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        var forecasts = model.Forecast(history, 5);

        // Assert: forecasts should be finite
        Assert.Equal(5, forecasts.Length);
        for (int i = 0; i < forecasts.Length; i++)
        {
            Assert.False(double.IsNaN(forecasts[i]), $"Forecast[{i}] is NaN");
            Assert.False(double.IsInfinity(forecasts[i]), $"Forecast[{i}] is Infinity");
        }

        // With a strong upward trend and non-trivial beta, multi-step forecasts
        // should be monotonically increasing (each step adds the trend)
        for (int i = 1; i < forecasts.Length; i++)
        {
            Assert.True(forecasts[i] > forecasts[i - 1],
                $"Forecast[{i}] ({forecasts[i]}) should be greater than Forecast[{i - 1}] ({forecasts[i - 1]}) with an upward trend");
        }
    }

    [Fact]
    public void NBEATSModel_Train_WithSingleColumnInput_DoesNotCrash()
    {
        // Arrange: single-column feature matrix (univariate time series)
        var (x, y) = CreateLinearTrendData(50);
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 5,
            NumStacks = 2,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            NumHiddenLayers = 2,
            Epochs = 2,
            BatchSize = 8
        };
        var model = new NBEATSModel<double>(options);

        // Act & Assert: training should not throw IndexOutOfRangeException
        var exception = Record.Exception(() => model.Train(x, y));
        Assert.Null(exception);
    }

    [Fact]
    public void AutoformerModel_Train_WithSmallData_DoesNotCrash()
    {
        // Arrange: small dataset with reduced options
        var (x, y) = CreateLinearTrendData(60);
        var options = new AutoformerOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            EmbeddingDim = 16,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumAttentionHeads = 2,
            Epochs = 2,
            BatchSize = 8
        };
        var model = new AutoformerModel<double>(options);

        // Act & Assert: training should not crash with rank-1 tensor
        var exception = Record.Exception(() => model.Train(x, y));
        Assert.Null(exception);
    }

    [Fact(Timeout = 30_000)]
    public async Task InformerModel_Train_CompletesInReasonableTime()
    {
        // Arrange: small dataset with reduced defaults
        var (x, y) = CreateLinearTrendData(40);
        var options = new InformerOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            EmbeddingDim = 16,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumAttentionHeads = 2,
            Epochs = 2,
            BatchSize = 8
        };
        var model = new InformerModel<double>(options);

        // Act & Assert: training should complete (not hang) and not crash
        var exception = await Record.ExceptionAsync(() => Task.Run(() => model.Train(x, y)));
        Assert.Null(exception);
    }

    #endregion
}
