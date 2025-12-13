using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Unit tests for TemporalFusionTransformer (TFT - Interpretable Multi-horizon Forecasting).
/// </summary>
public class TemporalFusionTransformerTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultOptions_CreatesValidModel()
    {
        var model = new TemporalFusionTransformer<double>();

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidModel()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 24,
            ForecastHorizon = 12,
            HiddenSize = 64,
            NumAttentionHeads = 4,
            NumLayers = 2,
            Epochs = 10,
            BatchSize = 16,
            QuantileLevels = new[] { 0.1, 0.5, 0.9 }
        };

        var model = new TemporalFusionTransformer<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithZeroLookbackWindow_ThrowsArgumentException()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 0
        };

        Assert.Throws<ArgumentException>(() => new TemporalFusionTransformer<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroForecastHorizon_ThrowsArgumentException()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            ForecastHorizon = 0
        };

        Assert.Throws<ArgumentException>(() => new TemporalFusionTransformer<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroHiddenSize_ThrowsArgumentException()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            HiddenSize = 0
        };

        Assert.Throws<ArgumentException>(() => new TemporalFusionTransformer<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroAttentionHeads_ThrowsArgumentException()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            NumAttentionHeads = 0
        };

        Assert.Throws<ArgumentException>(() => new TemporalFusionTransformer<double>(options));
    }

    [Fact]
    public void Constructor_WithHiddenSizeNotDivisibleByHeads_ThrowsArgumentException()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            HiddenSize = 65, // Not divisible by 4
            NumAttentionHeads = 4
        };

        Assert.Throws<ArgumentException>(() => new TemporalFusionTransformer<double>(options));
    }

    [Fact]
    public void Constructor_WithEmptyQuantileLevels_ThrowsArgumentException()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            QuantileLevels = Array.Empty<double>()
        };

        Assert.Throws<ArgumentException>(() => new TemporalFusionTransformer<double>(options));
    }

    [Fact]
    public void Constructor_WithInvalidQuantileLevel_ThrowsArgumentException()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            QuantileLevels = new[] { -0.1, 0.5, 0.9 } // -0.1 is invalid
        };

        Assert.Throws<ArgumentException>(() => new TemporalFusionTransformer<double>(options));
    }

    #endregion

    #region Training Tests

    [Fact]
    public void Train_WithValidData_CompletesWithoutError()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenSize = 16,
            NumAttentionHeads = 2,
            NumLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            QuantileLevels = new[] { 0.5 }
        };

        var model = new TemporalFusionTransformer<double>(options);
        var trainingData = GenerateSyntheticData(50, options.LookbackWindow);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    [Fact]
    public void Train_WithMinimalData_CompletesWithoutError()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 4,
            ForecastHorizon = 2,
            HiddenSize = 8,
            NumAttentionHeads = 2,
            NumLayers = 1,
            Epochs = 1,
            BatchSize = 1,
            QuantileLevels = new[] { 0.5 }
        };

        var model = new TemporalFusionTransformer<double>(options);
        var trainingData = GenerateSyntheticData(20, options.LookbackWindow);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    #endregion

    #region Prediction Tests

    [Fact]
    public void PredictSingle_AfterTraining_ReturnsValidPrediction()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenSize = 16,
            NumAttentionHeads = 2,
            NumLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            QuantileLevels = new[] { 0.5 }
        };

        var model = new TemporalFusionTransformer<double>(options);
        var trainingData = GenerateSyntheticData(50, options.LookbackWindow);
        model.Train(trainingData.inputs, trainingData.targets);

        var input = new Vector<double>(options.LookbackWindow);
        for (int i = 0; i < options.LookbackWindow; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * i / 10);
        }

        var prediction = model.PredictSingle(input);

        Assert.False(double.IsNaN(prediction), "Prediction is NaN");
        Assert.False(double.IsInfinity(prediction), "Prediction is Infinity");
    }

    [Fact]
    public void PredictSingle_ReturnsFiniteValues()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenSize = 16,
            NumAttentionHeads = 2,
            NumLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            QuantileLevels = new[] { 0.5 }
        };

        var model = new TemporalFusionTransformer<double>(options);
        var trainingData = GenerateSyntheticData(50, options.LookbackWindow);
        model.Train(trainingData.inputs, trainingData.targets);

        var input = new Vector<double>(options.LookbackWindow);
        for (int i = 0; i < options.LookbackWindow; i++)
        {
            input[i] = i * 0.1;
        }

        var prediction = model.PredictSingle(input);

        Assert.False(double.IsNaN(prediction), "Prediction contains NaN");
        Assert.False(double.IsInfinity(prediction), "Prediction contains Infinity");
    }

    [Fact]
    public void Predict_WithMatrix_ReturnsValidPredictions()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenSize = 16,
            NumAttentionHeads = 2,
            NumLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            QuantileLevels = new[] { 0.5 }
        };

        var model = new TemporalFusionTransformer<double>(options);
        var trainingData = GenerateSyntheticData(50, options.LookbackWindow);
        model.Train(trainingData.inputs, trainingData.targets);

        var inputMatrix = new Matrix<double>(1, options.LookbackWindow);
        for (int i = 0; i < options.LookbackWindow; i++)
        {
            inputMatrix[0, i] = Math.Sin(2 * Math.PI * i / 10);
        }

        var predictions = model.Predict(inputMatrix);

        Assert.NotNull(predictions);
        Assert.True(predictions.Length > 0);
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void Serialize_AndDeserialize_PreservesModel()
    {
        var options = new TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenSize = 16,
            NumAttentionHeads = 2,
            NumLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            QuantileLevels = new[] { 0.5 }
        };

        var model = new TemporalFusionTransformer<double>(options);
        var trainingData = GenerateSyntheticData(50, options.LookbackWindow);
        model.Train(trainingData.inputs, trainingData.targets);

        var serialized = model.Serialize();
        Assert.NotNull(serialized);
        Assert.NotEmpty(serialized);

        var deserializedModel = new TemporalFusionTransformer<double>(options);
        deserializedModel.Deserialize(serialized);

        var input = new Vector<double>(options.LookbackWindow);
        for (int i = 0; i < options.LookbackWindow; i++)
        {
            input[i] = i * 0.1;
        }

        var originalPrediction = model.PredictSingle(input);
        var deserializedPrediction = deserializedModel.PredictSingle(input);

        Assert.Equal(originalPrediction, deserializedPrediction, 6);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void Constructor_WithFloatType_CreatesValidModel()
    {
        var options = new TemporalFusionTransformerOptions<float>
        {
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenSize = 16,
            NumAttentionHeads = 2,
            NumLayers = 1,
            QuantileLevels = new[] { 0.5 }
        };

        var model = new TemporalFusionTransformer<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> inputs, Vector<double> targets) GenerateSyntheticData(int numSamples, int lookback)
    {
        var inputs = new Matrix<double>(numSamples, lookback);
        var targets = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            double baseValue = i * 0.1;
            for (int j = 0; j < lookback; j++)
            {
                inputs[i, j] = Math.Sin(2 * Math.PI * (baseValue + j) / 10) + 0.1 * (j % 3);
            }
            targets[i] = Math.Sin(2 * Math.PI * (baseValue + lookback) / 10) + 0.1 * (lookback % 3);
        }

        return (inputs, targets);
    }

    #endregion
}
