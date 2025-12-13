using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Unit tests for NHiTSModel (N-HiTS - Neural Hierarchical Interpolation for Time Series).
/// </summary>
public class NHiTSModelTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultOptions_CreatesValidModel()
    {
        var model = new NHiTSModel<double>();

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidModel()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 3,
            NumBlocksPerStack = 2,
            LookbackWindow = 24,
            ForecastHorizon = 12,
            HiddenLayerSize = 64,
            PoolingKernelSizes = new[] { 2, 4, 8 }
        };

        var model = new NHiTSModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithZeroStacks_ThrowsArgumentException()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 0,
            PoolingKernelSizes = new int[] { }
        };

        Assert.Throws<ArgumentException>(() => new NHiTSModel<double>(options));
    }

    [Fact]
    public void Constructor_WithNegativeStacks_ThrowsArgumentException()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = -1,
            PoolingKernelSizes = new[] { 2 }
        };

        Assert.Throws<ArgumentException>(() => new NHiTSModel<double>(options));
    }

    [Fact]
    public void Constructor_WithMismatchedPoolingKernelSizes_ThrowsArgumentException()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 3,
            PoolingKernelSizes = new[] { 2, 4 } // Only 2 sizes but 3 stacks
        };

        Assert.Throws<ArgumentException>(() => new NHiTSModel<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroLookbackWindow_ThrowsArgumentException()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 3,
            LookbackWindow = 0,
            PoolingKernelSizes = new[] { 2, 4, 8 }
        };

        Assert.Throws<ArgumentException>(() => new NHiTSModel<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroForecastHorizon_ThrowsArgumentException()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 3,
            ForecastHorizon = 0,
            PoolingKernelSizes = new[] { 2, 4, 8 }
        };

        Assert.Throws<ArgumentException>(() => new NHiTSModel<double>(options));
    }

    #endregion

    #region Training Tests

    [Fact]
    public void Train_WithValidData_CompletesWithoutError()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 2,
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            PoolingKernelSizes = new[] { 2, 4 }
        };

        var model = new NHiTSModel<double>(options);
        var trainingData = GenerateSyntheticData(50);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    [Fact]
    public void Train_WithMinimalData_CompletesWithoutError()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 2,
            LookbackWindow = 4,
            ForecastHorizon = 2,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1,
            Epochs = 1,
            BatchSize = 1,
            PoolingKernelSizes = new[] { 2, 2 }
        };

        var model = new NHiTSModel<double>(options);
        var trainingData = GenerateSyntheticData(20);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    #endregion

    #region Prediction Tests

    [Fact]
    public void PredictSingle_AfterTraining_ReturnsValidPrediction()
    {
        var options = new NHiTSOptions<double>
        {
            NumStacks = 2,
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            PoolingKernelSizes = new[] { 2, 4 }
        };

        var model = new NHiTSModel<double>(options);
        var trainingData = GenerateSyntheticData(50);
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
        var options = new NHiTSOptions<double>
        {
            NumStacks = 2,
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            PoolingKernelSizes = new[] { 2, 4 }
        };

        var model = new NHiTSModel<double>(options);
        var trainingData = GenerateSyntheticData(50);
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
        var options = new NHiTSOptions<double>
        {
            NumStacks = 2,
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            PoolingKernelSizes = new[] { 2, 4 }
        };

        var model = new NHiTSModel<double>(options);
        var trainingData = GenerateSyntheticData(50);
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
        var options = new NHiTSOptions<double>
        {
            NumStacks = 2,
            LookbackWindow = 8,
            ForecastHorizon = 4,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 2,
            BatchSize = 2,
            PoolingKernelSizes = new[] { 2, 4 }
        };

        var model = new NHiTSModel<double>(options);
        var trainingData = GenerateSyntheticData(50);
        model.Train(trainingData.inputs, trainingData.targets);

        var serialized = model.Serialize();
        Assert.NotNull(serialized);
        Assert.NotEmpty(serialized);

        var deserializedModel = new NHiTSModel<double>(options);
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
        var options = new NHiTSOptions<float>
        {
            NumStacks = 2,
            LookbackWindow = 8,
            ForecastHorizon = 4,
            PoolingKernelSizes = new[] { 2, 4 }
        };

        var model = new NHiTSModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> inputs, Vector<double> targets) GenerateSyntheticData(int numSamples)
    {
        int lookback = 8;

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
