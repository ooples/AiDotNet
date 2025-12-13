using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Unit tests for ChronosFoundationModel (Zero-shot Time Series Forecasting Foundation Model).
/// </summary>
public class ChronosFoundationModelTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultOptions_CreatesValidModel()
    {
        var model = new ChronosFoundationModel<double>();

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidModel()
    {
        var options = new ChronosOptions<double>
        {
            ContextLength = 64,
            ForecastHorizon = 16,
            VocabularySize = 512,
            EmbeddingDim = 32,
            NumLayers = 2,
            NumHeads = 4,
            Epochs = 10
        };

        var model = new ChronosFoundationModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithVocabularySizeLessThan2_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            VocabularySize = 1
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroEmbeddingDim_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            EmbeddingDim = 0
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Constructor_WithNegativeEmbeddingDim_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            EmbeddingDim = -10
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroNumHeads_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            NumHeads = 0
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Constructor_WithEmbeddingDimNotDivisibleByNumHeads_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            EmbeddingDim = 65, // Not divisible by 8
            NumHeads = 8
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroNumLayers_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            NumLayers = 0
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroContextLength_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            ContextLength = 0
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Constructor_WithZeroForecastHorizon_ThrowsArgumentException()
    {
        var options = new ChronosOptions<double>
        {
            ForecastHorizon = 0
        };

        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    #endregion

    #region Training Tests

    [Fact]
    public void Train_WithValidData_CompletesWithoutError()
    {
        var options = new ChronosOptions<double>
        {
            ContextLength = 16,
            ForecastHorizon = 4,
            VocabularySize = 64,
            EmbeddingDim = 16,
            NumLayers = 1,
            NumHeads = 2,
            Epochs = 2
        };

        var model = new ChronosFoundationModel<double>(options);
        var trainingData = GenerateSyntheticData(50, options.ContextLength);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    [Fact]
    public void Train_WithMinimalData_CompletesWithoutError()
    {
        var options = new ChronosOptions<double>
        {
            ContextLength = 8,
            ForecastHorizon = 2,
            VocabularySize = 32,
            EmbeddingDim = 8,
            NumLayers = 1,
            NumHeads = 2,
            Epochs = 1
        };

        var model = new ChronosFoundationModel<double>(options);
        var trainingData = GenerateSyntheticData(20, options.ContextLength);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    #endregion

    #region Prediction Tests

    [Fact]
    public void PredictSingle_AfterTraining_ReturnsFiniteValue()
    {
        var options = new ChronosOptions<double>
        {
            ContextLength = 16,
            ForecastHorizon = 4,
            VocabularySize = 64,
            EmbeddingDim = 16,
            NumLayers = 1,
            NumHeads = 2,
            Epochs = 2
        };

        var model = new ChronosFoundationModel<double>(options);
        var trainingData = GenerateSyntheticData(50, options.ContextLength);
        model.Train(trainingData.inputs, trainingData.targets);

        var input = new Vector<double>(options.ContextLength);
        for (int i = 0; i < options.ContextLength; i++)
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
        var options = new ChronosOptions<double>
        {
            ContextLength = 16,
            ForecastHorizon = 4,
            VocabularySize = 64,
            EmbeddingDim = 16,
            NumLayers = 1,
            NumHeads = 2,
            Epochs = 2
        };

        var model = new ChronosFoundationModel<double>(options);
        var trainingData = GenerateSyntheticData(50, options.ContextLength);
        model.Train(trainingData.inputs, trainingData.targets);

        var input = new Vector<double>(options.ContextLength);
        for (int i = 0; i < options.ContextLength; i++)
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
        var options = new ChronosOptions<double>
        {
            ContextLength = 16,
            ForecastHorizon = 4,
            VocabularySize = 64,
            EmbeddingDim = 16,
            NumLayers = 1,
            NumHeads = 2,
            Epochs = 2
        };

        var model = new ChronosFoundationModel<double>(options);
        var trainingData = GenerateSyntheticData(50, options.ContextLength);
        model.Train(trainingData.inputs, trainingData.targets);

        var inputMatrix = new Matrix<double>(1, options.ContextLength);
        for (int i = 0; i < options.ContextLength; i++)
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
        var options = new ChronosOptions<double>
        {
            ContextLength = 16,
            ForecastHorizon = 4,
            VocabularySize = 64,
            EmbeddingDim = 16,
            NumLayers = 1,
            NumHeads = 2,
            Epochs = 2
        };

        var model = new ChronosFoundationModel<double>(options);
        var trainingData = GenerateSyntheticData(50, options.ContextLength);
        model.Train(trainingData.inputs, trainingData.targets);

        var serialized = model.Serialize();
        Assert.NotNull(serialized);
        Assert.NotEmpty(serialized);

        var deserializedModel = new ChronosFoundationModel<double>(options);
        deserializedModel.Deserialize(serialized);

        var input = new Vector<double>(options.ContextLength);
        for (int i = 0; i < options.ContextLength; i++)
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
        var options = new ChronosOptions<float>
        {
            ContextLength = 16,
            ForecastHorizon = 4,
            VocabularySize = 64,
            EmbeddingDim = 16,
            NumLayers = 1,
            NumHeads = 2
        };

        var model = new ChronosFoundationModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> inputs, Vector<double> targets) GenerateSyntheticData(int numSamples, int contextLen)
    {
        var inputs = new Matrix<double>(numSamples, contextLen);
        var targets = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            double baseValue = i * 0.1;
            for (int j = 0; j < contextLen; j++)
            {
                inputs[i, j] = Math.Sin(2 * Math.PI * (baseValue + j) / 10) + 0.1 * (j % 3);
            }
            targets[i] = Math.Sin(2 * Math.PI * (baseValue + contextLen) / 10) + 0.1 * (contextLen % 3);
        }

        return (inputs, targets);
    }

    #endregion
}
