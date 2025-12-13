using AiDotNet.TimeSeries.AnomalyDetection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Unit tests for LSTMVAE (LSTM Variational Autoencoder for Anomaly Detection).
/// </summary>
public class LSTMVAETests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultOptions_CreatesValidModel()
    {
        var model = new LSTMVAE<double>();

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidModel()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 30,
            LatentDim = 16,
            HiddenSize = 32,
            LearningRate = 0.001,
            Epochs = 10,
            BatchSize = 16
        };

        var model = new LSTMVAE<double>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Training Tests

    [Fact]
    public void Train_WithValidData_CompletesWithoutError()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 8,
            LatentDim = 4,
            HiddenSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new LSTMVAE<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    [Fact]
    public void Train_WithMinimalData_CompletesWithoutError()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 4,
            LatentDim = 2,
            HiddenSize = 4,
            LearningRate = 0.001,
            Epochs = 1,
            BatchSize = 1
        };

        var model = new LSTMVAE<double>(options);
        var trainingData = GenerateSyntheticData(20, options.WindowSize);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    #endregion

    #region Prediction Tests

    [Fact]
    public void PredictSingle_AfterTraining_ReturnsFiniteValue()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 8,
            LatentDim = 4,
            HiddenSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new LSTMVAE<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        var input = new Vector<double>(options.WindowSize);
        for (int i = 0; i < options.WindowSize; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * i / 10);
        }

        var prediction = model.PredictSingle(input);

        Assert.False(double.IsNaN(prediction), "Prediction is NaN");
        Assert.False(double.IsInfinity(prediction), "Prediction is Infinity");
    }

    [Fact]
    public void Predict_WithMatrix_ReturnsValidPredictions()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 8,
            LatentDim = 4,
            HiddenSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new LSTMVAE<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        var inputMatrix = new Matrix<double>(1, options.WindowSize);
        for (int i = 0; i < options.WindowSize; i++)
        {
            inputMatrix[0, i] = i * 0.1;
        }

        var predictions = model.Predict(inputMatrix);

        Assert.NotNull(predictions);
        Assert.True(predictions.Length > 0);
    }

    #endregion

    #region Anomaly Detection Tests

    [Fact]
    public void DetectAnomalies_AfterTraining_ReturnsValidResults()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 8,
            LatentDim = 4,
            HiddenSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new LSTMVAE<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        // Create test data matrix
        var testData = new Matrix<double>(5, options.WindowSize);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < options.WindowSize; j++)
            {
                testData[i, j] = Math.Sin(2 * Math.PI * (i + j) / 10);
            }
        }

        var anomalies = model.DetectAnomalies(testData);

        Assert.NotNull(anomalies);
        Assert.Equal(testData.Rows, anomalies.Length);
    }

    [Fact]
    public void ComputeAnomalyScores_AfterTraining_ReturnsNonNegativeValues()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 8,
            LatentDim = 4,
            HiddenSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new LSTMVAE<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        // Create test data matrix
        var testData = new Matrix<double>(5, options.WindowSize);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < options.WindowSize; j++)
            {
                testData[i, j] = Math.Sin(2 * Math.PI * (i + j) / 10);
            }
        }

        var scores = model.ComputeAnomalyScores(testData);

        Assert.NotNull(scores);
        Assert.Equal(testData.Rows, scores.Length);

        // Reconstruction errors should be non-negative
        foreach (var score in scores)
        {
            Assert.True(score >= 0, "Anomaly score should be non-negative");
        }
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void Serialize_AndDeserialize_PreservesModel()
    {
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 8,
            LatentDim = 4,
            HiddenSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new LSTMVAE<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        var serialized = model.Serialize();
        Assert.NotNull(serialized);
        Assert.NotEmpty(serialized);

        var deserializedModel = new LSTMVAE<double>(options);
        deserializedModel.Deserialize(serialized);

        var input = new Vector<double>(options.WindowSize);
        for (int i = 0; i < options.WindowSize; i++)
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
        var options = new LSTMVAEOptions<float>
        {
            WindowSize = 8,
            LatentDim = 4,
            HiddenSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new LSTMVAE<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> inputs, Vector<double> targets) GenerateSyntheticData(int numSamples, int windowSize)
    {
        var inputs = new Matrix<double>(numSamples, windowSize);
        var targets = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            double baseValue = i * 0.1;
            for (int j = 0; j < windowSize; j++)
            {
                inputs[i, j] = Math.Sin(2 * Math.PI * (baseValue + j) / 10) + 0.1 * (j % 3);
            }
            // Target is the next value after the window
            targets[i] = Math.Sin(2 * Math.PI * (baseValue + windowSize) / 10) + 0.1 * (windowSize % 3);
        }

        return (inputs, targets);
    }

    #endregion
}
