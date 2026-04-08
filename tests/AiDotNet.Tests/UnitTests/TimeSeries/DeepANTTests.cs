using AiDotNet.TimeSeries.AnomalyDetection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Unit tests for DeepANT (Deep Learning for Anomaly Detection in Time Series).
/// </summary>
public class DeepANTTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultOptions_CreatesValidModel()
    {
        var model = new DeepANT<double>();

        Assert.NotNull(model);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidModel()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 20,
            LearningRate = 0.0001,
            Epochs = 10,
            BatchSize = 16
        };

        var model = new DeepANT<double>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Training Tests

    [Fact]
    public void Train_WithValidData_CompletesWithoutError()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new DeepANT<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    [Fact]
    public void Train_WithMinimalData_CompletesWithoutError()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 4,
            LearningRate = 0.001,
            Epochs = 1,
            BatchSize = 1
        };

        var model = new DeepANT<double>(options);
        var trainingData = GenerateSyntheticData(20, options.WindowSize);

        var exception = Record.Exception(() => model.Train(trainingData.inputs, trainingData.targets));

        Assert.Null(exception);
    }

    #endregion

    #region Prediction Tests

    [Fact]
    public void PredictSingle_AfterTraining_ReturnsFiniteValue()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new DeepANT<double>(options);
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
        var options = new DeepANTOptions<double>
        {
            WindowSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new DeepANT<double>(options);
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
        var options = new DeepANTOptions<double>
        {
            WindowSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new DeepANT<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        // Create test data longer than window size
        var testData = new Vector<double>(20);
        for (int i = 0; i < testData.Length; i++)
        {
            testData[i] = Math.Sin(2 * Math.PI * i / 10);
        }

        var anomalies = model.DetectAnomalies(testData);

        Assert.NotNull(anomalies);
        Assert.Equal(testData.Length - options.WindowSize, anomalies.Length);
    }

    [Fact]
    public void DetectAnomalies_WithDataShorterThanWindowSize_ThrowsArgumentException()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 10
        };

        var model = new DeepANT<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        // Create data shorter than window size
        var shortData = new Vector<double>(5);
        for (int i = 0; i < shortData.Length; i++)
        {
            shortData[i] = i * 0.1;
        }

        Assert.Throws<ArgumentException>(() => model.DetectAnomalies(shortData));
    }

    [Fact]
    public void ComputeAnomalyScores_AfterTraining_ReturnsValidScores()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new DeepANT<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        // Create test data
        var testData = new Vector<double>(20);
        for (int i = 0; i < testData.Length; i++)
        {
            testData[i] = Math.Sin(2 * Math.PI * i / 10);
        }

        var scores = model.ComputeAnomalyScores(testData);

        Assert.NotNull(scores);
        Assert.Equal(testData.Length - options.WindowSize, scores.Length);

        // Scores should be non-negative
        foreach (var score in scores)
        {
            Assert.True(score >= 0, "Anomaly score should be non-negative");
        }
    }

    [Fact]
    public void ComputeAnomalyScores_WithDataShorterThanWindowSize_ThrowsArgumentException()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 10
        };

        var model = new DeepANT<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        var shortData = new Vector<double>(5);

        Assert.Throws<ArgumentException>(() => model.ComputeAnomalyScores(shortData));
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void Serialize_AndDeserialize_PreservesModel()
    {
        var options = new DeepANTOptions<double>
        {
            WindowSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new DeepANT<double>(options);
        var trainingData = GenerateSyntheticData(50, options.WindowSize);
        model.Train(trainingData.inputs, trainingData.targets);

        var serialized = model.Serialize();
        Assert.NotNull(serialized);
        Assert.NotEmpty(serialized);

        var deserializedModel = new DeepANT<double>(options);
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
        var options = new DeepANTOptions<float>
        {
            WindowSize = 8,
            LearningRate = 0.001,
            Epochs = 2,
            BatchSize = 2
        };

        var model = new DeepANT<float>(options);

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
