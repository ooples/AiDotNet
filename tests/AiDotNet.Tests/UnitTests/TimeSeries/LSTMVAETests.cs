using AiDotNet.TimeSeries.AnomalyDetection;
using Xunit;
using System.Threading.Tasks;
using System.Reflection;
using System.Collections.Generic;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Unit tests for LSTMVAE (LSTM Variational Autoencoder for Anomaly Detection).
/// </summary>
public class LSTMVAETests
{
    #region Constructor Tests

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithDefaultOptions_CreatesValidModel()
    {
        var model = new LSTMVAE<double>();

        Assert.NotNull(model);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithCustomOptions_CreatesValidModel()
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

    [Fact(Timeout = 60000)]
    public async Task Train_WithValidData_CompletesWithoutError()
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

    [Fact(Timeout = 60000)]
    public async Task Train_WithMinimalData_CompletesWithoutError()
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

    [Fact(Timeout = 60000)]
    public async Task Train_WithExtremeEncoderLogVariance_KeepsParametersFinite()
    {
        await Task.Yield();
        var options = new LSTMVAEOptions<double>
        {
            WindowSize = 2,
            LatentDim = 1,
            HiddenSize = 2,
            LearningRate = 1e-6,
            KLWeight = 0.001,
            Epochs = 1,
            BatchSize = 1,
        };
        var model = new LSTMVAE<double>(options);

        // Put the encoder in the state that motivated the clamp: raw log-variance is far beyond
        // double's exp range. The forward already clamps this value; the regression is that the KL
        // gradient must use the same bounded variance instead of evaluating exp(1000).
        var encoderField = typeof(LSTMVAE<double>).GetField("_encoder", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(encoderField);
        object? encoderValue = encoderField!.GetValue(model);
        Assert.NotNull(encoderValue);
        object encoder = encoderValue!;
        FieldInfo? logVarBiasField = encoder.GetType().GetField("_logVarBias", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(logVarBiasField);
        var logVarBias = Assert.IsType<Tensor<double>>(logVarBiasField!.GetValue(encoder));
        logVarBias[0] = 1000.0;

        string[] parameterFields = ["_weights", "_bias", "_meanWeights", "_meanBias", "_logVarWeights", "_logVarBias"];
        var parametersBefore = new Dictionary<string, double[]>();
        foreach (string fieldName in parameterFields)
        {
            FieldInfo? parameterField = encoder.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(parameterField);
            var parameter = Assert.IsType<Tensor<double>>(parameterField!.GetValue(encoder));
            parametersBefore[fieldName] = parameter.ToArray();
        }

        var inputs = new Matrix<double>(2, 2);
        inputs[0, 0] = 0.25;
        inputs[0, 1] = -0.5;
        inputs[1, 0] = -0.125;
        inputs[1, 1] = 0.75;
        model.Train(inputs, new Vector<double>(new[] { 0.0, 0.0 }));

        bool anyParameterMoved = false;
        foreach (string fieldName in parameterFields)
        {
            FieldInfo? parameterField = encoder.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(parameterField);
            var parameter = Assert.IsType<Tensor<double>>(parameterField!.GetValue(encoder));
            for (int i = 0; i < parameter.Length; i++)
            {
                Assert.True(!double.IsNaN(parameter[i]) && !double.IsInfinity(parameter[i]),
                    $"Encoder parameter {fieldName}[{i}] became non-finite after the bounded-log-variance step: {parameter[i]}.");
                anyParameterMoved |= parameter[i] != parametersBefore[fieldName][i];
            }
        }

        Assert.True(anyParameterMoved,
            "The extreme-log-variance training step stayed finite but did not update any encoder parameter.");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void DecoderBackward_MirrorsForwardLatentPaddingAndTruncation(int latentLength)
    {
        using var decoder = new LSTMDecoderTensor<double>(latentDim: 2, outputSize: 2, hiddenSize: 2);
        using var latent = new Tensor<double>([latentLength]);
        for (int i = 0; i < latent.Length; i++) latent[i] = 0.25 * (i + 1);

        var (output, hidden) = decoder.DecodeWithCache(latent);
        using (output)
        using (hidden)
        using (var dOutput = new Tensor<double>([2]))
        {
            dOutput[0] = 0.5;
            dOutput[1] = -0.25;
            using var dLatent = decoder.AccumulateGradients(latent, hidden, dOutput);

            Assert.Equal(2, dLatent.Length);
            Assert.All(dLatent.ToArray(), value =>
                Assert.True(!double.IsNaN(value) && !double.IsInfinity(value),
                    $"Decoder latent gradient was non-finite: {value}."));
        }
    }

    [Fact]
    public void DecoderAccumulateGradients_PublishesTheAccumulatedGradients()
    {
        // The decoder trains outside the autodiff tape, so nothing scatters into its gradient
        // surface. Without publishing, GetParameterGradients reported an empty "never computed"
        // vector after a step that had computed real gradients (PR #2295 review).
        using var decoder = new LSTMDecoderTensor<double>(latentDim: 2, outputSize: 2, hiddenSize: 2);
        using var latent = new Tensor<double>([2]);
        latent[0] = 0.25;
        latent[1] = -0.5;

        var (output, hidden) = decoder.DecodeWithCache(latent);
        using (output)
        using (hidden)
        using (var dOutput = new Tensor<double>([2]))
        {
            dOutput[0] = 0.5;
            dOutput[1] = -0.25;
            using (decoder.AccumulateGradients(latent, hidden, dOutput)) { }
            using (decoder.AccumulateGradients(latent, hidden, dOutput)) { }

            var gradients = decoder.GetParameterGradients();

            // [W_d 2x2 | b_d 2 | W_o 2x2 | b_o 2]: db_o = dOutput per example, summed over two.
            Assert.Equal(12, gradients.Length);
            Assert.Equal(1.0, gradients[10], 12);
            Assert.Equal(-0.5, gradients[11], 12);

            decoder.ResetGradients();
            Assert.Equal(0, decoder.GetParameterGradients().Length);
        }
    }

    [Fact]
    public void EncoderAccumulateGradients_PublishesTheAccumulatedGradients()
    {
        using var encoder = new LSTMEncoderTensor<double>(inputSize: 3, latentDim: 2, hiddenSize: 2);
        var input = new Vector<double>(new[] { 0.1, -0.2, 0.3 });

        var (mean, logVar, hidden) = encoder.EncodeWithCache(input);
        using (mean)
        using (logVar)
        using (hidden)
        using (var dMean = new Tensor<double>([2]))
        using (var dLogVar = new Tensor<double>([2]))
        {
            dMean[0] = 0.75;
            dMean[1] = -0.125;
            dLogVar[0] = 0.0625;
            dLogVar[1] = -1.5;
            encoder.AccumulateGradients(input, hidden, dMean, dLogVar);

            var gradients = encoder.GetParameterGradients();

            // [W 2x3 | b 2 | W_mu 2x2 | b_mu 2 | W_logVar 2x2 | b_logVar 2]
            Assert.Equal(20, gradients.Length);
            Assert.Equal(0.75, gradients[12], 12);
            Assert.Equal(-0.125, gradients[13], 12);
            Assert.Equal(0.0625, gradients[18], 12);
            Assert.Equal(-1.5, gradients[19], 12);
        }
    }
    #endregion

    #region Prediction Tests

    [Fact(Timeout = 60000)]
    public async Task PredictSingle_AfterTraining_ReturnsFiniteValue()
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

    [Fact(Timeout = 60000)]
    public async Task Predict_WithMatrix_ReturnsValidPredictions()
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

    [Fact(Timeout = 60000)]
    public async Task DetectAnomalies_AfterTraining_ReturnsValidResults()
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

    [Fact(Timeout = 60000)]
    public async Task ComputeAnomalyScores_AfterTraining_ReturnsNonNegativeValues()
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

    [Fact(Timeout = 60000)]
    public async Task Serialize_AndDeserialize_PreservesModel()
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

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithFloatType_CreatesValidModel()
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
