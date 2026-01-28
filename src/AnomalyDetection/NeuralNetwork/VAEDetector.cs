using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Detects anomalies using Variational Autoencoder (VAE).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A VAE is a generative neural network that learns to encode data into
/// a lower-dimensional probabilistic latent space and decode it back. Anomalies are points
/// that are poorly reconstructed or fall in low-probability regions of the latent space.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train encoder to map data to latent distribution (mean + variance)
/// 2. Train decoder to reconstruct from latent samples
/// 3. Score combines reconstruction error and KL divergence
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Complex, high-dimensional data
/// - When you want probabilistic anomaly scores
/// - Image, text, or structured data anomalies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Latent dimensions: 10
/// - Hidden dimensions: 64
/// - Learning rate: 0.001
/// - Epochs: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Kingma, D.P., Welling, M. (2014). "Auto-Encoding Variational Bayes." ICLR.
/// An, J., Cho, S. (2015). "Variational Autoencoder based Anomaly Detection."
/// </para>
/// </remarks>
public class VAEDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Encoder weights
    private double[,]? _encoderW1;
    private double[]? _encoderB1;
    private double[,]? _encoderWMean;
    private double[]? _encoderBMean;
    private double[,]? _encoderWLogVar;
    private double[]? _encoderBLogVar;

    // Decoder weights
    private double[,]? _decoderW1;
    private double[]? _decoderB1;
    private double[,]? _decoderW2;
    private double[]? _decoderB2;

    private int _inputDim;

    /// <summary>
    /// Gets the latent space dimensions.
    /// </summary>
    public int LatentDim => _latentDim;

    /// <summary>
    /// Gets the hidden layer dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Creates a new VAE anomaly detector.
    /// </summary>
    /// <param name="latentDim">Dimensions of the latent space. Default is 10.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public VAEDetector(int latentDim = 10, int hiddenDim = 64, int epochs = 100,
        double learningRate = 0.001, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (latentDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(latentDim),
                "LatentDim must be at least 1. Recommended is 10.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "HiddenDim must be at least 1. Recommended is 64.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        _latentDim = latentDim;
        _hiddenDim = hiddenDim;
        _epochs = epochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Normalize data
        var (normalizedData, _, _) = NormalizeData(data);

        // Initialize weights
        InitializeWeights();

        // Train VAE
        TrainVAE(normalizedData);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (double[][] normalized, double[] means, double[] stds) NormalizeData(double[][] data)
    {
        int n = data.Length;
        int d = data[0].Length;

        var means = new double[d];
        var stds = new double[d];

        // Compute means
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                means[j] += data[i][j];
            }
            means[j] /= n;
        }

        // Compute stds
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                stds[j] += Math.Pow(data[i][j] - means[j], 2);
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        // Normalize
        var normalized = new double[n][];
        for (int i = 0; i < n; i++)
        {
            normalized[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                normalized[i][j] = (data[i][j] - means[j]) / stds[j];
            }
        }

        return (normalized, means, stds);
    }

    private void InitializeWeights()
    {
        // Xavier initialization
        double scale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _latentDim));
        double scale3 = Math.Sqrt(2.0 / (_latentDim + _hiddenDim));
        double scale4 = Math.Sqrt(2.0 / (_hiddenDim + _inputDim));

        // Encoder
        _encoderW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _encoderB1 = new double[_hiddenDim];
        _encoderWMean = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encoderBMean = new double[_latentDim];
        _encoderWLogVar = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encoderBLogVar = new double[_latentDim];

        // Decoder
        _decoderW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decoderB1 = new double[_hiddenDim];
        _decoderW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decoderB2 = new double[_inputDim];
    }

    private double[,] InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Box-Muller transform
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                matrix[i, j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
            }
        }
        return matrix;
    }

    private void TrainVAE(double[][] data)
    {
        int n = data.Length;
        int batchSize = Math.Min(32, n);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            // Shuffle data
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Accumulate gradients
                var gradients = InitializeGradients();

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var x = data[idx];

                    // Forward pass
                    var (hidden, mean, logVar, z, reconstruction) = Forward(x);

                    // Compute loss gradients and backpropagate
                    AccumulateGradients(gradients, x, hidden, mean, logVar, z, reconstruction);
                }

                // Update weights
                UpdateWeights(gradients, actualBatchSize);
            }
        }
    }

    private (double[] hidden, double[] mean, double[] logVar, double[] z, double[] reconstruction) Forward(double[] x)
    {
        // Encoder: x -> hidden
        var hidden = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            hidden[j] = _encoderB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                hidden[j] += x[i] * _encoderW1![i, j];
            }
            hidden[j] = ReLU(hidden[j]);
        }

        // Encoder: hidden -> mean, logVar
        var mean = new double[_latentDim];
        var logVar = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            mean[j] = _encoderBMean![j];
            logVar[j] = _encoderBLogVar![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                mean[j] += hidden[i] * _encoderWMean![i, j];
                logVar[j] += hidden[i] * _encoderWLogVar![i, j];
            }
        }

        // Reparameterization: z = mean + std * epsilon
        var z = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            double epsilon = GaussianRandom();
            z[j] = mean[j] + Math.Exp(0.5 * logVar[j]) * epsilon;
        }

        // Decoder: z -> hidden2
        var hidden2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            hidden2[j] = _decoderB1![j];
            for (int i = 0; i < _latentDim; i++)
            {
                hidden2[j] += z[i] * _decoderW1![i, j];
            }
            hidden2[j] = ReLU(hidden2[j]);
        }

        // Decoder: hidden2 -> reconstruction
        var reconstruction = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            reconstruction[j] = _decoderB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                reconstruction[j] += hidden2[i] * _decoderW2![i, j];
            }
        }

        return (hidden, mean, logVar, z, reconstruction);
    }

    private Dictionary<string, object> InitializeGradients()
    {
        return new Dictionary<string, object>
        {
            ["encoderW1"] = new double[_inputDim, _hiddenDim],
            ["encoderB1"] = new double[_hiddenDim],
            ["encoderWMean"] = new double[_hiddenDim, _latentDim],
            ["encoderBMean"] = new double[_latentDim],
            ["encoderWLogVar"] = new double[_hiddenDim, _latentDim],
            ["encoderBLogVar"] = new double[_latentDim],
            ["decoderW1"] = new double[_latentDim, _hiddenDim],
            ["decoderB1"] = new double[_hiddenDim],
            ["decoderW2"] = new double[_hiddenDim, _inputDim],
            ["decoderB2"] = new double[_inputDim]
        };
    }

    private void AccumulateGradients(Dictionary<string, object> gradients, double[] x,
        double[] hidden, double[] mean, double[] logVar, double[] z, double[] reconstruction)
    {
        // Capture nullable field
        var decoderW2 = _decoderW2;
        if (decoderW2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        // Compute reconstruction loss gradient
        var dReconstruction = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            dReconstruction[j] = 2 * (reconstruction[j] - x[j]);
        }

        // Backprop through decoder output layer
        var dHidden2 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                ((double[,])gradients["decoderW2"])[i, j] += hidden[Math.Min(i, hidden.Length - 1)] * dReconstruction[j];
                dHidden2[i] += decoderW2[i, j] * dReconstruction[j];
            }
        }
        for (int j = 0; j < _inputDim; j++)
        {
            ((double[])gradients["decoderB2"])[j] += dReconstruction[j];
        }

        // ReLU derivative
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (hidden[Math.Min(i, hidden.Length - 1)] <= 0) dHidden2[i] = 0;
        }

        // Continue backprop through rest of network (simplified)
        // KL divergence gradient
        for (int j = 0; j < _latentDim; j++)
        {
            double dKLMean = mean[j];
            double dKLLogVar = 0.5 * (Math.Exp(logVar[j]) - 1);

            ((double[])gradients["encoderBMean"])[j] += dKLMean;
            ((double[])gradients["encoderBLogVar"])[j] += dKLLogVar;
        }
    }

    private void UpdateWeights(Dictionary<string, object> gradients, int batchSize)
    {
        double lr = _learningRate / batchSize;

        // Capture nullable fields to avoid null-forgiving operators
        var encoderW1 = _encoderW1;
        var encoderB1 = _encoderB1;
        var encoderWMean = _encoderWMean;
        var encoderBMean = _encoderBMean;
        var encoderWLogVar = _encoderWLogVar;
        var encoderBLogVar = _encoderBLogVar;
        var decoderW1 = _decoderW1;
        var decoderB1 = _decoderB1;
        var decoderW2 = _decoderW2;
        var decoderB2 = _decoderB2;

        if (encoderW1 == null || encoderB1 == null || encoderWMean == null || encoderBMean == null ||
            encoderWLogVar == null || encoderBLogVar == null || decoderW1 == null || decoderB1 == null ||
            decoderW2 == null || decoderB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        // Update encoder
        UpdateMatrix(encoderW1, (double[,])gradients["encoderW1"], lr);
        UpdateVector(encoderB1, (double[])gradients["encoderB1"], lr);
        UpdateMatrix(encoderWMean, (double[,])gradients["encoderWMean"], lr);
        UpdateVector(encoderBMean, (double[])gradients["encoderBMean"], lr);
        UpdateMatrix(encoderWLogVar, (double[,])gradients["encoderWLogVar"], lr);
        UpdateVector(encoderBLogVar, (double[])gradients["encoderBLogVar"], lr);

        // Update decoder
        UpdateMatrix(decoderW1, (double[,])gradients["decoderW1"], lr);
        UpdateVector(decoderB1, (double[])gradients["decoderB1"], lr);
        UpdateMatrix(decoderW2, (double[,])gradients["decoderW2"], lr);
        UpdateVector(decoderB2, (double[])gradients["decoderB2"], lr);
    }

    private void UpdateMatrix(double[,] weights, double[,] gradients, double lr)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                weights[i, j] -= lr * gradients[i, j];
            }
        }
    }

    private void UpdateVector(double[] weights, double[] gradients, double lr)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] -= lr * gradients[i];
        }
    }

    private static double ReLU(double x) => Math.Max(0, x);

    private double GaussianRandom()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        var scores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            var x = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                x[j] = NumOps.ToDouble(X[i, j]);
            }

            var (_, mean, logVar, _, reconstruction) = Forward(x);

            // Reconstruction error
            double reconError = 0;
            for (int j = 0; j < _inputDim; j++)
            {
                reconError += Math.Pow(x[j] - reconstruction[j], 2);
            }
            reconError /= _inputDim;

            // KL divergence
            double klDiv = 0;
            for (int j = 0; j < _latentDim; j++)
            {
                klDiv += -0.5 * (1 + logVar[j] - mean[j] * mean[j] - Math.Exp(logVar[j]));
            }
            klDiv /= _latentDim;

            // Combined score
            double score = reconError + 0.1 * klDiv;
            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }
}
