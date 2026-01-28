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
    private Matrix<T>? _encoderW1;
    private Vector<T>? _encoderB1;
    private Matrix<T>? _encoderWMean;
    private Vector<T>? _encoderBMean;
    private Matrix<T>? _encoderWLogVar;
    private Vector<T>? _encoderBLogVar;

    // Decoder weights
    private Matrix<T>? _decoderW1;
    private Vector<T>? _decoderB1;
    private Matrix<T>? _decoderW2;
    private Vector<T>? _decoderB2;

    // Normalization parameters
    private Vector<T>? _dataMeans;
    private Vector<T>? _dataStds;

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

        if (learningRate <= 0 || double.IsNaN(learningRate) || double.IsInfinity(learningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be a positive, finite number. Recommended is 0.001.");
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

        // Normalize data and store parameters
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize weights
        InitializeWeights();

        // Train VAE
        TrainVAE(normalizedData);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (Matrix<T> normalized, Vector<T> means, Vector<T> stds) NormalizeData(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        var means = new Vector<T>(d);
        var stds = new Vector<T>(d);

        for (int j = 0; j < d; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, data[i, j]);
            }
            means[j] = NumOps.Divide(sum, NumOps.FromDouble(n));

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            double stdVal = Math.Sqrt(NumOps.ToDouble(variance) / n);
            if (stdVal < 1e-10) stdVal = 1;
            stds[j] = NumOps.FromDouble(stdVal);
        }

        var normalized = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                normalized[i, j] = NumOps.Divide(diff, stds[j]);
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
        _encoderB1 = InitializeVector(_hiddenDim);
        _encoderWMean = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encoderBMean = InitializeVector(_latentDim);
        _encoderWLogVar = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encoderBLogVar = InitializeVector(_latentDim);

        // Decoder
        _decoderW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decoderB1 = InitializeVector(_hiddenDim);
        _decoderW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decoderB2 = InitializeVector(_inputDim);
    }

    private Matrix<T> InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Box-Muller transform
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double val = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
                matrix[i, j] = NumOps.FromDouble(val);
            }
        }
        return matrix;
    }

    private Vector<T> InitializeVector(int size)
    {
        var vector = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = NumOps.Zero;
        }
        return vector;
    }

    private void TrainVAE(Matrix<T> data)
    {
        int n = data.Rows;
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
                    var x = data.GetRow(idx);

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

    private (Vector<T> hidden, Vector<T> mean, Vector<T> logVar, Vector<T> z, Vector<T> reconstruction) Forward(Vector<T> x)
    {
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

        // Encoder: x -> hidden
        var hidden = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = encoderB1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], encoderW1[i, j]));
            }
            double reluVal = ReLU(NumOps.ToDouble(sum));
            hidden[j] = NumOps.FromDouble(reluVal);
        }

        // Encoder: hidden -> mean, logVar
        var mean = new Vector<T>(_latentDim);
        var logVar = new Vector<T>(_latentDim);
        for (int j = 0; j < _latentDim; j++)
        {
            T sumMean = encoderBMean[j];
            T sumLogVar = encoderBLogVar[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sumMean = NumOps.Add(sumMean, NumOps.Multiply(hidden[i], encoderWMean[i, j]));
                sumLogVar = NumOps.Add(sumLogVar, NumOps.Multiply(hidden[i], encoderWLogVar[i, j]));
            }
            mean[j] = sumMean;
            logVar[j] = sumLogVar;
        }

        // Reparameterization: z = mean + std * epsilon
        var z = new Vector<T>(_latentDim);
        for (int j = 0; j < _latentDim; j++)
        {
            double epsilon = GaussianRandom();
            double meanVal = NumOps.ToDouble(mean[j]);
            double logVarVal = NumOps.ToDouble(logVar[j]);
            double zVal = meanVal + Math.Exp(0.5 * logVarVal) * epsilon;
            z[j] = NumOps.FromDouble(zVal);
        }

        // Decoder: z -> hidden2
        var hidden2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = decoderB1[j];
            for (int i = 0; i < _latentDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(z[i], decoderW1[i, j]));
            }
            double reluVal = ReLU(NumOps.ToDouble(sum));
            hidden2[j] = NumOps.FromDouble(reluVal);
        }

        // Decoder: hidden2 -> reconstruction
        var reconstruction = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = decoderB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(hidden2[i], decoderW2[i, j]));
            }
            reconstruction[j] = sum;
        }

        return (hidden, mean, logVar, z, reconstruction);
    }

    private class GradientAccumulators
    {
        public required double[,] encoderW1 { get; init; }
        public required double[] encoderB1 { get; init; }
        public required double[,] encoderWMean { get; init; }
        public required double[] encoderBMean { get; init; }
        public required double[,] encoderWLogVar { get; init; }
        public required double[] encoderBLogVar { get; init; }
        public required double[,] decoderW1 { get; init; }
        public required double[] decoderB1 { get; init; }
        public required double[,] decoderW2 { get; init; }
        public required double[] decoderB2 { get; init; }
    }

    private GradientAccumulators InitializeGradients()
    {
        return new GradientAccumulators
        {
            encoderW1 = new double[_inputDim, _hiddenDim],
            encoderB1 = new double[_hiddenDim],
            encoderWMean = new double[_hiddenDim, _latentDim],
            encoderBMean = new double[_latentDim],
            encoderWLogVar = new double[_hiddenDim, _latentDim],
            encoderBLogVar = new double[_latentDim],
            decoderW1 = new double[_latentDim, _hiddenDim],
            decoderB1 = new double[_hiddenDim],
            decoderW2 = new double[_hiddenDim, _inputDim],
            decoderB2 = new double[_inputDim]
        };
    }

    private void AccumulateGradients(GradientAccumulators gradients, Vector<T> x,
        Vector<T> hidden, Vector<T> mean, Vector<T> logVar, Vector<T> z, Vector<T> reconstruction)
    {
        var decoderW2 = _decoderW2;
        var decoderW1 = _decoderW1;
        var encoderWMean = _encoderWMean;
        var encoderWLogVar = _encoderWLogVar;
        var encoderW1 = _encoderW1;

        if (decoderW2 == null || decoderW1 == null || encoderWMean == null ||
            encoderWLogVar == null || encoderW1 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        // Compute reconstruction loss gradient: d/dRecon MSE = 2 * (recon - x)
        var dReconstruction = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            dReconstruction[j] = 2 * (NumOps.ToDouble(reconstruction[j]) - NumOps.ToDouble(x[j]));
        }

        // Recompute hidden2 for backprop
        var decoderB1Local = _decoderB1;
        if (decoderB1Local == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var hidden2Pre = new double[_hiddenDim];
        var hidden2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            hidden2Pre[j] = NumOps.ToDouble(decoderB1Local[j]);
            for (int i = 0; i < _latentDim; i++)
            {
                hidden2Pre[j] += NumOps.ToDouble(z[i]) * NumOps.ToDouble(decoderW1[i, j]);
            }
            hidden2[j] = ReLU(hidden2Pre[j]);
        }

        // Backprop through decoder output layer
        var dHidden2 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                gradients.decoderW2[i, j] += hidden2[i] * dReconstruction[j];
                dHidden2[i] += NumOps.ToDouble(decoderW2[i, j]) * dReconstruction[j];
            }
        }
        for (int j = 0; j < _inputDim; j++)
        {
            gradients.decoderB2[j] += dReconstruction[j];
        }

        // ReLU derivative for hidden2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (hidden2Pre[i] <= 0) dHidden2[i] = 0;
        }

        // Backprop through decoder layer 1
        var dZ = new double[_latentDim];
        for (int i = 0; i < _latentDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                gradients.decoderW1[i, j] += NumOps.ToDouble(z[i]) * dHidden2[j];
                dZ[i] += NumOps.ToDouble(decoderW1[i, j]) * dHidden2[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            gradients.decoderB1[j] += dHidden2[j];
        }

        // Gradients through reparameterization: z = mean + exp(0.5*logVar) * epsilon
        // dL/dMean = dL/dZ
        // dL/dLogVar = dL/dZ * epsilon * 0.5 * exp(0.5*logVar)
        // For simplicity, we'll use the standard VAE gradient formulas

        // KL divergence gradient
        var dMean_KL = new double[_latentDim];
        var dLogVar_KL = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            double meanVal = NumOps.ToDouble(mean[j]);
            double logVarVal = NumOps.ToDouble(logVar[j]);
            dMean_KL[j] = meanVal;
            dLogVar_KL[j] = 0.5 * (Math.Exp(logVarVal) - 1);
        }

        // Reconstruction gradient through reparameterization
        var dMean_recon = new double[_latentDim];
        var dLogVar_recon = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            dMean_recon[j] = dZ[j];
            // Simplified: assume epsilon contribution averages out
            double logVarVal = NumOps.ToDouble(logVar[j]);
            dLogVar_recon[j] = dZ[j] * 0.5 * Math.Exp(0.5 * logVarVal);
        }

        // Total gradient
        var dMean = new double[_latentDim];
        var dLogVar = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            dMean[j] = dMean_recon[j] + dMean_KL[j];
            dLogVar[j] = dLogVar_recon[j] + dLogVar_KL[j];
        }

        // Backprop through encoder mean/logvar layers
        var dHidden = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _latentDim; j++)
            {
                gradients.encoderWMean[i, j] += NumOps.ToDouble(hidden[i]) * dMean[j];
                gradients.encoderWLogVar[i, j] += NumOps.ToDouble(hidden[i]) * dLogVar[j];
                dHidden[i] += NumOps.ToDouble(encoderWMean[i, j]) * dMean[j];
                dHidden[i] += NumOps.ToDouble(encoderWLogVar[i, j]) * dLogVar[j];
            }
        }
        for (int j = 0; j < _latentDim; j++)
        {
            gradients.encoderBMean[j] += dMean[j];
            gradients.encoderBLogVar[j] += dLogVar[j];
        }

        // Recompute hidden layer pre-activation
        var encoderB1Local = _encoderB1;
        if (encoderB1Local == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var hiddenPre = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            hiddenPre[j] = NumOps.ToDouble(encoderB1Local[j]);
            for (int i = 0; i < _inputDim; i++)
            {
                hiddenPre[j] += NumOps.ToDouble(x[i]) * NumOps.ToDouble(encoderW1[i, j]);
            }
        }

        // ReLU derivative for hidden
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (hiddenPre[i] <= 0) dHidden[i] = 0;
        }

        // Backprop through encoder layer 1
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                gradients.encoderW1[i, j] += NumOps.ToDouble(x[i]) * dHidden[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            gradients.encoderB1[j] += dHidden[j];
        }
    }

    private void UpdateWeights(GradientAccumulators gradients, int batchSize)
    {
        double lr = _learningRate / batchSize;

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
        UpdateMatrixWeights(encoderW1, gradients.encoderW1, lr);
        UpdateVectorWeights(encoderB1, gradients.encoderB1, lr);
        UpdateMatrixWeights(encoderWMean, gradients.encoderWMean, lr);
        UpdateVectorWeights(encoderBMean, gradients.encoderBMean, lr);
        UpdateMatrixWeights(encoderWLogVar, gradients.encoderWLogVar, lr);
        UpdateVectorWeights(encoderBLogVar, gradients.encoderBLogVar, lr);

        // Update decoder
        UpdateMatrixWeights(decoderW1, gradients.decoderW1, lr);
        UpdateVectorWeights(decoderB1, gradients.decoderB1, lr);
        UpdateMatrixWeights(decoderW2, gradients.decoderW2, lr);
        UpdateVectorWeights(decoderB2, gradients.decoderB2, lr);
    }

    private void UpdateMatrixWeights(Matrix<T> weights, double[,] grads, double lr)
    {
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                weights[i, j] = NumOps.Subtract(weights[i, j], NumOps.FromDouble(lr * grads[i, j]));
            }
        }
    }

    private void UpdateVectorWeights(Vector<T> weights, double[] grads, double lr)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.Subtract(weights[i], NumOps.FromDouble(lr * grads[i]));
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

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        if (dataMeans == null || dataStds == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters missing.");
        }

        int n = X.Rows;
        var scores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Normalize input using stored parameters
            var x = new Vector<T>(_inputDim);
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(X[i, j], dataMeans[j]);
                x[j] = NumOps.Divide(diff, dataStds[j]);
            }

            var (_, mean, logVar, _, reconstruction) = Forward(x);

            // Reconstruction error
            T reconError = NumOps.Zero;
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(x[j], reconstruction[j]);
                reconError = NumOps.Add(reconError, NumOps.Multiply(diff, diff));
            }
            double reconErrorVal = NumOps.ToDouble(reconError) / _inputDim;

            // KL divergence
            double klDiv = 0;
            for (int j = 0; j < _latentDim; j++)
            {
                double meanVal = NumOps.ToDouble(mean[j]);
                double logVarVal = NumOps.ToDouble(logVar[j]);
                klDiv += -0.5 * (1 + logVarVal - meanVal * meanVal - Math.Exp(logVarVal));
            }
            klDiv /= _latentDim;

            // Combined score
            double score = reconErrorVal + 0.1 * klDiv;
            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }
}
