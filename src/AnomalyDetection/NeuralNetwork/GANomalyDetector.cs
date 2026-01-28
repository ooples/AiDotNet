using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements GANomaly for anomaly detection using GAN-based reconstruction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GANomaly learns to encode, decode, and re-encode data.
/// Anomalies are detected when the encoding of the original differs from the
/// encoding of the reconstruction, indicating the model cannot properly represent the data.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Encoder maps input to latent space z
/// 2. Decoder reconstructs from z
/// 3. Second encoder re-encodes the reconstruction
/// 4. Anomaly score is the difference between original encoding and re-encoding
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Image anomaly detection
/// - When reconstruction error alone is insufficient
/// - Semi-supervised anomaly detection with only normal examples
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Latent dimensions: 32
/// - Hidden dimensions: 64
/// - Epochs: 100
/// - Learning rate: 0.0002
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Akcay, S., Atapour-Abarghouei, A., and Breckon, T. P. (2018).
/// "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training." ACCV.
/// </para>
/// </remarks>
public class GANomalyDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Encoder weights
    private Matrix<T>? _encW1;
    private Vector<T>? _encB1;
    private Matrix<T>? _encW2;
    private Vector<T>? _encB2;

    // Decoder weights
    private Matrix<T>? _decW1;
    private Vector<T>? _decB1;
    private Matrix<T>? _decW2;
    private Vector<T>? _decB2;

    // Re-encoder weights (separate encoder for reconstruction)
    private Matrix<T>? _reEncW1;
    private Vector<T>? _reEncB1;
    private Matrix<T>? _reEncW2;
    private Vector<T>? _reEncB2;

    private int _inputDim;

    // Normalization parameters
    private Vector<T>? _dataMeans;
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the latent dimensions.
    /// </summary>
    public int LatentDim => _latentDim;

    /// <summary>
    /// Gets the hidden dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Creates a new GANomaly anomaly detector.
    /// </summary>
    /// <param name="latentDim">Dimensions of latent space. Default is 32.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.0002.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public GANomalyDetector(int latentDim = 32, int hiddenDim = 64, int epochs = 100,
        double learningRate = 0.0002, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (latentDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(latentDim),
                "Latent dimensions must be at least 1. Recommended is 32.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.0002.");
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

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize weights
        InitializeWeights();

        // Train
        Train(normalizedData);

        // Calculate scores for training data
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
        double scale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _latentDim));
        double scale3 = Math.Sqrt(2.0 / (_latentDim + _hiddenDim));
        double scale4 = Math.Sqrt(2.0 / (_hiddenDim + _inputDim));

        // Encoder: input -> hidden -> latent
        _encW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _encB1 = InitializeVector(_hiddenDim);
        _encW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encB2 = InitializeVector(_latentDim);

        // Decoder: latent -> hidden -> input
        _decW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decB1 = InitializeVector(_hiddenDim);
        _decW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decB2 = InitializeVector(_inputDim);

        // Re-encoder: input -> hidden -> latent
        _reEncW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _reEncB1 = InitializeVector(_hiddenDim);
        _reEncW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _reEncB2 = InitializeVector(_latentDim);
    }

    private Matrix<T> InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
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

    private void Train(Matrix<T> data)
    {
        int n = data.Rows;
        int batchSize = Math.Min(32, n);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Accumulate gradients (use double for intermediate computation)
                var gradients = InitializeGradients();

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var x = data.GetRow(idx);

                    // Forward pass
                    var z = Encode(x);
                    var xRecon = Decode(z);
                    var zRecon = ReEncode(xRecon);

                    // Compute gradients
                    AccumulateGradients(x, z, xRecon, zRecon, gradients);
                }

                // Update weights
                UpdateWeights(gradients, actualBatchSize);
            }
        }
    }

    private Vector<T> Encode(Vector<T> x)
    {
        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var encB2 = _encB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null)
        {
            throw new InvalidOperationException("Encoder weights not initialized.");
        }

        // Layer 1
        var h = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = encB1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], encW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var z = new Vector<T>(_latentDim);
        for (int j = 0; j < _latentDim; j++)
        {
            T sum = encB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h[i], encW2[i, j]));
            }
            z[j] = sum;
        }

        return z;
    }

    private Vector<T> Decode(Vector<T> z)
    {
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var decB2 = _decB2;

        if (decW1 == null || decB1 == null || decW2 == null || decB2 == null)
        {
            throw new InvalidOperationException("Decoder weights not initialized.");
        }

        // Layer 1
        var h = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = decB1[j];
            for (int i = 0; i < _latentDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(z[i], decW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var xRecon = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = decB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h[i], decW2[i, j]));
            }
            xRecon[j] = sum;
        }

        return xRecon;
    }

    private Vector<T> ReEncode(Vector<T> xRecon)
    {
        var reEncW1 = _reEncW1;
        var reEncB1 = _reEncB1;
        var reEncW2 = _reEncW2;
        var reEncB2 = _reEncB2;

        if (reEncW1 == null || reEncB1 == null || reEncW2 == null || reEncB2 == null)
        {
            throw new InvalidOperationException("Re-encoder weights not initialized.");
        }

        // Layer 1
        var h = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = reEncB1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(xRecon[i], reEncW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var zRecon = new Vector<T>(_latentDim);
        for (int j = 0; j < _latentDim; j++)
        {
            T sum = reEncB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h[i], reEncW2[i, j]));
            }
            zRecon[j] = sum;
        }

        return zRecon;
    }

    private static double LeakyReLU(double x, double alpha = 0.2)
    {
        return x >= 0 ? x : alpha * x;
    }

    private GradientAccumulators InitializeGradients()
    {
        return new GradientAccumulators
        {
            encW1 = new double[_inputDim, _hiddenDim],
            encB1 = new double[_hiddenDim],
            encW2 = new double[_hiddenDim, _latentDim],
            encB2 = new double[_latentDim],
            decW1 = new double[_latentDim, _hiddenDim],
            decB1 = new double[_hiddenDim],
            decW2 = new double[_hiddenDim, _inputDim],
            decB2 = new double[_inputDim],
            reEncW1 = new double[_inputDim, _hiddenDim],
            reEncB1 = new double[_hiddenDim],
            reEncW2 = new double[_hiddenDim, _latentDim],
            reEncB2 = new double[_latentDim]
        };
    }

    private class GradientAccumulators
    {
        public required double[,] encW1 { get; init; }
        public required double[] encB1 { get; init; }
        public required double[,] encW2 { get; init; }
        public required double[] encB2 { get; init; }
        public required double[,] decW1 { get; init; }
        public required double[] decB1 { get; init; }
        public required double[,] decW2 { get; init; }
        public required double[] decB2 { get; init; }
        public required double[,] reEncW1 { get; init; }
        public required double[] reEncB1 { get; init; }
        public required double[,] reEncW2 { get; init; }
        public required double[] reEncB2 { get; init; }
    }

    private void AccumulateGradients(Vector<T> x, Vector<T> z, Vector<T> xRecon, Vector<T> zRecon,
        GradientAccumulators gradients)
    {
        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var reEncW1 = _reEncW1;
        var reEncB1 = _reEncB1;
        var reEncW2 = _reEncW2;

        if (encW1 == null || encB1 == null || encW2 == null ||
            decW1 == null || decB1 == null || decW2 == null ||
            reEncW1 == null || reEncB1 == null || reEncW2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        // Weights for the two losses
        double reconWeight = 1.0;
        double latentWeight = 1.0;

        // === Forward pass with caching ===
        // Encoder layer 1
        var encH1Pre = new double[_hiddenDim];
        var encH1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            encH1Pre[j] = NumOps.ToDouble(encB1[j]);
            for (int i = 0; i < _inputDim; i++)
            {
                encH1Pre[j] += NumOps.ToDouble(x[i]) * NumOps.ToDouble(encW1[i, j]);
            }
            encH1[j] = LeakyReLU(encH1Pre[j]);
        }

        // Decoder layer 1
        var decH1Pre = new double[_hiddenDim];
        var decH1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            decH1Pre[j] = NumOps.ToDouble(decB1[j]);
            for (int i = 0; i < _latentDim; i++)
            {
                decH1Pre[j] += NumOps.ToDouble(z[i]) * NumOps.ToDouble(decW1[i, j]);
            }
            decH1[j] = LeakyReLU(decH1Pre[j]);
        }

        // Re-encoder layer 1
        var reEncH1Pre = new double[_hiddenDim];
        var reEncH1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            reEncH1Pre[j] = NumOps.ToDouble(reEncB1[j]);
            for (int i = 0; i < _inputDim; i++)
            {
                reEncH1Pre[j] += NumOps.ToDouble(xRecon[i]) * NumOps.ToDouble(reEncW1[i, j]);
            }
            reEncH1[j] = LeakyReLU(reEncH1Pre[j]);
        }

        // === Backprop for reconstruction loss: ||x - xRecon||^2 ===
        var dXRecon = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            dXRecon[j] = reconWeight * 2 * (NumOps.ToDouble(xRecon[j]) - NumOps.ToDouble(x[j]));
        }

        // Backprop through decoder layer 2
        var dDecH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                gradients.decW2[i, j] += decH1[i] * dXRecon[j];
                dDecH1[i] += NumOps.ToDouble(decW2[i, j]) * dXRecon[j];
            }
        }
        for (int j = 0; j < _inputDim; j++)
        {
            gradients.decB2[j] += dXRecon[j];
        }

        // LeakyReLU derivative for decoder h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (decH1Pre[i] < 0) dDecH1[i] *= 0.2;
        }

        // Backprop through decoder layer 1
        var dZ_fromRecon = new double[_latentDim];
        for (int i = 0; i < _latentDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                gradients.decW1[i, j] += NumOps.ToDouble(z[i]) * dDecH1[j];
                dZ_fromRecon[i] += NumOps.ToDouble(decW1[i, j]) * dDecH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            gradients.decB1[j] += dDecH1[j];
        }

        // === Backprop for latent consistency loss: ||z - zRecon||^2 ===
        var dZ_fromLatent = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            dZ_fromLatent[j] = latentWeight * 2 * (NumOps.ToDouble(z[j]) - NumOps.ToDouble(zRecon[j]));
        }

        var dZRecon = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            dZRecon[j] = latentWeight * 2 * (NumOps.ToDouble(zRecon[j]) - NumOps.ToDouble(z[j]));
        }

        // Backprop through re-encoder layer 2
        var dReEncH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _latentDim; j++)
            {
                gradients.reEncW2[i, j] += reEncH1[i] * dZRecon[j];
                dReEncH1[i] += NumOps.ToDouble(reEncW2[i, j]) * dZRecon[j];
            }
        }
        for (int j = 0; j < _latentDim; j++)
        {
            gradients.reEncB2[j] += dZRecon[j];
        }

        // LeakyReLU derivative for re-encoder h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (reEncH1Pre[i] < 0) dReEncH1[i] *= 0.2;
        }

        // Backprop through re-encoder layer 1
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                gradients.reEncW1[i, j] += NumOps.ToDouble(xRecon[i]) * dReEncH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            gradients.reEncB1[j] += dReEncH1[j];
        }

        // === Backprop through encoder ===
        var dZ = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            dZ[j] = dZ_fromRecon[j] + dZ_fromLatent[j];
        }

        // Backprop through encoder layer 2
        var dEncH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _latentDim; j++)
            {
                gradients.encW2[i, j] += encH1[i] * dZ[j];
                dEncH1[i] += NumOps.ToDouble(encW2[i, j]) * dZ[j];
            }
        }
        for (int j = 0; j < _latentDim; j++)
        {
            gradients.encB2[j] += dZ[j];
        }

        // LeakyReLU derivative for encoder h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (encH1Pre[i] < 0) dEncH1[i] *= 0.2;
        }

        // Backprop through encoder layer 1
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                gradients.encW1[i, j] += NumOps.ToDouble(x[i]) * dEncH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            gradients.encB1[j] += dEncH1[j];
        }
    }

    private void UpdateWeights(GradientAccumulators gradients, int batchSize)
    {
        double lr = _learningRate / batchSize;

        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var encB2 = _encB2;
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var decB2 = _decB2;
        var reEncW1 = _reEncW1;
        var reEncB1 = _reEncB1;
        var reEncW2 = _reEncW2;
        var reEncB2 = _reEncB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            reEncW1 == null || reEncB1 == null || reEncW2 == null || reEncB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        UpdateMatrixWeights(encW1, gradients.encW1, lr);
        UpdateVectorWeights(encB1, gradients.encB1, lr);
        UpdateMatrixWeights(encW2, gradients.encW2, lr);
        UpdateVectorWeights(encB2, gradients.encB2, lr);
        UpdateMatrixWeights(decW1, gradients.decW1, lr);
        UpdateVectorWeights(decB1, gradients.decB1, lr);
        UpdateMatrixWeights(decW2, gradients.decW2, lr);
        UpdateVectorWeights(decB2, gradients.decB2, lr);
        UpdateMatrixWeights(reEncW1, gradients.reEncW1, lr);
        UpdateVectorWeights(reEncB1, gradients.reEncB1, lr);
        UpdateMatrixWeights(reEncW2, gradients.reEncW2, lr);
        UpdateVectorWeights(reEncB2, gradients.reEncB2, lr);
    }

    private void UpdateMatrixWeights(Matrix<T> w, double[,] grad, double lr)
    {
        for (int i = 0; i < w.Rows; i++)
        {
            for (int j = 0; j < w.Columns; j++)
            {
                w[i, j] = NumOps.Subtract(w[i, j], NumOps.FromDouble(lr * grad[i, j]));
            }
        }
    }

    private void UpdateVectorWeights(Vector<T> w, double[] grad, double lr)
    {
        for (int i = 0; i < w.Length; i++)
        {
            w[i] = NumOps.Subtract(w[i], NumOps.FromDouble(lr * grad[i]));
        }
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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Normalize
            var x = new Vector<T>(_inputDim);
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(X[i, j], dataMeans[j]);
                x[j] = NumOps.Divide(diff, dataStds[j]);
            }

            // Encode, decode, re-encode
            var z = Encode(x);
            var xRecon = Decode(z);
            var zRecon = ReEncode(xRecon);

            // Anomaly score: ||z - zRecon||^2 (latent consistency)
            T score = NumOps.Zero;
            for (int j = 0; j < _latentDim; j++)
            {
                T diff = NumOps.Subtract(z[j], zRecon[j]);
                score = NumOps.Add(score, NumOps.Multiply(diff, diff));
            }

            scores[i] = score;
        }

        return scores;
    }
}
