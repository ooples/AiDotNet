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
    private double[,]? _encW1;
    private double[]? _encB1;
    private double[,]? _encW2;
    private double[]? _encB2;

    // Decoder weights
    private double[,]? _decW1;
    private double[]? _decB1;
    private double[,]? _decW2;
    private double[]? _decB2;

    // Re-encoder weights (separate encoder for reconstruction)
    private double[,]? _reEncW1;
    private double[]? _reEncB1;
    private double[,]? _reEncW2;
    private double[]? _reEncB2;

    private int _inputDim;

    // Normalization parameters
    private double[]? _dataMeans;
    private double[]? _dataStds;

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
        var (normalizedData, means, stds) = NormalizeData(data);
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

    private (double[][] normalized, double[] means, double[] stds) NormalizeData(double[][] data)
    {
        int n = data.Length;
        int d = data[0].Length;

        var means = new double[d];
        var stds = new double[d];

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                means[j] += data[i][j];
            }
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                stds[j] += Math.Pow(data[i][j] - means[j], 2);
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

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
        double scale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _latentDim));
        double scale3 = Math.Sqrt(2.0 / (_latentDim + _hiddenDim));
        double scale4 = Math.Sqrt(2.0 / (_hiddenDim + _inputDim));

        // Encoder: input -> hidden -> latent
        _encW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _encB1 = new double[_hiddenDim];
        _encW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encB2 = new double[_latentDim];

        // Decoder: latent -> hidden -> input
        _decW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decB1 = new double[_hiddenDim];
        _decW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decB2 = new double[_inputDim];

        // Re-encoder: input -> hidden -> latent
        _reEncW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _reEncB1 = new double[_hiddenDim];
        _reEncW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _reEncB2 = new double[_latentDim];
    }

    private double[,] InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                matrix[i, j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
            }
        }
        return matrix;
    }

    private void Train(double[][] data)
    {
        int n = data.Length;
        int batchSize = Math.Min(32, n);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
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
                    var z = Encode(x);
                    var xRecon = Decode(z);
                    var zRecon = ReEncode(xRecon);

                    // Compute gradients for encoder-decoder reconstruction loss
                    // and latent consistency loss (z vs zRecon)
                    AccumulateGradients(x, z, xRecon, zRecon, gradients);
                }

                // Update weights
                UpdateWeights(gradients, actualBatchSize);
            }
        }
    }

    private double[] Encode(double[] x)
    {
        // Layer 1
        var h = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h[j] = _encB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                h[j] += x[i] * _encW1![i, j];
            }
            h[j] = LeakyReLU(h[j]);
        }

        // Layer 2
        var z = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            z[j] = _encB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                z[j] += h[i] * _encW2![i, j];
            }
        }

        return z;
    }

    private double[] Decode(double[] z)
    {
        // Layer 1
        var h = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h[j] = _decB1![j];
            for (int i = 0; i < _latentDim; i++)
            {
                h[j] += z[i] * _decW1![i, j];
            }
            h[j] = LeakyReLU(h[j]);
        }

        // Layer 2
        var xRecon = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            xRecon[j] = _decB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                xRecon[j] += h[i] * _decW2![i, j];
            }
        }

        return xRecon;
    }

    private double[] ReEncode(double[] xRecon)
    {
        // Layer 1
        var h = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h[j] = _reEncB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                h[j] += xRecon[i] * _reEncW1![i, j];
            }
            h[j] = LeakyReLU(h[j]);
        }

        // Layer 2
        var zRecon = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            zRecon[j] = _reEncB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                zRecon[j] += h[i] * _reEncW2![i, j];
            }
        }

        return zRecon;
    }

    private static double LeakyReLU(double x, double alpha = 0.2)
    {
        return x >= 0 ? x : alpha * x;
    }

    private Dictionary<string, object> InitializeGradients()
    {
        return new Dictionary<string, object>
        {
            ["encW1"] = new double[_inputDim, _hiddenDim],
            ["encB1"] = new double[_hiddenDim],
            ["encW2"] = new double[_hiddenDim, _latentDim],
            ["encB2"] = new double[_latentDim],
            ["decW1"] = new double[_latentDim, _hiddenDim],
            ["decB1"] = new double[_hiddenDim],
            ["decW2"] = new double[_hiddenDim, _inputDim],
            ["decB2"] = new double[_inputDim],
            ["reEncW1"] = new double[_inputDim, _hiddenDim],
            ["reEncB1"] = new double[_hiddenDim],
            ["reEncW2"] = new double[_hiddenDim, _latentDim],
            ["reEncB2"] = new double[_latentDim]
        };
    }

    private void AccumulateGradients(double[] x, double[] z, double[] xRecon, double[] zRecon,
        Dictionary<string, object> gradients)
    {
        // GANomaly has two losses:
        // 1. Reconstruction loss: ||x - xRecon||^2
        // 2. Latent consistency loss: ||z - zRecon||^2
        double reconWeight = 1.0;
        double latentWeight = 1.0;

        // Get gradient arrays
        var encW1Grad = (double[,])gradients["encW1"];
        var encB1Grad = (double[])gradients["encB1"];
        var encW2Grad = (double[,])gradients["encW2"];
        var encB2Grad = (double[])gradients["encB2"];
        var decW1Grad = (double[,])gradients["decW1"];
        var decB1Grad = (double[])gradients["decB1"];
        var decW2Grad = (double[,])gradients["decW2"];
        var decB2Grad = (double[])gradients["decB2"];
        var reEncW1Grad = (double[,])gradients["reEncW1"];
        var reEncB1Grad = (double[])gradients["reEncB1"];
        var reEncW2Grad = (double[,])gradients["reEncW2"];
        var reEncB2Grad = (double[])gradients["reEncB2"];

        // === Forward pass with caching ===
        // Encoder layer 1
        var encH1Pre = new double[_hiddenDim];
        var encH1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            encH1Pre[j] = _encB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                encH1Pre[j] += x[i] * _encW1![i, j];
            }
            encH1[j] = LeakyReLU(encH1Pre[j]);
        }

        // Decoder layer 1
        var decH1Pre = new double[_hiddenDim];
        var decH1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            decH1Pre[j] = _decB1![j];
            for (int i = 0; i < _latentDim; i++)
            {
                decH1Pre[j] += z[i] * _decW1![i, j];
            }
            decH1[j] = LeakyReLU(decH1Pre[j]);
        }

        // Re-encoder layer 1
        var reEncH1Pre = new double[_hiddenDim];
        var reEncH1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            reEncH1Pre[j] = _reEncB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                reEncH1Pre[j] += xRecon[i] * _reEncW1![i, j];
            }
            reEncH1[j] = LeakyReLU(reEncH1Pre[j]);
        }

        // === Backprop for reconstruction loss: ||x - xRecon||^2 ===
        // Gradient w.r.t. xRecon
        var dXRecon = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            dXRecon[j] = reconWeight * 2 * (xRecon[j] - x[j]);
        }

        // Backprop through decoder layer 2
        var dDecH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                decW2Grad[i, j] += decH1[i] * dXRecon[j];
                dDecH1[i] += _decW2![i, j] * dXRecon[j];
            }
        }
        for (int j = 0; j < _inputDim; j++)
        {
            decB2Grad[j] += dXRecon[j];
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
                decW1Grad[i, j] += z[i] * dDecH1[j];
                dZ_fromRecon[i] += _decW1![i, j] * dDecH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            decB1Grad[j] += dDecH1[j];
        }

        // === Backprop for latent consistency loss: ||z - zRecon||^2 ===
        // Gradient w.r.t. z
        var dZ_fromLatent = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            dZ_fromLatent[j] = latentWeight * 2 * (z[j] - zRecon[j]);
        }

        // Gradient w.r.t. zRecon
        var dZRecon = new double[_latentDim];
        for (int j = 0; j < _latentDim; j++)
        {
            dZRecon[j] = latentWeight * 2 * (zRecon[j] - z[j]);
        }

        // Backprop through re-encoder layer 2
        var dReEncH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _latentDim; j++)
            {
                reEncW2Grad[i, j] += reEncH1[i] * dZRecon[j];
                dReEncH1[i] += _reEncW2![i, j] * dZRecon[j];
            }
        }
        for (int j = 0; j < _latentDim; j++)
        {
            reEncB2Grad[j] += dZRecon[j];
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
                reEncW1Grad[i, j] += xRecon[i] * dReEncH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            reEncB1Grad[j] += dReEncH1[j];
        }

        // === Backprop through encoder ===
        // Total gradient w.r.t. z
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
                encW2Grad[i, j] += encH1[i] * dZ[j];
                dEncH1[i] += _encW2![i, j] * dZ[j];
            }
        }
        for (int j = 0; j < _latentDim; j++)
        {
            encB2Grad[j] += dZ[j];
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
                encW1Grad[i, j] += x[i] * dEncH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            encB1Grad[j] += dEncH1[j];
        }
    }

    private void UpdateWeights(Dictionary<string, object> gradients, int batchSize)
    {
        double lr = _learningRate / batchSize;

        UpdateMatrix(_encW1!, (double[,])gradients["encW1"], lr);
        UpdateVector(_encB1!, (double[])gradients["encB1"], lr);
        UpdateMatrix(_encW2!, (double[,])gradients["encW2"], lr);
        UpdateVector(_encB2!, (double[])gradients["encB2"], lr);
        UpdateMatrix(_decW1!, (double[,])gradients["decW1"], lr);
        UpdateVector(_decB1!, (double[])gradients["decB1"], lr);
        UpdateMatrix(_decW2!, (double[,])gradients["decW2"], lr);
        UpdateVector(_decB2!, (double[])gradients["decB2"], lr);
        UpdateMatrix(_reEncW1!, (double[,])gradients["reEncW1"], lr);
        UpdateVector(_reEncB1!, (double[])gradients["reEncB1"], lr);
        UpdateMatrix(_reEncW2!, (double[,])gradients["reEncW2"], lr);
        UpdateVector(_reEncB2!, (double[])gradients["reEncB2"], lr);
    }

    private static void UpdateMatrix(double[,] w, double[,] grad, double lr)
    {
        for (int i = 0; i < w.GetLength(0); i++)
        {
            for (int j = 0; j < w.GetLength(1); j++)
            {
                w[i, j] -= lr * grad[i, j];
            }
        }
    }

    private static void UpdateVector(double[] w, double[] grad, double lr)
    {
        for (int i = 0; i < w.Length; i++)
        {
            w[i] -= lr * grad[i];
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
            var x = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                x[j] = (NumOps.ToDouble(X[i, j]) - dataMeans[j]) / dataStds[j];
            }

            // Encode, decode, re-encode
            var z = Encode(x);
            var xRecon = Decode(z);
            var zRecon = ReEncode(xRecon);

            // Anomaly score: ||z - zRecon||^2 (latent consistency)
            double score = 0;
            for (int j = 0; j < _latentDim; j++)
            {
                score += Math.Pow(z[j] - zRecon[j], 2);
            }

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }
}
