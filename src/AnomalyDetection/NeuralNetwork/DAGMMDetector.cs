using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements DAGMM (Deep Autoencoding Gaussian Mixture Model) for anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DAGMM combines an autoencoder with a Gaussian Mixture Model
/// in an end-to-end trainable architecture. The autoencoder learns compressed representations,
/// and the GMM learns the density of normal data in the latent space.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compression network (autoencoder) learns latent representation
/// 2. Estimation network predicts GMM membership probabilities
/// 3. GMM models the distribution of latent codes + reconstruction features
/// 4. Anomaly score is the negative log-likelihood under the GMM
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Complex multivariate data
/// - When you want to model multiple modes of normal behavior
/// - When reconstruction error alone is insufficient
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Latent dimensions: 4
/// - Hidden dimensions: 64
/// - Number of mixtures: 4
/// - Epochs: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zong, B., et al. (2018).
/// "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection." ICLR.
/// </para>
/// </remarks>
public class DAGMMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _numMixtures;
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

    // Estimation network weights
    private Matrix<T>? _estW1;
    private Vector<T>? _estB1;
    private Matrix<T>? _estW2;
    private Vector<T>? _estB2;

    // GMM parameters (kept as double for numerical stability in probability computations)
    private double[]? _phi; // Mixture weights
    private double[][]? _mu; // Means
    private double[][][]? _sigma; // Covariances

    private int _inputDim;
    private int _zDim; // latent + reconstruction features

    // Normalization parameters
    private Vector<T>? _dataMeans;
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the latent dimensions.
    /// </summary>
    public int LatentDim => _latentDim;

    /// <summary>
    /// Gets the number of GMM components.
    /// </summary>
    public int NumMixtures => _numMixtures;

    /// <summary>
    /// Creates a new DAGMM anomaly detector.
    /// </summary>
    /// <param name="latentDim">Dimensions of latent space. Default is 4.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="numMixtures">Number of GMM components. Default is 4.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.0001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DAGMMDetector(int latentDim = 4, int hiddenDim = 64, int numMixtures = 4,
        int epochs = 100, double learningRate = 0.0001,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (latentDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(latentDim),
                "Latent dimensions must be at least 1. Recommended is 4.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (numMixtures < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numMixtures),
                "Number of mixtures must be at least 1. Recommended is 4.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (learningRate <= 0 || double.IsNaN(learningRate) || double.IsInfinity(learningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be a positive, finite number. Recommended is 0.0001.");
        }

        _latentDim = latentDim;
        _hiddenDim = hiddenDim;
        _numMixtures = numMixtures;
        _epochs = epochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;
        _zDim = _latentDim + 2; // latent + euclidean distance + cosine similarity

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize weights
        InitializeWeights();

        // Initialize GMM parameters
        InitializeGMM();

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

        // Compute means
        for (int j = 0; j < d; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, data[i, j]);
            }
            means[j] = NumOps.Divide(sum, NumOps.FromDouble(n));
        }

        // Compute standard deviations
        for (int j = 0; j < d; j++)
        {
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

        // Normalize
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
        double scale5 = Math.Sqrt(2.0 / (_zDim + _hiddenDim));
        double scale6 = Math.Sqrt(2.0 / (_hiddenDim + _numMixtures));

        // Encoder
        _encW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _encB1 = InitializeVector(_hiddenDim);
        _encW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encB2 = InitializeVector(_latentDim);

        // Decoder
        _decW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decB1 = InitializeVector(_hiddenDim);
        _decW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decB2 = InitializeVector(_inputDim);

        // Estimation network
        _estW1 = InitializeMatrix(_zDim, _hiddenDim, scale5);
        _estB1 = InitializeVector(_hiddenDim);
        _estW2 = InitializeMatrix(_hiddenDim, _numMixtures, scale6);
        _estB2 = InitializeVector(_numMixtures);
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

    private void InitializeGMM()
    {
        _phi = new double[_numMixtures];
        _mu = new double[_numMixtures][];
        _sigma = new double[_numMixtures][][];

        for (int k = 0; k < _numMixtures; k++)
        {
            _phi[k] = 1.0 / _numMixtures;
            _mu[k] = new double[_zDim];
            _sigma[k] = new double[_zDim][];

            for (int j = 0; j < _zDim; j++)
            {
                _mu[k][j] = (_random.NextDouble() - 0.5) * 0.1;
                _sigma[k][j] = new double[_zDim];
                _sigma[k][j][j] = 1.0; // Identity covariance
            }
        }
    }

    private void Train(Matrix<T> data)
    {
        int n = data.Rows;
        int batchSize = Math.Min(64, n);
        double lr = _learningRate;
        double clipValue = 5.0;

        // Capture weights
        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var encB2 = _encB2;
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var decB2 = _decB2;
        var estW1 = _estW1;
        var estB1 = _estB1;
        var estW2 = _estW2;
        var estB2 = _estB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            estW1 == null || estB1 == null || estW2 == null || estB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();
            var allZ = new List<double[]>();
            var allGamma = new List<double[]>();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Gradient accumulators (use double for numerical stability during accumulation)
                var dEncW1 = new double[_inputDim, _hiddenDim];
                var dEncB1 = new double[_hiddenDim];
                var dEncW2 = new double[_hiddenDim, _latentDim];
                var dEncB2 = new double[_latentDim];
                var dDecW1 = new double[_latentDim, _hiddenDim];
                var dDecB1 = new double[_hiddenDim];
                var dDecW2 = new double[_hiddenDim, _inputDim];
                var dDecB2 = new double[_inputDim];
                var dEstW1 = new double[_zDim, _hiddenDim];
                var dEstB1 = new double[_hiddenDim];
                var dEstW2 = new double[_hiddenDim, _numMixtures];
                var dEstB2 = new double[_numMixtures];

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];

                    // Extract input
                    var x = new Vector<T>(_inputDim);
                    for (int i = 0; i < _inputDim; i++)
                    {
                        x[i] = data[idx, i];
                    }

                    // Forward pass with caching
                    var (z, zc, xRecon, gamma, encH, decH, estH) = ForwardPassWithCache(x);

                    // Store for GMM update (convert to double arrays)
                    var zcDouble = new double[_zDim];
                    for (int i = 0; i < _zDim; i++)
                    {
                        zcDouble[i] = NumOps.ToDouble(zc[i]);
                    }
                    allZ.Add(zcDouble);

                    var gammaDouble = new double[_numMixtures];
                    for (int i = 0; i < _numMixtures; i++)
                    {
                        gammaDouble[i] = NumOps.ToDouble(gamma[i]);
                    }
                    allGamma.Add(gammaDouble);

                    // Compute reconstruction loss gradient: dL/dxRecon = 2 * (xRecon - x) / inputDim
                    var dXRecon = new Vector<T>(_inputDim);
                    for (int i = 0; i < _inputDim; i++)
                    {
                        T diff = NumOps.Subtract(xRecon[i], x[i]);
                        dXRecon[i] = NumOps.Multiply(NumOps.FromDouble(2.0 / _inputDim), diff);
                    }

                    // Backprop through decoder output layer
                    var dDecH = new double[_hiddenDim];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        double decHVal = NumOps.ToDouble(decH[i]);
                        for (int j = 0; j < _inputDim; j++)
                        {
                            double dXReconVal = NumOps.ToDouble(dXRecon[j]);
                            dDecW2[i, j] += decHVal * dXReconVal;
                            dDecH[i] += NumOps.ToDouble(decW2[i, j]) * dXReconVal;
                        }
                    }
                    for (int j = 0; j < _inputDim; j++)
                    {
                        dDecB2[j] += NumOps.ToDouble(dXRecon[j]);
                    }

                    // Tanh derivative for decoder hidden layer
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        double h = NumOps.ToDouble(decH[i]);
                        dDecH[i] *= (1 - h * h);
                    }

                    // Backprop through decoder hidden layer
                    var dZ = new double[_latentDim];
                    for (int i = 0; i < _latentDim; i++)
                    {
                        double zVal = NumOps.ToDouble(z[i]);
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dDecW1[i, j] += zVal * dDecH[j];
                            dZ[i] += NumOps.ToDouble(decW1[i, j]) * dDecH[j];
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        dDecB1[j] += dDecH[j];
                    }

                    // Tanh derivative for z
                    for (int i = 0; i < _latentDim; i++)
                    {
                        double zVal = NumOps.ToDouble(z[i]);
                        dZ[i] *= (1 - zVal * zVal);
                    }

                    // Backprop through encoder output layer
                    var dEncH = new double[_hiddenDim];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        double encHVal = NumOps.ToDouble(encH[i]);
                        for (int j = 0; j < _latentDim; j++)
                        {
                            dEncW2[i, j] += encHVal * dZ[j];
                            dEncH[i] += NumOps.ToDouble(encW2[i, j]) * dZ[j];
                        }
                    }
                    for (int j = 0; j < _latentDim; j++)
                    {
                        dEncB2[j] += dZ[j];
                    }

                    // Tanh derivative for encoder hidden layer
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        double h = NumOps.ToDouble(encH[i]);
                        dEncH[i] *= (1 - h * h);
                    }

                    // Backprop through encoder hidden layer
                    for (int i = 0; i < _inputDim; i++)
                    {
                        double xVal = NumOps.ToDouble(x[i]);
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dEncW1[i, j] += xVal * dEncH[j];
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        dEncB1[j] += dEncH[j];
                    }

                    // Backprop through estimation network
                    var dGamma = new double[_numMixtures];
                    for (int k = 0; k < _numMixtures; k++)
                    {
                        double g = gammaDouble[k];
                        dGamma[k] = g * (1 - g) * 0.1;
                    }

                    var dEstH = new double[_hiddenDim];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        double estHVal = NumOps.ToDouble(estH[i]);
                        for (int j = 0; j < _numMixtures; j++)
                        {
                            dEstW2[i, j] += estHVal * dGamma[j];
                            dEstH[i] += NumOps.ToDouble(estW2[i, j]) * dGamma[j];
                        }
                    }
                    for (int j = 0; j < _numMixtures; j++)
                    {
                        dEstB2[j] += dGamma[j];
                    }

                    // Tanh derivative for estimation hidden
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        double h = NumOps.ToDouble(estH[i]);
                        dEstH[i] *= (1 - h * h);
                    }

                    for (int i = 0; i < _zDim; i++)
                    {
                        double zcVal = zcDouble[i];
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dEstW1[i, j] += zcVal * dEstH[j];
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        dEstB1[j] += dEstH[j];
                    }
                }

                // Apply weight updates with gradient clipping
                double scale = 1.0 / actualBatchSize;
                ApplyGradients(encW1, dEncW1, lr * scale, clipValue);
                ApplyGradients(encB1, dEncB1, lr * scale, clipValue);
                ApplyGradients(encW2, dEncW2, lr * scale, clipValue);
                ApplyGradients(encB2, dEncB2, lr * scale, clipValue);
                ApplyGradients(decW1, dDecW1, lr * scale, clipValue);
                ApplyGradients(decB1, dDecB1, lr * scale, clipValue);
                ApplyGradients(decW2, dDecW2, lr * scale, clipValue);
                ApplyGradients(decB2, dDecB2, lr * scale, clipValue);
                ApplyGradients(estW1, dEstW1, lr * scale, clipValue);
                ApplyGradients(estB1, dEstB1, lr * scale, clipValue);
                ApplyGradients(estW2, dEstW2, lr * scale, clipValue);
                ApplyGradients(estB2, dEstB2, lr * scale, clipValue);
            }

            // Update GMM parameters using all samples
            if (allZ.Count > 0)
            {
                UpdateGMM(allZ.ToArray(), allGamma.ToArray());
            }
        }
    }

    private void ApplyGradients(Matrix<T> weights, double[,] grads, double lr, double clipValue)
    {
        int rows = weights.Rows;
        int cols = weights.Columns;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double clipped = Math.Max(-clipValue, Math.Min(clipValue, grads[i, j]));
                weights[i, j] = NumOps.Subtract(weights[i, j], NumOps.FromDouble(lr * clipped));
            }
        }
    }

    private void ApplyGradients(Vector<T> weights, double[] grads, double lr, double clipValue)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            double clipped = Math.Max(-clipValue, Math.Min(clipValue, grads[i]));
            weights[i] = NumOps.Subtract(weights[i], NumOps.FromDouble(lr * clipped));
        }
    }

    private (Vector<T> z, Vector<T> zc, Vector<T> xRecon, Vector<T> gamma, Vector<T> encH, Vector<T> decH, Vector<T> estH) ForwardPassWithCache(Vector<T> x)
    {
        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var encB2 = _encB2;
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var decB2 = _decB2;
        var estW1 = _estW1;
        var estB1 = _estB1;
        var estW2 = _estW2;
        var estB2 = _estB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            estW1 == null || estB1 == null || estW2 == null || estB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        // Encoder layer 1
        var encH = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = encB1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], encW1[i, j]));
            }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            encH[j] = NumOps.FromDouble(tanhVal);
        }

        // Encoder layer 2
        var z = new Vector<T>(_latentDim);
        for (int j = 0; j < _latentDim; j++)
        {
            T sum = encB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(encH[i], encW2[i, j]));
            }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            z[j] = NumOps.FromDouble(tanhVal);
        }

        // Decoder layer 1
        var decH = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = decB1[j];
            for (int i = 0; i < _latentDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(z[i], decW1[i, j]));
            }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            decH[j] = NumOps.FromDouble(tanhVal);
        }

        // Decoder layer 2
        var xRecon = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = decB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(decH[i], decW2[i, j]));
            }
            xRecon[j] = sum;
        }

        // Compute reconstruction features
        T eucDistSq = NumOps.Zero;
        T dotProduct = NumOps.Zero;
        T normXSq = NumOps.Zero;
        T normReconSq = NumOps.Zero;

        for (int i = 0; i < _inputDim; i++)
        {
            T diff = NumOps.Subtract(x[i], xRecon[i]);
            eucDistSq = NumOps.Add(eucDistSq, NumOps.Multiply(diff, diff));
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(x[i], xRecon[i]));
            normXSq = NumOps.Add(normXSq, NumOps.Multiply(x[i], x[i]));
            normReconSq = NumOps.Add(normReconSq, NumOps.Multiply(xRecon[i], xRecon[i]));
        }

        double eucDist = Math.Sqrt(NumOps.ToDouble(eucDistSq));
        double normX = Math.Sqrt(NumOps.ToDouble(normXSq));
        double normRecon = Math.Sqrt(NumOps.ToDouble(normReconSq));
        double cosSim = (normX > 1e-10 && normRecon > 1e-10)
            ? NumOps.ToDouble(dotProduct) / (normX * normRecon)
            : 0;

        // Concatenate z with reconstruction features
        var zc = new Vector<T>(_zDim);
        for (int i = 0; i < _latentDim; i++)
        {
            zc[i] = z[i];
        }
        zc[_latentDim] = NumOps.FromDouble(eucDist);
        zc[_latentDim + 1] = NumOps.FromDouble(cosSim);

        // Estimation network layer 1
        var estH = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = estB1[j];
            for (int i = 0; i < _zDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(zc[i], estW1[i, j]));
            }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            estH[j] = NumOps.FromDouble(tanhVal);
        }

        // Estimation network layer 2
        var logits = new double[_numMixtures];
        double maxLogit = double.MinValue;
        for (int j = 0; j < _numMixtures; j++)
        {
            T sum = estB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(estH[i], estW2[i, j]));
            }
            logits[j] = NumOps.ToDouble(sum);
            if (logits[j] > maxLogit) maxLogit = logits[j];
        }

        // Softmax
        var gamma = new Vector<T>(_numMixtures);
        double sumExp = 0;
        for (int k = 0; k < _numMixtures; k++)
        {
            double expVal = Math.Exp(logits[k] - maxLogit);
            gamma[k] = NumOps.FromDouble(expVal);
            sumExp += expVal;
        }
        for (int k = 0; k < _numMixtures; k++)
        {
            gamma[k] = NumOps.Divide(gamma[k], NumOps.FromDouble(sumExp));
        }

        return (z, zc, xRecon, gamma, encH, decH, estH);
    }

    private (Vector<T> z, Vector<T> zc, Vector<T> xRecon, Vector<T> gamma) ForwardPass(Vector<T> x)
    {
        // Encode
        var z = Encode(x);

        // Decode
        var xRecon = Decode(z);

        // Compute reconstruction features
        T eucDistSq = NumOps.Zero;
        T dotProduct = NumOps.Zero;
        T normXSq = NumOps.Zero;
        T normReconSq = NumOps.Zero;

        for (int j = 0; j < _inputDim; j++)
        {
            T diff = NumOps.Subtract(x[j], xRecon[j]);
            eucDistSq = NumOps.Add(eucDistSq, NumOps.Multiply(diff, diff));
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(x[j], xRecon[j]));
            normXSq = NumOps.Add(normXSq, NumOps.Multiply(x[j], x[j]));
            normReconSq = NumOps.Add(normReconSq, NumOps.Multiply(xRecon[j], xRecon[j]));
        }

        double eucDist = Math.Sqrt(NumOps.ToDouble(eucDistSq));
        double normX = Math.Sqrt(NumOps.ToDouble(normXSq));
        double normRecon = Math.Sqrt(NumOps.ToDouble(normReconSq));
        double cosSim = (normX > 1e-10 && normRecon > 1e-10)
            ? NumOps.ToDouble(dotProduct) / (normX * normRecon)
            : 0;

        // Concatenate z with reconstruction features
        var zc = new Vector<T>(_zDim);
        for (int j = 0; j < _latentDim; j++)
        {
            zc[j] = z[j];
        }
        zc[_latentDim] = NumOps.FromDouble(eucDist);
        zc[_latentDim + 1] = NumOps.FromDouble(cosSim);

        // Estimate GMM membership
        var gamma = EstimateGamma(zc);

        return (z, zc, xRecon, gamma);
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
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            h[j] = NumOps.FromDouble(tanhVal);
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
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            z[j] = NumOps.FromDouble(tanhVal);
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
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            h[j] = NumOps.FromDouble(tanhVal);
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

    private Vector<T> EstimateGamma(Vector<T> zc)
    {
        var estW1 = _estW1;
        var estB1 = _estB1;
        var estW2 = _estW2;
        var estB2 = _estB2;

        if (estW1 == null || estB1 == null || estW2 == null || estB2 == null)
        {
            throw new InvalidOperationException("Estimation network weights not initialized.");
        }

        // Layer 1
        var h = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = estB1[j];
            for (int i = 0; i < _zDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(zc[i], estW1[i, j]));
            }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            h[j] = NumOps.FromDouble(tanhVal);
        }

        // Layer 2
        var logits = new double[_numMixtures];
        double maxLogit = double.MinValue;
        for (int j = 0; j < _numMixtures; j++)
        {
            T sum = estB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h[i], estW2[i, j]));
            }
            logits[j] = NumOps.ToDouble(sum);
            if (logits[j] > maxLogit) maxLogit = logits[j];
        }

        // Softmax
        var gamma = new Vector<T>(_numMixtures);
        double sumExp = 0;
        for (int k = 0; k < _numMixtures; k++)
        {
            double expVal = Math.Exp(logits[k] - maxLogit);
            gamma[k] = NumOps.FromDouble(expVal);
            sumExp += expVal;
        }
        for (int k = 0; k < _numMixtures; k++)
        {
            gamma[k] = NumOps.Divide(gamma[k], NumOps.FromDouble(sumExp));
        }

        return gamma;
    }

    private void UpdateGMM(double[][] allZ, double[][] allGamma)
    {
        int n = allZ.Length;

        // Capture nullable fields
        var phi = _phi;
        var mu = _mu;
        var sigma = _sigma;

        if (phi == null || mu == null || sigma == null)
        {
            throw new InvalidOperationException("GMM parameters not initialized.");
        }

        // Update phi (mixture weights)
        for (int k = 0; k < _numMixtures; k++)
        {
            double sumGamma = 0;
            for (int i = 0; i < n; i++)
            {
                sumGamma += allGamma[i][k];
            }
            phi[k] = sumGamma / n;
        }

        // Update mu (means)
        for (int k = 0; k < _numMixtures; k++)
        {
            double sumGamma = 0;
            for (int i = 0; i < n; i++)
            {
                sumGamma += allGamma[i][k];
            }

            for (int j = 0; j < _zDim; j++)
            {
                double sumWeighted = 0;
                for (int i = 0; i < n; i++)
                {
                    sumWeighted += allGamma[i][k] * allZ[i][j];
                }
                mu[k][j] = sumGamma > 0 ? sumWeighted / sumGamma : 0;
            }
        }

        // Update sigma (covariances) - diagonal only for simplicity
        for (int k = 0; k < _numMixtures; k++)
        {
            double sumGamma = 0;
            for (int i = 0; i < n; i++)
            {
                sumGamma += allGamma[i][k];
            }

            for (int j = 0; j < _zDim; j++)
            {
                double sumWeighted = 0;
                for (int i = 0; i < n; i++)
                {
                    double diff = allZ[i][j] - mu[k][j];
                    sumWeighted += allGamma[i][k] * diff * diff;
                }
                sigma[k][j][j] = sumGamma > 0 ? Math.Max(1e-6, sumWeighted / sumGamma) : 1.0;
            }
        }
    }

    private double ComputeEnergy(Vector<T> zc)
    {
        // Capture nullable fields
        var phi = _phi;
        var mu = _mu;
        var sigma = _sigma;

        if (phi == null || mu == null || sigma == null)
        {
            throw new InvalidOperationException("GMM parameters not initialized.");
        }

        // Compute negative log-likelihood under GMM
        double energy = 0;

        for (int k = 0; k < _numMixtures; k++)
        {
            double logDet = 0;
            double mahal = 0;

            for (int j = 0; j < _zDim; j++)
            {
                logDet += Math.Log(sigma[k][j][j]);
                double diff = NumOps.ToDouble(zc[j]) - mu[k][j];
                mahal += diff * diff / sigma[k][j][j];
            }

            double logPdf = -0.5 * (_zDim * Math.Log(2 * Math.PI) + logDet + mahal);
            energy += phi[k] * Math.Exp(logPdf);
        }

        return -Math.Log(energy + 1e-10);
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

            // Forward pass
            var (_, zc, _, _) = ForwardPass(x);

            // Compute energy (anomaly score)
            double energy = ComputeEnergy(zc);

            scores[i] = NumOps.FromDouble(energy);
        }

        return scores;
    }
}
