using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements DevNet (Deep Anomaly Detection Network) for end-to-end anomaly scoring.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DevNet learns to directly output anomaly scores using a deviation
/// network. It combines feature learning and anomaly scoring in a single network,
/// using reference points and deviation loss for training.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Learn feature representations through neural network
/// 2. Use Gaussian reference points to define normalcy
/// 3. Train with deviation loss to produce anomaly scores directly
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you want end-to-end anomaly scoring
/// - Tabular data with known anomaly labels (semi-supervised)
/// - When reconstruction-based methods don't work well
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Hidden dimensions: 64
/// - Output dimensions: 1
/// - Epochs: 50
/// - Learning rate: 0.001
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Pang, G., Shen, C., and van den Hengel, A. (2019).
/// "Deep Anomaly Detection with Deviation Networks." KDD.
/// </para>
/// </remarks>
public class DevNetDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _hiddenDim;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly double _marginScale;

    // Network weights
    private double[,]? _w1;
    private double[]? _b1;
    private double[,]? _w2;
    private double[]? _b2;
    private double[,]? _w3;
    private double[]? _b3;

    // Reference statistics
    private double _refMean;
    private double _refStd;
    private int _inputDim;

    // Normalization parameters
    private double[]? _dataMeans;
    private double[]? _dataStds;

    /// <summary>
    /// Gets the hidden dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Creates a new DevNet anomaly detector.
    /// </summary>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="epochs">Number of training epochs. Default is 50.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="marginScale">Scale for deviation margin. Default is 5.0.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DevNetDetector(int hiddenDim = 64, int epochs = 50,
        double learningRate = 0.001, double marginScale = 5.0,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 50.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.001.");
        }

        if (marginScale <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(marginScale),
                "Margin scale must be positive. Recommended is 5.0.");
        }

        _hiddenDim = hiddenDim;
        _epochs = epochs;
        _learningRate = learningRate;
        _marginScale = marginScale;
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

        // Initialize reference statistics (Gaussian prior)
        _refMean = 0.0;
        _refStd = 1.0;

        // Initialize weights
        InitializeWeights();

        // Train network with deviation loss
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
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double scale3 = Math.Sqrt(2.0 / (_hiddenDim + 1));

        _w1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _b1 = new double[_hiddenDim];
        _w2 = InitializeMatrix(_hiddenDim, _hiddenDim, scale2);
        _b2 = new double[_hiddenDim];
        _w3 = InitializeMatrix(_hiddenDim, 1, scale3);
        _b3 = new double[1];
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
        int batchSize = Math.Min(64, n);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Accumulate gradients
                var dW1 = new double[_inputDim, _hiddenDim];
                var dB1 = new double[_hiddenDim];
                var dW2 = new double[_hiddenDim, _hiddenDim];
                var dB2 = new double[_hiddenDim];
                var dW3 = new double[_hiddenDim, 1];
                var dB3 = new double[1];

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var x = data[idx];

                    // Forward pass
                    var (h1, h2, score) = ForwardWithCache(x);

                    // Sample reference score from Gaussian
                    double u1 = 1.0 - _random.NextDouble();
                    double u2 = 1.0 - _random.NextDouble();
                    double refScore = _refMean + _refStd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

                    // Deviation loss: push normal data close to reference
                    // For unsupervised, assume all training data is normal
                    // Normal samples should have low deviation from reference
                    // Loss = deviation^2 pushes scores toward reference
                    double deviation = score - refScore;

                    // L2 loss: ||score - refScore||^2
                    // This encourages normal data to have scores close to reference
                    double loss = deviation * deviation;

                    // Gradient of L2 loss: d/dscore (score - refScore)^2 = 2*(score - refScore)
                    double dScore = 2 * deviation;

                    // Gradient clipping
                    dScore = Math.Max(-10.0, Math.Min(10.0, dScore));

                    // Unused: loss value (for debugging/logging if needed)
                    _ = loss;

                    // Backprop through output layer
                    var dH2 = new double[_hiddenDim];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        dW3[i, 0] += h2[i] * dScore;
                        dH2[i] += _w3![i, 0] * dScore;
                    }
                    dB3[0] += dScore;

                    // ReLU derivative for h2
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        if (h2[i] <= 0) dH2[i] = 0;
                    }

                    // Backprop through layer 2
                    var dH1 = new double[_hiddenDim];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dW2[i, j] += h1[i] * dH2[j];
                            dH1[i] += _w2![i, j] * dH2[j];
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        dB2[j] += dH2[j];
                    }

                    // ReLU derivative for h1
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        if (h1[i] <= 0) dH1[i] = 0;
                    }

                    // Backprop through layer 1
                    for (int i = 0; i < _inputDim; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dW1[i, j] += x[i] * dH1[j];
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        dB1[j] += dH1[j];
                    }
                }

                // Update weights
                double lr = _learningRate / actualBatchSize;
                UpdateWeights(_w1!, dW1, lr);
                UpdateWeights(_b1!, dB1, lr);
                UpdateWeights(_w2!, dW2, lr);
                UpdateWeights(_b2!, dB2, lr);
                UpdateWeights(_w3!, dW3, lr);
                UpdateWeights(_b3!, dB3, lr);
            }
        }
    }

    private double Forward(double[] x)
    {
        // Layer 1
        var h1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h1[j] = _b1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                h1[j] += x[i] * _w1![i, j];
            }
            h1[j] = ReLU(h1[j]);
        }

        // Layer 2
        var h2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h2[j] = _b2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                h2[j] += h1[i] * _w2![i, j];
            }
            h2[j] = ReLU(h2[j]);
        }

        // Output layer (linear)
        double score = _b3![0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            score += h2[i] * _w3![i, 0];
        }

        return score;
    }

    private (double[] h1, double[] h2, double score) ForwardWithCache(double[] x)
    {
        // Layer 1
        var h1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h1[j] = _b1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                h1[j] += x[i] * _w1![i, j];
            }
            h1[j] = ReLU(h1[j]);
        }

        // Layer 2
        var h2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h2[j] = _b2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                h2[j] += h1[i] * _w2![i, j];
            }
            h2[j] = ReLU(h2[j]);
        }

        // Output layer
        double score = _b3![0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            score += h2[i] * _w3![i, 0];
        }

        return (h1, h2, score);
    }

    private static double ReLU(double x) => Math.Max(0, x);

    private static void UpdateWeights(double[,] weights, double[,] gradients, double lr)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                weights[i, j] -= lr * gradients[i, j];
            }
        }
    }

    private static void UpdateWeights(double[] weights, double[] gradients, double lr)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] -= lr * gradients[i];
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

            // Get anomaly score
            double score = Forward(x);

            // Convert to positive anomaly score (higher = more anomalous)
            // DevNet outputs deviation from reference
            scores[i] = NumOps.FromDouble(Math.Abs(score - _refMean));
        }

        return scores;
    }
}
