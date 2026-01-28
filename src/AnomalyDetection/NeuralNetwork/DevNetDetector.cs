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
    private Matrix<T>? _w1;
    private Vector<T>? _b1;
    private Matrix<T>? _w2;
    private Vector<T>? _b2;
    private Matrix<T>? _w3;
    private Vector<T>? _b3;

    // Reference statistics (initialized in Fit, default to zero before that)
    private T _refMean;
    private T _refStd;
    private int _inputDim;

    // Normalization parameters
    private Vector<T>? _dataMeans;
    private Vector<T>? _dataStds;

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

        // Initialize reference statistics to default values (will be set in Fit)
        _refMean = NumOps.Zero;
        _refStd = NumOps.One;
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

        // Initialize reference statistics (Gaussian prior)
        _refMean = NumOps.Zero;
        _refStd = NumOps.One;

        // Initialize weights
        InitializeWeights();

        // Train network with deviation loss
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
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double scale3 = Math.Sqrt(2.0 / (_hiddenDim + 1));

        _w1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _b1 = new Vector<T>(_hiddenDim);
        _w2 = InitializeMatrix(_hiddenDim, _hiddenDim, scale2);
        _b2 = new Vector<T>(_hiddenDim);
        _w3 = InitializeMatrix(_hiddenDim, 1, scale3);
        _b3 = new Vector<T>(1);

        // Initialize biases to zero
        for (int i = 0; i < _hiddenDim; i++)
        {
            _b1[i] = NumOps.Zero;
            _b2[i] = NumOps.Zero;
        }
        _b3[0] = NumOps.Zero;
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

    private void Train(Matrix<T> data)
    {
        // Capture nullable fields for proper null checking
        var w1 = _w1;
        var b1 = _b1;
        var w2 = _w2;
        var b2 = _b2;
        var w3 = _w3;
        var b3 = _b3;

        if (w1 == null || b1 == null || w2 == null || b2 == null || w3 == null || b3 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        int n = data.Rows;
        int batchSize = Math.Min(64, n);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Accumulate gradients
                var dW1 = new Matrix<T>(_inputDim, _hiddenDim);
                var dB1 = new Vector<T>(_hiddenDim);
                var dW2 = new Matrix<T>(_hiddenDim, _hiddenDim);
                var dB2 = new Vector<T>(_hiddenDim);
                var dW3 = new Matrix<T>(_hiddenDim, 1);
                var dB3 = new Vector<T>(1);

                // Initialize gradients to zero
                InitializeToZero(dW1, dB1);
                InitializeToZero(dW2, dB2);
                InitializeToZero(dW3, dB3);

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var x = data.GetRow(idx);

                    // Forward pass
                    var (h1, h2, score) = ForwardWithCache(x);

                    // Sample reference score from Gaussian
                    double u1 = 1.0 - _random.NextDouble();
                    double u2 = 1.0 - _random.NextDouble();
                    double refMean = NumOps.ToDouble(_refMean);
                    double refStd = NumOps.ToDouble(_refStd);
                    T refScore = NumOps.FromDouble(refMean + refStd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));

                    // Deviation loss: push normal data close to reference
                    T deviation = NumOps.Subtract(score, refScore);

                    // Gradient of L2 loss: d/dscore (score - refScore)^2 = 2*(score - refScore)
                    T dScore = NumOps.Multiply(NumOps.FromDouble(2), deviation);

                    // Gradient clipping
                    double dScoreVal = NumOps.ToDouble(dScore);
                    dScoreVal = Math.Max(-10.0, Math.Min(10.0, dScoreVal));
                    dScore = NumOps.FromDouble(dScoreVal);

                    // Backprop through output layer
                    var dH2 = new Vector<T>(_hiddenDim);
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        dH2[i] = NumOps.Zero;
                    }
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        dW3[i, 0] = NumOps.Add(dW3[i, 0], NumOps.Multiply(h2[i], dScore));
                        dH2[i] = NumOps.Add(dH2[i], NumOps.Multiply(w3[i, 0], dScore));
                    }
                    dB3[0] = NumOps.Add(dB3[0], dScore);

                    // ReLU derivative for h2
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        if (NumOps.ToDouble(h2[i]) <= 0) dH2[i] = NumOps.Zero;
                    }

                    // Backprop through layer 2
                    var dH1 = new Vector<T>(_hiddenDim);
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        dH1[i] = NumOps.Zero;
                    }
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dW2[i, j] = NumOps.Add(dW2[i, j], NumOps.Multiply(h1[i], dH2[j]));
                            dH1[i] = NumOps.Add(dH1[i], NumOps.Multiply(w2[i, j], dH2[j]));
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        dB2[j] = NumOps.Add(dB2[j], dH2[j]);
                    }

                    // ReLU derivative for h1
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        if (NumOps.ToDouble(h1[i]) <= 0) dH1[i] = NumOps.Zero;
                    }

                    // Backprop through layer 1
                    for (int i = 0; i < _inputDim; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dW1[i, j] = NumOps.Add(dW1[i, j], NumOps.Multiply(x[i], dH1[j]));
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        dB1[j] = NumOps.Add(dB1[j], dH1[j]);
                    }
                }

                // Update weights
                T lr = NumOps.FromDouble(_learningRate / actualBatchSize);
                UpdateWeights(w1, dW1, lr);
                UpdateWeights(b1, dB1, lr);
                UpdateWeights(w2, dW2, lr);
                UpdateWeights(b2, dB2, lr);
                UpdateWeights(w3, dW3, lr);
                UpdateWeights(b3, dB3, lr);
            }
        }
    }

    private void InitializeToZero(Matrix<T> matrix, Vector<T> vector)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Zero;
            }
        }
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = NumOps.Zero;
        }
    }

    private T Forward(Vector<T> x)
    {
        // Capture nullable fields for proper null checking
        var w1 = _w1;
        var b1 = _b1;
        var w2 = _w2;
        var b2 = _b2;
        var w3 = _w3;
        var b3 = _b3;

        if (w1 == null || b1 == null || w2 == null || b2 == null || w3 == null || b3 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        // Layer 1
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = b1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], w1[i, j]));
            }
            double val = NumOps.ToDouble(sum);
            h1[j] = NumOps.FromDouble(ReLU(val));
        }

        // Layer 2
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = b2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h1[i], w2[i, j]));
            }
            double val = NumOps.ToDouble(sum);
            h2[j] = NumOps.FromDouble(ReLU(val));
        }

        // Output layer (linear)
        T score = b3[0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            score = NumOps.Add(score, NumOps.Multiply(h2[i], w3[i, 0]));
        }

        return score;
    }

    private (Vector<T> h1, Vector<T> h2, T score) ForwardWithCache(Vector<T> x)
    {
        // Capture nullable fields for proper null checking
        var w1 = _w1;
        var b1 = _b1;
        var w2 = _w2;
        var b2 = _b2;
        var w3 = _w3;
        var b3 = _b3;

        if (w1 == null || b1 == null || w2 == null || b2 == null || w3 == null || b3 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        // Layer 1
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = b1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], w1[i, j]));
            }
            double val = NumOps.ToDouble(sum);
            h1[j] = NumOps.FromDouble(ReLU(val));
        }

        // Layer 2
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = b2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h1[i], w2[i, j]));
            }
            double val = NumOps.ToDouble(sum);
            h2[j] = NumOps.FromDouble(ReLU(val));
        }

        // Output layer
        T score = b3[0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            score = NumOps.Add(score, NumOps.Multiply(h2[i], w3[i, 0]));
        }

        return (h1, h2, score);
    }

    private static double ReLU(double x) => Math.Max(0, x);

    private void UpdateWeights(Matrix<T> weights, Matrix<T> gradients, T lr)
    {
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                T update = NumOps.Multiply(lr, gradients[i, j]);
                weights[i, j] = NumOps.Subtract(weights[i, j], update);
            }
        }
    }

    private void UpdateWeights(Vector<T> weights, Vector<T> gradients, T lr)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            T update = NumOps.Multiply(lr, gradients[i]);
            weights[i] = NumOps.Subtract(weights[i], update);
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

            // Get anomaly score
            T score = Forward(x);

            // Convert to positive anomaly score (higher = more anomalous)
            // DevNet outputs deviation from reference
            T deviation = NumOps.Subtract(score, _refMean);
            double absDeviation = Math.Abs(NumOps.ToDouble(deviation));
            scores[i] = NumOps.FromDouble(absDeviation);
        }

        return scores;
    }
}
