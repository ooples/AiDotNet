using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Detects anomalies using Deep SVDD (Support Vector Data Description).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Deep SVDD trains a neural network to map normal data points close to
/// a hypersphere center in the output space. Anomalies are points that map far from this center.
/// It combines deep learning with the classic SVDD concept.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Initialize network and compute center from initial encodings
/// 2. Train network to minimize distance of normal points to center
/// 3. Anomaly score is the distance to the center
/// </para>
/// <para>
/// <b>When to use:</b>
/// - One-class classification with deep learning
/// - When you have only normal examples for training
/// - Complex, high-dimensional data
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Hidden dimensions: 64
/// - Output dimensions: 32
/// - Learning rate: 0.001
/// - Epochs: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Ruff, L., et al. (2018). "Deep One-Class Classification." ICML.
/// </para>
/// </remarks>
public class DeepSVDDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _hiddenDim;
    private readonly int _outputDim;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Network weights
    private Matrix<T>? _w1;
    private Vector<T>? _b1;
    private Matrix<T>? _w2;
    private Vector<T>? _b2;
    private Matrix<T>? _w3;
    private Vector<T>? _b3;

    // Hypersphere center
    private Vector<T>? _center;
    private int _inputDim;

    // Normalization parameters
    private Vector<T>? _dataMeans;
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the hidden layer dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Gets the output dimensions.
    /// </summary>
    public int OutputDim => _outputDim;

    /// <summary>
    /// Creates a new Deep SVDD anomaly detector.
    /// </summary>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="outputDim">Dimensions of output (representation). Default is 32.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DeepSVDDDetector(int hiddenDim = 64, int outputDim = 32, int epochs = 100,
        double learningRate = 0.001, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "HiddenDim must be at least 1. Recommended is 64.");
        }

        if (outputDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(outputDim),
                "OutputDim must be at least 1. Recommended is 32.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.001.");
        }

        _hiddenDim = hiddenDim;
        _outputDim = outputDim;
        _epochs = epochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;

        // Normalize data and store normalization parameters
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize network weights
        InitializeWeights();

        // Compute initial center
        _center = ComputeCenter(normalizedData);

        // Train network
        TrainNetwork(normalizedData);

        // Recompute center after training
        _center = ComputeCenter(normalizedData);

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
        double scale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double scale3 = Math.Sqrt(2.0 / (_hiddenDim + _outputDim));

        _w1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _b1 = new Vector<T>(_hiddenDim);
        _w2 = InitializeMatrix(_hiddenDim, _hiddenDim, scale2);
        _b2 = new Vector<T>(_hiddenDim);
        _w3 = InitializeMatrix(_hiddenDim, _outputDim, scale3);
        _b3 = new Vector<T>(_outputDim);

        // Initialize biases to zero
        for (int i = 0; i < _hiddenDim; i++)
        {
            _b1[i] = NumOps.Zero;
            _b2[i] = NumOps.Zero;
        }
        for (int i = 0; i < _outputDim; i++)
        {
            _b3[i] = NumOps.Zero;
        }
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

    private Vector<T> ComputeCenter(Matrix<T> data)
    {
        int n = data.Rows;
        var center = new Vector<T>(_outputDim);

        // Initialize center to zero
        for (int j = 0; j < _outputDim; j++)
        {
            center[j] = NumOps.Zero;
        }

        for (int i = 0; i < n; i++)
        {
            var x = data.GetRow(i);
            var output = Forward(x);
            for (int j = 0; j < _outputDim; j++)
            {
                center[j] = NumOps.Add(center[j], output[j]);
            }
        }

        T nT = NumOps.FromDouble(n);
        for (int j = 0; j < _outputDim; j++)
        {
            center[j] = NumOps.Divide(center[j], nT);
        }

        return center;
    }

    private Vector<T> Forward(Vector<T> x)
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
            // ReLU: convert to double at activation boundary
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
            // ReLU: convert to double at activation boundary
            double val = NumOps.ToDouble(sum);
            h2[j] = NumOps.FromDouble(ReLU(val));
        }

        // Output layer (no activation - linear)
        var output = new Vector<T>(_outputDim);
        for (int j = 0; j < _outputDim; j++)
        {
            T sum = b3[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h2[i], w3[i, j]));
            }
            output[j] = sum;
        }

        return output;
    }

    private (Vector<T> h1, Vector<T> h2, Vector<T> output) ForwardWithCache(Vector<T> x)
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

        var output = new Vector<T>(_outputDim);
        for (int j = 0; j < _outputDim; j++)
        {
            T sum = b3[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h2[i], w3[i, j]));
            }
            output[j] = sum;
        }

        return (h1, h2, output);
    }

    private void TrainNetwork(Matrix<T> data)
    {
        // Capture nullable fields for proper null checking
        var w1 = _w1;
        var b1 = _b1;
        var w2 = _w2;
        var b2 = _b2;
        var w3 = _w3;
        var b3 = _b3;
        var center = _center;

        if (w1 == null || b1 == null || w2 == null || b2 == null || w3 == null || b3 == null || center == null)
        {
            throw new InvalidOperationException("Weights or center not initialized.");
        }

        int n = data.Rows;
        int batchSize = Math.Min(32, n);

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
                var dW3 = new Matrix<T>(_hiddenDim, _outputDim);
                var dB3 = new Vector<T>(_outputDim);

                // Initialize gradients to zero
                InitializeToZero(dW1, dB1);
                InitializeToZero(dW2, dB2);
                InitializeToZero(dW3, dB3);

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var x = data.GetRow(idx);

                    var (h1, h2, output) = ForwardWithCache(x);

                    // Compute gradient of loss: ||output - center||^2
                    var dOutput = new Vector<T>(_outputDim);
                    for (int j = 0; j < _outputDim; j++)
                    {
                        T diff = NumOps.Subtract(output[j], center[j]);
                        dOutput[j] = NumOps.Multiply(NumOps.FromDouble(2), diff);
                    }

                    // Backprop through output layer
                    var dH2 = new Vector<T>(_hiddenDim);
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        dH2[i] = NumOps.Zero;
                    }
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _outputDim; j++)
                        {
                            dW3[i, j] = NumOps.Add(dW3[i, j], NumOps.Multiply(h2[i], dOutput[j]));
                            dH2[i] = NumOps.Add(dH2[i], NumOps.Multiply(w3[i, j], dOutput[j]));
                        }
                    }
                    for (int j = 0; j < _outputDim; j++)
                    {
                        dB3[j] = NumOps.Add(dB3[j], dOutput[j]);
                    }

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

    private static double ReLU(double x) => Math.Max(0, x);

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_inputDim} features.",
                nameof(X));
        }

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        var center = _center;
        if (dataMeans == null || dataStds == null || center == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters or center missing.");
        }

        int n = X.Rows;
        var scores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Apply same normalization as training data
            var x = new Vector<T>(_inputDim);
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(X[i, j], dataMeans[j]);
                x[j] = NumOps.Divide(diff, dataStds[j]);
            }

            var output = Forward(x);

            // Distance to center
            T dist = NumOps.Zero;
            for (int j = 0; j < _outputDim; j++)
            {
                T diff = NumOps.Subtract(output[j], center[j]);
                dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
            }

            scores[i] = dist;
        }

        return scores;
    }
}
