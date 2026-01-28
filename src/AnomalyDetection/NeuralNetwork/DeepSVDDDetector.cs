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
    private double[,]? _w1;
    private double[]? _b1;
    private double[,]? _w2;
    private double[]? _b2;
    private double[,]? _w3;
    private double[]? _b3;

    // Hypersphere center
    private double[]? _center;
    private int _inputDim;

    // Normalization parameters
    private double[]? _dataMeans;
    private double[]? _dataStds;

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

        // Normalize data and store normalization parameters
        var (normalizedData, means, stds) = NormalizeData(data);
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
        double scale3 = Math.Sqrt(2.0 / (_hiddenDim + _outputDim));

        _w1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _b1 = new double[_hiddenDim];
        _w2 = InitializeMatrix(_hiddenDim, _hiddenDim, scale2);
        _b2 = new double[_hiddenDim];
        _w3 = InitializeMatrix(_hiddenDim, _outputDim, scale3);
        _b3 = new double[_outputDim];
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

    private double[] ComputeCenter(double[][] data)
    {
        int n = data.Length;
        var center = new double[_outputDim];

        foreach (var x in data)
        {
            var output = Forward(x);
            for (int j = 0; j < _outputDim; j++)
            {
                center[j] += output[j];
            }
        }

        for (int j = 0; j < _outputDim; j++)
        {
            center[j] /= n;
        }

        return center;
    }

    private double[] Forward(double[] x)
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

        // Output layer (no activation - linear)
        var output = new double[_outputDim];
        for (int j = 0; j < _outputDim; j++)
        {
            output[j] = _b3![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                output[j] += h2[i] * _w3![i, j];
            }
        }

        return output;
    }

    private (double[] h1, double[] h2, double[] output) ForwardWithCache(double[] x)
    {
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

        var output = new double[_outputDim];
        for (int j = 0; j < _outputDim; j++)
        {
            output[j] = _b3![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                output[j] += h2[i] * _w3![i, j];
            }
        }

        return (h1, h2, output);
    }

    private void TrainNetwork(double[][] data)
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
                var dW1 = new double[_inputDim, _hiddenDim];
                var dB1 = new double[_hiddenDim];
                var dW2 = new double[_hiddenDim, _hiddenDim];
                var dB2 = new double[_hiddenDim];
                var dW3 = new double[_hiddenDim, _outputDim];
                var dB3 = new double[_outputDim];

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var x = data[idx];

                    var (h1, h2, output) = ForwardWithCache(x);

                    // Compute gradient of loss: ||output - center||^2
                    var dOutput = new double[_outputDim];
                    for (int j = 0; j < _outputDim; j++)
                    {
                        dOutput[j] = 2 * (output[j] - _center![j]);
                    }

                    // Backprop through output layer
                    var dH2 = new double[_hiddenDim];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _outputDim; j++)
                        {
                            dW3[i, j] += h2[i] * dOutput[j];
                            dH2[i] += _w3![i, j] * dOutput[j];
                        }
                    }
                    for (int j = 0; j < _outputDim; j++)
                    {
                        dB3[j] += dOutput[j];
                    }

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

    private void UpdateWeights(double[,] weights, double[,] gradients, double lr)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                weights[i, j] -= lr * gradients[i, j];
            }
        }
    }

    private void UpdateWeights(double[] weights, double[] gradients, double lr)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] -= lr * gradients[i];
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
            // Apply same normalization as training data
            var x = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                x[j] = (NumOps.ToDouble(X[i, j]) - dataMeans[j]) / dataStds[j];
            }

            var output = Forward(x);

            // Distance to center
            double dist = 0;
            for (int j = 0; j < _outputDim; j++)
            {
                dist += Math.Pow(output[j] - _center![j], 2);
            }

            scores[i] = NumOps.FromDouble(dist);
        }

        return scores;
    }
}
