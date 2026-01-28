using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Implements N-BEATS (Neural Basis Expansion Analysis for Time Series) for anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> N-BEATS is a deep neural architecture for time series forecasting
/// that uses stacked blocks with basis expansion. For anomaly detection, it predicts
/// the next value and uses the prediction error as the anomaly score.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Stack multiple blocks, each outputting a partial forecast and backcast
/// 2. Each block uses fully-connected layers with basis expansion
/// 3. Residual learning allows progressive refinement
/// 4. High prediction errors indicate anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Univariate or multivariate time series
/// - When interpretability of forecasts matters
/// - Long-horizon forecasting-based anomaly detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of stacks: 2
/// - Number of blocks per stack: 3
/// - Hidden dimensions: 64
/// - Lookback: 10
/// - Epochs: 50
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Oreshkin, B. N., et al. (2020).
/// "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting." ICLR.
/// </para>
/// </remarks>
public class NBEATSDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _numStacks;
    private readonly int _numBlocks;
    private readonly int _hiddenDim;
    private readonly int _lookback;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Blocks weights
    private List<BlockWeights>? _blocks;

    private int _inputDim;

    // Normalization parameters
    private double[]? _dataMeans;
    private double[]? _dataStds;

    /// <summary>
    /// Gets the number of stacks.
    /// </summary>
    public int NumStacks => _numStacks;

    /// <summary>
    /// Gets the number of blocks per stack.
    /// </summary>
    public int NumBlocks => _numBlocks;

    /// <summary>
    /// Gets the lookback window size.
    /// </summary>
    public int Lookback => _lookback;

    /// <summary>
    /// Creates a new N-BEATS anomaly detector.
    /// </summary>
    /// <param name="numStacks">Number of stacks. Default is 2.</param>
    /// <param name="numBlocks">Number of blocks per stack. Default is 3.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="lookback">Lookback window size. Default is 10.</param>
    /// <param name="epochs">Number of training epochs. Default is 50.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public NBEATSDetector(int numStacks = 2, int numBlocks = 3, int hiddenDim = 64,
        int lookback = 10, int epochs = 50, double learningRate = 0.001,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numStacks < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numStacks),
                "Number of stacks must be at least 1. Recommended is 2.");
        }

        if (numBlocks < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numBlocks),
                "Number of blocks must be at least 1. Recommended is 3.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (lookback < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(lookback),
                "Lookback must be at least 1. Recommended is 10.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 50.");
        }

        _numStacks = numStacks;
        _numBlocks = numBlocks;
        _hiddenDim = hiddenDim;
        _lookback = lookback;
        _epochs = epochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;

        if (n < _lookback + 1)
        {
            throw new ArgumentException(
                $"Not enough samples for lookback {_lookback}. Need at least {_lookback + 1} samples.",
                nameof(X));
        }

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

        // Create sequences
        var (inputs, targets) = CreateSequences(normalizedData);

        // Train
        Train(inputs, targets);

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
        int inputSize = _lookback * _inputDim;
        _blocks = new List<BlockWeights>();

        int totalBlocks = _numStacks * _numBlocks;

        for (int b = 0; b < totalBlocks; b++)
        {
            var block = new BlockWeights
            {
                W1 = InitializeMatrix(inputSize, _hiddenDim),
                B1 = new double[_hiddenDim],
                W2 = InitializeMatrix(_hiddenDim, _hiddenDim),
                B2 = new double[_hiddenDim],
                W3 = InitializeMatrix(_hiddenDim, _hiddenDim),
                B3 = new double[_hiddenDim],
                W4 = InitializeMatrix(_hiddenDim, _hiddenDim),
                B4 = new double[_hiddenDim],
                // Theta for backcast and forecast
                WTheta_b = InitializeMatrix(_hiddenDim, inputSize),
                WTheta_f = InitializeMatrix(_hiddenDim, _inputDim) // Forecast horizon = 1
            };
            _blocks.Add(block);
        }
    }

    private double[,] InitializeMatrix(int rows, int cols)
    {
        double scale = Math.Sqrt(2.0 / (rows + cols));
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

    private (double[][] inputs, double[][] targets) CreateSequences(double[][] data)
    {
        int n = data.Length - _lookback;
        var inputs = new double[n][];
        var targets = new double[n][];

        for (int i = 0; i < n; i++)
        {
            // Flatten lookback window
            inputs[i] = new double[_lookback * _inputDim];
            for (int t = 0; t < _lookback; t++)
            {
                for (int j = 0; j < _inputDim; j++)
                {
                    inputs[i][t * _inputDim + j] = data[i + t][j];
                }
            }
            targets[i] = data[i + _lookback];
        }

        return (inputs, targets);
    }

    private void Train(double[][] inputs, double[][] targets)
    {
        int n = inputs.Length;
        int batchSize = Math.Min(32, n);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var input = inputs[idx];
                    var target = targets[idx];

                    // Forward pass
                    var (forecast, _) = Forward(input);

                    // Compute loss gradient
                    var gradForecast = new double[_inputDim];
                    for (int j = 0; j < _inputDim; j++)
                    {
                        gradForecast[j] = 2 * (forecast[j] - target[j]);
                    }

                    // Simplified backward pass - update last block's theta
                    double lr = _learningRate / actualBatchSize;
                    var lastBlock = _blocks![_blocks.Count - 1];

                    // Update forecast theta
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _inputDim; j++)
                        {
                            lastBlock.WTheta_f![i, j] -= lr * gradForecast[j] * 0.01;
                        }
                    }
                }
            }
        }
    }

    private (double[] forecast, double[] backcast) Forward(double[] input)
    {
        var blocks = _blocks;
        if (blocks == null)
        {
            throw new InvalidOperationException("Model not initialized.");
        }

        int inputSize = input.Length;
        var residual = (double[])input.Clone();
        var totalForecast = new double[_inputDim];
        var totalBackcast = new double[inputSize];

        foreach (var block in blocks)
        {
            // FC stack (4 layers with ReLU)
            var h = ForwardFC(residual, block.W1!, block.B1!);
            h = ApplyReLU(h);
            h = ForwardFC(h, block.W2!, block.B2!);
            h = ApplyReLU(h);
            h = ForwardFC(h, block.W3!, block.B3!);
            h = ApplyReLU(h);
            h = ForwardFC(h, block.W4!, block.B4!);
            h = ApplyReLU(h);

            // Generate backcast and forecast
            var backcast = ForwardFC(h, block.WTheta_b!, new double[inputSize]);
            var forecast = ForwardFC(h, block.WTheta_f!, new double[_inputDim]);

            // Update residual (subtract backcast)
            for (int i = 0; i < inputSize; i++)
            {
                residual[i] -= backcast[i];
            }

            // Accumulate forecast
            for (int i = 0; i < _inputDim; i++)
            {
                totalForecast[i] += forecast[i];
            }

            // Accumulate backcast
            for (int i = 0; i < inputSize; i++)
            {
                totalBackcast[i] += backcast[i];
            }
        }

        return (totalForecast, totalBackcast);
    }

    private double[] ForwardFC(double[] input, double[,] W, double[] b)
    {
        int outputSize = W.GetLength(1);
        var output = new double[outputSize];

        for (int j = 0; j < outputSize; j++)
        {
            output[j] = b[j];
            for (int i = 0; i < input.Length; i++)
            {
                output[j] += input[i] * W[i, j];
            }
        }

        return output;
    }

    private static double[] ApplyReLU(double[] x)
    {
        var output = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            output[i] = Math.Max(0, x[i]);
        }
        return output;
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

        // Convert to normalized double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = (NumOps.ToDouble(X[i, j]) - dataMeans[j]) / dataStds[j];
            }
        }

        // Score each point
        for (int i = 0; i < n; i++)
        {
            double score;

            if (i < _lookback)
            {
                // Not enough history
                score = 0;
                for (int j = 0; j < _inputDim; j++)
                {
                    score += data[i][j] * data[i][j];
                }
            }
            else
            {
                // Build input from lookback window
                var input = new double[_lookback * _inputDim];
                for (int t = 0; t < _lookback; t++)
                {
                    for (int j = 0; j < _inputDim; j++)
                    {
                        input[t * _inputDim + j] = data[i - _lookback + t][j];
                    }
                }

                // Predict and compute error
                var (forecast, _) = Forward(input);
                score = 0;
                for (int j = 0; j < _inputDim; j++)
                {
                    score += Math.Pow(data[i][j] - forecast[j], 2);
                }
            }

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private class BlockWeights
    {
        public double[,]? W1 { get; set; }
        public double[]? B1 { get; set; }
        public double[,]? W2 { get; set; }
        public double[]? B2 { get; set; }
        public double[,]? W3 { get; set; }
        public double[]? B3 { get; set; }
        public double[,]? W4 { get; set; }
        public double[]? B4 { get; set; }
        public double[,]? WTheta_b { get; set; }
        public double[,]? WTheta_f { get; set; }
    }
}
