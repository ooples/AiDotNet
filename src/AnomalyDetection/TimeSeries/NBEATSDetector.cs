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
    private Vector<T>? _dataMeans;
    private Vector<T>? _dataStds;

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

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.001.");
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

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(X);
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
        int inputSize = _lookback * _inputDim;
        _blocks = new List<BlockWeights>();

        int totalBlocks = _numStacks * _numBlocks;

        for (int b = 0; b < totalBlocks; b++)
        {
            var block = new BlockWeights
            {
                W1 = InitializeMatrix(inputSize, _hiddenDim),
                B1 = InitializeVector(_hiddenDim),
                W2 = InitializeMatrix(_hiddenDim, _hiddenDim),
                B2 = InitializeVector(_hiddenDim),
                W3 = InitializeMatrix(_hiddenDim, _hiddenDim),
                B3 = InitializeVector(_hiddenDim),
                W4 = InitializeMatrix(_hiddenDim, _hiddenDim),
                B4 = InitializeVector(_hiddenDim),
                // Theta for backcast and forecast
                WTheta_b = InitializeMatrix(_hiddenDim, inputSize),
                WTheta_f = InitializeMatrix(_hiddenDim, _inputDim) // Forecast horizon = 1
            };
            _blocks.Add(block);
        }
    }

    private Matrix<T> InitializeMatrix(int rows, int cols)
    {
        double scale = Math.Sqrt(2.0 / (rows + cols));
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

    private Vector<T> CreateZeroVector(int size)
    {
        var v = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            v[i] = NumOps.Zero;
        }
        return v;
    }

    private (Vector<T>[] inputs, Vector<T>[] targets) CreateSequences(Matrix<T> data)
    {
        int n = data.Rows - _lookback;
        var inputs = new Vector<T>[n];
        var targets = new Vector<T>[n];

        for (int i = 0; i < n; i++)
        {
            // Flatten lookback window
            inputs[i] = new Vector<T>(_lookback * _inputDim);
            for (int t = 0; t < _lookback; t++)
            {
                for (int j = 0; j < _inputDim; j++)
                {
                    inputs[i][t * _inputDim + j] = data[i + t, j];
                }
            }
            targets[i] = data.GetRow(i + _lookback);
        }

        return (inputs, targets);
    }

    private void Train(Vector<T>[] inputs, Vector<T>[] targets)
    {
        int n = inputs.Length;
        int batchSize = Math.Min(32, n);
        int inputSize = _lookback * _inputDim;

        var blocks = _blocks;
        if (blocks == null)
        {
            throw new InvalidOperationException("Model not initialized.");
        }

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Initialize gradient accumulators for all blocks (use double for intermediate computation)
                var blockGradients = new List<BlockGradients>();
                for (int b = 0; b < blocks.Count; b++)
                {
                    blockGradients.Add(new BlockGradients
                    {
                        dW1 = new double[inputSize, _hiddenDim],
                        dB1 = new double[_hiddenDim],
                        dW2 = new double[_hiddenDim, _hiddenDim],
                        dB2 = new double[_hiddenDim],
                        dW3 = new double[_hiddenDim, _hiddenDim],
                        dB3 = new double[_hiddenDim],
                        dW4 = new double[_hiddenDim, _hiddenDim],
                        dB4 = new double[_hiddenDim],
                        dWTheta_b = new double[_hiddenDim, inputSize],
                        dWTheta_f = new double[_hiddenDim, _inputDim]
                    });
                }

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var input = inputs[idx];
                    var target = targets[idx];

                    // Forward pass with caching
                    var (forecast, residuals, hValues) = ForwardWithCache(input);

                    // Compute loss gradient (MSE loss)
                    var gradForecast = new Vector<T>(_inputDim);
                    for (int j = 0; j < _inputDim; j++)
                    {
                        T diff = NumOps.Subtract(forecast[j], target[j]);
                        gradForecast[j] = NumOps.Multiply(NumOps.FromDouble(2.0), diff);
                    }

                    // Backprop through all blocks (reverse order)
                    for (int blockIdx = blocks.Count - 1; blockIdx >= 0; blockIdx--)
                    {
                        var block = blocks[blockIdx];
                        var grad = blockGradients[blockIdx];
                        var h = hValues[blockIdx];
                        var residual = residuals[blockIdx];

                        // Capture weights with null checks
                        var wThetaF = block.WTheta_f;
                        var w1 = block.W1;
                        var b1 = block.B1;
                        var w2 = block.W2;
                        var b2 = block.B2;
                        var w3 = block.W3;
                        var b3 = block.B3;
                        var w4 = block.W4;
                        var b4 = block.B4;

                        if (wThetaF == null || w1 == null || b1 == null || w2 == null || b2 == null ||
                            w3 == null || b3 == null || w4 == null || b4 == null)
                        {
                            throw new InvalidOperationException("Block weights not initialized.");
                        }

                        // Gradient through WTheta_f (forecast theta)
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            for (int j = 0; j < _inputDim; j++)
                            {
                                grad.dWTheta_f[i, j] += NumOps.ToDouble(h[i]) * NumOps.ToDouble(gradForecast[j]);
                            }
                        }

                        // Gradient through h (from forecast)
                        var dH = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            T sum = NumOps.Zero;
                            for (int j = 0; j < _inputDim; j++)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(wThetaF[i, j], gradForecast[j]));
                            }
                            dH[i] = sum;
                        }

                        // Note: In full N-BEATS, backcast gradients would also contribute
                        // For simplicity, we focus on forecast gradients

                        // Backprop through FC4 (ReLU)
                        var h3 = ForwardFC(residual, w1, b1);
                        h3 = ApplyReLU(h3);
                        h3 = ForwardFC(h3, w2, b2);
                        h3 = ApplyReLU(h3);
                        h3 = ForwardFC(h3, w3, b3);
                        h3 = ApplyReLU(h3);

                        // ReLU derivative for h
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            if (NumOps.ToDouble(h[i]) <= 0) dH[i] = NumOps.Zero;
                        }

                        // Update W4, B4
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            double h3i = NumOps.ToDouble(h3[i]);
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                grad.dW4[i, j] += h3i * NumOps.ToDouble(dH[j]);
                            }
                            grad.dB4[i] += NumOps.ToDouble(dH[i]);
                        }

                        // Continue backprop through FC3
                        var dH3 = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            T sum = NumOps.Zero;
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(w4[i, j], dH[j]));
                            }
                            double h3i = NumOps.ToDouble(h3[i]);
                            dH3[i] = h3i <= 0 ? NumOps.Zero : sum;
                        }

                        var h2 = ForwardFC(residual, w1, b1);
                        h2 = ApplyReLU(h2);
                        h2 = ForwardFC(h2, w2, b2);
                        h2 = ApplyReLU(h2);

                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            double h2i = NumOps.ToDouble(h2[i]);
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                grad.dW3[i, j] += h2i * NumOps.ToDouble(dH3[j]);
                            }
                            grad.dB3[i] += NumOps.ToDouble(dH3[i]);
                        }

                        // Continue backprop through FC2
                        var dH2 = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            T sum = NumOps.Zero;
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(w3[i, j], dH3[j]));
                            }
                            double h2i = NumOps.ToDouble(h2[i]);
                            dH2[i] = h2i <= 0 ? NumOps.Zero : sum;
                        }

                        var h1 = ForwardFC(residual, w1, b1);
                        h1 = ApplyReLU(h1);

                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            double h1i = NumOps.ToDouble(h1[i]);
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                grad.dW2[i, j] += h1i * NumOps.ToDouble(dH2[j]);
                            }
                            grad.dB2[i] += NumOps.ToDouble(dH2[i]);
                        }

                        // Continue backprop through FC1
                        var dH1 = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            T sum = NumOps.Zero;
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(w2[i, j], dH2[j]));
                            }
                            double h1i = NumOps.ToDouble(h1[i]);
                            dH1[i] = h1i <= 0 ? NumOps.Zero : sum;
                        }

                        for (int i = 0; i < inputSize; i++)
                        {
                            double residualI = NumOps.ToDouble(residual[i]);
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                grad.dW1[i, j] += residualI * NumOps.ToDouble(dH1[j]);
                            }
                        }
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            grad.dB1[j] += NumOps.ToDouble(dH1[j]);
                        }
                    }
                }

                // Apply gradients to all blocks using NumOps
                double lr = _learningRate / actualBatchSize;
                double clipValue = 5.0;

                for (int blockIdx = 0; blockIdx < blocks.Count; blockIdx++)
                {
                    var block = blocks[blockIdx];
                    var grad = blockGradients[blockIdx];

                    // Capture weights with null checks
                    var w1 = block.W1;
                    var b1 = block.B1;
                    var w2 = block.W2;
                    var b2 = block.B2;
                    var w3 = block.W3;
                    var b3 = block.B3;
                    var w4 = block.W4;
                    var b4 = block.B4;
                    var wThetaF = block.WTheta_f;

                    if (w1 == null || b1 == null || w2 == null || b2 == null ||
                        w3 == null || b3 == null || w4 == null || b4 == null || wThetaF == null)
                    {
                        throw new InvalidOperationException("Block weights not initialized.");
                    }

                    // Update W1, B1
                    for (int i = 0; i < inputSize; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dW1[i, j]));
                            w1[i, j] = NumOps.Subtract(w1[i, j], NumOps.FromDouble(lr * clippedGrad));
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dB1[j]));
                        b1[j] = NumOps.Subtract(b1[j], NumOps.FromDouble(lr * clippedGrad));
                    }

                    // Update W2, B2
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dW2[i, j]));
                            w2[i, j] = NumOps.Subtract(w2[i, j], NumOps.FromDouble(lr * clippedGrad));
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dB2[j]));
                        b2[j] = NumOps.Subtract(b2[j], NumOps.FromDouble(lr * clippedGrad));
                    }

                    // Update W3, B3
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dW3[i, j]));
                            w3[i, j] = NumOps.Subtract(w3[i, j], NumOps.FromDouble(lr * clippedGrad));
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dB3[j]));
                        b3[j] = NumOps.Subtract(b3[j], NumOps.FromDouble(lr * clippedGrad));
                    }

                    // Update W4, B4
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dW4[i, j]));
                            w4[i, j] = NumOps.Subtract(w4[i, j], NumOps.FromDouble(lr * clippedGrad));
                        }
                    }
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dB4[j]));
                        b4[j] = NumOps.Subtract(b4[j], NumOps.FromDouble(lr * clippedGrad));
                    }

                    // Update WTheta_f
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _inputDim; j++)
                        {
                            double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, grad.dWTheta_f[i, j]));
                            wThetaF[i, j] = NumOps.Subtract(wThetaF[i, j], NumOps.FromDouble(lr * clippedGrad));
                        }
                    }
                }
            }
        }
    }

    private (Vector<T> forecast, Vector<T>[] residuals, Vector<T>[] hValues) ForwardWithCache(Vector<T> input)
    {
        var blocks = _blocks;
        if (blocks == null)
        {
            throw new InvalidOperationException("Model not initialized.");
        }

        int inputSize = input.Length;

        // Clone input for residual
        var residual = new Vector<T>(inputSize);
        for (int i = 0; i < inputSize; i++)
        {
            residual[i] = input[i];
        }

        var totalForecast = CreateZeroVector(_inputDim);

        var residuals = new Vector<T>[blocks.Count];
        var hValues = new Vector<T>[blocks.Count];

        for (int blockIdx = 0; blockIdx < blocks.Count; blockIdx++)
        {
            var block = blocks[blockIdx];

            // Clone residual for caching
            residuals[blockIdx] = new Vector<T>(inputSize);
            for (int i = 0; i < inputSize; i++)
            {
                residuals[blockIdx][i] = residual[i];
            }

            // Capture weights with null checks
            var w1 = block.W1;
            var b1 = block.B1;
            var w2 = block.W2;
            var b2 = block.B2;
            var w3 = block.W3;
            var b3 = block.B3;
            var w4 = block.W4;
            var b4 = block.B4;
            var wThetaB = block.WTheta_b;
            var wThetaF = block.WTheta_f;

            if (w1 == null || b1 == null || w2 == null || b2 == null ||
                w3 == null || b3 == null || w4 == null || b4 == null ||
                wThetaB == null || wThetaF == null)
            {
                throw new InvalidOperationException("Block weights not initialized.");
            }

            // FC stack (4 layers with ReLU)
            var h = ForwardFC(residual, w1, b1);
            h = ApplyReLU(h);
            h = ForwardFC(h, w2, b2);
            h = ApplyReLU(h);
            h = ForwardFC(h, w3, b3);
            h = ApplyReLU(h);
            h = ForwardFC(h, w4, b4);
            h = ApplyReLU(h);

            hValues[blockIdx] = h;

            // Generate backcast and forecast
            var backcast = ForwardFCNoOffset(h, wThetaB, inputSize);
            var forecast = ForwardFCNoOffset(h, wThetaF, _inputDim);

            // Update residual (subtract backcast)
            for (int i = 0; i < inputSize; i++)
            {
                residual[i] = NumOps.Subtract(residual[i], backcast[i]);
            }

            // Accumulate forecast
            for (int i = 0; i < _inputDim; i++)
            {
                totalForecast[i] = NumOps.Add(totalForecast[i], forecast[i]);
            }
        }

        return (totalForecast, residuals, hValues);
    }

    private class BlockGradients
    {
        public required double[,] dW1 { get; init; }
        public required double[] dB1 { get; init; }
        public required double[,] dW2 { get; init; }
        public required double[] dB2 { get; init; }
        public required double[,] dW3 { get; init; }
        public required double[] dB3 { get; init; }
        public required double[,] dW4 { get; init; }
        public required double[] dB4 { get; init; }
        public required double[,] dWTheta_b { get; init; }
        public required double[,] dWTheta_f { get; init; }
    }

    private (Vector<T> forecast, Vector<T> backcast) Forward(Vector<T> input)
    {
        var blocks = _blocks;
        if (blocks == null)
        {
            throw new InvalidOperationException("Model not initialized.");
        }

        int inputSize = input.Length;

        // Clone input for residual
        var residual = new Vector<T>(inputSize);
        for (int i = 0; i < inputSize; i++)
        {
            residual[i] = input[i];
        }

        var totalForecast = CreateZeroVector(_inputDim);
        var totalBackcast = CreateZeroVector(inputSize);

        foreach (var block in blocks)
        {
            // Capture weights with null checks
            var w1 = block.W1;
            var b1 = block.B1;
            var w2 = block.W2;
            var b2 = block.B2;
            var w3 = block.W3;
            var b3 = block.B3;
            var w4 = block.W4;
            var b4 = block.B4;
            var wThetaB = block.WTheta_b;
            var wThetaF = block.WTheta_f;

            if (w1 == null || b1 == null || w2 == null || b2 == null ||
                w3 == null || b3 == null || w4 == null || b4 == null ||
                wThetaB == null || wThetaF == null)
            {
                throw new InvalidOperationException("Block weights not initialized.");
            }

            // FC stack (4 layers with ReLU)
            var h = ForwardFC(residual, w1, b1);
            h = ApplyReLU(h);
            h = ForwardFC(h, w2, b2);
            h = ApplyReLU(h);
            h = ForwardFC(h, w3, b3);
            h = ApplyReLU(h);
            h = ForwardFC(h, w4, b4);
            h = ApplyReLU(h);

            // Generate backcast and forecast
            var backcast = ForwardFCNoOffset(h, wThetaB, inputSize);
            var forecast = ForwardFCNoOffset(h, wThetaF, _inputDim);

            // Update residual (subtract backcast)
            for (int i = 0; i < inputSize; i++)
            {
                residual[i] = NumOps.Subtract(residual[i], backcast[i]);
            }

            // Accumulate forecast
            for (int i = 0; i < _inputDim; i++)
            {
                totalForecast[i] = NumOps.Add(totalForecast[i], forecast[i]);
            }

            // Accumulate backcast
            for (int i = 0; i < inputSize; i++)
            {
                totalBackcast[i] = NumOps.Add(totalBackcast[i], backcast[i]);
            }
        }

        return (totalForecast, totalBackcast);
    }

    private Vector<T> ForwardFC(Vector<T> input, Matrix<T> W, Vector<T> b)
    {
        int outputSize = W.Columns;
        var output = new Vector<T>(outputSize);

        for (int j = 0; j < outputSize; j++)
        {
            T sum = b[j];
            for (int i = 0; i < input.Length; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(input[i], W[i, j]));
            }
            output[j] = sum;
        }

        return output;
    }

    private Vector<T> ForwardFCNoOffset(Vector<T> input, Matrix<T> W, int outputSize)
    {
        var output = new Vector<T>(outputSize);

        for (int j = 0; j < outputSize; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < input.Length; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(input[i], W[i, j]));
            }
            output[j] = sum;
        }

        return output;
    }

    private Vector<T> ApplyReLU(Vector<T> x)
    {
        var output = new Vector<T>(x.Length);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            output[i] = NumOps.FromDouble(Math.Max(0, val));
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

        // Normalize data into Matrix<T>
        var normalizedData = new Matrix<T>(n, _inputDim);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(X[i, j], dataMeans[j]);
                normalizedData[i, j] = NumOps.Divide(diff, dataStds[j]);
            }
        }

        // Score each point
        for (int i = 0; i < n; i++)
        {
            T score;

            if (i < _lookback)
            {
                // Not enough history
                score = NumOps.Zero;
                for (int j = 0; j < _inputDim; j++)
                {
                    T val = normalizedData[i, j];
                    score = NumOps.Add(score, NumOps.Multiply(val, val));
                }
            }
            else
            {
                // Build input from lookback window
                var input = new Vector<T>(_lookback * _inputDim);
                for (int t = 0; t < _lookback; t++)
                {
                    for (int j = 0; j < _inputDim; j++)
                    {
                        input[t * _inputDim + j] = normalizedData[i - _lookback + t, j];
                    }
                }

                // Predict and compute error
                var (forecast, _) = Forward(input);
                score = NumOps.Zero;
                for (int j = 0; j < _inputDim; j++)
                {
                    T diff = NumOps.Subtract(normalizedData[i, j], forecast[j]);
                    score = NumOps.Add(score, NumOps.Multiply(diff, diff));
                }
            }

            scores[i] = score;
        }

        return scores;
    }

    private class BlockWeights
    {
        public Matrix<T>? W1 { get; set; }
        public Vector<T>? B1 { get; set; }
        public Matrix<T>? W2 { get; set; }
        public Vector<T>? B2 { get; set; }
        public Matrix<T>? W3 { get; set; }
        public Vector<T>? B3 { get; set; }
        public Matrix<T>? W4 { get; set; }
        public Vector<T>? B4 { get; set; }
        public Matrix<T>? WTheta_b { get; set; }
        public Matrix<T>? WTheta_f { get; set; }
    }
}
