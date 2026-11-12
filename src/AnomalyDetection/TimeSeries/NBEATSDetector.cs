using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting", "https://doi.org/10.48550/arXiv.1905.10437", Year = 2020, Authors = "Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio")]
public partial class NBEATSDetector<T> : AnomalyDetectorBase<T>
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
    [Buffer]
    private Vector<T>? _dataMeans;
    [Buffer]
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
        // Per-feature mean and population std (std below 1e-10 replaced by 1), then (x - mean) / std
        var engine = Engine;
        var values = AnomalyTensorOps.ToTensor(data);
        var (std, mean) = engine.TensorStdMean(values, new[] { 0 }, 0, false);
        var tiny = engine.TensorLessThan(std, NumOps.FromDouble(1e-10));
        var stds = engine.TensorWhere(tiny, Tensor<T>.CreateOnes(new[] { data.Columns }), std);
        var normalized = AnomalyTensorOps.DivideRow(AnomalyTensorOps.SubtractRow(values, mean), stds);
        return (normalized.ToMatrix(), mean.ToVector(), stds.ToVector());
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
                W1 = Tensor<T>.FromMatrix(InitializeMatrix(inputSize, _hiddenDim)),
                B1 = Tensor<T>.CreateZeros(new[] { _hiddenDim }),
                W2 = Tensor<T>.FromMatrix(InitializeMatrix(_hiddenDim, _hiddenDim)),
                B2 = Tensor<T>.CreateZeros(new[] { _hiddenDim }),
                W3 = Tensor<T>.FromMatrix(InitializeMatrix(_hiddenDim, _hiddenDim)),
                B3 = Tensor<T>.CreateZeros(new[] { _hiddenDim }),
                W4 = Tensor<T>.FromMatrix(InitializeMatrix(_hiddenDim, _hiddenDim)),
                B4 = Tensor<T>.CreateZeros(new[] { _hiddenDim }),
                // Theta for backcast and forecast
                WTheta_b = Tensor<T>.FromMatrix(InitializeMatrix(_hiddenDim, inputSize)),
                WTheta_f = Tensor<T>.FromMatrix(InitializeMatrix(_hiddenDim, _inputDim)) // Forecast horizon = 1
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

    // Flattened lookback windows data[i .. i + lookback) ([count, lookback * features]) for i < count.
    private Tensor<T> LookbackWindows(Tensor<T> data, int count)
    {
        var engine = Engine;
        int lookback = _lookback;
        var rows = new int[count * lookback];
        for (int i = 0; i < count; i++)
        {
            for (int t = 0; t < lookback; t++)
            {
                rows[i * lookback + t] = i + t;
            }
        }

        var gathered = engine.TensorGather(data, Tensor<int>.FromVector(Vector<int>.Wrap(rows), new[] { rows.Length }), 0);
        return engine.Reshape(gathered, new[] { count, lookback * data.Shape[1] });
    }

    private (Tensor<T> inputs, Tensor<T> targets) CreateSequences(Matrix<T> data)
    {
        // Every flattened lookback window and the row that follows it
        int n = data.Rows - _lookback;
        var values = AnomalyTensorOps.ToTensor(data);
        var inputs = LookbackWindows(values, n);
        var targets = Engine.TensorSlice(values, new[] { _lookback, 0 }, new[] { n, data.Columns });
        return (inputs, targets);
    }

    private void Train(Tensor<T> inputs, Tensor<T> targets)
    {
        var engine = Engine;
        int n = inputs.Shape[0];
        int batchSize = Math.Min(32, n);
        var blocks = _blocks;
        if (blocks == null)
        {
            throw new InvalidOperationException("Model not initialized.");
        }

        T two = NumOps.FromDouble(2.0);
        T clipValue = NumOps.FromDouble(5.0);
        T negClipValue = NumOps.Negate(clipValue);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batch, batchRows, 0, actualBatchSize);
                var input = AnomalyTensorOps.SelectRows(inputs, batchRows);
                var target = AnomalyTensorOps.SelectRows(targets, batchRows);

                // Forward pass with caching; MSE loss gradient 2 * (forecast - target), one row per sample
                var (forecast, caches) = ForwardBatch(input);
                var gradForecast = engine.TensorMultiplyScalar(engine.TensorSubtract(forecast, target), two);

                // Backprop through all blocks (reverse order), summing the gradients over the batch
                var gradients = new BlockGradients[blocks.Count];
                for (int blockIdx = blocks.Count - 1; blockIdx >= 0; blockIdx--)
                {
                    var block = blocks[blockIdx];
                    var cache = caches[blockIdx];

                    // Gradient through WTheta_f (forecast theta), and through h from the forecast
                    var dWThetaF = engine.TensorMatMul(engine.TensorTranspose(cache.H4), gradForecast);
                    var dH = engine.TensorMatMul(gradForecast, engine.TensorTranspose(block.WTheta_f));

                    // Gradient through WTheta_b (backcast theta)
                    // The backcast affects residual learning: residual_next = residual - backcast
                    // Gradient flows from subsequent blocks back through the residual chain
                    // For the last block, backcast gradient is zero (no subsequent blocks)
                    // For earlier blocks, gradient comes from how backcast affects next block's forecast:
                    // dL/d(backcast) = -dL/d(residual_next) = -dH_next * W1_next^T (simplified)
                    var dWThetaB = Tensor<T>.CreateZeros(new[] { block.WTheta_b.Shape[0], block.WTheta_b.Shape[1] });
                    if (blockIdx < blocks.Count - 1)
                    {
                        var nextBlock = blocks[blockIdx + 1];
                        var nextH = caches[blockIdx + 1].H4;

                        // dH for the next block from its forecast contribution, through its ReLU
                        var dHNext = engine.TensorMultiply(
                            engine.TensorMatMul(gradForecast, engine.TensorTranspose(nextBlock.WTheta_f)),
                            engine.TensorGreaterThan(nextH, NumOps.Zero));

                        // Backprop through next block's first FC layer to get dResidual; the gradient for
                        // WTheta_b is h * (-dResidual)^T (negative because residual = prev_residual - backcast)
                        var negDResidual = engine.TensorNegate(
                            engine.TensorMatMul(dHNext, engine.TensorTranspose(nextBlock.W1)));
                        dWThetaB = engine.TensorMatMul(engine.TensorTranspose(cache.H4), negDResidual);

                        // Also add gradient through h from backcast path
                        dH = engine.TensorAdd(dH, engine.TensorMatMul(negDResidual, engine.TensorTranspose(block.WTheta_b)));
                    }

                    // ReLU derivative for h, then backprop through FC4 .. FC1 (each with ReLU)
                    dH = engine.TensorMultiply(dH, engine.TensorGreaterThan(cache.H4, NumOps.Zero));
                    var dH3 = engine.TensorMultiply(engine.TensorMatMul(dH, engine.TensorTranspose(block.W4)),
                        engine.TensorGreaterThan(cache.H3, NumOps.Zero));
                    var dH2 = engine.TensorMultiply(engine.TensorMatMul(dH3, engine.TensorTranspose(block.W3)),
                        engine.TensorGreaterThan(cache.H2, NumOps.Zero));
                    var dH1 = engine.TensorMultiply(engine.TensorMatMul(dH2, engine.TensorTranspose(block.W2)),
                        engine.TensorGreaterThan(cache.H1, NumOps.Zero));

                    gradients[blockIdx] = new BlockGradients
                    {
                        dW1 = engine.TensorMatMul(engine.TensorTranspose(cache.Residual), dH1),
                        dB1 = engine.ReduceSum(dH1, new[] { 0 }, false),
                        dW2 = engine.TensorMatMul(engine.TensorTranspose(cache.H1), dH2),
                        dB2 = engine.ReduceSum(dH2, new[] { 0 }, false),
                        dW3 = engine.TensorMatMul(engine.TensorTranspose(cache.H2), dH3),
                        dB3 = engine.ReduceSum(dH3, new[] { 0 }, false),
                        dW4 = engine.TensorMatMul(engine.TensorTranspose(cache.H3), dH),
                        dB4 = engine.ReduceSum(dH, new[] { 0 }, false),
                        dWTheta_b = dWThetaB,
                        dWTheta_f = dWThetaF
                    };
                }

                // Apply clipped gradients (clip to [-5, 5], step learningRate / batch size) to all blocks
                T lr = NumOps.FromDouble(_learningRate / actualBatchSize);
                for (int blockIdx = 0; blockIdx < blocks.Count; blockIdx++)
                {
                    var block = blocks[blockIdx];
                    var grad = gradients[blockIdx];
                    block.W1 = Step(block.W1, grad.dW1, lr, negClipValue, clipValue);
                    block.B1 = Step(block.B1, grad.dB1, lr, negClipValue, clipValue);
                    block.W2 = Step(block.W2, grad.dW2, lr, negClipValue, clipValue);
                    block.B2 = Step(block.B2, grad.dB2, lr, negClipValue, clipValue);
                    block.W3 = Step(block.W3, grad.dW3, lr, negClipValue, clipValue);
                    block.B3 = Step(block.B3, grad.dB3, lr, negClipValue, clipValue);
                    block.W4 = Step(block.W4, grad.dW4, lr, negClipValue, clipValue);
                    block.B4 = Step(block.B4, grad.dB4, lr, negClipValue, clipValue);
                    block.WTheta_f = Step(block.WTheta_f, grad.dWTheta_f, lr, negClipValue, clipValue);
                    block.WTheta_b = Step(block.WTheta_b, grad.dWTheta_b, lr, negClipValue, clipValue);
                }
            }
        }
    }

    // weights - lr * clamp(gradient, -clip, clip)
    private Tensor<T> Step(Tensor<T> weights, Tensor<T> gradient, T lr, T negClip, T clip)
    {
        var engine = Engine;
        return engine.TensorSubtract(weights, engine.TensorMultiplyScalar(engine.TensorClamp(gradient, negClip, clip), lr));
    }

    // Forward pass over a batch of flattened windows ([batch, lookback * features]): the summed forecast
    // ([batch, features]) and, per block, its input residual and FC activations.
    private (Tensor<T> forecast, BlockCache[] caches) ForwardBatch(Tensor<T> input)
    {
        var blocks = _blocks;
        if (blocks == null)
        {
            throw new InvalidOperationException("Model not initialized.");
        }

        var engine = Engine;
        int batch = input.Shape[0];
        var residual = input;
        var totalForecast = Tensor<T>.CreateZeros(new[] { batch, _inputDim });
        var caches = new BlockCache[blocks.Count];

        for (int blockIdx = 0; blockIdx < blocks.Count; blockIdx++)
        {
            var block = blocks[blockIdx];

            // FC stack (4 layers with ReLU)
            var h1 = ForwardFC(residual, block.W1, block.B1);
            var h2 = ForwardFC(h1, block.W2, block.B2);
            var h3 = ForwardFC(h2, block.W3, block.B3);
            var h4 = ForwardFC(h3, block.W4, block.B4);
            caches[blockIdx] = new BlockCache { Residual = residual, H1 = h1, H2 = h2, H3 = h3, H4 = h4 };

            // Generate backcast and forecast; subtract the backcast from the residual, accumulate the forecast
            residual = engine.TensorSubtract(residual, engine.TensorMatMul(h4, block.WTheta_b));
            totalForecast = engine.TensorAdd(totalForecast, engine.TensorMatMul(h4, block.WTheta_f));
        }

        return (totalForecast, caches);
    }

    // ReLU(input @ W + b) for a batch of rows.
    private Tensor<T> ForwardFC(Tensor<T> input, Tensor<T> W, Tensor<T> b)
    {
        var engine = Engine;
        var linear = engine.TensorAdd(engine.TensorMatMul(input, W),
            AnomalyTensorOps.BroadcastRow(b, input.Shape[0]));
        return engine.TensorMax(linear, NumOps.Zero);
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

        var engine = Engine;
        int n = X.Rows;

        // Normalize data
        var normalizedData = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());

        // Points without enough history for forecast-based scoring get the sentinel 0 (no anomaly score
        // available); the others score the squared error of the forecast from their lookback window
        int count = n - _lookback;
        if (count < 1)
        {
            return new Vector<T>(n);
        }

        var (forecast, _) = ForwardBatch(LookbackWindows(normalizedData, count));
        var actual = engine.TensorSlice(normalizedData, new[] { _lookback, 0 }, new[] { count, _inputDim });
        var errors = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(actual, forecast)), new[] { 1 }, false);
        return engine.TensorConcatenate(new[] { Tensor<T>.CreateZeros(new[] { _lookback }), errors }, 0).ToVector();
    }

    private class BlockCache
    {
        public required Tensor<T> Residual { get; init; }
        public required Tensor<T> H1 { get; init; }
        public required Tensor<T> H2 { get; init; }
        public required Tensor<T> H3 { get; init; }
        public required Tensor<T> H4 { get; init; }
    }

    private class BlockGradients
    {
        public required Tensor<T> dW1 { get; init; }
        public required Tensor<T> dB1 { get; init; }
        public required Tensor<T> dW2 { get; init; }
        public required Tensor<T> dB2 { get; init; }
        public required Tensor<T> dW3 { get; init; }
        public required Tensor<T> dB3 { get; init; }
        public required Tensor<T> dW4 { get; init; }
        public required Tensor<T> dB4 { get; init; }
        public required Tensor<T> dWTheta_b { get; init; }
        public required Tensor<T> dWTheta_f { get; init; }
    }

    private class BlockWeights
    {
        public required Tensor<T> W1 { get; set; }
        public required Tensor<T> B1 { get; set; }
        public required Tensor<T> W2 { get; set; }
        public required Tensor<T> B2 { get; set; }
        public required Tensor<T> W3 { get; set; }
        public required Tensor<T> B3 { get; set; }
        public required Tensor<T> W4 { get; set; }
        public required Tensor<T> B4 { get; set; }
        public required Tensor<T> WTheta_b { get; set; }
        public required Tensor<T> WTheta_f { get; set; }
    }
}
