using System.Collections.Concurrent;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Autoformer model for long-term time series forecasting with decomposition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Long-Term Forecasting Challenge:</b>
/// Long-term time series forecasting requires models that can capture both fine-grained seasonal
/// patterns and long-term trends. Traditional approaches struggle because:
/// - RNNs have difficulty with long-range dependencies
/// - Transformers treat time series like text, ignoring continuous nature
/// - Neither explicitly models trend and seasonality separately
/// </para>
/// <para>
/// <b>The Autoformer Solution (Wu et al., NeurIPS 2021):</b>
/// Autoformer introduces three key innovations:
///
/// 1. <b>Series Decomposition Block:</b>
///    Progressive separation of trend and seasonal components at each layer.
///    Uses moving average to extract trend, remainder is seasonal.
///    Formula: Trend = MovingAvg(X), Seasonal = X - Trend
///
/// 2. <b>Auto-Correlation Mechanism:</b>
///    Replaces point-wise self-attention with period-based dependencies.
///    Finds correlations between sub-series and aggregates similar sub-sequences
///    based on their correlation strength.
///
/// 3. <b>Progressive Decomposition Architecture:</b>
///    Each encoder/decoder layer further refines the decomposition.
///    Seasonal and trend branches are processed separately and accumulated.
/// </para>
/// <para>
/// <b>For Beginners:</b> Autoformer is like having two experts work together:
/// - One expert focuses on the long-term direction (trend)
/// - One expert focuses on repeating patterns (seasonality)
///
/// Instead of looking at individual data points, it looks at how patterns repeat over time.
/// If today's pattern looks like last week's pattern, that's useful information!
///
/// Example use cases:
/// - Electricity demand forecasting (daily/weekly patterns)
/// - Retail sales prediction (seasonal buying patterns)
/// - Traffic flow prediction (rush hour patterns)
/// </para>
/// <para>
/// <b>Training (tape-based automatic differentiation):</b>
/// The whole forward pass (<see cref="ForwardCore"/>) is expressed with batched
/// <c>Engine.Tensor*</c> ops, so a <see cref="Tensors.Engines.Autodiff.GradientTape{T}"/>
/// produces the entire backward pass automatically — there is no hand-derived gradient
/// code — and every op is GPU-dispatchable. Series decomposition (the defining
/// moving-average trend/seasonal split) and a tape-differentiable time-delay
/// auto-correlation are both retained. Inference uses the exact same ops as training
/// (no scalar/tape divergence). Windows of the z-normalized series are trained with
/// MSE over the full forecast horizon, one Adam step per mini-batch, and the
/// best-epoch parameters are snapshotted and restored.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var inputMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var trainingLabels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Create an Autoformer with auto-correlation for long-range forecasting
/// var options = new AutoformerOptions&lt;double&gt;
/// {
///     
///     EmbeddingDim = 512
/// };
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new AutoformerModel&lt;double&gt;(options))
///     .Build(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = result.Predict(inputMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", "https://arxiv.org/abs/2106.13008", Year = 2021, Authors = "Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long")]
public partial class AutoformerModel<T> : TimeSeriesModelBase<T>, ISupportsLossFunction<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Autoformer is a point forecaster: its head emits a single value per horizon step, so any
    /// pointwise loss is meaningful. Defaults to mean squared error when none is configured.
    /// </remarks>
    public void SetLossFunction(ILossFunction<T> lossFunction) => ApplyLossFunction(lossFunction);

    private readonly AutoformerOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Series decomposition components
    private readonly int _movingAvgKernel;

    // Input embedding
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _inputProjection;      // [embeddingDim, 1]
    [Buffer]
    private Tensor<T> _positionalEncoding;   // [maxLen, embeddingDim]
    // Host-side copy of the (constant) positional encoding. The forward assembles a per-window PE tensor from
    // it every step; indexing _positionalEncoding[i] on a GPU-resident tensor would sync per element (seqLen*dim
    // reads per forward). The PE never changes after construction, so cache one host download and read that.
    private T[]? _positionalEncodingHost;

    // Encoder components
    private readonly List<AutoformerEncoderLayer<T>> _encoderLayers;

    // Decoder components
    private readonly List<AutoformerDecoderLayer<T>> _decoderLayers;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _decoderSeasonalInit;  // [forecastHorizon, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _decoderTrendInit;     // [forecastHorizon, embeddingDim]

    // Output projections
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _seasonalProjection;   // [1, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _trendProjection;      // [1, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputBias;           // [forecastHorizon]

    // Normalization statistics computed during training (zero-mean / unit-variance of the
    // training series). Inputs are normalized before the network and the forecast is
    // denormalized at inference so gradient flow stays well-scaled (mirrors NBEATS / Informer).
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    /// <summary>
    /// Initializes a new instance of the Autoformer model with the specified options.
    /// </summary>
    /// <param name="options">Configuration options for the model. Uses defaults if null.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create an Autoformer like this:
    /// <code>
    /// var model = new AutoformerModel&lt;double&gt;(new AutoformerOptions&lt;double&gt;
    /// {
    ///     LookbackWindow = 96,    // Look at past 96 time steps
    ///     ForecastHorizon = 24,   // Predict next 24 time steps
    ///     MovingAverageKernel = 25 // Trend smoothing window
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    public AutoformerModel(AutoformerOptions<T>? options = null)
        : this(options ?? new AutoformerOptions<T>(), initializeModel: true)
    {
    }

    private AutoformerModel(AutoformerOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        Options = _options;

        // Validate options
        if (_options.EmbeddingDim <= 0)
            throw new ArgumentException("EmbeddingDim must be positive.", nameof(options));
        if (_options.NumEncoderLayers <= 0)
            throw new ArgumentException("NumEncoderLayers must be positive.", nameof(options));
        if (_options.NumDecoderLayers <= 0)
            throw new ArgumentException("NumDecoderLayers must be positive.", nameof(options));
        if (_options.NumAttentionHeads <= 0)
            throw new ArgumentException("NumAttentionHeads must be positive.", nameof(options));
        if (_options.MovingAverageKernel <= 0 || _options.MovingAverageKernel % 2 == 0)
            throw new ArgumentException("MovingAverageKernel must be a positive odd number.", nameof(options));

        _random = RandomHelper.CreateSeededRandom(SeedOr(42));
        _movingAvgKernel = _options.MovingAverageKernel;
        _encoderLayers = new List<AutoformerEncoderLayer<T>>();
        _decoderLayers = new List<AutoformerDecoderLayer<T>>();

        // Initialize with default tensors
        _inputProjection = new Tensor<T>(new[] { 1, 1 });
        _positionalEncoding = new Tensor<T>(new[] { 1, 1 });
        _decoderSeasonalInit = new Tensor<T>(new[] { 1, 1 });
        _decoderTrendInit = new Tensor<T>(new[] { 1, 1 });
        _seasonalProjection = new Tensor<T>(new[] { 1, 1 });
        _trendProjection = new Tensor<T>(new[] { 1, 1 });
        _outputBias = new Tensor<T>(new[] { 1 });

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        double stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);
        var random = RandomHelper.CreateSeededRandom(SeedOr(42));

        // Input projection: maps single time step values to embedding dimension
        _inputProjection = InitTensor(new[] { _options.EmbeddingDim, 1 }, stddev, random);

        // Sinusoidal positional encoding
        int maxLen = Math.Max(_options.LookbackWindow, _options.ForecastHorizon) * 2;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);
        _positionalEncodingHost = _positionalEncoding.GetCpuData();

        // Encoder layers with series decomposition and auto-correlation
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new AutoformerEncoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _movingAvgKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                SeedOr(42) + i));
        }

        // Decoder layers
        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            _decoderLayers.Add(new AutoformerDecoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _movingAvgKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                SeedOr(42) + _options.NumEncoderLayers + i));
        }

        // Decoder initialization tensors (learnable)
        _decoderSeasonalInit = InitTensor(new[] { _options.ForecastHorizon, _options.EmbeddingDim }, stddev * 0.1, random);
        _decoderTrendInit = InitTensor(new[] { _options.ForecastHorizon, _options.EmbeddingDim }, stddev * 0.1, random);

        // Output projections for seasonal and trend components
        _seasonalProjection = InitTensor(new[] { 1, _options.EmbeddingDim }, stddev, random);
        _trendProjection = InitTensor(new[] { 1, _options.EmbeddingDim }, stddev, random);
        _outputBias = new Tensor<T>(new[] { _options.ForecastHorizon });
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    private Tensor<T> CreateSinusoidalPositionalEncoding(int maxLen, int embeddingDim)
    {
        var encoding = new Tensor<T>(new[] { maxLen, embeddingDim });
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2.0)) / embeddingDim);
                double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                encoding[pos * embeddingDim + i] = _numOps.FromDouble(value);
            }
        }
        return encoding;
    }

    /// <summary>
    /// Trains the Autoformer with tape-based automatic differentiation and the Adam optimizer.
    /// </summary>
    /// <remarks>
    /// The forward pass (<see cref="ForwardCore"/>) is expressed entirely with <c>Engine.Tensor*</c>
    /// ops, so a <see cref="Tensors.Engines.Autodiff.GradientTape{T}"/> produces the whole backward
    /// pass automatically (no hand-derived gradients) and every op is GPU-dispatchable. The series is
    /// z-normalized for stable gradient flow; each mini-batch accumulates the per-sample gradients
    /// (each computed on its own tape, disposed right after its backward) and takes ONE averaged Adam
    /// step per batch (so Adam's moment estimates do not thrash the way a per-sample step would).
    /// Best-epoch parameters are snapshotted and restored so a noisy late-epoch step cannot degrade
    /// the returned model.
    ///
    /// NOTE: samples in a batch are run through <see cref="ForwardCore"/> individually and their
    /// gradients summed then averaged. This yields the exact mini-batch gradient (mean over the batch)
    /// with all ops on the tape and GPU-dispatchable, while keeping peak memory at a single forward.
    /// Folding the batch into a single leading tensor dimension for one large GEMM per op is a
    /// throughput optimization left to the separate GPU-residency work (the auto-correlation top-k
    /// selection is per-sample data-dependent, so a true batched layout needs a per-batch-row gather
    /// that the current engine ops do not express).
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Store training series BEFORE the loop for cancellation safety / in-sample predictions.
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++) _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _numOps.FromDouble(y.Length);

        int lookback = _options.LookbackWindow;
        int horizon = _options.ForecastHorizon;

        // z-normalize the series for stable gradient flow (mirrors NBEATS / Informer).
        T yMean = _numOps.Zero;
        for (int i = 0; i < y.Length; i++) yMean = _numOps.Add(yMean, y[i]);
        yMean = _numOps.Divide(yMean, _numOps.FromDouble(y.Length));
        T yVar = _numOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T diff = _numOps.Subtract(y[i], yMean);
            yVar = _numOps.Add(yVar, _numOps.Multiply(diff, diff));
        }
        yVar = _numOps.Divide(yVar, _numOps.FromDouble(y.Length));
        T yStd = _numOps.Sqrt(yVar);
        if (_numOps.LessThanOrEquals(yStd, _numOps.FromDouble(1e-10))) yStd = _numOps.One;
        _normMean = yMean;
        _normStd = yStd;

        var yNorm = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            yNorm[i] = _numOps.Divide(_numOps.Subtract(y[i], yMean), yStd);

        // Valid sample = index with a complete lookback AND horizon window (idx in [L, N-H]).
        var validIndices = new List<int>();
        for (int i = lookback; i + horizon <= y.Length; i++) validIndices.Add(i);
        if (validIndices.Count == 0)
        {
            throw new ArgumentException(
                $"Not enough data to build a single training sample. Require at least " +
                $"{lookback + horizon} points, got {y.Length}.", nameof(y));
        }

        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
        var mseLoss = TrainingLoss;
        var allParams = CollectTrainableParameters();

        int batchSize = Math.Max(1, _options.BatchSize);
        bool timeBounded = _options.MaxTrainingTimeSeconds > 0;
        int maxEpochs = timeBounded ? int.MaxValue : _options.Epochs;

        double bestLoss = double.PositiveInfinity;
        List<Tensor<T>>? bestSnapshot = null;

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
            TrainingCancellationToken.ThrowIfCancellationRequested();

            var shuffled = validIndices.OrderBy(_ => _random.Next()).ToList();
            double epochLossSum = 0.0;
            int epochCount = 0;

            for (int start = 0; start < shuffled.Count; start += batchSize)
            {
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
                TrainingCancellationToken.ThrowIfCancellationRequested();

                int end = Math.Min(start + batchSize, shuffled.Count);
                int b = end - start;

                // Input [B, L], target [B, H]. The batched forward runs every window at once, each still choosing
                // its own top-k delays on the device; the loss, backward and Adam update are one fused compiled
                // plan when it applies (CPU or GPU), the eager tape otherwise (TrainTapeBatch).
                var inputData = new T[b * lookback];
                var targetData = new T[b * horizon];
                for (int bi = 0; bi < b; bi++)
                {
                    int idx = shuffled[start + bi];
                    for (int t = 0; t < lookback; t++) inputData[bi * lookback + t] = yNorm[idx - lookback + t];
                    for (int h = 0; h < horizon; h++) targetData[bi * horizon + h] = yNorm[idx + h];
                }
                var batchInput = new Tensor<T>(new[] { b, lookback }, new Vector<T>(inputData));
                var batchTarget = new Tensor<T>(new[] { b, horizon }, new Vector<T>(targetData));

                T lossValue = TrainTapeBatch(
                    Array.Empty<ITrainableLayer<T>>(), batchInput, batchTarget,
                    ForwardBatched, mseLoss.ComputeTapeLoss, optimizer, extraParameters: allParams);

                epochLossSum += Convert.ToDouble(lossValue) * b;
                epochCount += b;
            }

            if (epochCount > 0)
            {
                double epochLoss = epochLossSum / epochCount;
                if (!double.IsNaN(epochLoss) && !double.IsInfinity(epochLoss) && epochLoss < bestLoss)
                {
                    bestLoss = epochLoss;
                    bestSnapshot = allParams.Select(p => p.Clone()).ToList();
                }

                // Report after checkpointing so an early stop still leaves the best weights to
                // restore below.
                if (!ReportEpoch(epoch, timeBounded ? 0 : _options.Epochs, NumOps.FromDouble(epochLoss)))
                {
                    break;
                }
            }
        }

        // Restore best-epoch parameters (copy values into the live tensors the model uses).
        if (bestSnapshot is not null)
        {
            for (int i = 0; i < allParams.Count; i++)
            {
                var src = bestSnapshot[i];
                var dst = allParams[i];
                for (int k = 0; k < Math.Min(src.Length, dst.Length); k++) dst[k] = src[k];
            }
        }
    }

    // ── Inference forward (same Engine ops as training) ─────────────────────────────
    // Normalize the raw lookback with the training statistics, run the tape-differentiable
    // core, and denormalize the forecast. Returns [forecastHorizon, 1].
    private Tensor<T> ForwardEngine(Vector<T> rawInput)
    {
        int seqLen = Math.Min(rawInput.Length, _options.LookbackWindow);
        var norm = new Vector<T>(seqLen);
        for (int t = 0; t < seqLen; t++)
            norm[t] = _numOps.Divide(_numOps.Subtract(rawInput[t], _normMean), _normStd);

        var outNorm = ForwardCore(norm);
        int horizon = _options.ForecastHorizon;
        var result = new Tensor<T>(new[] { horizon, 1 });
        for (int h = 0; h < Math.Min(horizon, outNorm.Length); h++)
            result[h] = _numOps.Add(_numOps.Multiply(outNorm[h], _normStd), _normMean);
        return result;
    }

    // Forward on ONE already z-normalized lookback window: the batched forward with a batch of one, returned as
    // the normalized [forecastHorizon, 1] forecast. Training and inference run the same graph.
    internal Tensor<T> ForwardCore(Vector<T> input)
    {
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);
        var inData = new Vector<T>(seqLen);
        for (int t = 0; t < seqLen; t++) inData[t] = input[t];
        var forecast = ForwardBatched(new Tensor<T>(new[] { 1, seqLen }, inData));
        return Engine.Reshape(forecast, new[] { _options.ForecastHorizon, 1 });
    }
    /// <summary>
    /// Batched core forward on ALREADY z-normalized windows: <c>[B, L]</c> -> normalized forecast <c>[B, H]</c>.
    /// </summary>
    /// <remarks>
    /// The same model as the per-sample forward, one rank wider, with nothing read back to the host: each window's
    /// delays are chosen on the device (a rank mask over its own correlation spectrum) and its rolled-value
    /// aggregation is a one-hot-weighted sum of constant shift operators, so one compiled plan replays it for every
    /// batch with each window still picking its own delays.
    /// </remarks>
    internal Tensor<T> ForwardBatched(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int embDim = _options.EmbeddingDim;
        int horizon = _options.ForecastHorizon;

        // Embedding: x @ inputProjᵀ plus the positional encoding of each step, [B, L, D].
        var flatIn = Engine.Reshape(input, new[] { batch * seqLen, 1 });
        var embedded = Engine.Reshape(
            Engine.TensorMatMul(flatIn, Engine.Reshape(_inputProjection, new[] { 1, embDim })),
            new[] { batch, seqLen, embDim });
        embedded = Engine.TensorAdd(embedded, PositionalEncodingSlice(seqLen));

        // Series decomposition, then the encoder.
        var trend = MovingAverageBatched(embedded, _movingAvgKernel, seqLen);
        var seasonal = Engine.TensorSubtract(embedded, trend);
        for (int i = 0; i < _encoderLayers.Count; i++)
            (seasonal, trend) = EncoderLayerBatched(seasonal, trend, i);

        // Decoder init (Autoformer §3.2): the learned seasonal placeholder, and a trend init that is the mean of
        // each window's embedded input plus a learned refinement, broadcast over the horizon.
        var embMean = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(embedded, new[] { 1 }, keepDims: true), _numOps.FromDouble(1.0 / seqLen)); // [B, 1, D]
        var decTrend = Engine.TensorAdd(Engine.Reshape(_decoderTrendInit, new[] { 1, horizon, embDim }), embMean);
        var decSeasonal = Engine.TensorAdd(
            Engine.Reshape(_decoderSeasonalInit, new[] { 1, horizon, embDim }), new Tensor<T>(new[] { batch, 1, 1 }));
        for (int i = 0; i < _decoderLayers.Count; i++)
            (decSeasonal, decTrend) = DecoderLayerBatched(decSeasonal, decTrend, seasonal, trend, i);

        // Output projection: seasonal·projᵀ + trend·projᵀ + bias, [B, H].
        var output = Engine.TensorAdd(
            Engine.TensorMatMul(Engine.Reshape(decSeasonal, new[] { batch * horizon, embDim }),
                Engine.TensorTranspose(_seasonalProjection)),
            Engine.TensorMatMul(Engine.Reshape(decTrend, new[] { batch * horizon, embDim }),
                Engine.TensorTranspose(_trendProjection)));
        return Engine.TensorAdd(
            Engine.Reshape(output, new[] { batch, horizon }), Engine.Reshape(_outputBias, new[] { 1, horizon }));
    }

    /// <summary>The first <paramref name="seqLen"/> rows of the positional encoding as a constant <c>[1, L, D]</c>.</summary>
    private Tensor<T> PositionalEncodingSlice(int seqLen)
    {
        int embDim = _options.EmbeddingDim;
        var peHost = _positionalEncodingHost ??= _positionalEncoding.GetCpuData();
        var posData = new Vector<T>(seqLen * embDim);
        for (int i = 0; i < seqLen * embDim; i++) posData[i] = peHost[i];
        return new Tensor<T>(new[] { 1, seqLen, embDim }, posData);
    }

    /// <summary>Applies <paramref name="weight"/> to the last axis of a <c>[B, S, D]</c> tensor.</summary>
    private Tensor<T> ProjectBatched(Tensor<T> x, Tensor<T> weight)
    {
        int batch = x.Shape[0], seq = x.Shape[1];
        var flat = Engine.TensorMatMul(Engine.Reshape(x, new[] { batch * seq, x.Shape[2] }), weight);
        return Engine.Reshape(flat, new[] { batch, seq, flat.Shape[1] });
    }

    /// <summary>LayerNorm over the feature axis of a <c>[B, S, D]</c> tensor.</summary>
    private Tensor<T> LayerNormBatched(Tensor<T> x, Tensor<T> gamma, Tensor<T> beta)
    {
        int batch = x.Shape[0], seq = x.Shape[1], dim = x.Shape[2];
        var normed = Engine.LayerNorm(Engine.Reshape(x, new[] { batch * seq, dim }), gamma, beta, 1e-6, out _, out _);
        return Engine.Reshape(normed, new[] { batch, seq, dim });
    }

    /// <summary>The position-wise feed-forward block of a layer on <c>[B, S, D]</c>.</summary>
    private Tensor<T> FeedForwardBatched(Tensor<T> x, Tensor<T> ff1Weight, Tensor<T> ff1Bias, Tensor<T> ff2Weight, Tensor<T> ff2Bias)
    {
        int batch = x.Shape[0], seq = x.Shape[1], dim = x.Shape[2];
        var flat = Engine.Reshape(x, new[] { batch * seq, dim });
        var hidden = Engine.ReLU(AddBias(Engine.TensorMatMul(flat, Engine.TensorTranspose(ff1Weight)), ff1Bias));
        var output = AddBias(Engine.TensorMatMul(hidden, Engine.TensorTranspose(ff2Weight)), ff2Bias);
        return Engine.Reshape(output, new[] { batch, seq, dim });
    }

    // Constant shift operators keyed by (corrLen, lq, lk): row `lag` holds S_lag with S_lag[t, s] = 1 when
    // s == (t + lag) mod lk, flattened to [corrLen, lq*lk].
    [Scratch]
    private readonly ConcurrentDictionary<(int CorrLen, int Lq, int Lk), Tensor<T>> _shiftOperatorCache = new();

    private Tensor<T> ShiftOperators(int corrLen, int lq, int lk)
    {
        return _shiftOperatorCache.GetOrAdd((corrLen, lq, lk), _ =>
        {
            var data = new Vector<T>(corrLen * lq * lk);
            for (int lag = 0; lag < corrLen; lag++)
                for (int t = 0; t < lq; t++)
                    data[lag * (lq * lk) + t * lk + (t + lag) % lk] = _numOps.One;
            return new Tensor<T>(new[] { corrLen, lq * lk }, data);
        });
    }

    /// <summary>
    /// Auto-correlation attention (Wu et al. 2021, §3.1) for <c>[B, lq, D]</c> queries and <c>[B, lk, D]</c>
    /// keys/values, every window choosing its own top-k delays on the device.
    /// </summary>
    /// <remarks>
    /// out[b, t] = Σ_i softmax(R_b)_i · v[b, (t + lag_{b,i}) mod lk]. The selection is non-differentiable, as in
    /// the paper; the gradient flows through the softmax over the selected correlation values (into q and k) and
    /// through the aggregated values. Writing the aggregation as a[b, lag] · S_lag, with a the softmax weights at
    /// the selected lags and zero elsewhere, keeps it a constant-shape graph of batched matmuls.
    /// </remarks>
    private Tensor<T> AutoCorrelationBatched(Tensor<T> q, Tensor<T> k, Tensor<T> v)
    {
        int batch = q.Shape[0], lq = q.Shape[1], d = q.Shape[2];
        int lk = k.Shape[1];
        int corrLen = Math.Min(lq, lk);
        if (corrLen <= 0) return v;

        int topK = Math.Max(1, (int)Math.Round(_options.AutoCorrelationFactor * Math.Log(Math.Max(2.0, corrLen))));
        topK = Math.Min(topK, corrLen);

        var corr = CorrelationSpectrumBatched(q, k, corrLen, d);                                   // [B, corrLen]

        // Each window's top-k delays and their softmax weights, scattered to their lags: a[b, lag].
        var lagWeights = TopKLagWeights(corr, topK);                                               // [B, corrLen]

        // W_b = Σ_lag a[b, lag] · S_lag, then out_b = W_b · v_b.
        var mix = Engine.TensorMatMul(
            lagWeights, ShiftOperators(corrLen, lq, lk));                                          // [B, lq*lk]
        return Engine.TensorBatchMatMul(Engine.Reshape(mix, new[] { batch, lq, lk }), v);          // [B, lq, D]
    }

    /// <summary>
    /// Softmax weights over each row's top-<paramref name="topK"/> correlation values, placed at their lags with
    /// zero elsewhere: <c>[B, C]</c> in, <c>[B, C]</c> out.
    /// </summary>
    /// <remarks>
    /// Lag i is selected when fewer than k lags beat it: a strictly larger value, or an equal value at a lower lag
    /// (the order a stable descending sort gives). Built from same-type comparisons rather than an index-returning
    /// top-k, so a compiled plan can capture it and each replay re-selects from the new batch. The mask carries no
    /// gradient; the selected values do, through the softmax, as in the paper.
    /// </remarks>
    internal Tensor<T> TopKLagWeights(Tensor<T> corr, int topK)
    {
        int batch = corr.Shape[0], lags = corr.Shape[1];
        var expand = new Tensor<T>(new[] { 1, lags, lags });
        var own = Engine.TensorAdd(Engine.Reshape(corr, new[] { batch, lags, 1 }), expand);       // [b, i, j] = c_i
        var other = Engine.TensorAdd(Engine.Reshape(corr, new[] { batch, 1, lags }), expand);     // [b, i, j] = c_j
        var beats = Engine.TensorAdd(
            Engine.TensorGreaterThan(other, own),
            Engine.TensorMultiply(Engine.TensorEquals(other, own), EarlierLagMask(lags)));
        var rank = Engine.ReduceSum(beats, new[] { 2 }, keepDims: false);                         // [B, C]
        var selected = Engine.TensorLessThan(rank, _numOps.FromDouble(topK));                     // [B, C] 0/1

        var shifted = Engine.TensorSubtract(corr, Engine.ReduceMax(corr, new[] { 1 }, keepDims: true));
        var exp = Engine.TensorMultiply(Engine.TensorExp(shifted), selected);
        return Engine.TensorDivide(exp, Engine.ReduceSum(exp, new[] { 1 }, keepDims: true));
    }

    // [1, C, C] with 1 where j < i: an equal value at an earlier lag ranks ahead.
    [Scratch]
    private readonly ConcurrentDictionary<int, Tensor<T>> _earlierLagMaskCache = new();

    private Tensor<T> EarlierLagMask(int lags)
    {
        return _earlierLagMaskCache.GetOrAdd(lags, _ =>
        {
            var data = new Vector<T>(lags * lags);
            for (int i = 0; i < lags; i++)
                for (int j = 0; j < i; j++)
                    data[i * lags + j] = _numOps.One;
            return new Tensor<T>(new[] { 1, lags, lags }, data);
        });
    }
    private (Tensor<T> seasonal, Tensor<T> trend) EncoderLayerBatched(Tensor<T> seasonal, Tensor<T> trend, int layerIdx)
    {
        var layer = _encoderLayers[layerIdx];
        int seqLen = seasonal.Shape[1];

        var attn = AutoCorrelationBatched(
            ProjectBatched(seasonal, layer.GetQueryProjection()),
            ProjectBatched(seasonal, layer.GetKeyProjection()),
            ProjectBatched(seasonal, layer.GetValueProjection()));
        var normalized = LayerNormBatched(
            Engine.TensorAdd(seasonal, ProjectBatched(attn, layer.GetOutputProjection())),
            layer.GetLayerNorm1Gamma(), layer.GetLayerNorm1Beta());
        var ffOutput = FeedForwardBatched(normalized,
            layer.GetFF1Weight(), layer.GetFF1Bias(), layer.GetFF2Weight(), layer.GetFF2Bias());
        var newSeasonal = LayerNormBatched(Engine.TensorAdd(normalized, ffOutput),
            layer.GetLayerNorm2Gamma(), layer.GetLayerNorm2Beta());

        var newTrend = MovingAverageBatched(newSeasonal, _movingAvgKernel, seqLen);
        return (Engine.TensorSubtract(newSeasonal, newTrend), Engine.TensorAdd(trend, newTrend));
    }

    private (Tensor<T> seasonal, Tensor<T> trend) DecoderLayerBatched(
        Tensor<T> decSeasonal, Tensor<T> decTrend, Tensor<T> encSeasonal, Tensor<T> encTrend, int layerIdx)
    {
        var layer = _decoderLayers[layerIdx];
        int seqLen = decSeasonal.Shape[1];

        var selfAttn = AutoCorrelationBatched(
            ProjectBatched(decSeasonal, layer.GetSelfQueryProjection()),
            ProjectBatched(decSeasonal, layer.GetSelfKeyProjection()),
            ProjectBatched(decSeasonal, layer.GetSelfValueProjection()));
        var norm1 = LayerNormBatched(
            Engine.TensorAdd(decSeasonal, ProjectBatched(selfAttn, layer.GetSelfOutputProjection())),
            layer.GetLayerNorm1Gamma(), layer.GetLayerNorm1Beta());

        var crossAttn = AutoCorrelationBatched(
            ProjectBatched(norm1, layer.GetCrossQueryProjection()),
            ProjectBatched(encSeasonal, layer.GetCrossKeyProjection()),
            ProjectBatched(encSeasonal, layer.GetCrossValueProjection()));
        var norm2 = LayerNormBatched(
            Engine.TensorAdd(norm1, ProjectBatched(crossAttn, layer.GetCrossOutputProjection())),
            layer.GetLayerNorm2Gamma(), layer.GetLayerNorm2Beta());

        var ffOutput = FeedForwardBatched(norm2,
            layer.GetFF1Weight(), layer.GetFF1Bias(), layer.GetFF2Weight(), layer.GetFF2Bias());
        var newSeasonal = LayerNormBatched(Engine.TensorAdd(norm2, ffOutput),
            layer.GetLayerNorm3Gamma(), layer.GetLayerNorm3Beta());

        var newTrend = MovingAverageBatched(newSeasonal, _movingAvgKernel, seqLen);
        return (Engine.TensorSubtract(newSeasonal, newTrend), Engine.TensorAdd(decTrend, newTrend));
    }

    /// <summary>
    /// The paper's series-decomposition moving average (replication pad, stride-1 windowed mean) over the time
    /// axis of [B, S, D], built from engine ops so the tape differentiates the trend.
    /// </summary>
    /// <remarks>
    /// The replication padding is taken per sample from that sample's OWN first and last step (narrow on
    /// dim 1), so padding never bleeds across the batch boundary — flattening to [B*S, D] and padding there
    /// would splice one window's tail onto the next window's head and silently corrupt the trend.
    /// </remarks>
    internal Tensor<T> MovingAverageBatched(Tensor<T> x, int kernelSize, int seqLen)
    {
        int leftPad = kernelSize / 2;
        int rightPad = kernelSize - 1 - leftPad;
        Tensor<T> padded;
        if (leftPad + rightPad == 0)
        {
            padded = x;
        }
        else
        {
            var front = Engine.TensorNarrow(x, 1, 0, 1);            // [B, 1, D] — each sample's own first step
            var back = Engine.TensorNarrow(x, 1, seqLen - 1, 1);    // [B, 1, D] — each sample's own last step
            var parts = new Tensor<T>[leftPad + 1 + rightPad];
            int idx = 0;
            for (int i = 0; i < leftPad; i++) parts[idx++] = front;
            parts[idx++] = x;
            for (int i = 0; i < rightPad; i++) parts[idx++] = back;
            padded = Engine.Concat(parts, 1);
        }

        Tensor<T> acc = Engine.TensorNarrow(padded, 1, 0, seqLen);
        for (int j = 1; j < kernelSize; j++)
            acc = Engine.TensorAdd(acc, Engine.TensorNarrow(padded, 1, j, seqLen));
        return Engine.TensorMultiplyScalar(acc, _numOps.FromDouble(1.0 / kernelSize));
    }

    // Broadcast-add a [D] bias across the sequence dim of a [S, D] tensor
    // (Engine.TensorAdd requires equal shapes; the FFN biases need broadcasting).
    private Tensor<T> AddBias(Tensor<T> x, Tensor<T> bias)
        => Engine.TensorAdd(x, Engine.Reshape(bias, new[] { 1, bias.Shape[0] }));

    // Auto-correlation attention (Wu et al. 2021, "Autoformer", §3.1) on 2-D [seq, embDim] tensors,
    // built from IEngine ops so the gradient tape differentiates Autoformer's DEFINING time-delay
    // aggregation rather than the generic scaled-dot-product attention it would otherwise reduce to.
    // For each delay lag, the series auto-correlation R(lag) = mean over (t, d) of q[t]·k[t+lag]; the
    // top-k = round(factor·ln(L)) delays are softmax-weighted and used to aggregate the VALUE series
    // rolled by those delays (out[t] = Σ_i softmax(R)_i · v[(t+lag_i) mod L]). The top-k SELECTION is
    // data-dependent (non-differentiable, exactly as in the official implementation), but the gradient
    // still flows through the softmax weights (gathered R values → q, k) and through the rolled v.
    // Cache for the constant diagonal-sum operator used by the matmul spectrum, keyed by (lq, lk, corrLen, d).
    // Shape [corrLen, lq*lk]; ~110 KB at the default 24x24x512, and it never changes for a given model.
    [Scratch]
    private readonly ConcurrentDictionary<(int Lq, int Lk, int CorrLen, int D), Tensor<T>> _diagOperatorCache = new();

    /// <summary>
    /// Builds the constant operator A with A[lag, t*lk + (t+lag)] = 1/((corrLen-lag)*d) for t &lt; corrLen-lag,
    /// so that A @ vec(Q.K^T) is exactly the correlation spectrum R.
    /// </summary>
    private Tensor<T> DiagonalMeanOperator(int lq, int lk, int corrLen, int d)
    {
        return _diagOperatorCache.GetOrAdd((lq, lk, corrLen, d), _ =>
        {
            var data = new Vector<T>(corrLen * lq * lk);
            for (int lag = 0; lag < corrLen; lag++)
            {
                int valid = corrLen - lag;
                T w = _numOps.FromDouble(1.0 / ((double)valid * d));
                for (int t = 0; t < valid; t++)
                    data[lag * (lq * lk) + t * lk + (t + lag)] = w;
            }

            return new Tensor<T>(new[] { corrLen, lq * lk }, data);
        });
    }

    /// <summary>
    /// Correlation spectrum for <c>[B, lq, D]</c> queries and <c>[B, lk, D]</c> keys: R[b, lag] = mean over
    /// (t &lt; corrLen-lag, dim) of q[b,t,:]·k[b,t+lag,:]. Returns <c>[B, corrLen]</c>.
    /// </summary>
    /// <remarks>
    /// R[b, lag] is the mean of the lag-th diagonal of M_b = Q_b·K_bᵀ. Summing a diagonal is linear in M, so the
    /// whole spectrum is one batched matmul and one constant operator: R = vec(M) · Aᵀ. A per-lag loop issued
    /// corrLen x ~5 dispatches at each of the four call sites per forward. The products are summed in a different
    /// order than that loop, so the two agree to rounding, not bit for bit.
    /// </remarks>
    internal Tensor<T> CorrelationSpectrumBatched(Tensor<T> q, Tensor<T> k, int corrLen, int d)
    {
        int batch = q.Shape[0], lq = q.Shape[1], lk = k.Shape[1];
        var scores = Engine.TensorBatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })); // [B, lq, lk]
        return Engine.TensorMatMul(
            Engine.Reshape(scores, new[] { batch, lq * lk }),
            Engine.TensorTranspose(DiagonalMeanOperator(lq, lk, corrLen, d)));                // [B, corrLen]
    }
    // All trainable parameter tensors (model + encoder/decoder layers), in a
    // stable order, for the GradientTape sources and the optimizer step.
    internal List<Tensor<T>> CollectTrainableParameters()
    {
        var p = new List<Tensor<T>>
        {
            _inputProjection, _seasonalProjection, _trendProjection, _outputBias,
            _decoderSeasonalInit, _decoderTrendInit
        };
        foreach (var l in _encoderLayers)
        {
            p.Add(l.GetQueryProjection()); p.Add(l.GetKeyProjection()); p.Add(l.GetValueProjection());
            p.Add(l.GetOutputProjection()); p.Add(l.GetLayerNorm1Gamma()); p.Add(l.GetLayerNorm1Beta());
            p.Add(l.GetFF1Weight()); p.Add(l.GetFF1Bias()); p.Add(l.GetFF2Weight()); p.Add(l.GetFF2Bias());
            p.Add(l.GetLayerNorm2Gamma()); p.Add(l.GetLayerNorm2Beta());
        }
        foreach (var l in _decoderLayers)
        {
            p.Add(l.GetSelfQueryProjection()); p.Add(l.GetSelfKeyProjection()); p.Add(l.GetSelfValueProjection());
            p.Add(l.GetSelfOutputProjection()); p.Add(l.GetCrossQueryProjection()); p.Add(l.GetCrossKeyProjection());
            p.Add(l.GetCrossValueProjection()); p.Add(l.GetCrossOutputProjection());
            p.Add(l.GetFF1Weight()); p.Add(l.GetFF1Bias()); p.Add(l.GetFF2Weight()); p.Add(l.GetFF2Bias());
            p.Add(l.GetLayerNorm1Gamma()); p.Add(l.GetLayerNorm1Beta());
            p.Add(l.GetLayerNorm2Gamma()); p.Add(l.GetLayerNorm2Beta());
            p.Add(l.GetLayerNorm3Gamma()); p.Add(l.GetLayerNorm3Beta());
        }
        return p;
    }

    /// <summary>
    /// Predicts one-step-ahead values for each row of the input.
    /// </summary>
    // Rows per batched prediction forward (bounds activation memory; see Predict).
    private const int PredictChunkRows = 256;

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryPredictFromTimeIndexCalibration(input, _trainingSeries, out var calibratedPredictions))
        {
            return calibratedPredictions;
        }

        int n = input.Rows;
        var predictions = new Vector<T>(n);
        int lookback = _options.LookbackWindow;

        // In-sample evaluation: when asked to predict over the training inputs
        // (row count matches the observed series), forecast each position from the
        // model's PROPER lookback window of the observed series. Autoformer is a
        // sequence forecaster — feeding a single scalar row zeros the series
        // decomposition (seasonal = x - MovingAvg(x) = 0 when seqLen=1), collapsing
        // the forecast to a constant. This is a genuine one-step-ahead forecast
        // from real history, NOT a shortcut that returns the memorized target value.
        bool inSample = _trainingSeries.Length == n && n > 0;

        // Each row's window: the in-sample lookback of the observed series, or (out-of-sample / no history yet)
        // the row itself. ForwardEngine consumes the first min(length, lookback) values of a window.
        var windows = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            int w = inSample ? Math.Min(lookback, i) : 0;
            if (w > 0)
            {
                var window = new Vector<T>(w);
                for (int t = 0; t < w; t++) window[t] = _trainingSeries[i - w + t];
                windows[i] = window;
            }
            else
            {
                windows[i] = input.GetRow(i);
            }
        }

        // One batched forward per window length instead of one per row: AiModelBuilder predicts the whole dataset
        // after fitting, which made per-row inference a large share of a production fit. Same per-row values
        // (AutoformerBatchedForwardTests: ForwardBatched == ForwardCore row by row).
        foreach (var group in Enumerable.Range(0, n)
                     .Where(i => windows[i].Length > 0)
                     .GroupBy(i => Math.Min(windows[i].Length, lookback)))
        {
            int len = group.Key;
            var all = group.ToArray();
            // Bounded chunks: activations scale with rows x length x embedding (x4 in the FFN), and a whole
            // in-sample dataset in one forward exhausted memory at the default embedding width.
            for (int c0 = 0; c0 < all.Length; c0 += PredictChunkRows)
            {
                var rows = all.Skip(c0).Take(PredictChunkRows).ToArray();
                // A nested arena per chunk: its scratch is recycled when the chunk ends without touching any arena
                // the caller has open (whose tensors may still be live).
                using var chunkArena = AiDotNet.Tensors.Helpers.TensorArena.Create();
                var data = new Vector<T>(rows.Length * len);
                for (int r = 0; r < rows.Length; r++)
                    for (int t = 0; t < len; t++)
                        data[r * len + t] = _numOps.Divide(_numOps.Subtract(windows[rows[r]][t], _normMean), _normStd);
                var outNorm = ForwardBatched(new Tensor<T>(new[] { rows.Length, len }, data));      // [rows, horizon]
                for (int r = 0; r < rows.Length; r++)
                    predictions[rows[r]] = _numOps.Add(_numOps.Multiply(outNorm[r, 0], _normStd), _normMean);
            }
        }

        return predictions;
    }

    /// <summary>
    /// Predicts the next single value in the time series.
    /// </summary>
    public override T PredictSingle(Vector<T> input)
    {
        // Use the SAME Engine-op forward as training (no train/predict divergence).
        var output = ForwardEngine(input);
        return output.Length > 0 ? output[0] : _numOps.Zero;
    }

    /// <summary>
    /// Predicts multiple time steps ahead using the Autoformer architecture.
    /// </summary>
    /// <param name="input">Input sequence of historical values.</param>
    /// <returns>Vector of predictions for the entire forecast horizon.</returns>
    /// <remarks>
    /// <para>
    /// This method returns predictions for all steps in the forecast horizon,
    /// as Autoformer is designed for multi-horizon forecasting. It uses the SAME
    /// Engine-op forward as training and <see cref="PredictSingle"/>.
    /// </para>
    /// </remarks>
    public Vector<T> PredictMultiple(Vector<T> input)
    {
        int forecastHorizon = _options.ForecastHorizon;
        var predictions = new Vector<T>(forecastHorizon);

        var output = ForwardEngine(input);
        // Fail fast on a short forecast: silently repeating the last value (or zero-filling)
        // would mask a broken [ForecastHorizon, 1] output contract and return fabricated steps.
        if (output.Length < forecastHorizon)
        {
            throw new InvalidOperationException(
                $"ForwardEngine returned {output.Length} values, expected at least {forecastHorizon}.");
        }
        for (int h = 0; h < forecastHorizon; h++)
        {
            predictions[h] = output[h];
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["Architecture"] = "Autoformer",
                ["LookbackWindow"] = _options.LookbackWindow,
                ["ForecastHorizon"] = _options.ForecastHorizon,
                ["EmbeddingDim"] = _options.EmbeddingDim,
                ["NumEncoderLayers"] = _options.NumEncoderLayers,
                ["NumDecoderLayers"] = _options.NumDecoderLayers,
                ["NumAttentionHeads"] = _options.NumAttentionHeads,
                ["MovingAverageKernel"] = _options.MovingAverageKernel,
                ["AutoCorrelationFactor"] = _options.AutoCorrelationFactor
            }
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new AutoformerModel<T>(new AutoformerOptions<T>(_options), initializeModel: false);
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
        {
            writer.Write(dim);
        }
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(_numOps.ToDouble(tensor[i]));
        }
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        return tensor;
    }
}

/// <summary>
/// Autoformer encoder layer. Its trainable tensors are consumed by the model-level
/// tape forward (<c>AutoformerModel.EncoderLayerBatched</c>); the layer holds parameters
/// and their (de)serialization only — it does not run its own forward pass.
/// </summary>
// TWO inputs and TWO outputs - the (seasonal, trend) pair Autoformer's series decomposition carries
// through every layer. See AutoformerModel.EncoderLayerBatched, whose signature is literally
// (Tensor seasonal, Tensor trend) -> (Tensor seasonal, Tensor trend). Both streams are [B, T, emb] and
// stay that way: the auto-correlation block and the FFN each end in a residual add, and decomposition
// splits a stream into two streams of the SAME width rather than widening one.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal partial class AutoformerEncoderLayer<T> : NeuralNetworks.Layers.LayerBase<T>, IMultiOutputShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// Declines: this layer has two inputs and two outputs, and its own <c>ForwardTraced</c> throws
    /// because the real forward is <c>AutoformerModel.EncoderLayerBatched</c>.
    /// <see cref="OutputsFor"/> is the contract.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Both outputs are shaped like the incoming seasonal stream on port 0. Nothing widens: the
    /// residual adds pin every axis, and decomposition yields two streams of the same width.
    /// </para>
    /// <para>
    /// THE CONSTRUCTOR'S <c>embeddingDim * 2</c> WAS A FICTION. This layer declared
    /// <c>base(new[] { embeddingDim }, new[] { embeddingDim * 2 })</c>, packing the pair into one
    /// number, and no code anywhere produces a tensor of that width. It stayed undeclarable for as
    /// long as a contract could describe only ONE output - and declaring the packed number would have
    /// meant describing a tensor that does not exist, which is worse than declaring nothing. With
    /// <c>OutputsFor</c> the pair can be stated as what it actually is.
    /// </para>
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<OutputAxisContract>>? OutputsFor(IReadOnlyList<int> inputRanks)
    {
        if (inputRanks is null || inputRanks.Count == 0) return null;
        if (inputRanks[0] != 3) return null;

        return new[] { SeriesStreamAxes(), SeriesStreamAxes() };
    }

    /// <summary>One decomposition stream: [Batch, Time, Features], carried through unchanged.</summary>
    internal static IReadOnlyList<OutputAxisContract> SeriesStreamAxes() => new[]
    {
        new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
        new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
    };

    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _movingAvgKernel;
    private readonly int _autoCorrelationFactor;
    private readonly double _dropoutRate;

    // Auto-correlation parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _queryProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _keyProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _valueProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputProj;

    // Feed-forward parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff1Weight;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff1Bias;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff2Weight;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff2Bias;

    // Layer normalization parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm1Gamma;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm1Beta;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm2Gamma;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm2Beta;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => throw new NotSupportedException(
        "Autoformer runs its forward pass at the model level (AutoformerModel.ForwardCore); the layer-level Forward is unused.");

    /// <summary>Construction state: the 'seed' the layer was built with.</summary>
    private readonly int _seed;

    public AutoformerEncoderLayer(int embeddingDim, int numHeads, int movingAvgKernel,
        int autoCorrelationFactor, double dropoutRate, int seed)
        : base(new[] { embeddingDim }, new[] { embeddingDim * 2 })
    {
        _seed = seed;
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _movingAvgKernel = movingAvgKernel;
        _autoCorrelationFactor = autoCorrelationFactor;
        _dropoutRate = dropoutRate;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        // Initialize auto-correlation projections
        _queryProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _keyProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _valueProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _outputProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);

        // Initialize feed-forward
        int ffDim = embeddingDim * 4;
        _ff1Weight = InitTensor(new[] { ffDim, embeddingDim }, stddev, random);
        _ff1Bias = new Tensor<T>(new[] { ffDim });
        _ff2Weight = InitTensor(new[] { embeddingDim, ffDim }, stddev, random);
        _ff2Bias = new Tensor<T>(new[] { embeddingDim });

        // Initialize layer normalization
        _layerNorm1Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
        for (int i = 0; i < embeddingDim; i++)
        {
            _layerNorm1Gamma[i] = NumOps.One;
            _layerNorm2Gamma[i] = NumOps.One;
        }
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    // Getters for the model-level autodiff forward.
    public Tensor<T> GetQueryProjection() => _queryProj;
    public Tensor<T> GetKeyProjection() => _keyProj;
    public Tensor<T> GetValueProjection() => _valueProj;
    public Tensor<T> GetOutputProjection() => _outputProj;
    public Tensor<T> GetFF1Weight() => _ff1Weight;
    public Tensor<T> GetFF1Bias() => _ff1Bias;
    public Tensor<T> GetFF2Weight() => _ff2Weight;
    public Tensor<T> GetFF2Bias() => _ff2Bias;
    public Tensor<T> GetLayerNorm1Gamma() => _layerNorm1Gamma;
    public Tensor<T> GetLayerNorm1Beta() => _layerNorm1Beta;
    public Tensor<T> GetLayerNorm2Gamma() => _layerNorm2Gamma;
    public Tensor<T> GetLayerNorm2Beta() => _layerNorm2Beta;

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
        {
            writer.Write(dim);
        }
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(NumOps.ToDouble(tensor[i]));
        }
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        return tensor;
    }
}

/// <summary>
/// Autoformer decoder layer. Its trainable tensors are consumed by the model-level
/// tape forward (<c>AutoformerModel.DecoderLayerBatched</c>); the layer holds parameters
/// and their (de)serialization only — it does not run its own forward pass.
/// </summary>
// The decoder half of the same pair. DecoderLayerBatched takes FOUR tensors - the decoder's own
// (seasonal, trend) plus the encoder's - and returns TWO, both shaped like the decoder seasonal stream
// on port 0. Cross-correlation reads the encoder streams but is sized by the decoder queries.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal partial class AutoformerDecoderLayer<T> : NeuralNetworks.Layers.LayerBase<T>, IMultiOutputShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// Declines: four inputs, two outputs, and <c>ForwardTraced</c> throws - the real forward is
    /// <c>AutoformerModel.DecoderLayerBatched</c>. <see cref="OutputsFor"/> is the contract.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Both outputs follow port 0, the decoder's seasonal stream. As in the encoder, the constructor's
    /// declared output width of <c>embeddingDim * 2</c> is a packed placeholder no code produces.
    /// </para>
    /// <para>
    /// NOTE, not fixed here: the constructor declares THREE input shapes
    /// (<c>base(new int[][] { [emb], [emb], [emb] }, ...)</c>) while <c>DecoderLayerBatched</c> consumes
    /// FOUR tensors. This contract sizes from port 0 and so is unaffected either way, but the declared
    /// port count and the real one disagree.
    /// </para>
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<OutputAxisContract>>? OutputsFor(IReadOnlyList<int> inputRanks)
    {
        if (inputRanks is null || inputRanks.Count == 0) return null;
        if (inputRanks[0] != 3) return null;

        return new[]
        {
            AutoformerEncoderLayer<T>.SeriesStreamAxes(),
            AutoformerEncoderLayer<T>.SeriesStreamAxes(),
        };
    }

    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _movingAvgKernel;
    private readonly int _autoCorrelationFactor;
    private readonly double _dropoutRate;

    // Self auto-correlation parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _selfQueryProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _selfKeyProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _selfValueProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _selfOutputProj;

    // Cross auto-correlation parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _crossQueryProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _crossKeyProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _crossValueProj;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _crossOutputProj;

    // Feed-forward parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff1Weight;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff1Bias;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff2Weight;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ff2Bias;

    // Layer normalization
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm1Gamma;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm1Beta;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm2Gamma;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm2Beta;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm3Gamma;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm3Beta;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => throw new NotSupportedException(
        "Autoformer runs its forward pass at the model level (AutoformerModel.ForwardCore); the layer-level Forward is unused.");

    /// <summary>Construction state: the 'seed' the layer was built with.</summary>
    private readonly int _seed;

    public AutoformerDecoderLayer(int embeddingDim, int numHeads, int movingAvgKernel,
        int autoCorrelationFactor, double dropoutRate, int seed)
        : base(new int[][] { new[] { embeddingDim }, new[] { embeddingDim }, new[] { embeddingDim } }, new[] { embeddingDim * 2 })
    {
        _seed = seed;
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _movingAvgKernel = movingAvgKernel;
        _autoCorrelationFactor = autoCorrelationFactor;
        _dropoutRate = dropoutRate;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        // Initialize self auto-correlation
        _selfQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _selfKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _selfValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _selfOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);

        // Initialize cross auto-correlation
        _crossQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _crossKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _crossValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);
        _crossOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, stddev, random);

        // Initialize feed-forward
        int ffDim = embeddingDim * 4;
        _ff1Weight = InitTensor(new[] { ffDim, embeddingDim }, stddev, random);
        _ff1Bias = new Tensor<T>(new[] { ffDim });
        _ff2Weight = InitTensor(new[] { embeddingDim, ffDim }, stddev, random);
        _ff2Bias = new Tensor<T>(new[] { embeddingDim });

        // Initialize layer normalization
        _layerNorm1Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm3Gamma = new Tensor<T>(new[] { embeddingDim });
        _layerNorm3Beta = new Tensor<T>(new[] { embeddingDim });
        for (int i = 0; i < embeddingDim; i++)
        {
            _layerNorm1Gamma[i] = NumOps.One;
            _layerNorm2Gamma[i] = NumOps.One;
            _layerNorm3Gamma[i] = NumOps.One;
        }
    }

    private Tensor<T> InitTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    // Getters for the model-level autodiff forward.
    public Tensor<T> GetSelfQueryProjection() => _selfQueryProj;
    public Tensor<T> GetSelfKeyProjection() => _selfKeyProj;
    public Tensor<T> GetSelfValueProjection() => _selfValueProj;
    public Tensor<T> GetSelfOutputProjection() => _selfOutputProj;
    public Tensor<T> GetCrossQueryProjection() => _crossQueryProj;
    public Tensor<T> GetCrossKeyProjection() => _crossKeyProj;
    public Tensor<T> GetCrossValueProjection() => _crossValueProj;
    public Tensor<T> GetCrossOutputProjection() => _crossOutputProj;
    public Tensor<T> GetFF1Weight() => _ff1Weight;
    public Tensor<T> GetFF1Bias() => _ff1Bias;
    public Tensor<T> GetFF2Weight() => _ff2Weight;
    public Tensor<T> GetFF2Bias() => _ff2Bias;
    public Tensor<T> GetLayerNorm1Gamma() => _layerNorm1Gamma;
    public Tensor<T> GetLayerNorm1Beta() => _layerNorm1Beta;
    public Tensor<T> GetLayerNorm2Gamma() => _layerNorm2Gamma;
    public Tensor<T> GetLayerNorm2Beta() => _layerNorm2Beta;
    public Tensor<T> GetLayerNorm3Gamma() => _layerNorm3Gamma;
    public Tensor<T> GetLayerNorm3Beta() => _layerNorm3Beta;

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
        {
            writer.Write(dim);
        }
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(NumOps.ToDouble(tensor[i]));
        }
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        return tensor;
    }
}
