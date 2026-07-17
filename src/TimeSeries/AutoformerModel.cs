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
/// // Create an Autoformer with auto-correlation for long-range forecasting
/// var options = new AutoformerOptions&lt;double&gt;
/// {
///     InputLength = 96, PredictionLength = 24,
///     EmbeddingDim = 512, NumHeads = 8
/// };
/// var autoformer = new AutoformerModel&lt;double&gt;(options);
/// autoformer.Train(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = autoformer.Predict(inputMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", "https://arxiv.org/abs/2106.13008", Year = 2021, Authors = "Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long")]
public class AutoformerModel<T> : TimeSeriesModelBase<T>, ISupportsLossFunction<T>
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
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Series decomposition components
    private readonly int _movingAvgKernel;

    // Input embedding
    private Tensor<T> _inputProjection;      // [embeddingDim, 1]
    private Tensor<T> _positionalEncoding;   // [maxLen, embeddingDim]

    // Encoder components
    private readonly List<AutoformerEncoderLayer<T>> _encoderLayers;

    // Decoder components
    private readonly List<AutoformerDecoderLayer<T>> _decoderLayers;
    private Tensor<T> _decoderSeasonalInit;  // [forecastHorizon, embeddingDim]
    private Tensor<T> _decoderTrendInit;     // [forecastHorizon, embeddingDim]

    // Output projections
    private Tensor<T> _seasonalProjection;   // [1, embeddingDim]
    private Tensor<T> _trendProjection;      // [1, embeddingDim]
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

        _random = RandomHelper.CreateSeededRandom(42);
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
        var random = RandomHelper.CreateSeededRandom(42);

        // Input projection: maps single time step values to embedding dimension
        _inputProjection = InitTensor(new[] { _options.EmbeddingDim, 1 }, stddev, random);

        // Sinusoidal positional encoding
        int maxLen = Math.Max(_options.LookbackWindow, _options.ForecastHorizon) * 2;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);

        // Encoder layers with series decomposition and auto-correlation
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new AutoformerEncoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _movingAvgKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                42 + i));
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
                42 + _options.NumEncoderLayers + i));
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

                // Mini-batch gradient by accumulating each sample's gradient across SEPARATE
                // tapes. Every sample runs its own small forward graph under its own tape which
                // is disposed immediately after its backward — so peak memory stays at one
                // forward, not the whole batch, and the auto-correlation graph (hundreds of
                // small tensor nodes per forward) is not held B times over. The per-sample
                // gradients are summed and averaged, then ONE Adam step is taken per batch (so
                // Adam's moment estimates do not thrash the way a per-sample step would). This
                // is the exact mini-batch gradient (mean over the batch); the accumulation
                // arithmetic runs eagerly outside any tape.
                var accum = new Dictionary<Tensor<T>, Tensor<T>>(
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                double batchLossSum = 0.0;
                for (int bi = 0; bi < b; bi++)
                {
                    int idx = shuffled[start + bi];
                    var window = new Vector<T>(lookback);
                    for (int t = 0; t < lookback; t++) window[t] = yNorm[idx - lookback + t];
                    var targetData = new Vector<T>(horizon);
                    for (int h = 0; h < horizon; h++) targetData[h] = yNorm[idx + h];
                    var targetTensor = new Tensor<T>(new[] { horizon, 1 }, targetData);

                    Dictionary<Tensor<T>, Tensor<T>> sampleGrads;
                    T sampleLoss;
                    using (var tape = new Tensors.Engines.Autodiff.GradientTape<T>())
                    {
                        var pred = ForwardCore(window);                       // [horizon, 1], normalized
                        var l = mseLoss.ComputeTapeLoss(pred, targetTensor);  // scalar
                        sampleGrads = tape.ComputeGradients(l, sources: null);
                        sampleLoss = l.Length > 0 ? l[0] : _numOps.Zero;
                    }

                    foreach (var param in allParams)
                    {
                        if (!sampleGrads.TryGetValue(param, out var g)) continue;
                        accum[param] = accum.TryGetValue(param, out var acc)
                            ? Engine.TensorAdd(acc, g)
                            : g.Clone();
                    }
                    batchLossSum += Convert.ToDouble(sampleLoss);
                }

                var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                T invB = _numOps.FromDouble(1.0 / b);
                foreach (var kv in accum)
                    grads[kv.Key] = Engine.TensorMultiplyScalar(kv.Value, invB);

                T lossValue = _numOps.FromDouble(batchLossSum / b);
                Tensor<T> ComputeForward(Tensor<T> a, Tensor<T> t) => a;
                Tensor<T> ComputeLoss(Tensor<T> p, Tensor<T> t) => mseLoss.ComputeTapeLoss(p, t);
                var placeholder = new Tensor<T>(new[] { 1 });
                var context = new Tensors.Engines.Autodiff.TapeStepContext<T>(
                    allParams, grads, lossValue, placeholder, placeholder, ComputeForward, ComputeLoss, null);
                optimizer.Step(context);

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

    // Core Engine forward on an ALREADY z-normalized lookback window, returning the normalized
    // [forecastHorizon, 1] forecast. Built entirely from Engine.Tensor* ops so a GradientTape
    // differentiates it automatically and every op is GPU-dispatchable.
    private Tensor<T> ForwardCore(Vector<T> input)
    {
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);
        int embDim = _options.EmbeddingDim;
        int forecastHorizon = _options.ForecastHorizon;

        // Input + positional encoding as constant data tensors (no grad needed).
        var inData = new Vector<T>(seqLen);
        for (int t = 0; t < seqLen; t++) inData[t] = input[t];
        var inputTensor = new Tensor<T>(new[] { seqLen, 1 }, inData);

        var posData = new Vector<T>(seqLen * embDim);
        for (int i = 0; i < seqLen * embDim; i++) posData[i] = _positionalEncoding[i];
        var posTensor = new Tensor<T>(new[] { seqLen, embDim }, posData);

        // Embedding: input @ inputProj(1×embDim) + positional encoding.
        var embedded = Engine.TensorAdd(
            Engine.TensorMatMul(inputTensor, Engine.Reshape(_inputProjection, new[] { 1, embDim })),
            posTensor);

        // Series decomposition.
        var trend = MovingAverageEngine(embedded, _movingAvgKernel, seqLen, embDim);
        var seasonal = Engine.TensorSubtract(embedded, trend);

        // Encoder.
        for (int i = 0; i < _encoderLayers.Count; i++)
            (seasonal, trend) = EncoderLayerEngine(seasonal, trend, i);

        // Decoder init (Autoformer §3.2). The seasonal placeholder is the learned
        // init; the trend-cyclical init is DATA-DEPENDENT — the mean of the encoder
        // input broadcast over the decoder length — plus a small learned refinement.
        // This is what lets the forecast track each window's level: without the data
        // mean the decoder trend is a constant and every window decodes to the same
        // value (flat forecast, R² ≈ 0).
        Tensor<T> decSeasonal = _decoderSeasonalInit;
        var embMean = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(embedded, new[] { 0 }, keepDims: true),
            _numOps.FromDouble(1.0 / seqLen));
        Tensor<T> decTrend = Engine.TensorBroadcastAdd(_decoderTrendInit, embMean);
        for (int i = 0; i < _decoderLayers.Count; i++)
            (decSeasonal, decTrend) = DecoderLayerEngine(decSeasonal, decTrend, seasonal, trend, i);

        // Output projection: seasonal·projᵀ + trend·projᵀ + bias.
        var output = Engine.TensorAdd(
            Engine.TensorMatMul(decSeasonal, Engine.TensorTranspose(_seasonalProjection)),
            Engine.TensorMatMul(decTrend, Engine.TensorTranspose(_trendProjection)));
        return Engine.TensorAdd(output, Engine.Reshape(_outputBias, new[] { forecastHorizon, 1 }));
    }

    // Paper-faithful series-decomposition moving average (replication-pad +
    // stride-1 windowed mean) built from IEngine ops so the tape differentiates
    // the trend. Vectorized — O(kernelSize) tensor ops, no per-element indexing.
    private Tensor<T> MovingAverageEngine(Tensor<T> x, int kernelSize, int seqLen, int embDim)
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
            var front = Engine.TensorNarrow(x, 0, 0, 1);
            var back = Engine.TensorNarrow(x, 0, seqLen - 1, 1);
            var parts = new Tensor<T>[leftPad + 1 + rightPad];
            int idx = 0;
            for (int i = 0; i < leftPad; i++) parts[idx++] = front;
            parts[idx++] = x;
            for (int i = 0; i < rightPad; i++) parts[idx++] = back;
            padded = Engine.Concat(parts, 0);
        }
        Tensor<T> acc = Engine.TensorNarrow(padded, 0, 0, seqLen);
        for (int j = 1; j < kernelSize; j++)
            acc = Engine.TensorAdd(acc, Engine.TensorNarrow(padded, 0, j, seqLen));
        return Engine.TensorMultiplyScalar(acc, _numOps.FromDouble(1.0 / kernelSize));
    }

    // Broadcast-add a [D] bias across the sequence dim of a [S, D] tensor
    // (Engine.TensorAdd requires equal shapes; the FFN biases need broadcasting).
    private Tensor<T> AddBias(Tensor<T> x, Tensor<T> bias)
        => Engine.TensorBroadcastAdd(x, Engine.Reshape(bias, new[] { 1, bias.Shape[0] }));

    // Auto-correlation attention (Wu et al. 2021, "Autoformer", §3.1) on 2-D [seq, embDim] tensors,
    // built from IEngine ops so the gradient tape differentiates Autoformer's DEFINING time-delay
    // aggregation rather than the generic scaled-dot-product attention it would otherwise reduce to.
    // For each delay lag, the series auto-correlation R(lag) = mean over (t, d) of q[t]·k[t+lag]; the
    // top-k = round(factor·ln(L)) delays are softmax-weighted and used to aggregate the VALUE series
    // rolled by those delays (out[t] = Σ_i softmax(R)_i · v[(t+lag_i) mod L]). The top-k SELECTION is
    // data-dependent (non-differentiable, exactly as in the official implementation), but the gradient
    // still flows through the softmax weights (gathered R values → q, k) and through the rolled v.
    private Tensor<T> AutoCorrelationEngine(Tensor<T> q, Tensor<T> k, Tensor<T> v)
    {
        int lq = q.Shape[0];
        int d = q.Shape[1];
        int lk = k.Shape[0];
        int corrLen = Math.Min(lq, lk);
        if (corrLen <= 0) return v;

        int topK = Math.Max(1, (int)Math.Round(_options.AutoCorrelationFactor * Math.Log(Math.Max(2.0, corrLen))));
        topK = Math.Min(topK, corrLen);

        // R[lag] = mean_{t < corrLen-lag, dim} q[t]·k[t+lag], lag in [0, corrLen). Tape-tracked.
        var corrParts = new Tensor<T>[corrLen];
        for (int lag = 0; lag < corrLen; lag++)
        {
            int valid = corrLen - lag;
            var qSlice = Engine.TensorNarrow(q, 0, 0, valid);
            var kSlice = Engine.TensorNarrow(k, 0, lag, valid);
            var summed = Engine.ReduceSum(Engine.TensorMultiply(qSlice, kSlice), new[] { 0, 1 }, keepDims: false);
            corrParts[lag] = Engine.TensorMultiplyScalar(
                Engine.Reshape(summed, new[] { 1 }), _numOps.FromDouble(1.0 / (valid * d)));
        }
        var corr = Engine.Concat(corrParts, 0); // [corrLen]

        // Top-k delays by correlation value (host read of the forward values; the index choice is
        // non-differentiable, like the paper's topk over the autocorrelation spectrum).
        var corrVals = new double[corrLen];
        for (int lag = 0; lag < corrLen; lag++) corrVals[lag] = _numOps.ToDouble(corr[lag]);
        var topLags = Enumerable.Range(0, corrLen)
            .OrderByDescending(i => corrVals[i])
            .Take(topK)
            .ToArray();

        // Softmax over the gathered top-k correlation values → aggregation weights (tape-tracked).
        var gatheredParts = new Tensor<T>[topLags.Length];
        for (int i = 0; i < topLags.Length; i++)
            gatheredParts[i] = Engine.TensorNarrow(corr, 0, topLags[i], 1);
        var weights = Engine.Softmax(Engine.Concat(gatheredParts, 0)); // [topK]

        // Aggregate: out[t,d] = Σ_i weights[i] · v[(t + lag_i) mod lk, d], for t in [0, lq).
        var w0 = Engine.Reshape(Engine.TensorNarrow(weights, 0, 0, 1), new[] { 1, 1 });
        var agg = Engine.TensorBroadcastMultiply(RollAndFit(v, topLags[0], lk, lq), w0);
        for (int i = 1; i < topLags.Length; i++)
        {
            var rolled = RollAndFit(v, topLags[i], lk, lq);
            var wi = Engine.Reshape(Engine.TensorNarrow(weights, 0, i, 1), new[] { 1, 1 });
            agg = Engine.TensorAdd(agg, Engine.TensorBroadcastMultiply(rolled, wi));
        }
        return agg;
    }

    // Circular roll of a [lk, d] sequence by `lag` so result[t] = v[(t + lag) mod lk], then fit to
    // length outLen (truncate, or tile-then-truncate when outLen > lk). Built from Narrow/Concat —
    // no per-element indexing — so it stays on the gradient tape.
    private Tensor<T> RollAndFit(Tensor<T> v, int lag, int lk, int outLen)
    {
        lag %= lk;
        Tensor<T> rolled = lag == 0
            ? v
            : Engine.Concat(new[] { Engine.TensorNarrow(v, 0, lag, lk - lag), Engine.TensorNarrow(v, 0, 0, lag) }, 0);
        if (outLen == lk) return rolled;
        if (outLen < lk) return Engine.TensorNarrow(rolled, 0, 0, outLen);
        int reps = (outLen + lk - 1) / lk;
        var parts = new Tensor<T>[reps];
        for (int r = 0; r < reps; r++) parts[r] = rolled;
        return Engine.TensorNarrow(Engine.Concat(parts, 0), 0, 0, outLen);
    }

    private (Tensor<T> seasonal, Tensor<T> trend) EncoderLayerEngine(Tensor<T> seasonal, Tensor<T> trend, int layerIdx)
    {
        var layer = _encoderLayers[layerIdx];
        int seqLen = seasonal.Shape[0];
        int embDim = seasonal.Shape[1];

        var q = Engine.TensorMatMul(seasonal, layer.GetQueryProjection());
        var k = Engine.TensorMatMul(seasonal, layer.GetKeyProjection());
        var v = Engine.TensorMatMul(seasonal, layer.GetValueProjection());
        var attn = AutoCorrelationEngine(q, k, v);
        var projected = Engine.TensorMatMul(attn, layer.GetOutputProjection());
        var residual = Engine.TensorAdd(seasonal, projected);
        var normalized = Engine.LayerNorm(residual, layer.GetLayerNorm1Gamma(), layer.GetLayerNorm1Beta(),
            1e-6, out _, out _);

        var ffHidden = Engine.ReLU(AddBias(
            Engine.TensorMatMul(normalized, Engine.TensorTranspose(layer.GetFF1Weight())), layer.GetFF1Bias()));
        var ffOutput = AddBias(
            Engine.TensorMatMul(ffHidden, Engine.TensorTranspose(layer.GetFF2Weight())), layer.GetFF2Bias());
        var ffResidual = Engine.TensorAdd(normalized, ffOutput);
        var newSeasonal = Engine.LayerNorm(ffResidual, layer.GetLayerNorm2Gamma(), layer.GetLayerNorm2Beta(),
            1e-6, out _, out _);

        var newTrend = MovingAverageEngine(newSeasonal, _movingAvgKernel, seqLen, embDim);
        newSeasonal = Engine.TensorSubtract(newSeasonal, newTrend);
        return (newSeasonal, Engine.TensorAdd(trend, newTrend));
    }

    private (Tensor<T> seasonal, Tensor<T> trend) DecoderLayerEngine(
        Tensor<T> decSeasonal, Tensor<T> decTrend, Tensor<T> encSeasonal, Tensor<T> encTrend, int layerIdx)
    {
        var layer = _decoderLayers[layerIdx];
        int seqLen = decSeasonal.Shape[0];
        int embDim = decSeasonal.Shape[1];

        var sq = Engine.TensorMatMul(decSeasonal, layer.GetSelfQueryProjection());
        var sk = Engine.TensorMatMul(decSeasonal, layer.GetSelfKeyProjection());
        var sv = Engine.TensorMatMul(decSeasonal, layer.GetSelfValueProjection());
        var selfAttn = AutoCorrelationEngine(sq, sk, sv);
        var selfProjected = Engine.TensorMatMul(selfAttn, layer.GetSelfOutputProjection());
        var norm1 = Engine.LayerNorm(Engine.TensorAdd(decSeasonal, selfProjected),
            layer.GetLayerNorm1Gamma(), layer.GetLayerNorm1Beta(), 1e-6, out _, out _);

        var cq = Engine.TensorMatMul(norm1, layer.GetCrossQueryProjection());
        var ck = Engine.TensorMatMul(encSeasonal, layer.GetCrossKeyProjection());
        var cv = Engine.TensorMatMul(encSeasonal, layer.GetCrossValueProjection());
        var crossAttn = AutoCorrelationEngine(cq, ck, cv);
        var crossProjected = Engine.TensorMatMul(crossAttn, layer.GetCrossOutputProjection());
        var norm2 = Engine.LayerNorm(Engine.TensorAdd(norm1, crossProjected),
            layer.GetLayerNorm2Gamma(), layer.GetLayerNorm2Beta(), 1e-6, out _, out _);

        var ffHidden = Engine.ReLU(AddBias(
            Engine.TensorMatMul(norm2, Engine.TensorTranspose(layer.GetFF1Weight())), layer.GetFF1Bias()));
        var ffOutput = AddBias(
            Engine.TensorMatMul(ffHidden, Engine.TensorTranspose(layer.GetFF2Weight())), layer.GetFF2Bias());
        var newSeasonal = Engine.LayerNorm(Engine.TensorAdd(norm2, ffOutput),
            layer.GetLayerNorm3Gamma(), layer.GetLayerNorm3Beta(), 1e-6, out _, out _);

        var newTrend = MovingAverageEngine(newSeasonal, _movingAvgKernel, seqLen, embDim);
        newSeasonal = Engine.TensorSubtract(newSeasonal, newTrend);
        return (newSeasonal, Engine.TensorAdd(decTrend, newTrend));
    }

    // All trainable parameter tensors (model + encoder/decoder layers), in a
    // stable order, for the GradientTape sources and the optimizer step.
    private List<Tensor<T>> CollectTrainableParameters()
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
        for (int i = 0; i < n; i++)
        {
            if (inSample)
            {
                int w = Math.Min(lookback, i);
                if (w > 0)
                {
                    var window = new Vector<T>(w);
                    for (int t = 0; t < w; t++) window[t] = _trainingSeries[i - w + t];
                    var fc = ForwardEngine(window);
                    predictions[i] = fc.Length > 0 ? fc[0] : _numOps.Zero;
                    continue;
                }
            }
            // Out-of-sample (or no history yet): forecast from the row itself.
            predictions[i] = PredictSingle(input.GetRow(i));
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
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write options
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.MovingAverageKernel);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.LearningRate);
        writer.Write(_options.Epochs);
        writer.Write(_options.BatchSize);
        writer.Write(_options.AutoCorrelationFactor);

        // Write tensors
        WriteTensor(writer, _inputProjection);
        WriteTensor(writer, _positionalEncoding);
        WriteTensor(writer, _decoderSeasonalInit);
        WriteTensor(writer, _decoderTrendInit);
        WriteTensor(writer, _seasonalProjection);
        WriteTensor(writer, _trendProjection);
        WriteTensor(writer, _outputBias);

        // Write encoder layers
        foreach (var layer in _encoderLayers)
        {
            layer.Serialize(writer);
        }

        // Write decoder layers
        foreach (var layer in _decoderLayers)
        {
            layer.Serialize(writer);
        }

        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++)
            writer.Write(_numOps.ToDouble(_trainingSeries[i]));

        // Normalization statistics (appended; older files without them fall back to 0/1).
        writer.Write(_numOps.ToDouble(_normMean));
        writer.Write(_numOps.ToDouble(_normStd));
    }

    /// <inheritdoc/>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read options (skip, they're set via constructor)
        _ = reader.ReadInt32(); // LookbackWindow
        _ = reader.ReadInt32(); // ForecastHorizon
        _ = reader.ReadInt32(); // EmbeddingDim
        _ = reader.ReadInt32(); // NumEncoderLayers
        _ = reader.ReadInt32(); // NumDecoderLayers
        _ = reader.ReadInt32(); // NumAttentionHeads
        _ = reader.ReadInt32(); // MovingAverageKernel
        _ = reader.ReadDouble(); // DropoutRate
        _ = reader.ReadDouble(); // LearningRate
        _ = reader.ReadInt32(); // Epochs
        _ = reader.ReadInt32(); // BatchSize
        _ = reader.ReadInt32(); // AutoCorrelationFactor

        // Read tensors
        _inputProjection = ReadTensor(reader);
        _positionalEncoding = ReadTensor(reader);
        _decoderSeasonalInit = ReadTensor(reader);
        _decoderTrendInit = ReadTensor(reader);
        _seasonalProjection = ReadTensor(reader);
        _trendProjection = ReadTensor(reader);
        _outputBias = ReadTensor(reader);

        // Reinitialize layers
        _encoderLayers.Clear();
        _decoderLayers.Clear();

        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            var layer = new AutoformerEncoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _options.MovingAverageKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                42 + i);
            layer.Deserialize(reader);
            _encoderLayers.Add(layer);
        }

        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            var layer = new AutoformerDecoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                _options.MovingAverageKernel,
                _options.AutoCorrelationFactor,
                _options.DropoutRate,
                42 + _options.NumEncoderLayers + i);
            layer.Deserialize(reader);
            _decoderLayers.Add(layer);
        }

        try
        {
            int tsLen = reader.ReadInt32();
            _trainingSeries = new Vector<T>(tsLen);
            for (int i = 0; i < tsLen; i++)
                _trainingSeries[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        catch (EndOfStreamException)
        {
            _trainingSeries = Vector<T>.Empty();
        }

        // Normalization statistics (present in models serialized after the tape rewrite).
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            _normMean = _numOps.FromDouble(reader.ReadDouble());
            _normStd = _numOps.FromDouble(reader.ReadDouble());
        }
    }

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
/// tape forward (<c>AutoformerModel.EncoderLayerEngine</c>); the layer holds parameters
/// and their (de)serialization only — it does not run its own forward pass.
/// </summary>
internal class AutoformerEncoderLayer<T> : NeuralNetworks.Layers.LayerBase<T>
{
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _movingAvgKernel;
    private readonly int _autoCorrelationFactor;
    private readonly double _dropoutRate;

    // Auto-correlation parameters
    private Tensor<T> _queryProj;
    private Tensor<T> _keyProj;
    private Tensor<T> _valueProj;
    private Tensor<T> _outputProj;

    // Feed-forward parameters
    private Tensor<T> _ff1Weight;
    private Tensor<T> _ff1Bias;
    private Tensor<T> _ff2Weight;
    private Tensor<T> _ff2Bias;

    // Layer normalization parameters
    private Tensor<T> _layerNorm1Gamma;
    private Tensor<T> _layerNorm1Beta;
    private Tensor<T> _layerNorm2Gamma;
    private Tensor<T> _layerNorm2Beta;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { }

    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        foreach (var t in new[] { _queryProj, _keyProj, _valueProj, _outputProj, _ff1Weight, _ff1Bias, _ff2Weight, _ff2Bias, _layerNorm1Gamma, _layerNorm1Beta, _layerNorm2Gamma, _layerNorm2Beta })
            for (int i = 0; i < t.Length; i++) p.Add(t[i]);
        return new Vector<T>(p.ToArray());
    }

    public override Tensor<T> Forward(Tensor<T> input) => throw new NotSupportedException(
        "Autoformer runs its forward pass at the model level (AutoformerModel.ForwardCore); the layer-level Forward is unused.");

    public AutoformerEncoderLayer(int embeddingDim, int numHeads, int movingAvgKernel,
        int autoCorrelationFactor, double dropoutRate, int seed)
        : base(new[] { embeddingDim }, new[] { embeddingDim * 2 })
    {
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

    public override void Serialize(BinaryWriter writer)
    {
        WriteTensor(writer, _queryProj);
        WriteTensor(writer, _keyProj);
        WriteTensor(writer, _valueProj);
        WriteTensor(writer, _outputProj);
        WriteTensor(writer, _ff1Weight);
        WriteTensor(writer, _ff1Bias);
        WriteTensor(writer, _ff2Weight);
        WriteTensor(writer, _ff2Bias);
        WriteTensor(writer, _layerNorm1Gamma);
        WriteTensor(writer, _layerNorm1Beta);
        WriteTensor(writer, _layerNorm2Gamma);
        WriteTensor(writer, _layerNorm2Beta);
    }

    public override void Deserialize(BinaryReader reader)
    {
        _queryProj = ReadTensor(reader);
        _keyProj = ReadTensor(reader);
        _valueProj = ReadTensor(reader);
        _outputProj = ReadTensor(reader);
        _ff1Weight = ReadTensor(reader);
        _ff1Bias = ReadTensor(reader);
        _ff2Weight = ReadTensor(reader);
        _ff2Bias = ReadTensor(reader);
        _layerNorm1Gamma = ReadTensor(reader);
        _layerNorm1Beta = ReadTensor(reader);
        _layerNorm2Gamma = ReadTensor(reader);
        _layerNorm2Beta = ReadTensor(reader);
    }

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
/// tape forward (<c>AutoformerModel.DecoderLayerEngine</c>); the layer holds parameters
/// and their (de)serialization only — it does not run its own forward pass.
/// </summary>
internal class AutoformerDecoderLayer<T> : NeuralNetworks.Layers.LayerBase<T>
{
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _movingAvgKernel;
    private readonly int _autoCorrelationFactor;
    private readonly double _dropoutRate;

    // Self auto-correlation parameters
    private Tensor<T> _selfQueryProj;
    private Tensor<T> _selfKeyProj;
    private Tensor<T> _selfValueProj;
    private Tensor<T> _selfOutputProj;

    // Cross auto-correlation parameters
    private Tensor<T> _crossQueryProj;
    private Tensor<T> _crossKeyProj;
    private Tensor<T> _crossValueProj;
    private Tensor<T> _crossOutputProj;

    // Feed-forward parameters
    private Tensor<T> _ff1Weight;
    private Tensor<T> _ff1Bias;
    private Tensor<T> _ff2Weight;
    private Tensor<T> _ff2Bias;

    // Layer normalization
    private Tensor<T> _layerNorm1Gamma;
    private Tensor<T> _layerNorm1Beta;
    private Tensor<T> _layerNorm2Gamma;
    private Tensor<T> _layerNorm2Beta;
    private Tensor<T> _layerNorm3Gamma;
    private Tensor<T> _layerNorm3Beta;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { }

    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        foreach (var t in new[] { _selfQueryProj, _selfKeyProj, _selfValueProj, _selfOutputProj,
            _crossQueryProj, _crossKeyProj, _crossValueProj, _crossOutputProj,
            _ff1Weight, _ff1Bias, _ff2Weight, _ff2Bias,
            _layerNorm1Gamma, _layerNorm1Beta, _layerNorm2Gamma, _layerNorm2Beta, _layerNorm3Gamma, _layerNorm3Beta })
            for (int i = 0; i < t.Length; i++) p.Add(t[i]);
        return new Vector<T>(p.ToArray());
    }

    public override Tensor<T> Forward(Tensor<T> input) => throw new NotSupportedException(
        "Autoformer runs its forward pass at the model level (AutoformerModel.ForwardCore); the layer-level Forward is unused.");

    public AutoformerDecoderLayer(int embeddingDim, int numHeads, int movingAvgKernel,
        int autoCorrelationFactor, double dropoutRate, int seed)
        : base(new int[][] { new[] { embeddingDim }, new[] { embeddingDim }, new[] { embeddingDim } }, new[] { embeddingDim * 2 })
    {
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

    public override void Serialize(BinaryWriter writer)
    {
        WriteTensor(writer, _selfQueryProj);
        WriteTensor(writer, _selfKeyProj);
        WriteTensor(writer, _selfValueProj);
        WriteTensor(writer, _selfOutputProj);
        WriteTensor(writer, _crossQueryProj);
        WriteTensor(writer, _crossKeyProj);
        WriteTensor(writer, _crossValueProj);
        WriteTensor(writer, _crossOutputProj);
        WriteTensor(writer, _ff1Weight);
        WriteTensor(writer, _ff1Bias);
        WriteTensor(writer, _ff2Weight);
        WriteTensor(writer, _ff2Bias);
        WriteTensor(writer, _layerNorm1Gamma);
        WriteTensor(writer, _layerNorm1Beta);
        WriteTensor(writer, _layerNorm2Gamma);
        WriteTensor(writer, _layerNorm2Beta);
        WriteTensor(writer, _layerNorm3Gamma);
        WriteTensor(writer, _layerNorm3Beta);
    }

    public override void Deserialize(BinaryReader reader)
    {
        _selfQueryProj = ReadTensor(reader);
        _selfKeyProj = ReadTensor(reader);
        _selfValueProj = ReadTensor(reader);
        _selfOutputProj = ReadTensor(reader);
        _crossQueryProj = ReadTensor(reader);
        _crossKeyProj = ReadTensor(reader);
        _crossValueProj = ReadTensor(reader);
        _crossOutputProj = ReadTensor(reader);
        _ff1Weight = ReadTensor(reader);
        _ff1Bias = ReadTensor(reader);
        _ff2Weight = ReadTensor(reader);
        _ff2Bias = ReadTensor(reader);
        _layerNorm1Gamma = ReadTensor(reader);
        _layerNorm1Beta = ReadTensor(reader);
        _layerNorm2Gamma = ReadTensor(reader);
        _layerNorm2Beta = ReadTensor(reader);
        _layerNorm3Gamma = ReadTensor(reader);
        _layerNorm3Beta = ReadTensor(reader);
    }

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
