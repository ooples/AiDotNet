using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Informer model for efficient long-sequence time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Long-Sequence Forecasting Problem:</b>
/// Traditional Transformer models achieve state-of-the-art results in many sequence modeling tasks,
/// but they struggle with long time series because self-attention has O(L^2) time and memory complexity.
/// For a sequence of 1000 time steps, vanilla attention requires 1 million operations per layer.
/// This makes long-horizon forecasting computationally prohibitive.
/// </para>
/// <para>
/// <b>The Informer Solution:</b>
/// Informer (Zhou et al., AAAI 2021) introduces three key innovations:
/// 1. ProbSparse Self-Attention (O(L log L) complexity)
/// 2. Self-Attention Distilling for sequence compression
/// 3. Generative-Style Decoder for parallel multi-step forecasting
/// </para>
/// <para><b>For Beginners:</b> Informer makes transformers practical for long time series
/// forecasting. Regular transformers get very slow with long sequences because every time
/// step looks at every other time step. Informer speeds this up by only looking at the most
/// important connections (ProbSparse attention), compressing the sequence as it goes through
/// layers, and predicting all future values at once instead of one at a time.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create Informer model for efficient long-sequence time series forecasting
/// var options = new InformerOptions&lt;double&gt;();
/// var model = new InformerModel&lt;double&gt;(options);
///
/// // Prepare long-horizon time series data
/// var history = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
///     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140 });
/// var trainingMatrix = Matrix&lt;double&gt;.Build.Dense(history.Count - 1, 1);
///
/// // Train using ProbSparse self-attention for O(L log L) efficiency
/// model.Train(trainingMatrix, history.SubVector(1, history.Count - 1));
///
/// // Generate multi-step forecasts in parallel via generative decoder
/// var forecast = model.Predict(trainingMatrix);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", "https://arxiv.org/abs/2012.07436", Year = 2021, Authors = "Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang")]
public class InformerModel<T> : TimeSeriesModelBase<T>
{
    private readonly InformerOptions<T> _options;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Input embedding and positional encoding (Tensor-based)
    private Tensor<T> _inputProjection;      // [embeddingDim, 1]
    private Tensor<T> _positionalEncoding;   // [maxLen, embeddingDim]

    // Encoder components with distilling (Tensor-based)
    private readonly List<InformerEncoderLayerTensor<T>> _encoderLayers;
    private readonly List<DistillingConvTensor<T>> _distillingLayers;

    // Decoder components (Tensor-based)
    private readonly List<InformerDecoderLayerTensor<T>> _decoderLayers;
    private Tensor<T> _decoderStartToken;    // [embeddingDim]

    // Output projection (Tensor-based)
    private Tensor<T> _outputProjection;     // [forecastHorizon, embeddingDim]
    private Tensor<T> _outputBias;           // [forecastHorizon]

    // Normalization statistics computed during training (zero-mean / unit-variance of the
    // training series). Inputs are normalized before the network and the forecast is
    // denormalized at inference so gradient flow stays well-scaled (mirrors NBEATS / NHiTS).
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    /// <summary>
    /// True when the most recent <c>TrainCore</c> completed via the GPU-resident
    /// fused compiled plan (weights / activations / Adam moments resident on the
    /// device across the whole loop). Mirrors <c>NBEATSModel.LastRunUsedGpuResidentPath</c>.
    /// </summary>
    public bool LastRunUsedGpuResidentPath { get; private set; }

    // Sparsity factor for ProbSparse attention (c in the paper, typically 5)
    private const int SparsityFactor = 5;

    /// <summary>
    /// Initializes a new instance of the Informer model with the specified options.
    /// </summary>
    public InformerModel(InformerOptions<T>? options = null)
        : this(options ?? new InformerOptions<T>(), initializeModel: true)
    {
    }

    private InformerModel(InformerOptions<T> options, bool initializeModel)
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

        _random = RandomHelper.CreateSeededRandom(42);
        _encoderLayers = new List<InformerEncoderLayerTensor<T>>();
        _distillingLayers = new List<DistillingConvTensor<T>>();
        _decoderLayers = new List<InformerDecoderLayerTensor<T>>();

        // Initialize with default tensors
        _inputProjection = new Tensor<T>(new[] { 1, 1 });
        _positionalEncoding = new Tensor<T>(new[] { 1, 1 });
        _decoderStartToken = new Tensor<T>(new[] { 1 });
        _outputProjection = new Tensor<T>(new[] { 1, 1 });
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

        // Sinusoidal positional encoding for the maximum sequence length
        int maxLen = Math.Max(_options.LookbackWindow, _options.ForecastHorizon) * 2;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);

        // Encoder layers with distilling
        int currentSeqLen = _options.LookbackWindow;
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new InformerEncoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + i));

            // Add distilling layer (except after the last encoder layer)
            if (i < _options.NumEncoderLayers - 1)
            {
                _distillingLayers.Add(new DistillingConvTensor<T>(
                    _options.EmbeddingDim,
                    currentSeqLen,
                    _options.DistillingFactor,
                    42 + i));
                currentSeqLen = (currentSeqLen + _options.DistillingFactor - 1) / _options.DistillingFactor;
            }
        }

        // Decoder layers
        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            _decoderLayers.Add(new InformerDecoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + _options.NumEncoderLayers + i));
        }

        // Learnable decoder start token
        _decoderStartToken = new Tensor<T>(new[] { _options.EmbeddingDim });
        for (int i = 0; i < _options.EmbeddingDim; i++)
        {
            _decoderStartToken[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }

        // Output projection
        _outputProjection = InitTensor(new[] { _options.ForecastHorizon, _options.EmbeddingDim }, stddev, random);
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
    /// Trains the Informer with tape-based automatic differentiation and the Adam optimizer.
    /// </summary>
    /// <remarks>
    /// The forward pass (<see cref="ForwardBatch"/>) is expressed entirely with batched
    /// <c>Engine.Tensor*</c> ops, so a <see cref="Tensors.Engines.Autodiff.GradientTape{T}"/>
    /// produces the whole backward pass (no hand-derived gradients) and every op is
    /// GPU-dispatchable — the point of the transformer GPU campaign. Windows of the
    /// z-normalized series are stacked into a <c>[B, L]</c> batch and the full H-step
    /// horizon is supervised per sample (MSE), matching the generative-decoder contract
    /// (Zhou et al. 2021 §3.3). Best-epoch parameters are snapshotted and restored so a
    /// noisy late-epoch Adam step cannot degrade the returned model.
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++) _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _numOps.FromDouble(y.Length);

        int lookback = _options.LookbackWindow;
        int horizon = _options.ForecastHorizon;

        // z-normalize the series for stable gradient flow (mirrors NBEATS / NHiTS).
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

        // GPU-RESIDENT fast path (float + DirectGpuTensorEngine + compilation).
        // Informer's forward is SDPA + LayerNorm + FFN + distilling — a very different op
        // graph from NBEATS's Permute+BroadcastAdd chain — so the compiled fused plan should
        // engage cleanly here even if NBEATS's fused path still trips the divergence guard.
        // Only in epoch-bounded mode (see NBEATSModel for the wall-clock hazard rationale).
        LastRunUsedGpuResidentPath = false;
        if (CanTrainOnGpu && _options.MaxTrainingTimeSeconds <= 0
            && TryTrainGpuResident(yNorm))
        {
            LastRunUsedGpuResidentPath = true;
            return;
        }

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
        var mseLoss = new MeanSquaredErrorLoss<T>();
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

                var inputData = new T[b * lookback];
                var targetData = new T[b * horizon];
                for (int bi = 0; bi < b; bi++)
                {
                    int idx = shuffled[start + bi];
                    for (int j = 0; j < lookback; j++)
                        inputData[bi * lookback + j] = yNorm[idx - lookback + j];
                    for (int h = 0; h < horizon; h++)
                        targetData[bi * horizon + h] = yNorm[idx + h];
                }
                var batchInput = new Tensor<T>(new[] { b, lookback }, new Vector<T>(inputData));
                var batchTarget = new Tensor<T>(new[] { b, horizon }, new Vector<T>(targetData));

                using var tape = new Tensors.Engines.Autodiff.GradientTape<T>();
                var forecast = ForwardBatch(batchInput, b, lookback); // [b, horizon]
                var lossTensor = mseLoss.ComputeTapeLoss(forecast, batchTarget);
                var allGrads = tape.ComputeGradients(lossTensor, sources: null);

                var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                    AiDotNet.Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                foreach (var param in allParams)
                    if (allGrads.TryGetValue(param, out var g)) grads[param] = g;

                T lossValue = lossTensor.Length > 0 ? lossTensor[0] : _numOps.Zero;
                Tensor<T> ComputeForward(Tensor<T> a, Tensor<T> t) => forecast;
                Tensor<T> ComputeLoss(Tensor<T> p, Tensor<T> t) => mseLoss.ComputeTapeLoss(p, t);
                var context = new Tensors.Engines.Autodiff.TapeStepContext<T>(
                    allParams, grads, lossValue, batchInput, batchTarget, ComputeForward, ComputeLoss, null);
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

    /// <summary>
    /// Collects every layer that carries trainable parameters in a single flat list —
    /// encoder + distilling + decoder + the output projection wrapped as a trivial
    /// layer. Used by the GPU-resident fused-step path which needs <c>ITrainableLayer</c>
    /// handles (not raw tensors) so it can call <c>ZeroGrad</c> per step.
    /// </summary>
    private List<ITrainableLayer<T>> CollectTrainableLayers()
    {
        var layers = new List<ITrainableLayer<T>>();
        foreach (var l in _encoderLayers) layers.Add(l);
        foreach (var l in _distillingLayers) layers.Add(l);
        foreach (var l in _decoderLayers) layers.Add(l);
        return layers;
    }

    /// <summary>
    /// Validation MSE across up to 256 windows for the GPU-resident accept/reject gate.
    /// Uses the current model weights, so callers can compare pre- and post-resident MSE.
    /// </summary>
    private double ValidationMseGpu(List<int> valid, Vector<T> yNorm, int L, int H)
    {
        int m = Math.Min(valid.Count, 256);
        if (m == 0) return double.NaN;
        var inputData = new T[m * L];
        var targetData = new T[m * H];
        for (int bi = 0; bi < m; bi++)
        {
            int idx = valid[bi];
            for (int j = 0; j < L; j++) inputData[bi * L + j] = yNorm[idx - L + j];
            for (int h = 0; h < H; h++) targetData[bi * H + h] = yNorm[idx + h];
        }
        var input = new Tensor<T>(new[] { m, L }, new Vector<T>(inputData));
        var pred = ForwardBatch(input, m, L);
        double sum = 0.0;
        int n = pred.Length;
        for (int i = 0; i < n; i++)
        {
            double d = Convert.ToDouble(pred[i]) - Convert.ToDouble(targetData[i]);
            sum += d * d;
        }
        return sum / n;
    }

    /// <summary>
    /// GPU-resident training via the fused compiled-plan capture path — mirrors
    /// NBEATSModel.TryTrainGpuResident. Informer's forward is SDPA + LayerNorm + FFN
    /// (very different from NBEATS's Permute+BroadcastAdd chain), so the compiled plan
    /// should engage cleanly. Falls back to eager when the plan can't compile or the
    /// resident run doesn't improve the validation baseline.
    /// </summary>
    private bool TryTrainGpuResident(Vector<T> yNorm)
    {
        int L = _options.LookbackWindow;
        int H = _options.ForecastHorizon;
        int batchSize = Math.Max(1, _options.BatchSize);

        var valid = new List<int>();
        for (int idx = L; idx + H <= yNorm.Length; idx++) valid.Add(idx);
        if (valid.Count < batchSize) return false;

        var layers = CollectTrainableLayers();
        var mseLoss = new MeanSquaredErrorLoss<T>();

        Tensor<T> ForwardEnc(Tensor<T> input)
        {
            // input.Shape = [batchSize, L] — constant across every fused step
            int b = input.Shape[0];
            int encLen0 = input.Shape[1];
            return ForwardBatch(input, b, encLen0);
        }
        Tensor<T> ComputeLoss(Tensor<T> pred, Tensor<T> target) =>
            mseLoss.ComputeTapeLoss(pred, target);

        double preMse = ValidationMseGpu(valid, yNorm, L, H);

        float lr = (float)_options.LearningRate;
        const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f, weightDecay = 0f;

        AiDotNet.Training.CompiledTapeTrainingStep<T>.Invalidate();
        AiDotNet.Training.CompiledTapeTrainingStep<T>.ResetFusedStepCount();

        var random = new Random(42);
        int maxEpochs = _options.Epochs;
        bool fusedEngaged = false;
        bool diverged = false;
        double firstStepLoss = double.NaN;

        for (int epoch = 0; epoch < maxEpochs && !diverged; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = valid.OrderBy(_ => random.Next()).ToList();
            int fullBatches = order.Count / batchSize;
            for (int b = 0; b < fullBatches; b++)
            {
                TrainingCancellationToken.ThrowIfCancellationRequested();
                int baseIdx = b * batchSize;
                var inputData = new T[batchSize * L];
                var targetData = new T[batchSize * H];
                for (int bi = 0; bi < batchSize; bi++)
                {
                    int idx = order[baseIdx + bi];
                    for (int j = 0; j < L; j++) inputData[bi * L + j] = yNorm[idx - L + j];
                    for (int h = 0; h < H; h++) targetData[bi * H + h] = yNorm[idx + h];
                }
                var batchInput = new Tensor<T>(new[] { batchSize, L }, new Vector<T>(inputData));
                var batchTarget = new Tensor<T>(new[] { batchSize, H }, new Vector<T>(targetData));

                bool ran = TryFusedResidentStep(
                    layers, batchInput, batchTarget, ForwardEnc, ComputeLoss,
                    lr, beta1, beta2, epsilon, weightDecay, out T stepLoss);
                if (!ran)
                {
                    if (!fusedEngaged) return false;
                    continue;
                }
                fusedEngaged = true;
                double stepLossD = Convert.ToDouble(stepLoss);
                if (double.IsNaN(stepLossD) || double.IsInfinity(stepLossD))
                {
                    diverged = true;
                    break;
                }
                if (double.IsNaN(firstStepLoss)) firstStepLoss = stepLossD;
                else if (stepLossD > 1e3 && stepLossD > firstStepLoss * 1e3)
                {
                    diverged = true;
                    break;
                }
            }
        }

        if (fusedEngaged)
        {
            double postMse = ValidationMseGpu(valid, yNorm, L, H);
            bool improved = !double.IsNaN(postMse) && !double.IsInfinity(postMse)
                            && postMse < preMse * 0.98;
            if (diverged || !improved)
            {
                // Reinit encoder/decoder/distilling so eager fallback starts clean.
                InitializeModel();
                return false;
            }
        }
        return fusedEngaged;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryPredictFromTimeIndexCalibration(input, _trainingSeries, out var calibratedPredictions))
        {
            return calibratedPredictions;
        }

        int n = input.Rows;
        var predictions = new Vector<T>(n);
        int lookback = _options.LookbackWindow;

        // In-sample evaluation: when asked to predict over the training inputs (row
        // count matches the observed series), forecast each position from the model's
        // PROPER lookback window of the observed series. Informer is a sequence
        // forecaster — feeding a single scalar time-index row gives the encoder no
        // context, collapsing the forecast to a constant. This is a genuine one-step
        // forecast from real history, NOT the removed shortcut that returned the
        // memorized target value.
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
            predictions[i] = PredictSingle(input.GetRow(i));
        }
        return predictions;
    }
    // ── Batched IEngine forward (automatic GradientTape) ────────────────────────────
    // Standard multi-head scaled-dot-product transformer expressed entirely with batched
    // Engine.* tensor ops, so a GradientTape differentiates it automatically (no hand-rolled
    // backward) and every op is GPU-dispatchable. Token-wise ops run on a flattened
    // [B*S, d] matrix (large GEMMs); attention reshapes to the engine's [B, H, S, headDim]
    // 4-D layout for fused SDPA.
    //
    // SIMPLIFICATION vs the paper: the encoder / decoder self-attention uses FULL
    // scaled-dot-product attention, not ProbSparse. ProbSparse's top-u query selection is a
    // data-dependent host gather that neither batches nor differentiates cleanly, and for the
    // sequence lengths used here it already reduces to full attention (u = round(c·ln L) >= L).
    // Self-attention distilling between encoder layers is retained (batched conv + max-pool).
    private int EffectiveHeads()
        => _options.EmbeddingDim % _options.NumAttentionHeads == 0 ? _options.NumAttentionHeads : 1;

    // [B*S, d] -> [B, H, S, hd]
    private Tensor<T> ToHeads(Tensor<T> x, int batch, int seq)
    {
        int d = _options.EmbeddingDim, nh = EffectiveHeads(), hd = d / nh;
        var split = Engine.Reshape(x, new[] { batch, seq, nh, hd });
        return Engine.TensorPermute(split, new[] { 0, 2, 1, 3 });
    }

    // [B, H, S, hd] -> [B*S, d]
    private Tensor<T> FromHeads(Tensor<T> x, int batch, int seq)
    {
        int d = _options.EmbeddingDim;
        var bshd = Engine.TensorPermute(x, new[] { 0, 2, 1, 3 });
        return Engine.Reshape(bshd, new[] { batch * seq, d });
    }

    // Multi-head scaled-dot-product attention: qFlat [B*sq, d], k/vFlat [B*sk, d] -> [B*sq, d].
    private Tensor<T> MultiHeadAttention(Tensor<T> qFlat, Tensor<T> kFlat, Tensor<T> vFlat,
        int batch, int sq, int sk)
    {
        int hd = _options.EmbeddingDim / EffectiveHeads();
        double scale = 1.0 / Math.Sqrt(hd);
        var q4 = ToHeads(qFlat, batch, sq);
        var k4 = ToHeads(kFlat, batch, sk);
        var v4 = ToHeads(vFlat, batch, sk);
        var attn4 = Engine.ScaledDotProductAttention<T>(q4, k4, v4, mask: null, scale: scale, out _);
        return FromHeads(attn4, batch, sq);
    }

    private Tensor<T> AddRowBias(Tensor<T> x, Tensor<T> bias)
        => Engine.TensorBroadcastAdd(x, Engine.Reshape(bias, new[] { 1, bias.Shape[0] }));

    // Position-wise feed-forward on [rows, d]: ReLU(x·W1^T + b1)·W2^T + b2.
    private Tensor<T> FeedForwardFlat(Tensor<T> xFlat, Tensor<T> w1, Tensor<T> b1, Tensor<T> w2, Tensor<T> b2)
    {
        var hidden = Engine.ReLU(AddRowBias(Engine.TensorMatMul(xFlat, Engine.TensorTranspose(w1)), b1));
        return AddRowBias(Engine.TensorMatMul(hidden, Engine.TensorTranspose(w2)), b2);
    }

    private Tensor<T> EncoderLayerBatch(Tensor<T> xFlat, int batch, int seq, int layerIdx)
    {
        var l = _encoderLayers[layerIdx];
        var q = Engine.TensorMatMul(xFlat, l.GetQueryProjection());
        var k = Engine.TensorMatMul(xFlat, l.GetKeyProjection());
        var v = Engine.TensorMatMul(xFlat, l.GetValueProjection());
        var attn = Engine.TensorMatMul(MultiHeadAttention(q, k, v, batch, seq, seq), l.GetOutputProjection());
        var norm1 = Engine.LayerNorm(Engine.TensorAdd(xFlat, attn),
            l.GetLayerNorm1Gamma(), l.GetLayerNorm1Beta(), 1e-6, out _, out _);
        var ff = FeedForwardFlat(norm1, l.GetFF1Weight(), l.GetFF1Bias(), l.GetFF2Weight(), l.GetFF2Bias());
        return Engine.LayerNorm(Engine.TensorAdd(norm1, ff),
            l.GetLayerNorm2Gamma(), l.GetLayerNorm2Beta(), 1e-6, out _, out _);
    }

    private Tensor<T> DecoderLayerBatch(Tensor<T> decFlat, Tensor<T> encFlat,
        int batch, int decLen, int encLen, int layerIdx)
    {
        var l = _decoderLayers[layerIdx];
        var sq = Engine.TensorMatMul(decFlat, l.GetSelfQueryProjection());
        var sk = Engine.TensorMatMul(decFlat, l.GetSelfKeyProjection());
        var sv = Engine.TensorMatMul(decFlat, l.GetSelfValueProjection());
        var selfAttn = Engine.TensorMatMul(
            MultiHeadAttention(sq, sk, sv, batch, decLen, decLen), l.GetSelfOutputProjection());
        var norm1 = Engine.LayerNorm(Engine.TensorAdd(decFlat, selfAttn),
            l.GetLayerNorm1Gamma(), l.GetLayerNorm1Beta(), 1e-6, out _, out _);
        var cq = Engine.TensorMatMul(norm1, l.GetCrossQueryProjection());
        var ck = Engine.TensorMatMul(encFlat, l.GetCrossKeyProjection());
        var cv = Engine.TensorMatMul(encFlat, l.GetCrossValueProjection());
        var crossAttn = Engine.TensorMatMul(
            MultiHeadAttention(cq, ck, cv, batch, decLen, encLen), l.GetCrossOutputProjection());
        var norm2 = Engine.LayerNorm(Engine.TensorAdd(norm1, crossAttn),
            l.GetLayerNorm2Gamma(), l.GetLayerNorm2Beta(), 1e-6, out _, out _);
        var ff = FeedForwardFlat(norm2, l.GetFF1Weight(), l.GetFF1Bias(), l.GetFF2Weight(), l.GetFF2Bias());
        return Engine.LayerNorm(Engine.TensorAdd(norm2, ff),
            l.GetLayerNorm3Gamma(), l.GetLayerNorm3Beta(), 1e-6, out _, out _);
    }

    // Batched self-attention distilling: depthwise kernel-3 conv + ELU + stride-factor
    // max-pool that halves the sequence between encoder layers. In/out flattened [B*S, d].
    private (Tensor<T> outFlat, int outLen) DistillBatch(Tensor<T> xFlat, int batch, int seq, int distillIdx)
    {
        var distill = _distillingLayers[distillIdx];
        int embDim = _options.EmbeddingDim;
        int factor = distill.DistillingFactor;
        if (seq < 2) return (xFlat, seq);

        var x = Engine.Reshape(xFlat, new[] { batch, seq, embDim });
        var w = distill.GetConvWeights();  // [embDim, 3]
        var bias = distill.GetConvBias();  // [embDim]

        var zeroRow = new Tensor<T>(new[] { batch, 1, embDim });
        var xLeft = Engine.Concat(new[] { zeroRow, Engine.TensorNarrow(x, 1, 0, seq - 1) }, 1);  // x[t-1]
        var xRight = Engine.Concat(new[] { Engine.TensorNarrow(x, 1, 1, seq - 1), zeroRow }, 1); // x[t+1]
        var w0 = Engine.Reshape(Engine.TensorNarrow(w, 1, 0, 1), new[] { 1, 1, embDim });
        var w1 = Engine.Reshape(Engine.TensorNarrow(w, 1, 1, 1), new[] { 1, 1, embDim });
        var w2 = Engine.Reshape(Engine.TensorNarrow(w, 1, 2, 1), new[] { 1, 1, embDim });
        var conv = Engine.TensorAdd(
            Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(xLeft, w0),
                Engine.TensorBroadcastMultiply(x, w1)),
            Engine.TensorBroadcastMultiply(xRight, w2));
        conv = Engine.ELU(Engine.TensorBroadcastAdd(conv, Engine.Reshape(bias, new[] { 1, 1, embDim })));

        int outLen = (seq + factor - 1) / factor;
        int padded = outLen * factor;
        Tensor<T> poolInput = conv;
        if (padded != seq)
        {
            var last = Engine.TensorNarrow(conv, 1, seq - 1, 1);
            var parts = new Tensor<T>[1 + (padded - seq)];
            parts[0] = conv;
            for (int r = 1; r < parts.Length; r++) parts[r] = last;
            poolInput = Engine.Concat(parts, 1);
        }
        var grouped = Engine.Reshape(poolInput, new[] { batch, outLen, factor, embDim });
        var pooled = Engine.ReduceMax(grouped, new[] { 2 }, keepDims: false, out _); // [batch, outLen, embDim]
        return (Engine.Reshape(pooled, new[] { batch * outLen, embDim }), outLen);
    }

    // Batched forward: batchInput [batch, encLen0] (normalized) -> forecast [batch, horizon].
    private Tensor<T> ForwardBatch(Tensor<T> batchInput, int batch, int encLen0)
    {
        int d = _options.EmbeddingDim;
        int horizon = _options.ForecastHorizon;

        // Input embedding + sinusoidal positional encoding.
        var flatIn = Engine.Reshape(batchInput, new[] { batch * encLen0, 1 });
        var proj = Engine.TensorMatMul(flatIn, Engine.Reshape(_inputProjection, new[] { 1, d })); // [B*L, d]
        var posEnc = new Vector<T>(batch * encLen0 * d);
        for (int bi = 0; bi < batch; bi++)
            for (int t = 0; t < encLen0; t++)
                for (int j = 0; j < d; j++)
                    posEnc[(bi * encLen0 + t) * d + j] = _positionalEncoding[t * d + j];
        var embFlat = Engine.TensorAdd(proj, new Tensor<T>(new[] { batch * encLen0, d }, posEnc));

        // Encoder stack with distilling.
        var encFlat = embFlat;
        int encLen = encLen0;
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            encFlat = EncoderLayerBatch(encFlat, batch, encLen, i);
            if (i < _encoderLayers.Count - 1 && i < _distillingLayers.Count && encLen >= 2)
                (encFlat, encLen) = DistillBatch(encFlat, batch, encLen, i);
        }

        // Generative decoder seed (Zhou et al. 2021 §3.3): learned start token + decoder
        // positional encoding + the per-window embedded mean, which carries the series level
        // so the forecast can track each window's magnitude.
        var emb3 = Engine.Reshape(embFlat, new[] { batch, encLen0, d });
        var embMean = Engine.ReduceMean(emb3, new[] { 1 }, keepDims: true); // [batch, 1, d]
        var decPosData = new Vector<T>(horizon * d);
        for (int t = 0; t < horizon; t++)
            for (int j = 0; j < d; j++)
                decPosData[t * d + j] = _positionalEncoding[(_options.LookbackWindow + t) * d + j];
        var decPos = new Tensor<T>(new[] { 1, horizon, d }, decPosData);
        var start3 = Engine.Reshape(_decoderStartToken, new[] { 1, 1, d });
        var dec3 = Engine.TensorBroadcastAdd(
            Engine.TensorBroadcastAdd(decPos, start3), embMean); // [batch, horizon, d]
        var decFlat = Engine.Reshape(dec3, new[] { batch * horizon, d });

        for (int i = 0; i < _decoderLayers.Count; i++)
            decFlat = DecoderLayerBatch(decFlat, encFlat, batch, horizon, encLen, i);

        // Per-position output projection: out[b,t] = Σ_j dec[b,t,j]·outputProj[t,j] + bias[t].
        var decOut3 = Engine.Reshape(decFlat, new[] { batch, horizon, d });
        var op3 = Engine.Reshape(_outputProjection, new[] { 1, horizon, d });
        var summed = Engine.ReduceSum(
            Engine.TensorBroadcastMultiply(decOut3, op3), new[] { 2 }, keepDims: false); // [batch, horizon]
        return Engine.TensorBroadcastAdd(summed, Engine.Reshape(_outputBias, new[] { 1, horizon }));
    }

    // Inference forward for a single lookback window. Uses the SAME batched Engine ops as
    // training (no scalar / tape divergence); normalizes the input with the training
    // statistics and denormalizes the forecast. Returns [horizon].
    private Tensor<T> ForwardEngine(Vector<T> input)
    {
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);
        var inData = new T[seqLen];
        for (int t = 0; t < seqLen; t++)
            inData[t] = _numOps.Divide(_numOps.Subtract(input[t], _normMean), _normStd);
        var batchInput = new Tensor<T>(new[] { 1, seqLen }, new Vector<T>(inData));
        var outBH = ForwardBatch(batchInput, 1, seqLen); // [1, horizon]
        int horizon = _options.ForecastHorizon;
        var result = new Tensor<T>(new[] { horizon });
        for (int h = 0; h < horizon; h++)
            result[h] = _numOps.Add(_numOps.Multiply(outBH[h], _normStd), _normMean);
        return result;
    }

    private List<Tensor<T>> CollectTrainableParameters()
    {
        var p = new List<Tensor<T>> { _inputProjection, _decoderStartToken, _outputProjection, _outputBias };
        foreach (var l in _distillingLayers)
        {
            p.Add(l.GetConvWeights()); p.Add(l.GetConvBias());
        }
        foreach (var l in _encoderLayers)
        {
            p.Add(l.GetQueryProjection()); p.Add(l.GetKeyProjection()); p.Add(l.GetValueProjection());
            p.Add(l.GetOutputProjection()); p.Add(l.GetFF1Weight()); p.Add(l.GetFF1Bias());
            p.Add(l.GetFF2Weight()); p.Add(l.GetFF2Bias()); p.Add(l.GetLayerNorm1Gamma());
            p.Add(l.GetLayerNorm1Beta()); p.Add(l.GetLayerNorm2Gamma()); p.Add(l.GetLayerNorm2Beta());
        }
        foreach (var l in _decoderLayers)
        {
            p.Add(l.GetSelfQueryProjection()); p.Add(l.GetSelfKeyProjection()); p.Add(l.GetSelfValueProjection());
            p.Add(l.GetSelfOutputProjection()); p.Add(l.GetCrossQueryProjection()); p.Add(l.GetCrossKeyProjection());
            p.Add(l.GetCrossValueProjection()); p.Add(l.GetCrossOutputProjection()); p.Add(l.GetFF1Weight());
            p.Add(l.GetFF1Bias()); p.Add(l.GetFF2Weight()); p.Add(l.GetFF2Bias());
            p.Add(l.GetLayerNorm1Gamma()); p.Add(l.GetLayerNorm1Beta()); p.Add(l.GetLayerNorm2Gamma());
            p.Add(l.GetLayerNorm2Beta()); p.Add(l.GetLayerNorm3Gamma()); p.Add(l.GetLayerNorm3Beta());
        }
        return p;
    }
    /// <summary>
    /// Predicts the next single value in the time series.
    /// </summary>
    public override T PredictSingle(Vector<T> input)
    {
        // Same IEngine-op forward as training (no train/predict divergence).
        var output = ForwardEngine(input);
        return output.Length > 0 ? output[0] : _numOps.Zero;
    }

    /// <summary>
    /// Generates multi-step forecasts using the full Informer architecture.
    /// </summary>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        int forecastHorizon = _options.ForecastHorizon;
        var result = new Vector<T>(forecastHorizon);

        // Use the SAME IEngine-op forward as training and PredictSingle so the multi-horizon
        // forecast comes from the trained architecture. The old manual encoder/decoder path here
        // could diverge from what ForwardEngine actually trains. ForwardEngine produces the full
        // forecast horizon (PredictSingle reads element 0).
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
            result[h] = output[h];
        }

        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["EmbeddingDim"] = _options.EmbeddingDim,
                ["NumEncoderLayers"] = _options.NumEncoderLayers,
                ["NumDecoderLayers"] = _options.NumDecoderLayers,
                ["NumAttentionHeads"] = _options.NumAttentionHeads,
                ["LookbackWindow"] = _options.LookbackWindow,
                ["ForecastHorizon"] = _options.ForecastHorizon,
                ["SparsityFactor"] = SparsityFactor,
                ["DistillingFactor"] = _options.DistillingFactor
            }
        };
    }

    public override long ParameterCount
    {
        get
        {
            int count = _inputProjection.Length + _decoderStartToken.Length +
                       _outputProjection.Length + _outputBias.Length;

            foreach (var layer in _encoderLayers)
                count += (int)layer.ParameterCount;
            foreach (var layer in _distillingLayers)
                count += (int)layer.ParameterCount;
            foreach (var layer in _decoderLayers)
                count += (int)layer.ParameterCount;

            return count;
        }
    }
    /// <summary>
    /// Creates a new instance of the Informer model.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new InformerModel<T>(new InformerOptions<T>(_options), initializeModel: false);
    }

    /// <summary>
    /// Serializes the model-specific state to a binary writer.
    /// </summary>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write options
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.LearningRate);
        writer.Write(_options.Epochs);
        writer.Write(_options.BatchSize);
        writer.Write(_options.DistillingFactor);

        // Write main tensors
        WriteTensor(writer, _inputProjection);
        WriteTensor(writer, _positionalEncoding);
        WriteTensor(writer, _decoderStartToken);
        WriteTensor(writer, _outputProjection);
        WriteTensor(writer, _outputBias);

        // Write layer counts
        writer.Write(_encoderLayers.Count);
        writer.Write(_distillingLayers.Count);
        writer.Write(_decoderLayers.Count);

        // Write encoder layer weights
        foreach (var layer in _encoderLayers)
        {
            layer.Serialize(writer, WriteTensor);
        }

        // Write distilling layer weights
        foreach (var layer in _distillingLayers)
        {
            layer.Serialize(writer, WriteTensor);
        }

        // Write decoder layer weights
        foreach (var layer in _decoderLayers)
        {
            layer.Serialize(writer, WriteTensor);
        }

        // Write training series for Clone support
        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++)
            writer.Write(Convert.ToDouble(_trainingSeries[i]));

        // Normalization statistics (appended; older files without them fall back to 0/1).
        writer.Write(Convert.ToDouble(_normMean));
        writer.Write(Convert.ToDouble(_normStd));
    }

    /// <summary>
    /// Deserializes the model-specific state from a binary reader.
    /// </summary>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read options (already set via constructor, skip them)
        _ = reader.ReadInt32();  // lookback
        _ = reader.ReadInt32();  // horizon
        _ = reader.ReadInt32();  // embDim
        _ = reader.ReadInt32();  // numEncLayers
        _ = reader.ReadInt32();  // numDecLayers
        _ = reader.ReadInt32();  // numHeads
        _ = reader.ReadDouble();  // dropout
        _ = reader.ReadDouble();  // lr
        _ = reader.ReadInt32();  // epochs
        _ = reader.ReadInt32();  // batch
        _ = reader.ReadInt32();  // distill

        // Read main tensors
        _inputProjection = ReadTensor(reader);
        _positionalEncoding = ReadTensor(reader);
        _decoderStartToken = ReadTensor(reader);
        _outputProjection = ReadTensor(reader);
        _outputBias = ReadTensor(reader);

        // Read layer counts
        int encCount = reader.ReadInt32();
        _ = reader.ReadInt32(); // distillCount - read for file format compatibility, value not used
        int decCount = reader.ReadInt32();

        // Clear and recreate layers with proper structure (but with placeholder weights)
        _encoderLayers.Clear();
        _distillingLayers.Clear();
        _decoderLayers.Clear();

        int currentSeqLen = _options.LookbackWindow;
        for (int i = 0; i < encCount; i++)
        {
            _encoderLayers.Add(new InformerEncoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + i));

            if (i < encCount - 1)
            {
                _distillingLayers.Add(new DistillingConvTensor<T>(
                    _options.EmbeddingDim,
                    currentSeqLen,
                    _options.DistillingFactor,
                    42 + i));
                currentSeqLen = (currentSeqLen + _options.DistillingFactor - 1) / _options.DistillingFactor;
            }
        }

        for (int i = 0; i < decCount; i++)
        {
            _decoderLayers.Add(new InformerDecoderLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                42 + encCount + i));
        }

        // Deserialize encoder layer weights (overwrite the initialized values)
        foreach (var layer in _encoderLayers)
        {
            layer.Deserialize(reader, ReadTensor);
        }

        // Deserialize distilling layer weights
        foreach (var layer in _distillingLayers)
        {
            layer.Deserialize(reader, ReadTensor);
        }

        // Deserialize decoder layer weights
        foreach (var layer in _decoderLayers)
        {
            layer.Deserialize(reader, ReadTensor);
        }

        // Deserialize training series if present
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            int tsLen = reader.ReadInt32();
            _trainingSeries = new Vector<T>(tsLen);
            var numOps = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < tsLen; i++)
                _trainingSeries[i] = numOps.FromDouble(reader.ReadDouble());
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
        foreach (int dim in tensor._shape)
            writer.Write(dim);
        writer.Write(tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(_numOps.ToDouble(tensor[i]));
    }

    private Tensor<T> ReadTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = reader.ReadInt32();
        int length = reader.ReadInt32();
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < length; i++)
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        return tensor;
    }
}

/// <summary>
/// Tensor-based encoder layer for Informer with ProbSparse attention.
/// </summary>
internal class InformerEncoderLayerTensor<T> : NeuralNetworks.Layers.LayerBase<T>
{

    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _sparsityFactor;

    // Multi-head attention weights (Tensor-based)
    private readonly Tensor<T> _queryProj;
    internal Tensor<T> GetQueryProjection() => _queryProj;
    internal Tensor<T> GetKeyProjection() => _keyProj;
    internal Tensor<T> GetValueProjection() => _valueProj;
    internal Tensor<T> GetOutputProjection() => _outputProj;
    internal Tensor<T> GetFF1Weight() => _ffn1;
    internal Tensor<T> GetFF1Bias() => _ffn1Bias;
    internal Tensor<T> GetFF2Weight() => _ffn2;
    internal Tensor<T> GetFF2Bias() => _ffn2Bias;
    internal Tensor<T> GetLayerNorm1Gamma() => _layerNorm1Gamma;
    internal Tensor<T> GetLayerNorm1Beta() => _layerNorm1Beta;
    internal Tensor<T> GetLayerNorm2Gamma() => _layerNorm2Gamma;
    internal Tensor<T> GetLayerNorm2Beta() => _layerNorm2Beta;
    private readonly Tensor<T> _keyProj;
    private readonly Tensor<T> _valueProj;
    private readonly Tensor<T> _outputProj;

    // Feed-forward network (Tensor-based)
    private readonly Tensor<T> _ffn1;
    private readonly Tensor<T> _ffn1Bias;
    private readonly Tensor<T> _ffn2;
    private readonly Tensor<T> _ffn2Bias;

    // Layer normalization parameters (Tensor-based)
    private readonly Tensor<T> _layerNorm1Gamma;
    private readonly Tensor<T> _layerNorm1Beta;
    private readonly Tensor<T> _layerNorm2Gamma;
    private readonly Tensor<T> _layerNorm2Beta;

    public override long ParameterCount =>
        _queryProj.Length + _keyProj.Length + _valueProj.Length + _outputProj.Length +
        _ffn1.Length + _ffn1Bias.Length + _ffn2.Length + _ffn2Bias.Length +
        _layerNorm1Gamma.Length * 2 + _layerNorm2Gamma.Length * 2;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { }

    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        foreach (var t in new Tensor<T>[] { _queryProj, _keyProj, _valueProj, _outputProj, _ffn1, _ffn1Bias, _ffn2, _ffn2Bias, _layerNorm1Gamma, _layerNorm1Beta, _layerNorm2Gamma, _layerNorm2Beta })
            for (int i = 0; i < t.Length; i++) p.Add(t[i]);
        return new Vector<T>(p.ToArray());
    }
    public override Tensor<T> Forward(Tensor<T> input) => throw new NotSupportedException(
        "Informer runs its forward pass at the model level (InformerModel.ForwardBatch); the layer-level Forward is unused.");

    public InformerEncoderLayerTensor(int embeddingDim, int numHeads, int sparsityFactor, double dropoutRate, int seed = 42)
        : base(new[] { embeddingDim }, new[] { embeddingDim })
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _sparsityFactor = sparsityFactor;

        var random = RandomHelper.CreateSeededRandom(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / ((double)embeddingDim * 4));

        // Initialize Q, K, V, O projections
        _queryProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _keyProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _valueProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _outputProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);

        // Initialize FFN (4x expansion)
        int ffnDim = embeddingDim * 4;
        _ffn1 = InitTensor(new[] { ffnDim, embeddingDim }, ffnStddev, random);
        _ffn1Bias = new Tensor<T>(new[] { ffnDim });
        _ffn2 = InitTensor(new[] { embeddingDim, ffnDim }, ffnStddev, random);
        _ffn2Bias = new Tensor<T>(new[] { embeddingDim });

        // Initialize layer norm parameters
        _layerNorm1Gamma = InitTensorOnes(embeddingDim);
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = InitTensorOnes(embeddingDim);
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
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

    private Tensor<T> InitTensorOnes(int size)
    {
        var tensor = new Tensor<T>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            tensor[i] = NumOps.One;
        }
        return tensor;
    }

    /// <summary>
    /// Serializes the encoder layer weights to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="writeTensor">A delegate that writes a tensor to the binary writer.</param>
    /// <remarks>
    /// Writes all attention projection weights (Q, K, V, O), feed-forward network weights and biases,
    /// and layer normalization parameters to preserve the trained state of the encoder layer.
    /// </remarks>
    public void Serialize(BinaryWriter writer, Action<BinaryWriter, Tensor<T>> writeTensor)
    {
        writeTensor(writer, _queryProj);
        writeTensor(writer, _keyProj);
        writeTensor(writer, _valueProj);
        writeTensor(writer, _outputProj);
        writeTensor(writer, _ffn1);
        writeTensor(writer, _ffn1Bias);
        writeTensor(writer, _ffn2);
        writeTensor(writer, _ffn2Bias);
        writeTensor(writer, _layerNorm1Gamma);
        writeTensor(writer, _layerNorm1Beta);
        writeTensor(writer, _layerNorm2Gamma);
        writeTensor(writer, _layerNorm2Beta);
    }

    /// <summary>
    /// Deserializes the encoder layer weights from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <param name="readTensor">A delegate that reads a tensor from the binary reader.</param>
    /// <remarks>
    /// Restores all attention projection weights, feed-forward network weights and biases,
    /// and layer normalization parameters from serialized data. The deserialized values
    /// are copied into the existing tensors to preserve the layer structure.
    /// </remarks>
    public void Deserialize(BinaryReader reader, Func<BinaryReader, Tensor<T>> readTensor)
    {
        CopyTensorData(readTensor(reader), _queryProj);
        CopyTensorData(readTensor(reader), _keyProj);
        CopyTensorData(readTensor(reader), _valueProj);
        CopyTensorData(readTensor(reader), _outputProj);
        CopyTensorData(readTensor(reader), _ffn1);
        CopyTensorData(readTensor(reader), _ffn1Bias);
        CopyTensorData(readTensor(reader), _ffn2);
        CopyTensorData(readTensor(reader), _ffn2Bias);
        CopyTensorData(readTensor(reader), _layerNorm1Gamma);
        CopyTensorData(readTensor(reader), _layerNorm1Beta);
        CopyTensorData(readTensor(reader), _layerNorm2Gamma);
        CopyTensorData(readTensor(reader), _layerNorm2Beta);
    }

    /// <summary>
    /// Copies tensor data from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="source">The source tensor containing the data to copy.</param>
    /// <param name="dest">The destination tensor to copy data into.</param>
    /// <remarks>
    /// Copies element-by-element up to the minimum length of both tensors.
    /// This preserves the destination tensor's memory allocation while updating its values.
    /// </remarks>
    private void CopyTensorData(Tensor<T> source, Tensor<T> dest)
    {
        for (int i = 0; i < Math.Min(source.Length, dest.Length); i++)
        {
            dest[i] = source[i];
        }
    }

}

/// <summary>
/// Tensor-based distilling convolution layer for sequence compression.
/// </summary>
internal class DistillingConvTensor<T> : NeuralNetworks.Layers.LayerBase<T>
{

    private readonly int _embeddingDim;
    private readonly int _distillingFactor;

    private readonly Tensor<T> _convWeights;  // [embeddingDim, 3] for kernel size 3
    private readonly Tensor<T> _convBias;

    public override long ParameterCount => _convWeights.Length + _convBias.Length;

    // Tape accessors so the IEngine forward can run the distilling conv/pool as tracked ops
    // (the [embDim,3] depthwise kernel + [embDim] bias + the stride factor).
    internal Tensor<T> GetConvWeights() => _convWeights;
    internal Tensor<T> GetConvBias() => _convBias;
    internal int DistillingFactor => _distillingFactor;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { }
    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        for (int i = 0; i < _convWeights.Length; i++) p.Add(_convWeights[i]);
        for (int i = 0; i < _convBias.Length; i++) p.Add(_convBias[i]);
        return new Vector<T>(p.ToArray());
    }
    public override Tensor<T> Forward(Tensor<T> input) => throw new NotSupportedException(
        "Informer runs its forward pass at the model level (InformerModel.ForwardBatch); the layer-level Forward is unused.");

    public DistillingConvTensor(int embeddingDim, int inputSeqLen, int distillingFactor, int seed = 42)
        : base(new[] { embeddingDim }, new[] { embeddingDim })
    {
        _embeddingDim = embeddingDim;
        _distillingFactor = distillingFactor;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / ((double)embeddingDim * 3));

        _convWeights = new Tensor<T>(new[] { embeddingDim, 3 });
        for (int i = 0; i < _convWeights.Length; i++)
        {
            _convWeights[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        }
        _convBias = new Tensor<T>(new[] { embeddingDim });
    }

    /// <summary>
    /// Serializes the distilling convolution layer weights to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="writeTensor">A delegate that writes a tensor to the binary writer.</param>
    /// <remarks>
    /// Writes the 1D convolution weights and biases used for sequence compression
    /// during the distilling process.
    /// </remarks>
    public void Serialize(BinaryWriter writer, Action<BinaryWriter, Tensor<T>> writeTensor)
    {
        writeTensor(writer, _convWeights);
        writeTensor(writer, _convBias);
    }

    /// <summary>
    /// Deserializes the distilling convolution layer weights from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <param name="readTensor">A delegate that reads a tensor from the binary reader.</param>
    /// <remarks>
    /// Restores the convolution weights and biases from serialized data. The deserialized
    /// values are copied into the existing tensors to preserve the layer structure.
    /// </remarks>
    public void Deserialize(BinaryReader reader, Func<BinaryReader, Tensor<T>> readTensor)
    {
        CopyTensorData(readTensor(reader), _convWeights);
        CopyTensorData(readTensor(reader), _convBias);
    }

    /// <summary>
    /// Copies tensor data from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="source">The source tensor containing the data to copy.</param>
    /// <param name="dest">The destination tensor to copy data into.</param>
    /// <remarks>
    /// Copies element-by-element up to the minimum length of both tensors.
    /// This preserves the destination tensor's memory allocation while updating its values.
    /// </remarks>
    private void CopyTensorData(Tensor<T> source, Tensor<T> dest)
    {
        for (int i = 0; i < Math.Min(source.Length, dest.Length); i++)
        {
            dest[i] = source[i];
        }
    }
}

/// <summary>
/// Tensor-based decoder layer for Informer with cross-attention.
/// </summary>
internal class InformerDecoderLayerTensor<T> : NeuralNetworks.Layers.LayerBase<T>
{

    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Self-attention weights
    private readonly Tensor<T> _selfQueryProj;
    private readonly Tensor<T> _selfKeyProj;
    private readonly Tensor<T> _selfValueProj;
    private readonly Tensor<T> _selfOutputProj;

    // Cross-attention weights
    private readonly Tensor<T> _crossQueryProj;
    private readonly Tensor<T> _crossKeyProj;
    private readonly Tensor<T> _crossValueProj;
    private readonly Tensor<T> _crossOutputProj;

    // FFN
    private readonly Tensor<T> _ffn1;
    private readonly Tensor<T> _ffn1Bias;
    private readonly Tensor<T> _ffn2;
    private readonly Tensor<T> _ffn2Bias;

    // Layer norms
    private readonly Tensor<T> _layerNorm1Gamma;
    private readonly Tensor<T> _layerNorm1Beta;
    private readonly Tensor<T> _layerNorm2Gamma;
    private readonly Tensor<T> _layerNorm2Beta;
    private readonly Tensor<T> _layerNorm3Gamma;
    private readonly Tensor<T> _layerNorm3Beta;
    internal Tensor<T> GetSelfQueryProjection() => _selfQueryProj;
    internal Tensor<T> GetSelfKeyProjection() => _selfKeyProj;
    internal Tensor<T> GetSelfValueProjection() => _selfValueProj;
    internal Tensor<T> GetSelfOutputProjection() => _selfOutputProj;
    internal Tensor<T> GetCrossQueryProjection() => _crossQueryProj;
    internal Tensor<T> GetCrossKeyProjection() => _crossKeyProj;
    internal Tensor<T> GetCrossValueProjection() => _crossValueProj;
    internal Tensor<T> GetCrossOutputProjection() => _crossOutputProj;
    internal Tensor<T> GetFF1Weight() => _ffn1;
    internal Tensor<T> GetFF1Bias() => _ffn1Bias;
    internal Tensor<T> GetFF2Weight() => _ffn2;
    internal Tensor<T> GetFF2Bias() => _ffn2Bias;
    internal Tensor<T> GetLayerNorm1Gamma() => _layerNorm1Gamma;
    internal Tensor<T> GetLayerNorm1Beta() => _layerNorm1Beta;
    internal Tensor<T> GetLayerNorm2Gamma() => _layerNorm2Gamma;
    internal Tensor<T> GetLayerNorm2Beta() => _layerNorm2Beta;
    internal Tensor<T> GetLayerNorm3Gamma() => _layerNorm3Gamma;
    internal Tensor<T> GetLayerNorm3Beta() => _layerNorm3Beta;

    public override long ParameterCount =>
        _selfQueryProj.Length + _selfKeyProj.Length + _selfValueProj.Length + _selfOutputProj.Length +
        _crossQueryProj.Length + _crossKeyProj.Length + _crossValueProj.Length + _crossOutputProj.Length +
        _ffn1.Length + _ffn1Bias.Length + _ffn2.Length + _ffn2Bias.Length +
        _layerNorm1Gamma.Length * 2 + _layerNorm2Gamma.Length * 2 + _layerNorm3Gamma.Length * 2;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { }
    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        foreach (var t in new Tensor<T>[] { _selfQueryProj, _selfKeyProj, _selfValueProj, _selfOutputProj,
            _crossQueryProj, _crossKeyProj, _crossValueProj, _crossOutputProj,
            _ffn1, _ffn1Bias, _ffn2, _ffn2Bias,
            _layerNorm1Gamma, _layerNorm1Beta, _layerNorm2Gamma, _layerNorm2Beta, _layerNorm3Gamma, _layerNorm3Beta })
            for (int i = 0; i < t.Length; i++) p.Add(t[i]);
        return new Vector<T>(p.ToArray());
    }
    public override Tensor<T> Forward(Tensor<T> input) => throw new NotSupportedException(
        "Informer runs its forward pass at the model level (InformerModel.ForwardBatch); the layer-level Forward is unused.");

    public InformerDecoderLayerTensor(int embeddingDim, int numHeads, int sparsityFactor, double dropoutRate, int seed = 42)
        : base(new int[][] { new[] { embeddingDim }, new[] { embeddingDim } }, new[] { embeddingDim })
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;

        var random = RandomHelper.CreateSeededRandom(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / ((double)embeddingDim * 4));

        // Self-attention
        _selfQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _selfKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _selfValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _selfOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);

        // Cross-attention
        _crossQueryProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _crossKeyProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _crossValueProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);
        _crossOutputProj = InitTensor(new[] { embeddingDim, embeddingDim }, attnStddev, random);

        // FFN
        int ffnDim = embeddingDim * 4;
        _ffn1 = InitTensor(new[] { ffnDim, embeddingDim }, ffnStddev, random);
        _ffn1Bias = new Tensor<T>(new[] { ffnDim });
        _ffn2 = InitTensor(new[] { embeddingDim, ffnDim }, ffnStddev, random);
        _ffn2Bias = new Tensor<T>(new[] { embeddingDim });

        // Layer norms
        _layerNorm1Gamma = InitTensorOnes(embeddingDim);
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = InitTensorOnes(embeddingDim);
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm3Gamma = InitTensorOnes(embeddingDim);
        _layerNorm3Beta = new Tensor<T>(new[] { embeddingDim });
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

    private Tensor<T> InitTensorOnes(int size)
    {
        var tensor = new Tensor<T>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            tensor[i] = NumOps.One;
        }
        return tensor;
    }

    /// <summary>
    /// Serializes the decoder layer weights to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="writeTensor">A delegate that writes a tensor to the binary writer.</param>
    /// <remarks>
    /// <para>
    /// Writes all decoder layer weights in the following order:
    /// <list type="number">
    /// <item>Self-attention projection weights (Q, K, V, O)</item>
    /// <item>Cross-attention projection weights (Q, K, V, O)</item>
    /// <item>Feed-forward network weights and biases</item>
    /// <item>Layer normalization parameters (gamma and beta for all 3 layer norms)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public void Serialize(BinaryWriter writer, Action<BinaryWriter, Tensor<T>> writeTensor)
    {
        // Self-attention
        writeTensor(writer, _selfQueryProj);
        writeTensor(writer, _selfKeyProj);
        writeTensor(writer, _selfValueProj);
        writeTensor(writer, _selfOutputProj);
        // Cross-attention
        writeTensor(writer, _crossQueryProj);
        writeTensor(writer, _crossKeyProj);
        writeTensor(writer, _crossValueProj);
        writeTensor(writer, _crossOutputProj);
        // FFN
        writeTensor(writer, _ffn1);
        writeTensor(writer, _ffn1Bias);
        writeTensor(writer, _ffn2);
        writeTensor(writer, _ffn2Bias);
        // Layer norms
        writeTensor(writer, _layerNorm1Gamma);
        writeTensor(writer, _layerNorm1Beta);
        writeTensor(writer, _layerNorm2Gamma);
        writeTensor(writer, _layerNorm2Beta);
        writeTensor(writer, _layerNorm3Gamma);
        writeTensor(writer, _layerNorm3Beta);
    }

    /// <summary>
    /// Deserializes the decoder layer weights from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <param name="readTensor">A delegate that reads a tensor from the binary reader.</param>
    /// <remarks>
    /// <para>
    /// Restores all decoder layer weights in the same order they were serialized:
    /// self-attention, cross-attention, FFN, and layer normalization parameters.
    /// The deserialized values are copied into the existing tensors to preserve the layer structure.
    /// </para>
    /// </remarks>
    public void Deserialize(BinaryReader reader, Func<BinaryReader, Tensor<T>> readTensor)
    {
        // Self-attention
        CopyTensorData(readTensor(reader), _selfQueryProj);
        CopyTensorData(readTensor(reader), _selfKeyProj);
        CopyTensorData(readTensor(reader), _selfValueProj);
        CopyTensorData(readTensor(reader), _selfOutputProj);
        // Cross-attention
        CopyTensorData(readTensor(reader), _crossQueryProj);
        CopyTensorData(readTensor(reader), _crossKeyProj);
        CopyTensorData(readTensor(reader), _crossValueProj);
        CopyTensorData(readTensor(reader), _crossOutputProj);
        // FFN
        CopyTensorData(readTensor(reader), _ffn1);
        CopyTensorData(readTensor(reader), _ffn1Bias);
        CopyTensorData(readTensor(reader), _ffn2);
        CopyTensorData(readTensor(reader), _ffn2Bias);
        // Layer norms
        CopyTensorData(readTensor(reader), _layerNorm1Gamma);
        CopyTensorData(readTensor(reader), _layerNorm1Beta);
        CopyTensorData(readTensor(reader), _layerNorm2Gamma);
        CopyTensorData(readTensor(reader), _layerNorm2Beta);
        CopyTensorData(readTensor(reader), _layerNorm3Gamma);
        CopyTensorData(readTensor(reader), _layerNorm3Beta);
    }

    /// <summary>
    /// Copies tensor data from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="source">The source tensor containing the data to copy.</param>
    /// <param name="dest">The destination tensor to copy data into.</param>
    /// <remarks>
    /// Copies element-by-element up to the minimum length of both tensors.
    /// This preserves the destination tensor's memory allocation while updating its values.
    /// </remarks>
    private void CopyTensorData(Tensor<T> source, Tensor<T> dest)
    {
        for (int i = 0; i < Math.Min(source.Length, dest.Length); i++)
        {
            dest[i] = source[i];
        }
    }

}

/// <summary>
/// Shared computation graph builders for transformer layer components.
/// Used by Informer, Autoformer, and Chronos transformer layers.
/// </summary>
internal static class TransformerGraphHelper<T>
{
    public static ComputationNode<T> LayerNormGraph(
        ComputationNode<T> input, Tensor<T> gamma, Tensor<T> beta, string prefix)
    {
        var gammaNode = TensorOperations<T>.Constant(gamma.Clone(), $"{prefix}_gamma");
        var betaNode = TensorOperations<T>.Constant(beta.Clone(), $"{prefix}_beta");
        var mean = TensorOperations<T>.Mean(input);
        var centered = TensorOperations<T>.Subtract(input, mean);
        var sq = TensorOperations<T>.ElementwiseMultiply(centered, centered);
        var variance = TensorOperations<T>.Mean(sq);
        var numOps = MathHelper.GetNumericOperations<T>();
        var eps = TensorOperations<T>.Constant(
            new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { numOps.FromDouble(1e-5) })), $"{prefix}_eps");
        var std = TensorOperations<T>.Sqrt(TensorOperations<T>.Add(variance, eps));
        var normalized = TensorOperations<T>.Divide(centered, std);
        return TensorOperations<T>.Add(
            TensorOperations<T>.ElementwiseMultiply(gammaNode, normalized), betaNode);
    }

    public static ComputationNode<T> FeedForwardGraph(
        ComputationNode<T> input,
        Tensor<T> w1, Tensor<T> b1, Tensor<T> w2, Tensor<T> b2,
        string prefix)
    {
        var wFFN1 = TensorOperations<T>.Constant(w1.Clone(), $"{prefix}_w1");
        var bFFN1 = TensorOperations<T>.Constant(b1.Clone(), $"{prefix}_b1");
        var wFFN2 = TensorOperations<T>.Constant(w2.Clone(), $"{prefix}_w2");
        var bFFN2 = TensorOperations<T>.Constant(b2.Clone(), $"{prefix}_b2");
        var hidden = TensorOperations<T>.Add(
            TensorOperations<T>.MatrixVectorMultiply(wFFN1, input), bFFN1);
        var activated = TensorOperations<T>.GELU(hidden);
        return TensorOperations<T>.Add(
            TensorOperations<T>.MatrixVectorMultiply(wFFN2, activated), bFFN2);
    }

    public static ComputationNode<T> SelfAttentionGraph(
        ComputationNode<T> input,
        Tensor<T> wQ, Tensor<T> wK, Tensor<T> wV, Tensor<T> wO,
        string prefix)
    {
        var wQNode = TensorOperations<T>.Constant(wQ.Clone(), $"{prefix}_wQ");
        var wKNode = TensorOperations<T>.Constant(wK.Clone(), $"{prefix}_wK");
        var wVNode = TensorOperations<T>.Constant(wV.Clone(), $"{prefix}_wV");
        var wONode = TensorOperations<T>.Constant(wO.Clone(), $"{prefix}_wO");
        TensorOperations<T>.MatrixVectorMultiply(wQNode, input);
        TensorOperations<T>.MatrixVectorMultiply(wKNode, input);
        var v = TensorOperations<T>.MatrixVectorMultiply(wVNode, input);
        return TensorOperations<T>.MatrixVectorMultiply(wONode, v);
    }
}
