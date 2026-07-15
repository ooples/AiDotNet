#pragma warning disable CS8618 // Fields initialized in InitializeComponents called from constructor
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TimeSeries.TFT;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Temporal Fusion Transformer (TFT) per Lim et al. (2021).
/// Architecture retained here: input embedding → variable selection (softmax-weighted
/// feature combination) → static-enrichment GRN → interpretable multi-head attention →
/// post-attention gated skip (GRN) → forecast head.
/// </summary>
/// <remarks>
/// <para>
/// <b>Tape-based training.</b> The entire forward pass (<see cref="ForwardBatch"/>) is
/// expressed with batched <c>Engine.Tensor*</c> ops, so a
/// <see cref="Tensors.Engines.Autodiff.GradientTape{T}"/> produces the whole backward pass
/// automatically (no hand-derived gradients) and every op is GPU-dispatchable. Windows of
/// the z-normalized series are stacked into a <c>[B, L]</c> batch and the full H-step horizon
/// is supervised per sample with the pinball / quantile loss summed over every configured
/// <see cref="TemporalFusionTransformerOptions{T}.QuantileLevels"/> level, so the model learns a
/// genuine predictive spread (see <see cref="PredictQuantiles"/>); the point forecast is the
/// median-level head.
/// </para>
/// <para>
/// <b>Simplifications vs the full paper</b> (documented per the tape-conversion campaign):
/// (1) The sequential LSTM encoder-decoder is replaced by additive sinusoidal positional encodings feeding the
/// self-attention block (a recurrent scan differentiates poorly on the tape and does not
/// batch; positional encodings + attention recover temporal order). (2) For univariate series
/// (the only exogenous inputs available here) variable selection reduces to a learned
/// per-channel softmax gate over the embedded features rather than selection across multiple
/// exogenous variables. The GRN, variable-selection gating, and interpretable multi-head
/// attention — TFT's defining components — are all retained and fully differentiable.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", "https://arxiv.org/abs/1912.09363", Year = 2021, Authors = "Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister")]
public class TemporalFusionTransformer<T> : TimeSeriesModelBase<T>
{
    private readonly TemporalFusionTransformerOptions<T> _options;
    private readonly Random _random;
    private readonly int _hiddenSize;

    // Input embedding: scalar value at each timestep -> hiddenSize vector.
    private Tensor<T> _inputEmbeddingWeight; // [1, hiddenSize]
    private Tensor<T> _inputEmbeddingBias;   // [hiddenSize]

    // Sinusoidal positional encoding [maxLen, hiddenSize] (replaces the LSTM scan).
    private Tensor<T> _positionalEncoding;

    // Variable selection (softmax-weighted feature combination).
    private GatedResidualNetwork<T> _vsnFeatureGrn; // processes embedded features
    private GatedResidualNetwork<T> _vsnSelectGrn;  // produces selection logits

    // Static-enrichment GRN.
    private GatedResidualNetwork<T> _enrichmentGrn;

    // Interpretable multi-head attention: Q/K/V + output projection.
    private Tensor<T> _queryWeight, _keyWeight, _valueWeight, _attnOutputWeight; // each [hiddenSize, hiddenSize]

    // Post-attention gated skip connection.
    private GatedResidualNetwork<T> _postAttentionGrn;

    // Quantile forecast head: pooled hidden -> H-step forecast for each quantile level (quantile-major).
    private Tensor<T> _forecastWeight; // [hiddenSize, forecastHorizon * numQuantiles]
    private Tensor<T> _forecastBias;   // [forecastHorizon * numQuantiles]

    // Training state.
    private Vector<T> _trainingSeries = Vector<T>.Empty();
    private T _normMean;
    private T _normStd;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    public TemporalFusionTransformer(
        TemporalFusionTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(options ?? new TemporalFusionTransformerOptions<T>())
    {
        _options = options ?? new TemporalFusionTransformerOptions<T>();
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(42);
        _hiddenSize = _options.HiddenSize;
        _normMean = NumOps.Zero;
        _normStd = NumOps.One;
        _optimizer = optimizer;

        ValidateOptions();
        InitializeComponents();
    }

    private void ValidateOptions()
    {
        if (_options.LookbackWindow <= 0)
            throw new ArgumentException("LookbackWindow must be positive.", nameof(_options.LookbackWindow));
        if (_options.ForecastHorizon <= 0)
            throw new ArgumentException("ForecastHorizon must be positive.", nameof(_options.ForecastHorizon));
        if (_options.HiddenSize <= 0)
            throw new ArgumentException("HiddenSize must be positive.", nameof(_options.HiddenSize));
        if (_options.NumAttentionHeads <= 0)
            throw new ArgumentException("NumAttentionHeads must be positive.", nameof(_options.NumAttentionHeads));
        if (_options.HiddenSize % _options.NumAttentionHeads != 0)
            throw new ArgumentException(
                $"HiddenSize ({_options.HiddenSize}) must be divisible by NumAttentionHeads ({_options.NumAttentionHeads}).");
        if (_options.QuantileLevels == null || _options.QuantileLevels.Length == 0)
            throw new ArgumentException("QuantileLevels must contain at least one value.");
        foreach (var q in _options.QuantileLevels)
            if (q <= 0.0 || q >= 1.0)
                throw new ArgumentException(
                    $"Each quantile level must be strictly between 0 and 1; got {q}.", nameof(_options.QuantileLevels));
    }

    private void InitializeComponents()
    {
        int d = _hiddenSize;

        // Input embedding (scalar -> d).
        _inputEmbeddingWeight = CreateRandomTensor([1, d], Math.Sqrt(2.0));
        _inputEmbeddingBias = new Tensor<T>([d]);

        // Positional encoding for the full lookback window.
        _positionalEncoding = CreateSinusoidalPositionalEncoding(Math.Max(1, _options.LookbackWindow), d);

        // Variable selection GRNs.
        _vsnFeatureGrn = new GatedResidualNetwork<T>(d, d, d, seed: _random.Next());
        _vsnSelectGrn = new GatedResidualNetwork<T>(d, d, d, seed: _random.Next());

        // Static enrichment.
        _enrichmentGrn = new GatedResidualNetwork<T>(d, d, d, seed: _random.Next());

        // Attention weights.
        double stdAttn = Math.Sqrt(1.0 / d);
        _queryWeight = CreateRandomTensor([d, d], stdAttn);
        _keyWeight = CreateRandomTensor([d, d], stdAttn);
        _valueWeight = CreateRandomTensor([d, d], stdAttn);
        _attnOutputWeight = CreateRandomTensor([d, d], stdAttn);

        // Post-attention gated skip.
        _postAttentionGrn = new GatedResidualNetwork<T>(d, d, d, seed: _random.Next());

        // Quantile forecast head: projects the pooled features to H forecasts for EACH quantile level,
        // laid out quantile-major as [q0_h0..q0_h(H-1), q1_h0.., ...] (index = qi*H + h). Trained with the
        // pinball/quantile loss so each level learns a genuine predictive interval (TFT's defining feature).
        int horizon = _options.ForecastHorizon;
        int numQuantiles = _options.QuantileLevels.Length;
        int outDim = horizon * numQuantiles;
        _forecastWeight = CreateRandomTensor([d, outDim], Math.Sqrt(2.0 / (d + outDim)));
        _forecastBias = new Tensor<T>([outDim]);
    }

    // ── Training (tape + Adam) ──────────────────────────────────────────────────────
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++) _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = NumOps.FromDouble(y.Length);

        int lookback = _options.LookbackWindow;
        int horizon = _options.ForecastHorizon;

        // z-normalize for stable gradient flow (mirrors NBEATS / NHiTS / Informer).
        ComputeNormStats(y);
        var yNorm = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            yNorm[i] = NumOps.Divide(NumOps.Subtract(y[i], _normMean), _normStd);

        var validIndices = new List<int>();
        for (int i = lookback; i + horizon <= y.Length; i++) validIndices.Add(i);
        if (validIndices.Count == 0)
        {
            throw new ArgumentException(
                $"Not enough data to build a single training sample. Require at least " +
                $"{lookback + horizon} points, got {y.Length}.", nameof(y));
        }

        var optimizer = _optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
        // Pinball / quantile loss, one per configured level. Summing them over the quantile-major forecast
        // trains a genuine spread (lower levels below, upper above) instead of a single point head.
        int numQuantiles = _options.QuantileLevels.Length;
        var quantileLosses = new QuantileLoss<T>[numQuantiles];
        for (int qi = 0; qi < numQuantiles; qi++)
            quantileLosses[qi] = new QuantileLoss<T>(_options.QuantileLevels[qi]);

        // Sum the mean pinball loss of every quantile's [B, H] slice of the [B, H*Q] forecast (all on the tape).
        Tensor<T> QuantilePinballLoss(Tensor<T> pred, Tensor<T> tgt)
        {
            Tensor<T> total = null!;
            for (int qi = 0; qi < numQuantiles; qi++)
            {
                var sliceQi = Engine.TensorNarrow(pred, 1, qi * horizon, horizon); // [B, H]
                var ql = quantileLosses[qi].ComputeTapeLoss(sliceQi, tgt);
                total = qi == 0 ? ql : Engine.TensorAdd(total, ql);
            }
            return total;
        }
        var allParams = CollectAllTrainableParameters();

        int batchSize = Math.Max(1, _options.BatchSize);
        double bestLoss = double.PositiveInfinity;
        List<Tensor<T>>? bestSnapshot = null;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();

            var shuffled = validIndices.OrderBy(_ => _random.Next()).ToList();
            double epochLossSum = 0.0;
            int epochCount = 0;

            for (int start = 0; start < shuffled.Count; start += batchSize)
            {
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
                var batchInput = new Tensor<T>([b, lookback], new Vector<T>(inputData));
                var batchTarget = new Tensor<T>([b, horizon], new Vector<T>(targetData));

                using var tape = new GradientTape<T>();
                var forecast = ForwardBatch(batchInput, b, lookback); // [b, horizon*Q]
                var lossTensor = QuantilePinballLoss(forecast, batchTarget);
                var allGrads = tape.ComputeGradients(lossTensor, sources: null);

                var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                    TensorReferenceComparer<Tensor<T>>.Instance);
                foreach (var param in allParams)
                    if (allGrads.TryGetValue(param, out var g)) grads[param] = g;

                T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
                Tensor<T> ComputeForward(Tensor<T> a, Tensor<T> t) => forecast;
                Tensor<T> ComputeLoss(Tensor<T> p, Tensor<T> t) => QuantilePinballLoss(p, t);
                var context = new TapeStepContext<T>(
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

                // Report after checkpointing so an early stop still leaves the best weights to
                // restore below.
                if (!ReportEpoch(epoch, _options.Epochs, NumOps.FromDouble(epochLoss)))
                {
                    break;
                }
            }
        }

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

    /// <summary>Collects all trainable parameter tensors (embedding, VSN, enrichment, attention, post-GRN, head).</summary>
    private IReadOnlyList<Tensor<T>> CollectAllTrainableParameters()
    {
        var p = new List<Tensor<T>>
        {
            _inputEmbeddingWeight, _inputEmbeddingBias,
            _queryWeight, _keyWeight, _valueWeight, _attnOutputWeight,
            _forecastWeight, _forecastBias
        };
        foreach (var g in _vsnFeatureGrn.GetTrainableParameters()) p.Add(g);
        foreach (var g in _vsnSelectGrn.GetTrainableParameters()) p.Add(g);
        foreach (var g in _enrichmentGrn.GetTrainableParameters()) p.Add(g);
        foreach (var g in _postAttentionGrn.GetTrainableParameters()) p.Add(g);
        return p;
    }

    // ── Batched IEngine forward (automatic GradientTape) ────────────────────────────
    private int EffectiveHeads()
        => _hiddenSize % _options.NumAttentionHeads == 0 ? _options.NumAttentionHeads : 1;

    // [B*S, d] -> [B, H, S, hd]
    private Tensor<T> ToHeads(Tensor<T> x, int batch, int seq)
    {
        int d = _hiddenSize, nh = EffectiveHeads(), hd = d / nh;
        var split = Engine.Reshape(x, [batch, seq, nh, hd]);
        return Engine.TensorPermute(split, [0, 2, 1, 3]);
    }

    // [B, H, S, hd] -> [B*S, d]
    private Tensor<T> FromHeads(Tensor<T> x, int batch, int seq)
    {
        int d = _hiddenSize;
        var bshd = Engine.TensorPermute(x, [0, 2, 1, 3]);
        return Engine.Reshape(bshd, [batch * seq, d]);
    }

    // Row-wise affine map: x[N, in] · W[in, out] + bias[out] -> [N, out] (W stored [in, out]).
    private Tensor<T> MatMulBias(Tensor<T> x, Tensor<T> w, Tensor<T>? bias)
    {
        var r = Engine.TensorMatMul(x, w);
        if (bias != null) r = Engine.TensorBroadcastAdd(r, Engine.Reshape(bias, [1, bias.Shape[0]]));
        return r;
    }

    /// <summary>
    /// Batched forward: <c>batchInput [B, L]</c> (z-normalized) -> point forecast <c>[B, H]</c>.
    /// </summary>
    private Tensor<T> ForwardBatch(Tensor<T> batchInput, int batch, int seq)
    {
        int d = _hiddenSize;

        // 1. Input embedding + sinusoidal positional encoding -> [B*L, d].
        var flatIn = Engine.Reshape(batchInput, [batch * seq, 1]);
        var emb = Engine.TensorMatMul(flatIn, _inputEmbeddingWeight); // [B*L, d]
        emb = Engine.TensorBroadcastAdd(emb, Engine.Reshape(_inputEmbeddingBias, [1, d]));
        var posData = new Vector<T>(batch * seq * d);
        for (int bi = 0; bi < batch; bi++)
            for (int t = 0; t < seq; t++)
                for (int j = 0; j < d; j++)
                    posData[(bi * seq + t) * d + j] = _positionalEncoding[t * d + j];
        emb = Engine.TensorAdd(emb, new Tensor<T>([batch * seq, d], posData));

        // 2. Variable selection: softmax-gated combination of processed features.
        var selectionWeights = Engine.Softmax(_vsnSelectGrn.Forward(emb), 1); // [B*L, d]
        var processed = _vsnFeatureGrn.Forward(emb);                          // [B*L, d]
        var selected = Engine.TensorMultiply(processed, selectionWeights);

        // 3. Static enrichment.
        var enriched = _enrichmentGrn.Forward(selected); // [B*L, d]

        // 4. Interpretable multi-head self-attention.
        int hd = d / EffectiveHeads();
        double scale = 1.0 / Math.Sqrt(hd);
        var q = ToHeads(MatMulBias(enriched, _queryWeight, null), batch, seq);
        var k = ToHeads(MatMulBias(enriched, _keyWeight, null), batch, seq);
        var v = ToHeads(MatMulBias(enriched, _valueWeight, null), batch, seq);
        var attn4 = Engine.ScaledDotProductAttention<T>(q, k, v, mask: null, scale: scale, out _);
        var attn = FromHeads(attn4, batch, seq);
        var attnOut = MatMulBias(attn, _attnOutputWeight, null); // [B*L, d]

        // 5. Post-attention gated skip connection (GRN over attention + enriched residual).
        var gated = _postAttentionGrn.Forward(Engine.TensorAdd(enriched, attnOut)); // [B*L, d]

        // 6. Pool over the sequence and project to the H-step forecast.
        var gated3 = Engine.Reshape(gated, [batch, seq, d]);
        var pooled = Engine.ReduceMean(gated3, [1], keepDims: false); // [batch, d]
        return MatMulBias(pooled, _forecastWeight, _forecastBias);    // [batch, horizon*Q]
    }

    // Inference forward for a single lookback window using the SAME Engine ops as training.
    // Full quantile forecast for a single lookback window: [H*Q] denormalized (index = qi*H + h).
    private Tensor<T> ForwardEngineQuantiles(Vector<T> input)
    {
        int horizon = _options.ForecastHorizon;
        int numQuantiles = _options.QuantileLevels.Length;
        int outDim = horizon * numQuantiles;
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);
        if (seqLen <= 0) return new Tensor<T>([outDim]);
        var inData = new T[seqLen];
        for (int t = 0; t < seqLen; t++)
            inData[t] = NumOps.Divide(NumOps.Subtract(input[input.Length - seqLen + t], _normMean), _normStd);
        var batchInput = new Tensor<T>([1, seqLen], new Vector<T>(inData));
        var outBHQ = ForwardBatch(batchInput, 1, seqLen); // [1, H*Q]

        var result = new Tensor<T>([outDim]);
        for (int i = 0; i < outDim; i++)
            result[i] = NumOps.Add(NumOps.Multiply(outBHQ[i], _normStd), _normMean);
        return result;
    }

    // Point forecast = the median (closest-to-0.5) quantile's H-step forecast.
    private Tensor<T> ForwardEngine(Vector<T> input)
    {
        int horizon = _options.ForecastHorizon;
        var all = ForwardEngineQuantiles(input); // [H*Q]
        int medQi = MedianQuantileIndex();
        var result = new Tensor<T>([horizon]);
        for (int h = 0; h < horizon; h++)
            result[h] = all[medQi * horizon + h];
        return result;
    }

    // Index of the quantile level nearest 0.5 — used as the point forecast.
    private int MedianQuantileIndex()
    {
        var levels = _options.QuantileLevels;
        int best = 0;
        double bestDist = double.MaxValue;
        for (int qi = 0; qi < levels.Length; qi++)
        {
            double dist = Math.Abs(levels[qi] - 0.5);
            if (dist < bestDist) { bestDist = dist; best = qi; }
        }
        return best;
    }

    // ── Prediction ──────────────────────────────────────────────────────────────────
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryPredictFromTimeIndexCalibration(input, _trainingSeries, out var calibrated))
            return calibrated;

        int n = input.Rows;
        int lookback = _options.LookbackWindow;
        var predictions = new Vector<T>(n);

        // In-sample: forecast each position from its proper lookback window of the observed
        // series (a genuine one-step forecast, not a memorized target).
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
                    predictions[i] = fc.Length > 0 ? fc[0] : NumOps.Zero;
                    continue;
                }
            }
            predictions[i] = PredictSingle(input.GetRow(i));
        }
        return predictions;
    }

    public override T PredictSingle(Vector<T> input)
    {
        var output = ForwardEngine(input);
        return output.Length > 0 ? output[0] : NumOps.Zero;
    }

    /// <summary>
    /// Multi-horizon forecast for a single lookback window (H denormalized future values).
    /// </summary>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        var output = ForwardEngine(input);
        var result = new Vector<T>(output.Length);
        for (int i = 0; i < output.Length; i++) result[i] = output[i];
        return result;
    }

    /// <summary>
    /// Returns per-quantile forecasts as a flat vector laid out quantile-major:
    /// <c>[q0_h0..q0_h(H-1), q1_h0..q1_h(H-1), ...]</c> (index = <c>qi*ForecastHorizon + h</c>), in the same
    /// order as <see cref="TemporalFusionTransformerOptions{T}.QuantileLevels"/>. Each level is produced by a
    /// dedicated head column trained with the pinball loss, so the levels give a genuine predictive spread.
    /// </summary>
    public Vector<T> PredictQuantiles(Vector<T> input)
    {
        var all = ForwardEngineQuantiles(input); // [H*Q] denormalized, index = qi*H + h
        var result = new Vector<T>(all.Length);
        for (int i = 0; i < all.Length; i++)
            result[i] = all[i];
        return result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────
    private void ComputeNormStats(Vector<T> y)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < y.Length; i++) sum = NumOps.Add(sum, y[i]);
        _normMean = NumOps.Divide(sum, NumOps.FromDouble(y.Length));

        T sumSq = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T diff = NumOps.Subtract(y[i], _normMean);
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
        }
        T variance = NumOps.Divide(sumSq, NumOps.FromDouble(y.Length));
        _normStd = NumOps.FromDouble(Math.Max(Math.Sqrt(NumOps.ToDouble(variance)), 1e-8));
    }

    private Tensor<T> CreateRandomTensor(int[] shape, double stddev)
    {
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = NumOps.FromDouble(normal * stddev);
        }
        return new Tensor<T>(shape, new Vector<T>(data));
    }

    private Tensor<T> CreateSinusoidalPositionalEncoding(int maxLen, int embeddingDim)
    {
        var encoding = new Tensor<T>([maxLen, embeddingDim]);
        for (int pos = 0; pos < maxLen; pos++)
            for (int i = 0; i < embeddingDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2.0)) / embeddingDim);
                double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                encoding[pos * embeddingDim + i] = NumOps.FromDouble(value);
            }
        return encoding;
    }

    // ── Serialization ───────────────────────────────────────────────────────────────
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.HiddenSize);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.NumLayers);

        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++)
            writer.Write(NumOps.ToDouble(_trainingSeries[i]));

        writer.Write(NumOps.ToDouble(_normMean));
        writer.Write(NumOps.ToDouble(_normStd));

        foreach (var p in CollectAllTrainableParameters())
            SerializeTensor(writer, p);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.LookbackWindow = reader.ReadInt32();
        _options.ForecastHorizon = reader.ReadInt32();
        _options.HiddenSize = reader.ReadInt32();
        _options.NumAttentionHeads = reader.ReadInt32();
        _options.NumLayers = reader.ReadInt32();

        int seriesLen = reader.ReadInt32();
        _trainingSeries = new Vector<T>(seriesLen);
        for (int i = 0; i < seriesLen; i++)
            _trainingSeries[i] = NumOps.FromDouble(reader.ReadDouble());

        _normMean = NumOps.FromDouble(reader.ReadDouble());
        _normStd = NumOps.FromDouble(reader.ReadDouble());

        // Restore parameter values into the freshly-initialized tensors (shapes fixed by options).
        foreach (var p in CollectAllTrainableParameters())
        {
            var loaded = DeserializeTensor(reader);
            for (int k = 0; k < Math.Min(loaded.Length, p.Length); k++) p[k] = loaded[k];
        }
    }

    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor._shape.Length);
        foreach (var dim in tensor._shape) writer.Write(dim);
        var span = tensor.Data.Span;
        for (int i = 0; i < span.Length; i++) writer.Write(NumOps.ToDouble(span[i]));
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++) shape[i] = reader.ReadInt32();
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        for (int i = 0; i < size; i++) data[i] = NumOps.FromDouble(reader.ReadDouble());
        return new Tensor<T>(shape, new Vector<T>(data));
    }

    // ── Metadata ──────────────────────────────────────────────────────────────────
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "TemporalFusionTransformer",
            Description = "TFT (Lim et al. 2021) with GRN gating, variable selection, and interpretable attention; tape-trained point forecast.",
            Complexity = ParameterCount,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["HiddenSize"] = _options.HiddenSize,
                ["NumAttentionHeads"] = _options.NumAttentionHeads,
                ["NumLayers"] = _options.NumLayers,
                ["LookbackWindow"] = _options.LookbackWindow,
                ["ForecastHorizon"] = _options.ForecastHorizon,
                ["Architecture"] = "Embedding + VariableSelection(GRN) + Enrichment(GRN) + InterpretableMHA + PostGRN + ForecastHead"
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new TemporalFusionTransformer<T>(new TemporalFusionTransformerOptions<T>(_options));
    }

    public override long ParameterCount
    {
        get
        {
            long count = 0;
            foreach (var p in CollectAllTrainableParameters()) count += p.Length;
            return count;
        }
    }
}
