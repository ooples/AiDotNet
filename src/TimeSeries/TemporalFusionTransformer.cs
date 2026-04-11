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
/// Architecture: Input embedding → VSN → LSTM encoder-decoder → Static enrichment →
/// Interpretable multi-head attention → Gated skip connections → Quantile output.
/// </summary>
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

    // --- Paper components ---
    // Input embedding: projects each input variable to hiddenSize
    private Tensor<T> _inputEmbeddingWeight; // [hiddenSize, inputVarSize]
    private Tensor<T> _inputEmbeddingBias;   // [hiddenSize]

    // Variable Selection Network for observed past inputs
    private GatedResidualNetwork<T> _pastVsn;

    // LSTM encoder (processes observed past inputs)
    private Tensor<T> _lstmWi, _lstmWf, _lstmWc, _lstmWo; // input gate weights [hiddenSize, hiddenSize]
    private Tensor<T> _lstmUi, _lstmUf, _lstmUc, _lstmUo; // recurrent weights [hiddenSize, hiddenSize]
    private Tensor<T> _lstmBi, _lstmBf, _lstmBc, _lstmBo; // biases [hiddenSize]

    // Static enrichment GRN
    private GatedResidualNetwork<T> _enrichmentGrn;

    // Multi-head attention: Q, K, V projections + output projection
    private Tensor<T> _queryWeight, _keyWeight, _valueWeight, _outputWeight;

    // Post-attention GRN (gated skip connection)
    private GatedResidualNetwork<T> _postAttentionGrn;

    // Quantile output projection
    private Tensor<T> _quantileWeight; // [numQuantiles * forecastHorizon, hiddenSize]
    private Tensor<T> _quantileBias;   // [numQuantiles * forecastHorizon]

    // Training state
    private Vector<T> _trainingSeries = Vector<T>.Empty();
    private T _normMean;
    private T _normStd;

    // User-configurable optimizer (default: Adam per paper)
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
        {
            if (q <= 0.0 || q >= 1.0)
                throw new ArgumentException($"Quantile level {q} must be in (0, 1).");
        }
    }

    private void InitializeComponents()
    {
        int inputVarSize = Math.Max(_options.TimeVaryingUnknownSize, 1);

        // Input embedding: project each timestep's input to hiddenSize
        _inputEmbeddingWeight = CreateRandomTensor([_hiddenSize, inputVarSize], Math.Sqrt(2.0 / inputVarSize));
        _inputEmbeddingBias = new Tensor<T>([_hiddenSize]);

        // Variable selection for observed inputs (per the paper, uses GRN-based gating)
        // Simplified: single GRN that selects from lookback-embedded features
        _pastVsn = new GatedResidualNetwork<T>(_hiddenSize, _hiddenSize, _hiddenSize, seed: _random.Next());

        // LSTM encoder weights (single layer per paper default)
        InitializeLstmWeights();

        // Static enrichment GRN
        _enrichmentGrn = new GatedResidualNetwork<T>(_hiddenSize, _hiddenSize, _hiddenSize, seed: _random.Next());

        // Multi-head attention weights
        int headDim = _hiddenSize / _options.NumAttentionHeads;
        double stdAttn = Math.Sqrt(1.0 / _hiddenSize);
        _queryWeight = CreateRandomTensor([_hiddenSize, _hiddenSize], stdAttn);
        _keyWeight = CreateRandomTensor([_hiddenSize, _hiddenSize], stdAttn);
        // Paper: shared V weights across heads for interpretability
        _valueWeight = CreateRandomTensor([headDim, _hiddenSize], stdAttn);
        _outputWeight = CreateRandomTensor([_hiddenSize, _hiddenSize], stdAttn);

        // Post-attention gated skip connection
        _postAttentionGrn = new GatedResidualNetwork<T>(_hiddenSize, _hiddenSize, _hiddenSize, seed: _random.Next());

        // Quantile output projection
        int numQuantiles = _options.QuantileLevels.Length;
        int outputSize = numQuantiles * _options.ForecastHorizon;
        _quantileWeight = CreateRandomTensor([outputSize, _hiddenSize], Math.Sqrt(2.0 / (_hiddenSize + outputSize)));
        _quantileBias = new Tensor<T>([outputSize]);
    }

    private void InitializeLstmWeights()
    {
        double std = Math.Sqrt(2.0 / _hiddenSize);
        _lstmWi = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmWf = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmWc = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmWo = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmUi = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmUf = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmUc = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmUo = CreateRandomTensor([_hiddenSize, _hiddenSize], std);
        _lstmBi = new Tensor<T>([_hiddenSize]);
        _lstmBf = CreateConstantTensor([_hiddenSize], 1.0); // Forget gate bias = 1 (paper convention)
        _lstmBc = new Tensor<T>([_hiddenSize]);
        _lstmBo = new Tensor<T>([_hiddenSize]);
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];

        // Normalize
        ComputeNormStats(y);
        var yNorm = NormalizeVector(y);

        int lookback = _options.LookbackWindow;

        // Per Lim et al. 2021: train ALL parameters through backpropagation
        var allParams = CollectAllTrainableParameters();
        var optimizer = _optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate });
        var mseLoss = new MeanSquaredErrorLoss<T>();

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();

            for (int i = lookback; i < y.Length; i++)
            {
                if (i % 8 == 0) TrainingCancellationToken.ThrowIfCancellationRequested();

                var window = ExtractWindow(x, yNorm, i, lookback);
                T target = yNorm[i];
                var targetTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { target }));

                // Tape-tracked forward + backward + Adam step
                using var tape = new GradientTape<T>();

                var quantilePred = ForwardQuantiles(window);

                // Extract median prediction
                int medianIdx = Array.IndexOf(_options.QuantileLevels, 0.5);
                if (medianIdx < 0) medianIdx = _options.QuantileLevels.Length / 2;
                int predIdx = medianIdx * _options.ForecastHorizon;
                var sliceWeights = new T[quantilePred.Length];
                if (predIdx < sliceWeights.Length) sliceWeights[predIdx] = NumOps.One;
                var sliceTensor = new Tensor<T>(new[] { 1, quantilePred.Length }, new Vector<T>(sliceWeights));
                var predCol = Engine.Reshape(quantilePred, [quantilePred.Length, 1]);
                var prediction = Engine.Reshape(Engine.TensorMatMul(sliceTensor, predCol), [1]);

                var lossTensor = mseLoss.ComputeTapeLoss(prediction, targetTensor);

                var allGrads = tape.ComputeGradients(lossTensor, sources: null);
                var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                foreach (var param in allParams)
                {
                    if (allGrads.TryGetValue(param, out var grad))
                        grads[param] = grad;
                }

                T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

                Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> tgt) => prediction;
                Tensor<T> ComputeLoss(Tensor<T> pred, Tensor<T> tgt) =>
                    mseLoss.ComputeTapeLoss(pred, tgt);

                var context = new Tensors.Engines.Autodiff.TapeStepContext<T>(
                    allParams, grads, lossValue,
                    window, targetTensor, ComputeForward, ComputeLoss, null);

                optimizer.Step(context);
            }
        }
    }

    /// <summary>
    /// Collects all trainable parameter tensors from LSTM, GRN, attention, and output layers.
    /// </summary>
    private IReadOnlyList<Tensor<T>> CollectAllTrainableParameters()
    {
        var parameters = new List<Tensor<T>>();

        // Input embedding
        parameters.Add(_inputEmbeddingWeight);
        parameters.Add(_inputEmbeddingBias);

        // LSTM weights (4 gates × input + recurrent + bias = 12 tensors)
        parameters.AddRange(new[] { _lstmWi, _lstmWf, _lstmWc, _lstmWo });
        parameters.AddRange(new[] { _lstmUi, _lstmUf, _lstmUc, _lstmUo });
        parameters.AddRange(new[] { _lstmBi, _lstmBf, _lstmBc, _lstmBo });

        // Attention weights
        parameters.AddRange(new[] { _queryWeight, _keyWeight, _valueWeight, _outputWeight });

        // Quantile output
        parameters.Add(_quantileWeight);
        parameters.Add(_quantileBias);

        // GRN parameters
        foreach (var p in _pastVsn.GetTrainableParameters()) parameters.Add(p);
        foreach (var p in _enrichmentGrn.GetTrainableParameters()) parameters.Add(p);
        foreach (var p in _postAttentionGrn.GetTrainableParameters()) parameters.Add(p);

        return parameters;
    }

    /// <summary>
    /// Forward pass through the full TFT architecture per Lim et al. (2021):
    /// Input embedding → VSN → LSTM → Static enrichment → Attention → Output
    /// </summary>
    private Tensor<T> ForwardQuantiles(Tensor<T> lookbackWindow)
    {
        // 1. Input embedding: project lookback to sequence of hidden states
        int seqLen = Math.Max(1, lookbackWindow.Length / _hiddenSize);
        if (seqLen * _hiddenSize > lookbackWindow.Length)
            seqLen = 1;

        var embedded = EmbedInput(lookbackWindow);

        // 2. Variable Selection (GRN-based gating on embedded features)
        var selected = _pastVsn.Forward(embedded);

        // 3. LSTM encoder: process temporal sequence
        var lstmOutput = LstmForward(selected, seqLen);

        // 4. Static enrichment via GRN
        var enriched = _enrichmentGrn.Forward(lstmOutput);

        // 5. Interpretable multi-head attention (paper: shared V weights)
        var attended = InterpretableMultiHeadAttention(enriched, seqLen);

        // 6. Post-attention gated skip connection
        var gated = _postAttentionGrn.Forward(attended);

        // 7. Quantile output projection
        return QuantileProject(gated);
    }

    private Tensor<T> EmbedInput(Tensor<T> input)
    {
        int inSize = _inputEmbeddingWeight.Shape[1];
        int inputLen = Math.Min(input.Length, inSize);

        // Project input to hidden size
        var inputPadded = new Tensor<T>([inSize]);
        var padSpan = inputPadded.AsWritableSpan();
        var inSpan = input.Data.Span;
        for (int i = 0; i < inputLen; i++)
            padSpan[i] = inSpan[i];

        var inputCol = Engine.Reshape(inputPadded, [inSize, 1]);
        var matResult = Engine.TensorMatMul(_inputEmbeddingWeight, inputCol);
        var result = Engine.Reshape(matResult, [_hiddenSize]);
        return Engine.TensorAdd(result, _inputEmbeddingBias);
    }

    private Tensor<T> LstmForward(Tensor<T> input, int seqLen)
    {
        // Initialize hidden and cell states to zero
        var h = new Tensor<T>([_hiddenSize]);
        var c = new Tensor<T>([_hiddenSize]);
        var hSpan = h.AsWritableSpan();
        var cSpan = c.AsWritableSpan();

        // Process each timestep
        int stepSize = Math.Max(1, input.Length / seqLen);
        for (int t = 0; t < seqLen; t++)
        {
            // Extract timestep input
            var xt = new Tensor<T>([_hiddenSize]);
            var xtSpan = xt.AsWritableSpan();
            int offset = t * stepSize;
            var inSpan = input.Data.Span;
            for (int i = 0; i < _hiddenSize && (offset + i) < input.Length; i++)
                xtSpan[i] = inSpan[offset + i];

            // LSTM cell: i = σ(Wx·x + Uh·h + b)
            var xtCol = Engine.Reshape(xt, [_hiddenSize, 1]);
            var hCol = Engine.Reshape(h, [_hiddenSize, 1]);

            var ig = ApplySigmoid(AddAll(
                Engine.Reshape(Engine.TensorMatMul(_lstmWi, xtCol), [_hiddenSize]),
                Engine.Reshape(Engine.TensorMatMul(_lstmUi, hCol), [_hiddenSize]),
                _lstmBi));

            var fg = ApplySigmoid(AddAll(
                Engine.Reshape(Engine.TensorMatMul(_lstmWf, xtCol), [_hiddenSize]),
                Engine.Reshape(Engine.TensorMatMul(_lstmUf, hCol), [_hiddenSize]),
                _lstmBf));

            var cCandidate = ApplyTanh(AddAll(
                Engine.Reshape(Engine.TensorMatMul(_lstmWc, xtCol), [_hiddenSize]),
                Engine.Reshape(Engine.TensorMatMul(_lstmUc, hCol), [_hiddenSize]),
                _lstmBc));

            var og = ApplySigmoid(AddAll(
                Engine.Reshape(Engine.TensorMatMul(_lstmWo, xtCol), [_hiddenSize]),
                Engine.Reshape(Engine.TensorMatMul(_lstmUo, hCol), [_hiddenSize]),
                _lstmBo));

            // c = f ⊙ c + i ⊙ c_candidate
            c = Engine.TensorAdd(
                Engine.TensorMultiply(fg, c),
                Engine.TensorMultiply(ig, cCandidate));

            // h = o ⊙ tanh(c)
            h = Engine.TensorMultiply(og, ApplyTanh(c));
        }

        return h; // Return final hidden state
    }

    private Tensor<T> InterpretableMultiHeadAttention(Tensor<T> input, int seqLen)
    {
        int numHeads = _options.NumAttentionHeads;
        int headDim = _hiddenSize / numHeads;

        // Q, K projections: [hiddenSize, hiddenSize] @ input
        var inputCol = Engine.Reshape(input, [_hiddenSize, 1]);
        var q = Engine.Reshape(Engine.TensorMatMul(_queryWeight, inputCol), [_hiddenSize]);
        var k = Engine.Reshape(Engine.TensorMatMul(_keyWeight, inputCol), [_hiddenSize]);

        // V projection: shared across heads per paper (interpretable attention)
        // V weight is [headDim, hiddenSize]
        var v = Engine.Reshape(Engine.TensorMatMul(_valueWeight, inputCol), [headDim]);

        // Per-head attention: each head attends with its own Q/K slice but shares V
        var attended = new Tensor<T>([_hiddenSize]);
        var attSpan = attended.AsWritableSpan();
        var qSpan = q.Data.Span;
        var kSpan = k.Data.Span;
        var vSpan = v.Data.Span;

        for (int head = 0; head < numHeads; head++)
        {
            int hOffset = head * headDim;

            // Compute attention score: q_h · k_h / sqrt(d_k)
            T score = NumOps.Zero;
            for (int d = 0; d < headDim; d++)
            {
                score = NumOps.Add(score, NumOps.Multiply(qSpan[hOffset + d], kSpan[hOffset + d]));
            }
            T scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));
            T weight = NumOps.Multiply(score, scale);
            // For single-step: softmax over 1 position = 1.0
            // Apply weight to shared V
            for (int d = 0; d < headDim; d++)
            {
                attSpan[hOffset + d] = NumOps.Multiply(weight, vSpan[d]);
            }
        }

        // Output projection
        var attCol = Engine.Reshape(attended, [_hiddenSize, 1]);
        return Engine.Reshape(Engine.TensorMatMul(_outputWeight, attCol), [_hiddenSize]);
    }

    private Tensor<T> QuantileProject(Tensor<T> hidden)
    {
        int outSize = _quantileWeight.Shape[0];
        var hiddenCol = Engine.Reshape(hidden, [_hiddenSize, 1]);
        var result = Engine.Reshape(Engine.TensorMatMul(_quantileWeight, hiddenCol), [outSize]);
        return Engine.TensorAdd(result, _quantileBias);
    }

    private void UpdateQuantileWeights(Tensor<T> input, T error, double lr)
    {
        // SGD update on quantile projection (main trainable output layer)
        var lrT = NumOps.FromDouble(lr);
        var gradScale = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(error, lrT));

        var wSpan = _quantileWeight.AsWritableSpan();
        var bSpan = _quantileBias.AsWritableSpan();

        // Update bias
        for (int i = 0; i < bSpan.Length; i++)
        {
            bSpan[i] = NumOps.Subtract(bSpan[i], gradScale);
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        int lookback = _options.LookbackWindow;
        var predictions = new Vector<T>(n);

        var series = new T[_trainingSeries.Length + n];
        for (int i = 0; i < _trainingSeries.Length; i++)
            series[i] = _trainingSeries[i];
        T lastKnown = _trainingSeries.Length > 0 ? _trainingSeries[_trainingSeries.Length - 1] : NumOps.Zero;
        for (int i = 0; i < n; i++)
            series[_trainingSeries.Length + i] = lastKnown;

        for (int i = 0; i < n; i++)
        {
            int seriesPos = i;
            var lookbackWindow = new Vector<T>(lookback);
            for (int j = 0; j < lookback; j++)
            {
                int idx = seriesPos - lookback + 1 + j;
                lookbackWindow[j] = (idx >= 0 && idx < series.Length) ? series[idx] : NumOps.Zero;
            }
            predictions[i] = PredictSingle(lookbackWindow);
        }
        return predictions;
    }

    public override T PredictSingle(Vector<T> input)
    {
        int expectedLen = _options.LookbackWindow * Math.Max(_options.TimeVaryingUnknownSize, 1);
        var inputTensor = new Tensor<T>([expectedLen]);
        var tSpan = inputTensor.AsWritableSpan();
        for (int i = 0; i < Math.Min(input.Length, expectedLen); i++)
            tSpan[i] = NormalizeValue(input[i]);

        var quantilePredictions = ForwardQuantiles(inputTensor);

        int medianIdx = Array.IndexOf(_options.QuantileLevels, 0.5);
        if (medianIdx < 0) medianIdx = _options.QuantileLevels.Length / 2;
        int predIdx = medianIdx * _options.ForecastHorizon;
        T normalizedPred = predIdx < quantilePredictions.Length ? quantilePredictions[predIdx] : NumOps.Zero;

        // Denormalize
        return NumOps.Add(NumOps.Multiply(normalizedPred, _normStd), _normMean);
    }

    public Vector<T> PredictQuantiles(Vector<T> input)
    {
        var inputTensor = new Tensor<T>([input.Length]);
        var tSpan = inputTensor.AsWritableSpan();
        for (int i = 0; i < input.Length; i++)
            tSpan[i] = NormalizeValue(input[i]);

        var quantilePredictions = ForwardQuantiles(inputTensor);
        var result = new Vector<T>(quantilePredictions.Length);
        var qSpan = quantilePredictions.Data.Span;
        for (int i = 0; i < quantilePredictions.Length; i++)
            result[i] = NumOps.Add(NumOps.Multiply(qSpan[i], _normStd), _normMean);

        return result;
    }

    // --- Normalization helpers ---

    private void ComputeNormStats(Vector<T> y)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
            sum = NumOps.Add(sum, y[i]);
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

    private Vector<T> NormalizeVector(Vector<T> y)
    {
        var result = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            result[i] = NumOps.Divide(NumOps.Subtract(y[i], _normMean), _normStd);
        return result;
    }

    private T NormalizeValue(T val)
    {
        return NumOps.Divide(NumOps.Subtract(val, _normMean), _normStd);
    }

    private Tensor<T> ExtractWindow(Matrix<T> x, Vector<T> yNorm, int idx, int lookback)
    {
        int varSize = Math.Max(_options.TimeVaryingUnknownSize, 1);
        int windowLen = lookback * varSize;
        var window = new Tensor<T>([windowLen]);
        var wSpan = window.AsWritableSpan();

        if (x.Columns >= lookback)
        {
            for (int j = 0; j < lookback && j < x.Columns; j++)
                wSpan[j] = x[idx, j];
        }
        else
        {
            for (int j = 0; j < lookback; j++)
            {
                int yIdx = idx - lookback + j;
                wSpan[j] = (yIdx >= 0 && yIdx < yNorm.Length) ? yNorm[yIdx] : NumOps.Zero;
            }
        }
        return window;
    }

    // --- Tensor utility methods ---

    private Tensor<T> AddAll(Tensor<T> a, Tensor<T> b, Tensor<T> c)
    {
        return Engine.TensorAdd(Engine.TensorAdd(a, b), c);
    }

    private Tensor<T> ApplySigmoid(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        var xSpan = x.Data.Span;
        var rSpan = result.AsWritableSpan();
        for (int i = 0; i < xSpan.Length; i++)
        {
            double val = NumOps.ToDouble(xSpan[i]);
            double clamped = val < -20 ? -20 : (val > 20 ? 20 : val);
            rSpan[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-clamped)));
        }
        return result;
    }

    private Tensor<T> ApplyTanh(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        var xSpan = x.Data.Span;
        var rSpan = result.AsWritableSpan();
        for (int i = 0; i < xSpan.Length; i++)
        {
            rSpan[i] = NumOps.FromDouble(Math.Tanh(NumOps.ToDouble(xSpan[i])));
        }
        return result;
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

    private Tensor<T> CreateConstantTensor(int[] shape, double value)
    {
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        for (int i = 0; i < size; i++)
            data[i] = NumOps.FromDouble(value);
        return new Tensor<T>(shape, new Vector<T>(data));
    }

    // --- Serialization ---

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.HiddenSize);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.NumLayers);
        writer.Write(_options.QuantileLevels.Length);
        foreach (var q in _options.QuantileLevels)
            writer.Write(q);

        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++)
            writer.Write(NumOps.ToDouble(_trainingSeries[i]));

        writer.Write(NumOps.ToDouble(_normMean));
        writer.Write(NumOps.ToDouble(_normStd));

        SerializeTensor(writer, _quantileWeight);
        SerializeTensor(writer, _quantileBias);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.LookbackWindow = reader.ReadInt32();
        _options.ForecastHorizon = reader.ReadInt32();
        _options.HiddenSize = reader.ReadInt32();
        _options.NumAttentionHeads = reader.ReadInt32();
        _options.NumLayers = reader.ReadInt32();

        int numQuantiles = reader.ReadInt32();
        _options.QuantileLevels = new double[numQuantiles];
        for (int i = 0; i < numQuantiles; i++)
            _options.QuantileLevels[i] = reader.ReadDouble();

        int seriesLen = reader.ReadInt32();
        _trainingSeries = new Vector<T>(seriesLen);
        for (int i = 0; i < seriesLen; i++)
            _trainingSeries[i] = NumOps.FromDouble(reader.ReadDouble());

        _normMean = NumOps.FromDouble(reader.ReadDouble());
        _normStd = NumOps.FromDouble(reader.ReadDouble());

        _quantileWeight = DeserializeTensor(reader);
        _quantileBias = DeserializeTensor(reader);
    }

    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        var shapeArr = tensor._shape;
        writer.Write(shapeArr.Length);
        foreach (var dim in shapeArr)
            writer.Write(dim);
        var span = tensor.Data.Span;
        for (int i = 0; i < span.Length; i++)
            writer.Write(NumOps.ToDouble(span[i]));
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = reader.ReadInt32();
        int size = 1;
        foreach (var s in shape) size *= s;
        var data = new T[size];
        for (int i = 0; i < size; i++)
            data[i] = NumOps.FromDouble(reader.ReadDouble());
        return new Tensor<T>(shape, new Vector<T>(data));
    }

    // --- Metadata ---

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "TemporalFusionTransformer",
            Description = "TFT per Lim et al. (2021) with LSTM encoder, GRN gating, and interpretable attention",
            Complexity = ParameterCount,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["HiddenSize"] = _options.HiddenSize,
                ["NumAttentionHeads"] = _options.NumAttentionHeads,
                ["NumLayers"] = _options.NumLayers,
                ["LookbackWindow"] = _options.LookbackWindow,
                ["ForecastHorizon"] = _options.ForecastHorizon,
                ["QuantileLevels"] = string.Join(",", _options.QuantileLevels),
                ["Architecture"] = "LSTM + GRN + Interpretable Multi-Head Attention"
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new TemporalFusionTransformer<T>(new TemporalFusionTransformerOptions<T>(_options));
    }

    public override int ParameterCount
    {
        get
        {
            int count = _inputEmbeddingWeight.Length + _inputEmbeddingBias.Length;
            count += _queryWeight.Length + _keyWeight.Length + _valueWeight.Length + _outputWeight.Length;
            count += _quantileWeight.Length + _quantileBias.Length;
            // LSTM: 4 gates × (W + U + b) = 4 × (h² + h² + h) = 4(2h² + h)
            count += 4 * (2 * _hiddenSize * _hiddenSize + _hiddenSize);
            return count;
        }
    }
}
