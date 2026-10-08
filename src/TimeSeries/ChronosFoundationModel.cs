using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Chronos foundation model for zero-shot time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>What is a Foundation Model?</b>
/// A foundation model is a large neural network pretrained on vast amounts of data that can be
/// applied to new tasks without task-specific training (zero-shot) or with minimal fine-tuning.
/// GPT-3/4 are foundation models for text; Chronos is a foundation model for time series.
/// </para>
/// <para>
/// <b>The Chronos Approach:</b>
/// Chronos (Ansari et al., 2024) treats time series forecasting as a language modeling task.
/// The key insight is that if we can tokenize continuous time series values into discrete
/// tokens, we can apply the same powerful transformer architectures that work so well for text.
/// </para>
/// <para>
/// <b>Mean-Scaling Tokenization:</b>
/// Before tokenization, values are normalized by the mean absolute value of the context:
/// x_normalized = x / (mean(|context|) + epsilon)
/// This makes the model scale-invariant - it can handle time series of any magnitude.
/// Normalized values are then mapped to discrete tokens using a fixed vocabulary of
/// uniformly-spaced bins covering a reasonable range (e.g., -15 to 15).
/// </para>
/// <para>
/// <b>Causal Transformer Architecture:</b>
/// Chronos uses a decoder-only transformer (like GPT) with causal masking. Each position
/// can only attend to itself and previous positions, enabling autoregressive generation.
/// The architecture includes:
/// - Token embeddings mapping discrete tokens to dense vectors
/// - Sinusoidal positional encoding for temporal awareness
/// - Multiple transformer layers with multi-head causal self-attention
/// - Layer normalization and feed-forward networks
/// - Output projection to vocabulary logits
/// </para>
/// <para>
/// <b>Zero-Shot Forecasting:</b>
/// Once pretrained on diverse time series data (synthetic and real), Chronos can forecast
/// new time series it has never seen. The model learns general patterns of temporal dynamics
/// that transfer across domains - seasonality, trends, noise patterns, etc.
/// </para>
/// <para><b>For Beginners:</b> Imagine you've read thousands of different books about weather,
/// stock prices, store sales, and website traffic. After reading all these, you develop an
/// intuition for how numbers change over time. When someone shows you a new sequence of numbers
/// you've never seen, you can make educated guesses about what comes next.
///
/// Chronos does exactly this but with neural networks. It "reads" millions of time series during
/// training and learns patterns. Then it can forecast new time series without being specifically
/// trained on that type of data. This is incredibly powerful for real-world applications where
/// you might not have enough historical data to train a specialized model.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var historicalData = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var historicalLabels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var recentContext = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use a pre-trained Chronos foundation model for zero-shot time series forecasting
/// var options = new ChronosOptions&lt;double&gt;();
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ChronosFoundationModel&lt;double&gt;(options))
///     .Build(historicalData, historicalLabels);
/// Vector&lt;double&gt; forecast = result.Predict(recentContext);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Chronos: Learning the Language of Time Series", "https://arxiv.org/abs/2403.07815", Year = 2024, Authors = "Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Yuyang Wang")]
public partial class ChronosFoundationModel<T> : TimeSeriesModelBase<T>
{
    private readonly ChronosOptions<T> _options;
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Tokenization parameters
    private readonly int _vocabularySize;
    private double _binMin = -15.0;
    private double _binMax = 15.0;
    private double _binWidth;

    // Transformer components - now using Tensor<T>
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _tokenEmbeddings;      // [vocabularySize, embeddingDim]
    [Buffer]
    private Tensor<T> _positionalEncoding;   // [maxLen, embeddingDim]
    private List<ChronosTransformerLayerTensor<T>> _transformerLayers;
    private Tensor<T> _outputProjection;     // [vocabularySize, embeddingDim]
    private Tensor<T> _outputBias;           // [vocabularySize]

    // Layer normalization for final output
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _finalLayerNormGamma;  // [embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _finalLayerNormBeta;   // [embeddingDim]

    // Gradient accumulators for batch training
    [Scratch]
    private readonly Dictionary<string, Tensor<T>> _gradientAccumulators;
    private int _gradientCount;

    /// <summary>
    /// Initializes a new instance of the Chronos foundation model.
    /// </summary>
    public ChronosFoundationModel(ChronosOptions<T>? options = null)
        : this(options ?? new ChronosOptions<T>(), initializeModel: true)
    {
    }

    private ChronosFoundationModel(ChronosOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        Options = _options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = RandomHelper.CreateSeededRandom(SeedOr(42));

        ValidateOptions(options);

        _vocabularySize = _options.VocabularySize;
        _binWidth = (_binMax - _binMin) / _vocabularySize;
        _transformerLayers = new List<ChronosTransformerLayerTensor<T>>();
        _gradientAccumulators = new Dictionary<string, Tensor<T>>();
        _gradientCount = 0;

        // Initialize with empty tensors - will be properly initialized in InitializeModel
        _tokenEmbeddings = new Tensor<T>(new[] { 1, 1 });
        _positionalEncoding = new Tensor<T>(new[] { 1, 1 });
        _outputProjection = new Tensor<T>(new[] { 1, 1 });
        _outputBias = new Tensor<T>(new[] { 1 });
        _finalLayerNormGamma = new Tensor<T>(new[] { 1 });
        _finalLayerNormBeta = new Tensor<T>(new[] { 1 });

        if (initializeModel)
            InitializeModel();
    }

    private static void ValidateOptions(ChronosOptions<T> options)
    {
        if (options.VocabularySize < 2)
            throw new ArgumentException($"VocabularySize must be at least 2, got {options.VocabularySize}", nameof(options));

        if (options.EmbeddingDim <= 0)
            throw new ArgumentException($"EmbeddingDim must be positive, got {options.EmbeddingDim}", nameof(options));

        if (options.NumHeads <= 0)
            throw new ArgumentException($"NumHeads must be positive, got {options.NumHeads}", nameof(options));

        if (options.EmbeddingDim % options.NumHeads != 0)
            throw new ArgumentException($"EmbeddingDim ({options.EmbeddingDim}) must be divisible by NumHeads ({options.NumHeads})", nameof(options));

        if (options.NumLayers <= 0)
            throw new ArgumentException($"NumLayers must be positive, got {options.NumLayers}", nameof(options));

        if (options.ContextLength <= 0)
            throw new ArgumentException($"ContextLength must be positive, got {options.ContextLength}", nameof(options));

        if (options.ForecastHorizon <= 0)
            throw new ArgumentException($"ForecastHorizon must be positive, got {options.ForecastHorizon}", nameof(options));
    }

    private void InitializeModel()
    {
        double stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);

        // Token embeddings: [vocabularySize, embeddingDim]
        _tokenEmbeddings = new Tensor<T>(new[] { _vocabularySize, _options.EmbeddingDim });
        InitializeTensorXavier(_tokenEmbeddings, stddev);

        // Sinusoidal positional encoding for context + forecast length
        int maxLen = _options.ContextLength + _options.ForecastHorizon;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);

        // Transformer layers
        _transformerLayers.Clear();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _transformerLayers.Add(new ChronosTransformerLayerTensor<T>(
                _options.EmbeddingDim,
                _options.NumHeads,
                seed: SeedOr(42) + i * 1000
            ));
        }

        // Final layer normalization
        _finalLayerNormGamma = new Tensor<T>(new[] { _options.EmbeddingDim });
        _finalLayerNormBeta = new Tensor<T>(new[] { _options.EmbeddingDim });
        for (int i = 0; i < _options.EmbeddingDim; i++)
        {
            _finalLayerNormGamma[i] = _numOps.One;
            _finalLayerNormBeta[i] = _numOps.Zero;
        }

        // Output projection: [vocabularySize, embeddingDim]
        _outputProjection = new Tensor<T>(new[] { _vocabularySize, _options.EmbeddingDim });
        InitializeTensorXavier(_outputProjection, stddev);
        _outputBias = new Tensor<T>(new[] { _vocabularySize });

        // Initialize gradient accumulators
        InitializeGradientAccumulators();
    }

    private void InitializeTensorXavier(Tensor<T> tensor, double stddev)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = _numOps.FromDouble((_random.NextDouble() * 2 - 1) * stddev);
        }
    }

    private void InitializeGradientAccumulators()
    {
        _gradientAccumulators.Clear();
        _gradientAccumulators["tokenEmbeddings"] = new Tensor<T>(_tokenEmbeddings._shape);
        _gradientAccumulators["outputProjection"] = new Tensor<T>(_outputProjection._shape);
        _gradientAccumulators["outputBias"] = new Tensor<T>(_outputBias._shape);
        _gradientAccumulators["finalLayerNormGamma"] = new Tensor<T>(_finalLayerNormGamma._shape);
        _gradientAccumulators["finalLayerNormBeta"] = new Tensor<T>(_finalLayerNormBeta._shape);

        for (int l = 0; l < _transformerLayers.Count; l++)
        {
            _transformerLayers[l].InitializeGradientAccumulators(_gradientAccumulators, l);
        }
        _gradientCount = 0;
    }

    private Tensor<T> CreateSinusoidalPositionalEncoding(int maxLen, int embeddingDim)
    {
        var pe = new Tensor<T>(new[] { maxLen, embeddingDim });
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                // Integer division (i / 2) is intentional - pairs adjacent dimensions (sin/cos) with same frequency
                int dimPair = i / 2;
                double angle = pos / Math.Pow(10000.0, (2.0 * dimPair) / embeddingDim);
                double value = i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
                pe[pos, i] = _numOps.FromDouble(value);
            }
        }
        return pe;
    }

    private int Tokenize(T value, double scaleFactor)
    {
        double normalized = Convert.ToDouble(value) / scaleFactor;
        int token = (int)Math.Floor((normalized - _binMin) / _binWidth);
        return Math.Max(0, Math.Min(token, _vocabularySize - 1));
    }

    private T Detokenize(int tokenIdx, double scaleFactor)
    {
        double binCenter = _binMin + (tokenIdx + 0.5) * _binWidth;
        return _numOps.FromDouble(binCenter * scaleFactor);
    }

    private double ComputeScaleFactor(Vector<T> context)
    {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < context.Length; i++)
        {
            double val = Math.Abs(Convert.ToDouble(context[i]));
            if (!double.IsNaN(val) && !double.IsInfinity(val))
            {
                sum += val;
                count++;
            }
        }
        return count > 0 ? (sum / count) + 1e-8 : 1.0;
    }

    /// <summary>
    /// Trains the Chronos model using proper backpropagation through all parameters.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = _numOps.FromDouble(y.Length);

        // Build autoregressive training pairs from the SERIES itself: a window of past values maps to
        // the next value. The feature matrix x carries only the time index (as for the other
        // univariate time-series models), so training on its rows would feed the model a single index
        // value instead of a sequence — it would never see real history and could not learn to
        // forecast. This mirrors how Forecast/PredictSingle consume the series' own lookback window.
        int context = _options.ContextLength;
        var trainInputs = new List<Vector<T>>();
        var trainTargets = new List<T>();
        for (int idx = 1; idx < y.Length; idx++)
        {
            int start = Math.Max(0, idx - context);
            var window = new Vector<T>(idx - start);
            for (int i = start; i < idx; i++)
            {
                window[i - start] = y[i];
            }
            trainInputs.Add(window);
            trainTargets.Add(y[idx]);
        }

        T learningRate = _numOps.FromDouble(_options.LearningRate);
        int batchSize = Math.Min(32, Math.Max(1, trainInputs.Count));

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var indices = Enumerable.Range(0, trainInputs.Count).OrderBy(_ => _random.Next()).ToList();

            for (int batch = 0; batch < indices.Count; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, indices.Count - batch);

                // Reset gradient accumulators
                ResetGradientAccumulators();

                // Batch gradient: one tape forward/backward per group of equal-length windows (the first
                // ContextLength-1 training windows are shorter than the context). Mathematically the sum of the
                // per-sample gradients, computed batched -- and correct: the previous hand-written per-sample
                // backprop disagreed with finite differences on every transformer-layer parameter
                // (ChronosGradientFiniteDifferenceTests).
                var groups = new Dictionary<int, (List<Vector<T>> Windows, List<T> Targets)>();
                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    int len = Math.Min(trainInputs[idx].Length, _options.ContextLength);
                    if (!groups.TryGetValue(len, out var group))
                        groups[len] = group = (new List<Vector<T>>(), new List<T>());
                    group.Windows.Add(trainInputs[idx]);
                    group.Targets.Add(trainTargets[idx]);
                }

                foreach (var group in groups.Values)
                {
                    TrainingCancellationToken.ThrowIfCancellationRequested();
                    var gradients = ComputeBatchGradientsTape(group.Windows, group.Targets);
                    AccumulateGradients(gradients);

                    // Recycle this group's Engine-op scratch. TimeSeriesModelBase.Train runs TrainCore inside one
                    // TensorArena; the gradient tape's disposal resets it, and this explicit Reset keeps that true even
                    // if the group's work is ever done without a tape (the hand-written backprop that preceded this
                    // retained ~220 MB/s, 74 GB on an Ooples-sized fit). The gradients are heap copies already folded
                    // into the accumulators, so nothing arena-backed outlives the group.
                    AiDotNet.Tensors.Helpers.TensorArena.Current?.Reset();
                }

                // Apply accumulated gradients
                ApplyGradients(learningRate, actualBatchSize);
            }
        }
    }

    /// <summary>
    /// The training loss of one sample through the ORIGINAL per-token forward (the forward half of
    /// <see cref="ComputeGradients"/>): next-token cross-entropy at the last position. Test oracle for gradient checks.
    /// </summary>
    internal double ReferenceSampleLoss(Vector<T> input, T target)
    {
        double scaleFactor = ComputeScaleFactor(input);
        int seqLen = Math.Min(input.Length, _options.ContextLength);
        var embedded = new List<Tensor<T>>();
        for (int t = 0; t < seqLen; t++)
        {
            int token = Tokenize(input[input.Length - seqLen + t], scaleFactor);
            var emb = new Tensor<T>(new[] { _options.EmbeddingDim });
            for (int i = 0; i < _options.EmbeddingDim; i++)
                emb[i] = _numOps.Add(_tokenEmbeddings[token, i], _positionalEncoding[t, i]);
            embedded.Add(emb);
        }
        var current = embedded;
        foreach (var layer in _transformerLayers) current = layer.Forward(current);
        var (normalized, _) = ApplyLayerNormWithCache(current[current.Count - 1], _finalLayerNormGamma, _finalLayerNormBeta);
        var logits = new double[_vocabularySize];
        double max = double.NegativeInfinity;
        for (int i = 0; i < _vocabularySize; i++)
        {
            double s = _numOps.ToDouble(_outputBias[i]);
            for (int j = 0; j < _options.EmbeddingDim; j++) s += _numOps.ToDouble(_outputProjection[i, j]) * _numOps.ToDouble(normalized[j]);
            logits[i] = s; if (s > max) max = s;
        }
        double sum = 0; for (int i = 0; i < _vocabularySize; i++) sum += Math.Exp(logits[i] - max);
        int targetToken = Tokenize(target, scaleFactor);
        return -(logits[targetToken] - max - Math.Log(sum));
    }

    /// <summary>The named trainable tensors (accumulator keys) -- shared by the batched trainer and gradient tests.</summary>
    internal IEnumerable<(string Key, Tensor<T> Param)> NamedTrainableTensors()
    {
        yield return ("tokenEmbeddings", _tokenEmbeddings);
        yield return ("outputProjection", _outputProjection);
        yield return ("outputBias", _outputBias);
        yield return ("finalLayerNormGamma", _finalLayerNormGamma);
        yield return ("finalLayerNormBeta", _finalLayerNormBeta);
        for (int layerIndex = 0; layerIndex < _transformerLayers.Count; layerIndex++)
            foreach (var (key, param) in _transformerLayers[layerIndex].NamedParameters())
                yield return ($"layer{layerIndex}_{key}", param);
    }

    /// <summary>
    /// Summed gradient of the per-sample next-token cross-entropy over a group of training windows that all have the
    /// same length, computed in one batched tape forward/backward. Same model and loss as <see cref="ComputeGradients"/>
    /// (each sample: tokenize the window with its own scale, embed + positional encoding, the transformer stack, final
    /// layer norm on the LAST position, logits = W h + b, cross-entropy against the tokenized target), so the result
    /// equals the sum of ComputeGradients over the group; keys match the gradient accumulators.
    /// </summary>
    internal Dictionary<string, Tensor<T>> ComputeBatchGradientsTape(IReadOnlyList<Vector<T>> windows, IReadOnlyList<T> targets)
    {
        int b = windows.Count, l = Math.Min(windows[0].Length, _options.ContextLength), e = _options.EmbeddingDim;
        var tokenIdx = new int[b * l];
        var targetTokens = new Tensor<T>(new[] { b });
        for (int s = 0; s < b; s++)
        {
            var window = windows[s];
            double scale = ComputeScaleFactor(window);
            for (int t = 0; t < l; t++)
                tokenIdx[s * l + t] = Tokenize(window[window.Length - l + t], scale);
            targetTokens[s] = _numOps.FromDouble(Tokenize(targets[s], scale));
        }

        var named = new List<(string Key, Tensor<T> Param)>
        {
            ("tokenEmbeddings", _tokenEmbeddings), ("outputProjection", _outputProjection), ("outputBias", _outputBias),
            ("finalLayerNormGamma", _finalLayerNormGamma), ("finalLayerNormBeta", _finalLayerNormBeta),
        };
        for (int layerIndex = 0; layerIndex < _transformerLayers.Count; layerIndex++)
            foreach (var (key, param) in _transformerLayers[layerIndex].NamedParameters())
                named.Add(($"layer{layerIndex}_{key}", param));

        using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>();
        var indices = new Tensor<int>(tokenIdx, new[] { b * l });
        var embedded = Engine.Reshape(Engine.TensorGather(_tokenEmbeddings, indices, 0), new[] { b, l, e });
        var positions = Engine.TensorNarrow(_positionalEncoding, 0, 0, l);              // [L, E], constant
        var x = Engine.TensorAdd(embedded, Engine.Reshape(positions, new[] { 1, l, e }));
        foreach (var layer in _transformerLayers)
            x = layer.ForwardBatch(x);

        var last = Engine.Reshape(Engine.TensorNarrow(x, 1, l - 1, 1), new[] { b, e });
        var normalized = Engine.LayerNorm(last, _finalLayerNormGamma, _finalLayerNormBeta, 1e-6, out _, out _);
        var logits = Engine.TensorAdd(Engine.TensorMatMulTransposed(normalized, _outputProjection), _outputBias);
        var logProbs = Engine.TensorLogSoftmax(logits, 1);
        var picked = Engine.TensorGatherClassValues(logProbs, targetTokens);           // [B]
        var loss = Engine.TensorNegate(Engine.ReduceSum(picked, null));               // summed, like the per-sample sum

        var grads = tape.ComputeGradients(loss, named.Select(p => p.Param).ToArray());
        // Heap copies: disposing the tape resets the training arena, which would recycle arena-backed gradients
        // under the caller.
        var result = new Dictionary<string, Tensor<T>>(named.Count);
        foreach (var (key, param) in named)
        {
            if (!grads.TryGetValue(param, out var g)) continue;
            var copy = new Tensor<T>(g._shape);
            for (int i = 0; i < g.Length; i++) copy[i] = g[i];
            result[key] = copy;
        }
        return result;
    }

    private void ResetGradientAccumulators()
    {
        foreach (var tensor in _gradientAccumulators.Values)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = _numOps.Zero;
            }
        }
        _gradientCount = 0;
    }

    private (Tensor<T> output, LayerNormCache cache) ApplyLayerNormWithCache(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta)
    {
        double mean = 0;
        for (int i = 0; i < input.Length; i++)
            mean += Convert.ToDouble(input[i]);
        mean /= input.Length;

        double variance = 0;
        for (int i = 0; i < input.Length; i++)
        {
            double diff = Convert.ToDouble(input[i]) - mean;
            variance += diff * diff;
        }
        variance /= input.Length;

        double stddev = Math.Sqrt(variance + 1e-6);

        var normalized = new Tensor<T>(new[] { input.Length });
        var output = new Tensor<T>(new[] { input.Length });

        for (int i = 0; i < input.Length; i++)
        {
            double norm = (Convert.ToDouble(input[i]) - mean) / stddev;
            normalized[i] = _numOps.FromDouble(norm);
            output[i] = _numOps.Add(
                _numOps.Multiply(gamma[i], _numOps.FromDouble(norm)),
                beta[i]);
        }

        return (output, new LayerNormCache
        {
            Input = input,
            Normalized = normalized,
            Mean = mean,
            Variance = variance,
            Stddev = stddev
        });
    }

    private void AccumulateGradients(Dictionary<string, Tensor<T>> gradients)
    {
        foreach (var kvp in gradients)
        {
            if (_gradientAccumulators.TryGetValue(kvp.Key, out var accumulator))
            {
                for (int i = 0; i < Math.Min(accumulator.Length, kvp.Value.Length); i++)
                {
                    accumulator[i] = _numOps.Add(accumulator[i], kvp.Value[i]);
                }
            }
            else
            {
                // A heap copy, never the gradient itself: the gradient may be an arena tensor that the per-sample
                // TensorArena.Reset in TrainCore recycles, and an accumulator must outlive the sample.
                var accumulatorCopy = new Tensor<T>(kvp.Value._shape);
                for (int i = 0; i < kvp.Value.Length; i++)
                    accumulatorCopy[i] = kvp.Value[i];
                _gradientAccumulators[kvp.Key] = accumulatorCopy;
            }
        }
        _gradientCount++;
    }

    private void ApplyGradients(T learningRate, int batchSize)
    {
        T batchSizeT = _numOps.FromDouble(batchSize);

        // Apply to token embeddings
        ApplyGradientToTensor(_tokenEmbeddings, _gradientAccumulators["tokenEmbeddings"], learningRate, batchSizeT);

        // Apply to output projection
        ApplyGradientToTensor(_outputProjection, _gradientAccumulators["outputProjection"], learningRate, batchSizeT);
        ApplyGradientToTensor(_outputBias, _gradientAccumulators["outputBias"], learningRate, batchSizeT);

        // Apply to final layer norm
        ApplyGradientToTensor(_finalLayerNormGamma, _gradientAccumulators["finalLayerNormGamma"], learningRate, batchSizeT);
        ApplyGradientToTensor(_finalLayerNormBeta, _gradientAccumulators["finalLayerNormBeta"], learningRate, batchSizeT);

        // Apply to transformer layers
        for (int l = 0; l < _transformerLayers.Count; l++)
        {
            _transformerLayers[l].ApplyGradients(_gradientAccumulators, l, learningRate, batchSizeT, Engine);
        }
    }

    private void ApplyGradientToTensor(Tensor<T> tensor, Tensor<T> gradient, T learningRate, T batchSize)
    {
        // Average gradient over batch and apply
        var avgGrad = Engine.TensorDivideScalar(gradient, batchSize);
        var scaledGrad = Engine.TensorMultiplyScalar(avgGrad, learningRate);
        var result = Engine.TensorSubtract(tensor, scaledGrad);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = result[i];
        }
    }

    /// <summary>
    /// Predicts the next value in a time series.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryPredictFromTimeIndexCalibration(input, _trainingSeries, out var calibratedPredictions))
        {
            return calibratedPredictions;
        }

        int n = input.Rows;
        if (n > 1 && input.Columns > 0)
            return PredictRowsBatched(input);

        var predictions = new Vector<T>(n);
        // Forecast every row from its own lookback window (see DeepARModel.Predict: the prior
        // i < _trainingSeries.Length shortcut returned memorized training values for OOS rows).
        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }
        return predictions;
    }

    /// <summary>
    /// <see cref="Predict(Matrix{T})"/> for many rows in one batched forward: each row is tokenized with its own scale
    /// (as <see cref="PredictWithScale"/> does), the transformer stack runs once over [rows, L, E] (the same math as
    /// the per-token <c>Forward</c>), and each row takes the first maximal logit at the last position and
    /// detokenizes it. AiModelBuilder predicts the whole dataset after fitting; per-row scalar inference was most of a
    /// production Chronos fit once training was batched.
    /// </summary>
    private Vector<T> PredictRowsBatched(Matrix<T> input)
    {
        int n = input.Rows, cols = input.Columns, l = Math.Min(cols, _options.ContextLength), e = _options.EmbeddingDim;
        var tokenIdx = new int[n * l];
        var scales = new double[n];
        for (int r = 0; r < n; r++)
        {
            var row = input.GetRow(r);
            scales[r] = ComputeScaleFactor(row);
            for (int t = 0; t < l; t++)
                tokenIdx[r * l + t] = Tokenize(row[cols - l + t], scales[r]);
        }

        var indices = new Tensor<int>(tokenIdx, new[] { n * l });
        var embedded = Engine.Reshape(Engine.TensorGather(_tokenEmbeddings, indices, 0), new[] { n, l, e });
        var positions = Engine.TensorNarrow(_positionalEncoding, 0, 0, l);
        var x = Engine.TensorAdd(embedded, Engine.Reshape(positions, new[] { 1, l, e }));
        foreach (var layer in _transformerLayers)
            x = layer.ForwardBatch(x);
        var last = Engine.Reshape(Engine.TensorNarrow(x, 1, l - 1, 1), new[] { n, e });
        var normalized = Engine.LayerNorm(last, _finalLayerNormGamma, _finalLayerNormBeta, 1e-6, out _, out _);
        var logits = Engine.TensorAdd(Engine.TensorMatMulTransposed(normalized, _outputProjection), _outputBias);

        var predictions = new Vector<T>(n);
        for (int r = 0; r < n; r++)
        {
            int best = 0;
            double bestLogit = double.NegativeInfinity;
            for (int v = 0; v < _vocabularySize; v++)
            {
                double s = _numOps.ToDouble(logits[r, v]);
                if (s > bestLogit) { bestLogit = s; best = v; }
            }
            predictions[r] = Detokenize(best, scales[r]);
        }
        return predictions;
    }

    public override T PredictSingle(Vector<T> input)
    {
        return PredictWithScale(input, ComputeScaleFactor(input));
    }

    /// <summary>
    /// Predicts the next value from <paramref name="input"/> using an EXPLICIT mean-scale factor.
    /// </summary>
    /// <remarks>
    /// Multi-step forecasting must hold the scale fixed across the horizon (see <see cref="Forecast"/>):
    /// because <see cref="Detokenize"/> multiplies the de-tokenized value by the scale, recomputing the
    /// scale from a context that already contains earlier forecasts feeds it back into itself and the
    /// forecast amplifies geometrically. <see cref="PredictSingle"/> keeps the per-window scale for
    /// single-step / in-sample prediction.
    /// </remarks>
    private T PredictWithScale(Vector<T> input, double scaleFactor)
    {
        int seqLen = Math.Min(input.Length, _options.ContextLength);
        var tokens = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokens[i] = Tokenize(input[input.Length - seqLen + i], scaleFactor);
        }

        var embedded = new List<Tensor<T>>();
        for (int t = 0; t < seqLen; t++)
        {
            var emb = new Tensor<T>(new[] { _options.EmbeddingDim });
            for (int i = 0; i < _options.EmbeddingDim; i++)
            {
                emb[i] = _numOps.Add(
                    _tokenEmbeddings[tokens[t], i],
                    _positionalEncoding[t, i]);
            }
            embedded.Add(emb);
        }

        foreach (var layer in _transformerLayers)
        {
            embedded = layer.Forward(embedded);
        }

        var lastHidden = embedded[embedded.Count - 1];
        lastHidden = ApplyLayerNorm(lastHidden, _finalLayerNormGamma, _finalLayerNormBeta);

        var logits = new double[_vocabularySize];
        double maxLogit = double.NegativeInfinity;
        int predictedToken = 0;

        for (int i = 0; i < _vocabularySize; i++)
        {
            double sum = Convert.ToDouble(_outputBias[i]);
            for (int j = 0; j < _options.EmbeddingDim; j++)
            {
                sum += Convert.ToDouble(_outputProjection[i, j]) * Convert.ToDouble(lastHidden[j]);
            }
            logits[i] = sum;

            if (sum > maxLogit)
            {
                maxLogit = sum;
                predictedToken = i;
            }
        }

        return Detokenize(predictedToken, scaleFactor);
    }

    /// <summary>
    /// Forecasts <paramref name="steps"/> values ahead of <paramref name="history"/> autoregressively
    /// under Chronos mean-scaling, with the scale factor held FIXED across the whole horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Chronos tokenizes mean-scaled values (value / scale) and de-tokenizes by multiplying the bin
    /// center back by that scale. The scale must be computed once from the observed history and reused
    /// for every forecast step. The default iterative forecaster recomputes the scale from a context
    /// that grows with each appended forecast, so the scale feeds back through <see cref="Detokenize"/>
    /// and the multi-step forecast explodes geometrically. Fixing the scale keeps the forecast on the
    /// numeric scale of the data.
    /// </para>
    /// </remarks>
    public override Vector<T> Forecast(Vector<T> history, int steps)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("The model must be trained before forecasting.");
        }
        if (history == null)
        {
            throw new ArgumentNullException(nameof(history), "History cannot be null.");
        }
        if (steps <= 0)
        {
            throw new ArgumentException("Number of forecast steps must be positive.", nameof(steps));
        }

        // Mean-scale computed ONCE from the observed history and held fixed for the horizon.
        double scaleFactor = ComputeScaleFactor(history);

        var context = new List<T>(history.Length + steps);
        for (int i = 0; i < history.Length; i++)
        {
            context.Add(history[i]);
        }

        var forecasts = new Vector<T>(steps);
        for (int step = 0; step < steps; step++)
        {
            var window = new Vector<T>(context.Count);
            for (int i = 0; i < context.Count; i++)
            {
                window[i] = context[i];
            }

            T prediction = PredictWithScale(window, scaleFactor);
            forecasts[step] = prediction;
            context.Add(prediction);
        }

        return forecasts;
    }

    private Tensor<T> ApplyLayerNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta)
    {
        return Engine.LayerNorm(input, gamma, beta, 1e-6, out _, out _);
    }

    /// <summary>
    /// Generates probabilistic forecasts by sampling from the model.
    /// </summary>
    public Dictionary<double, Vector<T>> ForecastWithQuantiles(Vector<T> history, double[] quantiles, int numSamples = 100)
    {
        var samples = new List<Vector<T>>();
        double scaleFactor = ComputeScaleFactor(history);

        for (int s = 0; s < numSamples; s++)
        {
            var forecast = new Vector<T>(_options.ForecastHorizon);
            var context = history.Clone();

            for (int h = 0; h < _options.ForecastHorizon; h++)
            {
                T prediction = PredictWithTemperature(context, scaleFactor, 0.5 + _random.NextDouble() * 0.5);
                forecast[h] = prediction;

                var newContext = new Vector<T>(context.Length);
                for (int i = 0; i < context.Length - 1; i++)
                    newContext[i] = context[i + 1];
                newContext[context.Length - 1] = prediction;
                context = newContext;
            }

            samples.Add(forecast);
        }

        var result = new Dictionary<double, Vector<T>>();
        foreach (var q in quantiles)
        {
            var quantileForecast = new Vector<T>(_options.ForecastHorizon);
            for (int h = 0; h < _options.ForecastHorizon; h++)
            {
                var values = samples.Select(sample => Convert.ToDouble(sample[h])).OrderBy(v => v).ToList();
                int idx = (int)(q * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                quantileForecast[h] = _numOps.FromDouble(values[idx]);
            }
            result[q] = quantileForecast;
        }

        return result;
    }

    private T PredictWithTemperature(Vector<T> input, double scaleFactor, double temperature)
    {
        int seqLen = Math.Min(input.Length, _options.ContextLength);
        var tokens = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
            tokens[i] = Tokenize(input[input.Length - seqLen + i], scaleFactor);

        var embedded = new List<Tensor<T>>();
        for (int t = 0; t < seqLen; t++)
        {
            var emb = new Tensor<T>(new[] { _options.EmbeddingDim });
            for (int i = 0; i < _options.EmbeddingDim; i++)
                emb[i] = _numOps.Add(_tokenEmbeddings[tokens[t], i], _positionalEncoding[t, i]);
            embedded.Add(emb);
        }

        foreach (var layer in _transformerLayers)
            embedded = layer.Forward(embedded);

        var lastHidden = ApplyLayerNorm(embedded[embedded.Count - 1], _finalLayerNormGamma, _finalLayerNormBeta);

        var logits = new double[_vocabularySize];
        for (int i = 0; i < _vocabularySize; i++)
        {
            double sum = Convert.ToDouble(_outputBias[i]);
            for (int j = 0; j < _options.EmbeddingDim; j++)
                sum += Convert.ToDouble(_outputProjection[i, j]) * Convert.ToDouble(lastHidden[j]);
            logits[i] = sum / temperature;
        }

        double maxLogit = logits.Max();
        double sumExp = 0;
        for (int i = 0; i < _vocabularySize; i++)
        {
            logits[i] = Math.Exp(logits[i] - maxLogit);
            sumExp += logits[i];
        }

        double r = _random.NextDouble() * sumExp;
        double cumSum = 0;
        int sampledToken = _vocabularySize - 1;
        for (int i = 0; i < _vocabularySize; i++)
        {
            cumSum += logits[i];
            if (cumSum >= r)
            {
                sampledToken = i;
                break;
            }
        }

        return Detokenize(sampledToken, scaleFactor);
    }

    private const int SerializationVersion = 3;



    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
            writer.Write(dim);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(Convert.ToDouble(tensor[i]));
    }



    private void ValidateOption(int serialized, int expected, string name)
    {
        if (serialized != expected)
            throw new InvalidOperationException($"Serialized {name} ({serialized}) doesn't match options ({expected})");
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = reader.ReadInt32();

        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = _numOps.FromDouble(reader.ReadDouble());
        return tensor;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "Chronos Foundation Model",
            Description = "Foundation model for zero-shot time series forecasting with mean-scaling tokenization and causal transformer",
            Complexity = ParameterCount,
            FeatureCount = _options.ContextLength,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "VocabularySize", _vocabularySize },
                { "EmbeddingDim", _options.EmbeddingDim },
                { "NumLayers", _options.NumLayers },
                { "NumHeads", _options.NumHeads },
                { "ContextLength", _options.ContextLength },
                { "ForecastHorizon", _options.ForecastHorizon }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new ChronosFoundationModel<T>(new ChronosOptions<T>(_options));
    }

    // ParameterCount restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    private class LayerNormCache
    {
        public Tensor<T> Input { get; set; } = new Tensor<T>(new[] { 1 });
        public Tensor<T> Normalized { get; set; } = new Tensor<T>(new[] { 1 });
        public double Mean { get; set; }
        public double Variance { get; set; }
        public double Stddev { get; set; }
    }
}

/// <summary>
/// Options for Chronos foundation model.
/// </summary>
public class ChronosOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 64;
    public int VocabularySize { get; set; } = 4096;
    public int EmbeddingDim { get; set; } = 256;
    public int NumLayers { get; set; } = 6;
    public int NumHeads { get; set; } = 8;
    public double LearningRate { get; set; } = 0.0001;
    public int Epochs { get; set; } = 100;

    public ChronosOptions() { }

    public ChronosOptions(ChronosOptions<T> other)
        : base(other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        VocabularySize = other.VocabularySize;
        EmbeddingDim = other.EmbeddingDim;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;
        DecompositionMethod = other.DecompositionMethod;
        UseIntercept = other.UseIntercept;
    }
}

/// <summary>
/// Chronos transformer layer with causal multi-head self-attention and feed-forward network.
/// Now uses Tensor<T> and proper backpropagation.
/// </summary>
// Rank 1, in equals out - a pre-norm transformer block is shape-preserving by construction, and the
// code says so twice. Both constructors declare the same width on each side
// (`base(new[] { embeddingDim }, new[] { embeddingDim })`), and Forward ends on
// `AddResidual(_cachedResidual1, ffnOutput)`: a residual add can only return the shape it was added
// to, so attention and the 4x FFN expansion both come back to _embeddingDim before the layer exits.
// The single-tensor ForwardTraced wraps its argument as a one-position sequence and returns that
// sequence's last element, so the same relation holds on the traced path.
//
// No hand-written OutputAxesFor: with matching layouts the generator derives Same(Features), which
// is exactly the relation above.
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal partial class ChronosTransformerLayerTensor<T> : NeuralNetworks.Layers.LayerBase<T>, IShapeContract
{
    private int _embeddingDim;
    private int _numHeads;
    private int _headDim;

    // Self-attention weights - now using Tensor<T>
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _queryProj;     // [embeddingDim, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _keyProj;       // [embeddingDim, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _valueProj;     // [embeddingDim, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _outputProj;    // [embeddingDim, embeddingDim]

    // Feed-forward network
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ffn1;          // [ffnDim, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ffn1Bias;      // [ffnDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ffn2;          // [embeddingDim, ffnDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _ffn2Bias;      // [embeddingDim]

    // Layer normalization parameters
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm1Gamma;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm1Beta;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm2Gamma;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _layerNorm2Beta;

    // Forward pass cache for backpropagation
    [Scratch]
    private List<Tensor<T>>? _cachedInput;
    [Scratch]
    private List<Tensor<T>>? _cachedNorm1;
    [Scratch]
    private List<Tensor<T>>? _cachedAttentionOutput;
    [Scratch]
    private List<Tensor<T>>? _cachedResidual1;
    [Scratch]
    private List<Tensor<T>>? _cachedNorm2;
    [Scratch]
    private List<Tensor<T>>? _cachedFfnHidden;

    public override bool SupportsTraining => true;

    public override void ResetState()
    {
        _cachedInput = null;
        _cachedNorm1 = null;
        _cachedAttentionOutput = null;
        _cachedResidual1 = null;
        _cachedNorm2 = null;
        _cachedFfnHidden = null;
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Single-tensor interface: treat as single-position sequence
        var seqInput = new List<Tensor<T>> { input };
        var seqOutput = Forward(seqInput);
        return seqOutput.Count > 0 ? seqOutput[seqOutput.Count - 1] : input;
    }

    public ChronosTransformerLayerTensor(int embeddingDim, int numHeads, int seed = 42)
        : base(new[] { embeddingDim }, new[] { embeddingDim })
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;

        var random = RandomHelper.CreateSeededRandom(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / (embeddingDim * 4.0));

        // Initialize attention projections
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

        // Initialize layer norms
        _layerNorm1Gamma = InitTensorOnes(embeddingDim);
        _layerNorm1Beta = new Tensor<T>(new[] { embeddingDim });
        _layerNorm2Gamma = InitTensorOnes(embeddingDim);
        _layerNorm2Beta = new Tensor<T>(new[] { embeddingDim });
    }

    internal ChronosTransformerLayerTensor()
        : base(new[] { 1 }, new[] { 1 })
    {
        _embeddingDim = 0;
        _numHeads = 1;
        _headDim = 0;
        _queryProj = new Tensor<T>(new[] { 1, 1 });
        _keyProj = new Tensor<T>(new[] { 1, 1 });
        _valueProj = new Tensor<T>(new[] { 1, 1 });
        _outputProj = new Tensor<T>(new[] { 1, 1 });
        _ffn1 = new Tensor<T>(new[] { 1, 1 });
        _ffn1Bias = new Tensor<T>(new[] { 1 });
        _ffn2 = new Tensor<T>(new[] { 1, 1 });
        _ffn2Bias = new Tensor<T>(new[] { 1 });
        _layerNorm1Gamma = new Tensor<T>(new[] { 1 });
        _layerNorm1Beta = new Tensor<T>(new[] { 1 });
        _layerNorm2Gamma = new Tensor<T>(new[] { 1 });
        _layerNorm2Beta = new Tensor<T>(new[] { 1 });
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

    public void InitializeGradientAccumulators(Dictionary<string, Tensor<T>> accumulators, int layerIndex)
    {
        string prefix = $"layer{layerIndex}_";
        accumulators[prefix + "queryProj"] = new Tensor<T>(_queryProj._shape);
        accumulators[prefix + "keyProj"] = new Tensor<T>(_keyProj._shape);
        accumulators[prefix + "valueProj"] = new Tensor<T>(_valueProj._shape);
        accumulators[prefix + "outputProj"] = new Tensor<T>(_outputProj._shape);
        accumulators[prefix + "ffn1"] = new Tensor<T>(_ffn1._shape);
        accumulators[prefix + "ffn1Bias"] = new Tensor<T>(_ffn1Bias._shape);
        accumulators[prefix + "ffn2"] = new Tensor<T>(_ffn2._shape);
        accumulators[prefix + "ffn2Bias"] = new Tensor<T>(_ffn2Bias._shape);
        accumulators[prefix + "layerNorm1Gamma"] = new Tensor<T>(_layerNorm1Gamma._shape);
        accumulators[prefix + "layerNorm1Beta"] = new Tensor<T>(_layerNorm1Beta._shape);
        accumulators[prefix + "layerNorm2Gamma"] = new Tensor<T>(_layerNorm2Gamma._shape);
        accumulators[prefix + "layerNorm2Beta"] = new Tensor<T>(_layerNorm2Beta._shape);
    }

    /// <summary>
    /// Forward pass through the transformer layer with caching for backprop.
    /// </summary>
    /// <summary>
    /// This layer's trainable tensors under the gradient-accumulator key names <see cref="ApplyGradients"/> reads
    /// (unprefixed; the model adds <c>layer{l}_</c>).
    /// </summary>
    internal IEnumerable<(string Key, Tensor<T> Param)> NamedParameters()
    {
        yield return ("queryProj", _queryProj);
        yield return ("keyProj", _keyProj);
        yield return ("valueProj", _valueProj);
        yield return ("outputProj", _outputProj);
        yield return ("ffn1", _ffn1);
        yield return ("ffn1Bias", _ffn1Bias);
        yield return ("ffn2", _ffn2);
        yield return ("ffn2Bias", _ffn2Bias);
        yield return ("layerNorm1Gamma", _layerNorm1Gamma);
        yield return ("layerNorm1Beta", _layerNorm1Beta);
        yield return ("layerNorm2Gamma", _layerNorm2Gamma);
        yield return ("layerNorm2Beta", _layerNorm2Beta);
    }

    /// <summary>
    /// Batched, tape-differentiable form of <see cref="Forward"/>: x is [B, L, E] (B windows of the same length), and
    /// the math is the per-token path's exactly -- pre-norm (population variance, eps 1e-6), causal single-head
    /// attention over the full embedding scaled by 1/sqrt(headDim) with q/k/v/out = W x, residual, pre-norm,
    /// GELU FFN (W x + b), residual -- expressed as Engine ops so a GradientTape yields every parameter's gradient.
    /// </summary>
    internal Tensor<T> ForwardBatch(Tensor<T> x)
    {
        int b = x.Shape[0], l = x.Shape[1], e = _embeddingDim;
        var flat = Engine.Reshape(x, new[] { b * l, e });

        var n1 = Engine.LayerNorm(flat, _layerNorm1Gamma, _layerNorm1Beta, 1e-6, out _, out _);
        var q = Engine.Reshape(Engine.TensorMatMulTransposed(n1, _queryProj), new[] { b, l, e });
        var k = Engine.Reshape(Engine.TensorMatMulTransposed(n1, _keyProj), new[] { b, l, e });
        var v = Engine.Reshape(Engine.TensorMatMulTransposed(n1, _valueProj), new[] { b, l, e });
        var ctx = Engine.MultiHeadAttentionCore(q, k, v, numHeads: 1, scale: 1.0 / Math.Sqrt(_headDim), causal: true);
        var attn = Engine.TensorMatMulTransposed(Engine.Reshape(ctx, new[] { b * l, e }), _outputProj);
        var r1 = Engine.TensorAdd(flat, attn);

        var n2 = Engine.LayerNorm(r1, _layerNorm2Gamma, _layerNorm2Beta, 1e-6, out _, out _);
        var hidden = Engine.GELU(Engine.TensorAdd(Engine.TensorMatMulTransposed(n2, _ffn1), _ffn1Bias));
        var ffn = Engine.TensorAdd(Engine.TensorMatMulTransposed(hidden, _ffn2), _ffn2Bias);
        return Engine.Reshape(Engine.TensorAdd(r1, ffn), new[] { b, l, e });
    }

    public List<Tensor<T>> Forward(List<Tensor<T>> input)
    {
        _cachedInput = input;

        // Pre-norm + causal self-attention
        _cachedNorm1 = LayerNorm(input, _layerNorm1Gamma, _layerNorm1Beta);
        _cachedAttentionOutput = CausalSelfAttention(_cachedNorm1);
        _cachedResidual1 = AddResidual(input, _cachedAttentionOutput);

        // Pre-norm + FFN
        _cachedNorm2 = LayerNorm(_cachedResidual1, _layerNorm2Gamma, _layerNorm2Beta);
        var ffnOutput = FeedForward(_cachedNorm2);
        return AddResidual(_cachedResidual1, ffnOutput);
    }

    private List<Tensor<T>> CausalSelfAttention(List<Tensor<T>> input)
    {
        int seqLen = input.Count;
        double scale = 1.0 / Math.Sqrt(_headDim);

        var queries = input.Select(x => MatVecMul(_queryProj, x)).ToList();
        var keys = input.Select(x => MatVecMul(_keyProj, x)).ToList();
        var values = input.Select(x => MatVecMul(_valueProj, x)).ToList();

        var output = new List<Tensor<T>>();

        for (int q = 0; q < seqLen; q++)
        {
            var attnWeights = new double[q + 1];
            double maxScore = double.NegativeInfinity;

            for (int k = 0; k <= q; k++)
            {
                attnWeights[k] = Convert.ToDouble(DotProduct(queries[q], keys[k])) * scale;
                maxScore = Math.Max(maxScore, attnWeights[k]);
            }

            double sum = 0;
            for (int k = 0; k <= q; k++)
            {
                attnWeights[k] = Math.Exp(attnWeights[k] - maxScore);
                sum += attnWeights[k];
            }
            for (int k = 0; k <= q; k++)
                attnWeights[k] /= sum;

            var result = new Tensor<T>(new[] { _embeddingDim });
            for (int k = 0; k <= q; k++)
            {
                for (int d = 0; d < _embeddingDim; d++)
                {
                    result[d] = NumOps.Add(result[d],
                        NumOps.Multiply(NumOps.FromDouble(attnWeights[k]), values[k][d]));
                }
            }
            output.Add(MatVecMul(_outputProj, result));
        }

        return output;
    }

    private List<Tensor<T>> LayerNorm(List<Tensor<T>> input, Tensor<T> gamma, Tensor<T> beta)
    {
        var output = new List<Tensor<T>>();
        foreach (var vec in input)
        {
            double mean = 0;
            for (int i = 0; i < vec.Length; i++)
                mean += Convert.ToDouble(vec[i]);
            mean /= vec.Length;

            double variance = 0;
            for (int i = 0; i < vec.Length; i++)
            {
                double diff = Convert.ToDouble(vec[i]) - mean;
                variance += diff * diff;
            }
            variance /= vec.Length;

            double stddev = Math.Sqrt(variance + 1e-6);
            // Vectorized LayerNorm: normalized = gamma * ((vec - mean) / std) + beta
            var meanTensor = Tensor<T>.CreateDefault(new[] { vec.Length }, NumOps.FromDouble(mean));
            var centered = Engine.TensorSubtract(vec, meanTensor);
            var normTensor = Engine.TensorMultiplyScalar<T>(centered, NumOps.FromDouble(1.0 / stddev));
            var scaled = Engine.TensorMultiply(normTensor, gamma);
            var normalized = Engine.TensorAdd(scaled, beta);
            output.Add(normalized);
        }
        return output;
    }

    private List<Tensor<T>> FeedForward(List<Tensor<T>> input)
    {
        _cachedFfnHidden = new List<Tensor<T>>();
        var output = new List<Tensor<T>>();

        foreach (var vec in input)
        {
            var hidden = MatVecMul(_ffn1, vec);
            hidden = Engine.TensorAdd(hidden, _ffn1Bias);
            hidden = Engine.GELU(hidden);
            _cachedFfnHidden.Add(hidden);

            var result = MatVecMul(_ffn2, hidden);
            result = Engine.TensorAdd(result, _ffn2Bias);
            output.Add(result);
        }
        return output;
    }

    private T GELU(T x)
    {
        double xd = Convert.ToDouble(x);
        double gelu = xd * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (xd + 0.044715 * xd * xd * xd)));
        return NumOps.FromDouble(gelu);
    }

    private List<Tensor<T>> AddResidual(List<Tensor<T>> input, List<Tensor<T>> residual)
    {
        var output = new List<Tensor<T>>();
        for (int t = 0; t < input.Count; t++)
        {
            output.Add(Engine.TensorAdd(input[t], residual[t]));
        }
        return output;
    }

    private Tensor<T> MatVecMul(Tensor<T> matrix, Tensor<T> vec)
    {
        // Direct tensor matmul: matrix [rows, cols] @ vec [cols, 1] -> [rows, 1] -> [rows]
        int rows = matrix.Shape[0];
        var vecCol = vec.Reshape(vec.Length, 1);
        var result = Engine.TensorMatMul(matrix, vecCol);
        return result.Reshape(rows);
    }

    private Tensor<T> MatVecMulTranspose(Tensor<T> matrix, Tensor<T> vec)
    {
        int cols = matrix.Shape[1];

        // Direct: M^T @ v using tensor transpose + matmul
        var matT = matrix.Transpose([1, 0]);
        var vecCol = vec.Reshape(vec.Length, 1);
        var result = Engine.TensorMatMul(matT, vecCol);
        return result.Reshape(cols);
    }

    private T DotProduct(Tensor<T> a, Tensor<T> b)
    {
        return VectorHelper.DotProduct(a.ToVector(), b.ToVector());
    }

    public void ApplyGradients(Dictionary<string, Tensor<T>> accumulators, int layerIndex,
        T learningRate, T batchSize, IEngine engine)
    {
        string prefix = $"layer{layerIndex}_";

        ApplyGradient(_queryProj, accumulators[prefix + "queryProj"], learningRate, batchSize, engine);
        ApplyGradient(_keyProj, accumulators[prefix + "keyProj"], learningRate, batchSize, engine);
        ApplyGradient(_valueProj, accumulators[prefix + "valueProj"], learningRate, batchSize, engine);
        ApplyGradient(_outputProj, accumulators[prefix + "outputProj"], learningRate, batchSize, engine);
        ApplyGradient(_ffn1, accumulators[prefix + "ffn1"], learningRate, batchSize, engine);
        ApplyGradient(_ffn1Bias, accumulators[prefix + "ffn1Bias"], learningRate, batchSize, engine);
        ApplyGradient(_ffn2, accumulators[prefix + "ffn2"], learningRate, batchSize, engine);
        ApplyGradient(_ffn2Bias, accumulators[prefix + "ffn2Bias"], learningRate, batchSize, engine);
        ApplyGradient(_layerNorm1Gamma, accumulators[prefix + "layerNorm1Gamma"], learningRate, batchSize, engine);
        ApplyGradient(_layerNorm1Beta, accumulators[prefix + "layerNorm1Beta"], learningRate, batchSize, engine);
        ApplyGradient(_layerNorm2Gamma, accumulators[prefix + "layerNorm2Gamma"], learningRate, batchSize, engine);
        ApplyGradient(_layerNorm2Beta, accumulators[prefix + "layerNorm2Beta"], learningRate, batchSize, engine);
    }

    private void ApplyGradient(Tensor<T> tensor, Tensor<T> gradient, T learningRate, T batchSize, IEngine engine)
    {
        var avgGrad = engine.TensorDivideScalar(gradient, batchSize);
        var scaledGrad = engine.TensorMultiplyScalar(avgGrad, learningRate);
        var result = engine.TensorSubtract(tensor, scaledGrad);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = result[i];
        }
    }

    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
            writer.Write(dim);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(Convert.ToDouble(tensor[i]));
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = reader.ReadInt32();

        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble(reader.ReadDouble());
        return tensor;
    }
}
