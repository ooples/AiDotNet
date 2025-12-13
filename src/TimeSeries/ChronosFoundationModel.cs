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
public class ChronosFoundationModel<T> : TimeSeriesModelBase<T>
{
    private readonly ChronosOptions<T> _options;
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    // Tokenization parameters
    private int _vocabularySize;
    private double _binMin = -15.0;  // Minimum normalized value
    private double _binMax = 15.0;   // Maximum normalized value
    private double _binWidth;

    // Transformer components
    private Matrix<T> _tokenEmbeddings = new Matrix<T>(0, 0);
    private Matrix<T> _positionalEncoding = new Matrix<T>(0, 0);
    private List<ChronosTransformerLayer<T>> _transformerLayers = new List<ChronosTransformerLayer<T>>();
    private Matrix<T> _outputProjection = new Matrix<T>(0, 0);
    private Vector<T> _outputBias = new Vector<T>(0);

    // Layer normalization for final output
    private Vector<T> _finalLayerNormGamma = new Vector<T>(0);
    private Vector<T> _finalLayerNormBeta = new Vector<T>(0);

    /// <summary>
    /// Initializes a new instance of the Chronos foundation model.
    /// </summary>
    /// <param name="options">Configuration options. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>Model Architecture Setup:</b>
    /// The constructor initializes the complete Chronos architecture:
    /// </para>
    /// <list type="bullet">
    /// <item><b>Token Embeddings:</b> Maps discrete tokens to dense vectors of dimension EmbeddingDim</item>
    /// <item><b>Positional Encoding:</b> Sinusoidal encodings for sequence position awareness</item>
    /// <item><b>Transformer Layers:</b> NumLayers decoder layers with causal self-attention</item>
    /// <item><b>Output Layer:</b> Projects hidden states back to vocabulary logits</item>
    /// </list>
    /// <para>
    /// <b>Tokenization Bins:</b>
    /// The vocabulary represents uniformly-spaced bins from -15 to 15 in normalized space.
    /// Values outside this range are clipped. The bin width is: (30) / VocabularySize.
    /// For VocabularySize=4096, each bin spans ~0.0073 in normalized units.
    /// </para>
    /// <para><b>For Beginners:</b> The vocabulary size determines how precisely the model can
    /// represent values. A larger vocabulary means more precise representations but also more
    /// parameters to learn. The default of 4096 provides a good balance - it can distinguish
    /// values that differ by about 0.7% after normalization.
    /// </para>
    /// </remarks>
    public ChronosFoundationModel(ChronosOptions<T>? options = null)
        : this(options ?? new ChronosOptions<T>(), initializeModel: true)
    {
    }

    private ChronosFoundationModel(ChronosOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = new Random(42);

        // Validate options to prevent runtime failures
        ValidateOptions(options);

        _vocabularySize = _options.VocabularySize;
        _binWidth = (_binMax - _binMin) / _vocabularySize;
        _transformerLayers = new List<ChronosTransformerLayer<T>>();

        if (initializeModel)
            InitializeModel();
    }

    /// <summary>
    /// Validates configuration options to prevent division-by-zero and invalid dimensions.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when options contain invalid values.</exception>
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

        // Token embeddings: vocabulary_size x embedding_dim
        _tokenEmbeddings = new Matrix<T>(_vocabularySize, _options.EmbeddingDim);
        InitializeMatrix(_tokenEmbeddings, stddev);

        // Sinusoidal positional encoding for context + forecast length
        int maxLen = _options.ContextLength + _options.ForecastHorizon;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);

        // Transformer layers
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _transformerLayers.Add(new ChronosTransformerLayer<T>(
                _options.EmbeddingDim,
                _options.NumHeads,
                seed: 42 + i * 1000
            ));
        }

        // Final layer normalization
        _finalLayerNormGamma = new Vector<T>(_options.EmbeddingDim);
        _finalLayerNormBeta = new Vector<T>(_options.EmbeddingDim);
        for (int i = 0; i < _options.EmbeddingDim; i++)
        {
            _finalLayerNormGamma[i] = _numOps.One;
        }

        // Output projection: projects from embedding_dim to vocabulary_size
        _outputProjection = new Matrix<T>(_vocabularySize, _options.EmbeddingDim);
        InitializeMatrix(_outputProjection, stddev);
        _outputBias = new Vector<T>(_vocabularySize);
    }

    private void InitializeMatrix(Matrix<T> matrix, double stddev)
    {
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                matrix[i, j] = _numOps.FromDouble((_random.NextDouble() * 2 - 1) * stddev);
    }

    /// <summary>
    /// Creates sinusoidal positional encoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why Positional Encoding?</b>
    /// Self-attention is permutation-invariant - it doesn't know the order of tokens.
    /// Positional encoding injects position information by adding unique patterns to each position.
    /// </para>
    /// <para>
    /// <b>Sinusoidal Encoding:</b>
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    /// This creates unique patterns for each position while allowing the model to learn
    /// relative positions through dot products.
    /// </para>
    /// </remarks>
    private Matrix<T> CreateSinusoidalPositionalEncoding(int maxLen, int embeddingDim)
    {
        var pe = new Matrix<T>(maxLen, embeddingDim);
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                double angle = pos / Math.Pow(10000.0, (2.0 * (i / 2)) / embeddingDim);
                pe[pos, i] = _numOps.FromDouble(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        }
        return pe;
    }

    /// <summary>
    /// Tokenizes a continuous value using mean-scaling normalization.
    /// </summary>
    /// <param name="value">The value to tokenize.</param>
    /// <param name="scaleFactor">The scaling factor (mean absolute value of context).</param>
    /// <returns>Token index in [0, VocabularySize-1].</returns>
    /// <remarks>
    /// <para>
    /// <b>Mean-Scaling Tokenization:</b>
    /// The Chronos paper uses mean-scaling to make the model scale-invariant:
    /// 1. Compute scale = mean(|context values|) + epsilon
    /// 2. Normalize: x_norm = x / scale
    /// 3. Map to bin: token = floor((x_norm - bin_min) / bin_width)
    /// 4. Clip to valid range: token = clamp(token, 0, vocab_size - 1)
    /// </para>
    /// <para>
    /// This allows the same model to handle time series of vastly different magnitudes -
    /// from stock prices in thousands to temperature readings in tens.
    /// </para>
    /// </remarks>
    private int Tokenize(T value, double scaleFactor)
    {
        double normalized = Convert.ToDouble(value) / scaleFactor;
        int token = (int)Math.Floor((normalized - _binMin) / _binWidth);
        return Math.Max(0, Math.Min(token, _vocabularySize - 1));
    }

    /// <summary>
    /// Converts a token back to a continuous value.
    /// </summary>
    /// <param name="tokenIdx">The token index.</param>
    /// <param name="scaleFactor">The scaling factor to denormalize.</param>
    /// <returns>The reconstructed continuous value.</returns>
    private T Detokenize(int tokenIdx, double scaleFactor)
    {
        // Return bin center
        double binCenter = _binMin + (tokenIdx + 0.5) * _binWidth;
        return _numOps.FromDouble(binCenter * scaleFactor);
    }

    /// <summary>
    /// Computes the mean-scaling factor for a context window.
    /// </summary>
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
    /// Trains the Chronos model on time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Training Objective:</b>
    /// Chronos is trained with next-token prediction, similar to language models.
    /// Given tokens [t1, t2, ..., tn], the model learns to predict [t2, t3, ..., t(n+1)].
    /// The loss is cross-entropy between predicted and actual next tokens.
    /// </para>
    /// <para>
    /// <b>Pretraining vs Fine-tuning:</b>
    /// In the original Chronos paper, the model is pretrained on a large corpus of
    /// synthetic and real time series. This implementation supports both pretraining
    /// (training from scratch) and fine-tuning on domain-specific data.
    /// </para>
    /// <para>
    /// <b>This Implementation:</b>
    /// Uses stochastic coordinate descent with numerical gradient estimation for
    /// framework-agnostic training. For production use with large datasets, consider
    /// using a deep learning framework with GPU acceleration.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        T epsilon = _numOps.FromDouble(1e-5);
        T twoEpsilon = _numOps.Multiply(_numOps.FromDouble(2.0), epsilon);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            // Shuffle training order
            var indices = Enumerable.Range(0, x.Rows).OrderBy(_ => _random.Next()).ToList();

            foreach (int i in indices)
            {
                Vector<T> input = x.GetRow(i);
                T target = y[i];

                // Update all trainable parameters
                UpdateParameters(input, target, learningRate, epsilon, twoEpsilon);
            }
        }
    }

    private void UpdateParameters(Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon)
    {
        // Update output projection (most impactful)
        UpdateMatrixSubset(_outputProjection, input, target, learningRate, epsilon, twoEpsilon, 100);
        UpdateVectorSubset(_outputBias, input, target, learningRate, epsilon, twoEpsilon, 20);

        // Update token embeddings
        UpdateMatrixSubset(_tokenEmbeddings, input, target, learningRate, epsilon, twoEpsilon, 50);

        // Update transformer layers
        foreach (var layer in _transformerLayers)
        {
            layer.UpdateWeights(input, target, learningRate, epsilon, twoEpsilon, PredictSingle, 30);
        }

        // Update final layer norm
        UpdateVectorSubset(_finalLayerNormGamma, input, target, learningRate, epsilon, twoEpsilon, 10);
    }

    private void UpdateMatrixSubset(Matrix<T> matrix, Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon, int sampleSize)
    {
        int totalWeights = matrix.Rows * matrix.Columns;
        int actualSample = Math.Min(sampleSize, totalWeights);

        for (int s = 0; s < actualSample; s++)
        {
            int flatIdx = _random.Next(totalWeights);
            int i = flatIdx / matrix.Columns;
            int j = flatIdx % matrix.Columns;

            T original = matrix[i, j];

            matrix[i, j] = _numOps.Add(original, epsilon);
            T err1 = _numOps.Subtract(target, PredictSingle(input));
            T lossPlus = _numOps.Multiply(err1, err1);

            matrix[i, j] = _numOps.Subtract(original, epsilon);
            T err2 = _numOps.Subtract(target, PredictSingle(input));
            T lossMinus = _numOps.Multiply(err2, err2);

            matrix[i, j] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            matrix[i, j] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }
    }

    private void UpdateVectorSubset(Vector<T> vector, Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon, int sampleSize)
    {
        int actualSample = Math.Min(sampleSize, vector.Length);

        for (int s = 0; s < actualSample; s++)
        {
            int i = _random.Next(vector.Length);
            T original = vector[i];

            vector[i] = _numOps.Add(original, epsilon);
            T err1 = _numOps.Subtract(target, PredictSingle(input));
            T lossPlus = _numOps.Multiply(err1, err1);

            vector[i] = _numOps.Subtract(original, epsilon);
            T err2 = _numOps.Subtract(target, PredictSingle(input));
            T lossMinus = _numOps.Multiply(err2, err2);

            vector[i] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            vector[i] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }
    }

    /// <summary>
    /// Predicts the next value in a time series.
    /// </summary>
    /// <param name="input">The context window of historical values.</param>
    /// <returns>The predicted next value.</returns>
    /// <remarks>
    /// <para>
    /// <b>Prediction Pipeline:</b>
    /// 1. Compute scale factor from context (mean absolute value)
    /// 2. Tokenize each value in the context using mean-scaling
    /// 3. Embed tokens and add positional encoding
    /// 4. Process through transformer layers with causal attention
    /// 5. Apply final layer normalization
    /// 6. Project to vocabulary logits
    /// 7. Select most likely token (argmax) and detokenize
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Step 1: Compute scale factor
        double scaleFactor = ComputeScaleFactor(input);

        // Step 2: Tokenize input
        int seqLen = Math.Min(input.Length, _options.ContextLength);
        var tokens = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokens[i] = Tokenize(input[input.Length - seqLen + i], scaleFactor);
        }

        // Step 3: Embed tokens and add positional encoding
        var embedded = new List<Vector<T>>();
        for (int t = 0; t < seqLen; t++)
        {
            var emb = new Vector<T>(_options.EmbeddingDim);
            for (int i = 0; i < _options.EmbeddingDim; i++)
            {
                // Token embedding + positional encoding
                emb[i] = _numOps.Add(
                    _tokenEmbeddings[tokens[t], i],
                    _positionalEncoding[t, i]);
            }
            embedded.Add(emb);
        }

        // Step 4: Process through transformer layers
        foreach (var layer in _transformerLayers)
        {
            embedded = layer.Forward(embedded);
        }

        // Step 5: Get last hidden state and apply final layer norm
        var lastHidden = embedded[embedded.Count - 1];
        lastHidden = ApplyLayerNorm(lastHidden, _finalLayerNormGamma, _finalLayerNormBeta);

        // Step 6: Project to vocabulary logits
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

        // Step 7: Detokenize
        return Detokenize(predictedToken, scaleFactor);
    }

    private Vector<T> ApplyLayerNorm(Vector<T> input, Vector<T> gamma, Vector<T> beta)
    {
        // Validate dimensions match (should always be true after option validation)
        if (input.Length != gamma.Length || input.Length != beta.Length)
        {
            throw new InvalidOperationException(
                $"Layer normalization dimension mismatch: input={input.Length}, gamma={gamma.Length}, beta={beta.Length}");
        }

        // Compute mean
        double mean = 0;
        for (int i = 0; i < input.Length; i++)
            mean += Convert.ToDouble(input[i]);
        mean /= input.Length;

        // Compute variance
        double variance = 0;
        for (int i = 0; i < input.Length; i++)
        {
            double diff = Convert.ToDouble(input[i]) - mean;
            variance += diff * diff;
        }
        variance /= input.Length;

        // Normalize
        double stddev = Math.Sqrt(variance + 1e-6);
        var output = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            double normalized = (Convert.ToDouble(input[i]) - mean) / stddev;
            output[i] = _numOps.Add(
                _numOps.Multiply(gamma[i], _numOps.FromDouble(normalized)),
                beta[i]);
        }
        return output;
    }

    /// <summary>
    /// Generates probabilistic forecasts by sampling from the model.
    /// </summary>
    /// <param name="history">Historical time series values.</param>
    /// <param name="quantiles">Quantiles to compute (e.g., [0.1, 0.5, 0.9]).</param>
    /// <param name="numSamples">Number of samples for Monte Carlo estimation.</param>
    /// <returns>Dictionary mapping quantiles to forecast vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>Probabilistic Forecasting:</b>
    /// Unlike point forecasts, probabilistic forecasts provide uncertainty estimates.
    /// Chronos generates multiple forecast trajectories by sampling from the predicted
    /// token distribution, then computes quantiles across these trajectories.
    /// </para>
    /// <para>
    /// <b>Sampling Process:</b>
    /// 1. For each sample, autoregressively generate forecast tokens
    /// 2. At each step, sample from softmax(logits / temperature) instead of argmax
    /// 3. Collect all forecast trajectories
    /// 4. Compute pointwise quantiles across trajectories
    /// </para>
    /// </remarks>
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
                // Sample next value (with temperature sampling for diversity)
                T prediction = PredictWithTemperature(context, scaleFactor, 0.5 + _random.NextDouble() * 0.5);
                forecast[h] = prediction;

                // Update context
                var newContext = new Vector<T>(context.Length);
                for (int i = 0; i < context.Length - 1; i++)
                    newContext[i] = context[i + 1];
                newContext[context.Length - 1] = prediction;
                context = newContext;
            }

            samples.Add(forecast);
        }

        // Compute quantiles
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
        // Tokenize and embed (same as PredictSingle)
        int seqLen = Math.Min(input.Length, _options.ContextLength);
        var tokens = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
            tokens[i] = Tokenize(input[input.Length - seqLen + i], scaleFactor);

        var embedded = new List<Vector<T>>();
        for (int t = 0; t < seqLen; t++)
        {
            var emb = new Vector<T>(_options.EmbeddingDim);
            for (int i = 0; i < _options.EmbeddingDim; i++)
                emb[i] = _numOps.Add(_tokenEmbeddings[tokens[t], i], _positionalEncoding[t, i]);
            embedded.Add(emb);
        }

        foreach (var layer in _transformerLayers)
            embedded = layer.Forward(embedded);

        var lastHidden = ApplyLayerNorm(embedded[embedded.Count - 1], _finalLayerNormGamma, _finalLayerNormBeta);

        // Compute logits
        var logits = new double[_vocabularySize];
        for (int i = 0; i < _vocabularySize; i++)
        {
            double sum = Convert.ToDouble(_outputBias[i]);
            for (int j = 0; j < _options.EmbeddingDim; j++)
                sum += Convert.ToDouble(_outputProjection[i, j]) * Convert.ToDouble(lastHidden[j]);
            logits[i] = sum / temperature;
        }

        // Softmax and sample
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

    private const int SerializationVersion = 2;

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(SerializationVersion);

        // Serialize options
        writer.Write(_vocabularySize);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.ContextLength);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.NumLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_binMin);
        writer.Write(_binMax);

        // Serialize token embeddings
        SerializeMatrix(writer, _tokenEmbeddings);

        // Serialize positional encoding
        SerializeMatrix(writer, _positionalEncoding);

        // Serialize transformer layers
        writer.Write(_transformerLayers.Count);
        foreach (var layer in _transformerLayers)
            layer.Serialize(writer);

        // Serialize final layer norm
        SerializeVector(writer, _finalLayerNormGamma);
        SerializeVector(writer, _finalLayerNormBeta);

        // Serialize output projection
        SerializeMatrix(writer, _outputProjection);
        SerializeVector(writer, _outputBias);
    }

    private void SerializeMatrix(BinaryWriter writer, Matrix<T> matrix)
    {
        writer.Write(matrix.Rows);
        writer.Write(matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                writer.Write(Convert.ToDouble(matrix[i, j]));
    }

    private void SerializeVector(BinaryWriter writer, Vector<T> vector)
    {
        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
            writer.Write(Convert.ToDouble(vector[i]));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        int version = reader.ReadInt32();
        if (version != SerializationVersion)
            throw new NotSupportedException($"Unsupported serialization version: {version}");

        // Deserialize and validate options
        int vocabularySize = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        int contextLength = reader.ReadInt32();
        int forecastHorizon = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        _binMin = reader.ReadDouble();
        _binMax = reader.ReadDouble();

        ValidateOption(vocabularySize, _vocabularySize, "VocabularySize");
        ValidateOption(embeddingDim, _options.EmbeddingDim, "EmbeddingDim");
        ValidateOption(contextLength, _options.ContextLength, "ContextLength");
        ValidateOption(forecastHorizon, _options.ForecastHorizon, "ForecastHorizon");
        ValidateOption(numLayers, _options.NumLayers, "NumLayers");
        ValidateOption(numHeads, _options.NumHeads, "NumHeads");

        _binWidth = (_binMax - _binMin) / _vocabularySize;

        // Deserialize token embeddings
        _tokenEmbeddings = DeserializeMatrix(reader);

        // Deserialize positional encoding
        _positionalEncoding = DeserializeMatrix(reader);

        // Deserialize transformer layers
        int layerCount = reader.ReadInt32();
        _transformerLayers = new List<ChronosTransformerLayer<T>>(layerCount);
        for (int i = 0; i < layerCount; i++)
            _transformerLayers.Add(ChronosTransformerLayer<T>.Deserialize(reader));

        // Deserialize final layer norm
        _finalLayerNormGamma = DeserializeVector(reader);
        _finalLayerNormBeta = DeserializeVector(reader);

        // Deserialize output projection
        _outputProjection = DeserializeMatrix(reader);
        _outputBias = DeserializeVector(reader);
    }

    private void ValidateOption(int serialized, int expected, string name)
    {
        if (serialized != expected)
            throw new InvalidOperationException($"Serialized {name} ({serialized}) doesn't match options ({expected})");
    }

    private Matrix<T> DeserializeMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = _numOps.FromDouble(reader.ReadDouble());
        return matrix;
    }

    private Vector<T> DeserializeVector(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var vector = new Vector<T>(len);
        for (int i = 0; i < len; i++)
            vector[i] = _numOps.FromDouble(reader.ReadDouble());
        return vector;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "Chronos Foundation Model",
            ModelType = ModelType.TimeSeriesRegression,
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

    public override int ParameterCount
    {
        get
        {
            int count = _tokenEmbeddings.Rows * _tokenEmbeddings.Columns;
            count += _outputProjection.Rows * _outputProjection.Columns + _outputBias.Length;
            count += _finalLayerNormGamma.Length * 2;
            foreach (var layer in _transformerLayers)
                count += layer.ParameterCount;
            return count;
        }
    }
}

/// <summary>
/// Options for Chronos foundation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Key Configuration Parameters:</b>
/// </para>
/// <list type="bullet">
/// <item><b>VocabularySize:</b> Number of discrete tokens (bins) for value representation.
/// Larger values allow finer-grained representations but require more parameters.</item>
/// <item><b>EmbeddingDim:</b> Dimension of token embeddings and hidden states.
/// Larger values increase model capacity but also computation.</item>
/// <item><b>NumLayers:</b> Number of transformer layers. More layers allow learning
/// more complex patterns but risk overfitting on small datasets.</item>
/// <item><b>NumHeads:</b> Number of attention heads. Should divide EmbeddingDim evenly.</item>
/// <item><b>ContextLength:</b> Maximum number of historical values to consider.</item>
/// <item><b>ForecastHorizon:</b> Number of future steps to predict.</item>
/// </list>
/// </remarks>
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
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Causal Self-Attention:</b>
/// Unlike bidirectional attention in BERT-style models, causal attention only allows
/// each position to attend to previous positions (and itself). This is essential for
/// autoregressive generation where future values are unknown at prediction time.
/// </para>
/// <para>
/// <b>Multi-Head Attention:</b>
/// Instead of a single attention function, multi-head attention runs h parallel attention
/// operations with different learned projections. This allows the model to jointly attend
/// to information from different representation subspaces at different positions.
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
/// </para>
/// <para>
/// <b>Pre-Norm Architecture:</b>
/// Layer normalization is applied before (not after) attention and FFN sublayers.
/// This improves training stability and allows for deeper networks.
/// </para>
/// </remarks>
internal class ChronosTransformerLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Self-attention weights (Q, K, V projections)
    private Matrix<T> _queryProj;
    private Matrix<T> _keyProj;
    private Matrix<T> _valueProj;
    private Matrix<T> _outputProj;

    // Feed-forward network (4x expansion)
    private Matrix<T> _ffn1;
    private Vector<T> _ffn1Bias;
    private Matrix<T> _ffn2;
    private Vector<T> _ffn2Bias;

    // Layer normalization parameters
    private Vector<T> _layerNorm1Gamma;
    private Vector<T> _layerNorm1Beta;
    private Vector<T> _layerNorm2Gamma;
    private Vector<T> _layerNorm2Beta;

    public int ParameterCount =>
        _queryProj.Rows * _queryProj.Columns * 4 +
        _ffn1.Rows * _ffn1.Columns + _ffn1Bias.Length +
        _ffn2.Rows * _ffn2.Columns + _ffn2Bias.Length +
        _layerNorm1Gamma.Length * 4;

    public ChronosTransformerLayer(int embeddingDim, int numHeads, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = RandomHelper.CreateSeededRandom(seed);
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;

        var random = RandomHelper.CreateSeededRandom(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / (embeddingDim * 4));

        // Initialize attention projections
        _queryProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _keyProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _valueProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _outputProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);

        // Initialize FFN (4x expansion)
        int ffnDim = embeddingDim * 4;
        _ffn1 = InitMatrix(ffnDim, embeddingDim, ffnStddev, random);
        _ffn1Bias = new Vector<T>(ffnDim);
        _ffn2 = InitMatrix(embeddingDim, ffnDim, ffnStddev, random);
        _ffn2Bias = new Vector<T>(embeddingDim);

        // Initialize layer norms
        _layerNorm1Gamma = InitVector(embeddingDim, _numOps.One);
        _layerNorm1Beta = new Vector<T>(embeddingDim);
        _layerNorm2Gamma = InitVector(embeddingDim, _numOps.One);
        _layerNorm2Beta = new Vector<T>(embeddingDim);
    }

    private ChronosTransformerLayer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = RandomHelper.CreateSeededRandom(42);
        _embeddingDim = 0;
        _numHeads = 1;
        _headDim = 0;
        _queryProj = new Matrix<T>(0, 0);
        _keyProj = new Matrix<T>(0, 0);
        _valueProj = new Matrix<T>(0, 0);
        _outputProj = new Matrix<T>(0, 0);
        _ffn1 = new Matrix<T>(0, 0);
        _ffn1Bias = new Vector<T>(0);
        _ffn2 = new Matrix<T>(0, 0);
        _ffn2Bias = new Vector<T>(0);
        _layerNorm1Gamma = new Vector<T>(0);
        _layerNorm1Beta = new Vector<T>(0);
        _layerNorm2Gamma = new Vector<T>(0);
        _layerNorm2Beta = new Vector<T>(0);
    }

    private Matrix<T> InitMatrix(int rows, int cols, double stddev, Random random)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return matrix;
    }

    private Vector<T> InitVector(int size, T value)
    {
        var vector = new Vector<T>(size);
        for (int i = 0; i < size; i++)
            vector[i] = value;
        return vector;
    }

    /// <summary>
    /// Forward pass through the transformer layer.
    /// </summary>
    public List<Vector<T>> Forward(List<Vector<T>> input)
    {
        // Pre-norm + causal self-attention
        var normalized = LayerNorm(input, _layerNorm1Gamma, _layerNorm1Beta);
        var attended = CausalSelfAttention(normalized);
        var residual1 = AddResidual(input, attended);

        // Pre-norm + FFN
        normalized = LayerNorm(residual1, _layerNorm2Gamma, _layerNorm2Beta);
        var ffnOutput = FeedForward(normalized);
        return AddResidual(residual1, ffnOutput);
    }

    /// <summary>
    /// Causal multi-head self-attention.
    /// </summary>
    private List<Vector<T>> CausalSelfAttention(List<Vector<T>> input)
    {
        int seqLen = input.Count;
        double scale = 1.0 / Math.Sqrt(_headDim);

        // Compute Q, K, V for all positions
        var queries = input.Select(x => MatVecMul(_queryProj, x)).ToList();
        var keys = input.Select(x => MatVecMul(_keyProj, x)).ToList();
        var values = input.Select(x => MatVecMul(_valueProj, x)).ToList();

        var output = new List<Vector<T>>();

        for (int q = 0; q < seqLen; q++)
        {
            // Causal: only attend to positions <= q
            var attnWeights = new double[q + 1];
            double maxScore = double.NegativeInfinity;

            for (int k = 0; k <= q; k++)
            {
                attnWeights[k] = Convert.ToDouble(DotProduct(queries[q], keys[k])) * scale;
                maxScore = Math.Max(maxScore, attnWeights[k]);
            }

            // Softmax
            double sum = 0;
            for (int k = 0; k <= q; k++)
            {
                attnWeights[k] = Math.Exp(attnWeights[k] - maxScore);
                sum += attnWeights[k];
            }
            for (int k = 0; k <= q; k++)
                attnWeights[k] /= sum;

            // Weighted sum of values
            var result = new Vector<T>(_embeddingDim);
            for (int k = 0; k <= q; k++)
            {
                for (int d = 0; d < _embeddingDim; d++)
                {
                    result[d] = _numOps.Add(result[d],
                        _numOps.Multiply(_numOps.FromDouble(attnWeights[k]), values[k][d]));
                }
            }
            output.Add(MatVecMul(_outputProj, result));
        }

        return output;
    }

    private List<Vector<T>> LayerNorm(List<Vector<T>> input, Vector<T> gamma, Vector<T> beta)
    {
        var output = new List<Vector<T>>();
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
            var normalized = new Vector<T>(vec.Length);
            for (int i = 0; i < vec.Length && i < gamma.Length; i++)
            {
                double norm = (Convert.ToDouble(vec[i]) - mean) / stddev;
                normalized[i] = _numOps.Add(
                    _numOps.Multiply(gamma[i], _numOps.FromDouble(norm)),
                    beta[i]);
            }
            output.Add(normalized);
        }
        return output;
    }

    private List<Vector<T>> FeedForward(List<Vector<T>> input)
    {
        var output = new List<Vector<T>>();
        foreach (var vec in input)
        {
            // First linear + GELU
            var hidden = MatVecMul(_ffn1, vec);
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] = _numOps.Add(hidden[i], _ffn1Bias[i]);
                hidden[i] = GELU(hidden[i]);
            }

            // Second linear
            var result = MatVecMul(_ffn2, hidden);
            for (int i = 0; i < result.Length; i++)
                result[i] = _numOps.Add(result[i], _ffn2Bias[i]);
            output.Add(result);
        }
        return output;
    }

    private T GELU(T x)
    {
        double xd = Convert.ToDouble(x);
        double gelu = xd * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (xd + 0.044715 * xd * xd * xd)));
        return _numOps.FromDouble(gelu);
    }

    private List<Vector<T>> AddResidual(List<Vector<T>> input, List<Vector<T>> residual)
    {
        var output = new List<Vector<T>>();
        for (int t = 0; t < input.Count; t++)
        {
            var vec = new Vector<T>(input[t].Length);
            for (int i = 0; i < input[t].Length && i < residual[t].Length; i++)
                vec[i] = _numOps.Add(input[t][i], residual[t][i]);
            output.Add(vec);
        }
        return output;
    }

    private Vector<T> MatVecMul(Matrix<T> matrix, Vector<T> vec)
    {
        var result = new Vector<T>(matrix.Rows);
        for (int i = 0; i < matrix.Rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < Math.Min(matrix.Columns, vec.Length); j++)
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[i, j], vec[j]));
            result[i] = sum;
        }
        return result;
    }

    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            sum = _numOps.Add(sum, _numOps.Multiply(a[i], b[i]));
        return sum;
    }

    public void UpdateWeights(Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon,
        Func<Vector<T>, T> predict, int sampleSize)
    {
        // Use seeded Random for reproducibility
        var allMatrices = new[] { _queryProj, _keyProj, _valueProj, _outputProj, _ffn1, _ffn2 };

        foreach (var matrix in allMatrices)
        {
            int totalWeights = matrix.Rows * matrix.Columns;
            int actualSample = Math.Min(sampleSize / 6, totalWeights);

            for (int s = 0; s < actualSample; s++)
            {
                int flatIdx = _random.Next(totalWeights);
                int i = flatIdx / matrix.Columns;
                int j = flatIdx % matrix.Columns;

                T original = matrix[i, j];

                matrix[i, j] = _numOps.Add(original, epsilon);
                T err1 = _numOps.Subtract(target, predict(input));
                T lossPlus = _numOps.Multiply(err1, err1);

                matrix[i, j] = _numOps.Subtract(original, epsilon);
                T err2 = _numOps.Subtract(target, predict(input));
                T lossMinus = _numOps.Multiply(err2, err2);

                matrix[i, j] = original;

                T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
                matrix[i, j] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
            }
        }
    }

    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_embeddingDim);
        writer.Write(_numHeads);

        SerializeMatrix(writer, _queryProj);
        SerializeMatrix(writer, _keyProj);
        SerializeMatrix(writer, _valueProj);
        SerializeMatrix(writer, _outputProj);
        SerializeMatrix(writer, _ffn1);
        SerializeVector(writer, _ffn1Bias);
        SerializeMatrix(writer, _ffn2);
        SerializeVector(writer, _ffn2Bias);
        SerializeVector(writer, _layerNorm1Gamma);
        SerializeVector(writer, _layerNorm1Beta);
        SerializeVector(writer, _layerNorm2Gamma);
        SerializeVector(writer, _layerNorm2Beta);
    }

    private void SerializeMatrix(BinaryWriter writer, Matrix<T> matrix)
    {
        writer.Write(matrix.Rows);
        writer.Write(matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                writer.Write(Convert.ToDouble(matrix[i, j]));
    }

    private void SerializeVector(BinaryWriter writer, Vector<T> vector)
    {
        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
            writer.Write(Convert.ToDouble(vector[i]));
    }

    public static ChronosTransformerLayer<T> Deserialize(BinaryReader reader)
    {
        var layer = new ChronosTransformerLayer<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        int embeddingDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();

        typeof(ChronosTransformerLayer<T>).GetField("_embeddingDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, embeddingDim);
        typeof(ChronosTransformerLayer<T>).GetField("_numHeads", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, numHeads);
        typeof(ChronosTransformerLayer<T>).GetField("_headDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, embeddingDim / numHeads);

        layer._queryProj = DeserializeMatrix(reader, numOps);
        layer._keyProj = DeserializeMatrix(reader, numOps);
        layer._valueProj = DeserializeMatrix(reader, numOps);
        layer._outputProj = DeserializeMatrix(reader, numOps);
        layer._ffn1 = DeserializeMatrix(reader, numOps);
        layer._ffn1Bias = DeserializeVector(reader, numOps);
        layer._ffn2 = DeserializeMatrix(reader, numOps);
        layer._ffn2Bias = DeserializeVector(reader, numOps);
        layer._layerNorm1Gamma = DeserializeVector(reader, numOps);
        layer._layerNorm1Beta = DeserializeVector(reader, numOps);
        layer._layerNorm2Gamma = DeserializeVector(reader, numOps);
        layer._layerNorm2Beta = DeserializeVector(reader, numOps);

        return layer;
    }

    private static Matrix<T> DeserializeMatrix(BinaryReader reader, INumericOperations<T> numOps)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = numOps.FromDouble(reader.ReadDouble());
        return matrix;
    }

    private static Vector<T> DeserializeVector(BinaryReader reader, INumericOperations<T> numOps)
    {
        int len = reader.ReadInt32();
        var vector = new Vector<T>(len);
        for (int i = 0; i < len; i++)
            vector[i] = numOps.FromDouble(reader.ReadDouble());
        return vector;
    }
}
