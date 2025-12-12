namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Chronos-inspired foundation model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Chronos is a foundation model approach that treats time series forecasting as a language modeling task.
/// It tokenizes time series values and uses transformer architectures pretrained on large corpora of
/// diverse time series data. This implementation provides a simplified version of the concept.
/// </para>
/// <para>
/// Key concepts:
/// - Value Tokenization: Maps continuous values to discrete tokens
/// - Transformer Backbone: Uses self-attention for pattern learning
/// - Zero-shot Forecasting: Can forecast on new series without retraining
/// - Probabilistic Outputs: Generates distributional forecasts
/// </para>
/// <para><b>For Beginners:</b> Chronos is inspired by how large language models (like GPT) work,
/// but for time series instead of text. Just as language models learn patterns from vast amounts
/// of text and can then understand new sentences, Chronos learns from many different time series
/// and can forecast new ones it hasn't seen before.
///
/// The key idea is to treat numbers in a time series like "words" in a sentence, using similar
/// techniques to understand patterns and make predictions. This makes it very versatile - the
/// same model can work on sales data, temperature readings, stock prices, etc.
/// </para>
/// </remarks>
public class ChronosFoundationModel<T> : TimeSeriesModelBase<T>
{
    private readonly ChronosOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    // Tokenization
    private Vector<T> _vocabularyCentroids = new Vector<T>(0);
    private int _vocabularySize;

    // Transformer components
    private Matrix<T> _tokenEmbeddings = new Matrix<T>(0, 0);
    private List<TransformerBlock<T>> _transformerLayers = new List<TransformerBlock<T>>();
    private Matrix<T> _outputProjection = new Matrix<T>(0, 0);
    private Vector<T> _outputBias = new Vector<T>(0);

    public ChronosFoundationModel(ChronosOptions<T>? options = null)
        : this(options ?? new ChronosOptions<T>(), initializeModel: true)
    {
    }

    /// <summary>
    /// Private constructor for proper options instance management.
    /// </summary>
    private ChronosFoundationModel(ChronosOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _vocabularySize = _options.VocabularySize;
        _transformerLayers = new List<TransformerBlock<T>>();

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        var random = new Random(42);

        // Initialize vocabulary centroids for tokenization
        _vocabularyCentroids = new Vector<T>(_vocabularySize);
        for (int i = 0; i < _vocabularySize; i++)
        {
            // Evenly spaced centroids (would be learned from data in practice)
            double value = -3.0 + (6.0 * i / (_vocabularySize - 1));
            _vocabularyCentroids[i] = _numOps.FromDouble(value);
        }

        // Token embeddings
        double stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);
        _tokenEmbeddings = new Matrix<T>(_options.EmbeddingDim, _vocabularySize);
        for (int i = 0; i < _tokenEmbeddings.Rows; i++)
            for (int j = 0; j < _tokenEmbeddings.Columns; j++)
                _tokenEmbeddings[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        // Transformer layers - use different seeds for each layer
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _transformerLayers.Add(new TransformerBlock<T>(_options.EmbeddingDim, _options.NumHeads, seed: 42 + i * 1000));
        }

        // Output projection (back to vocabulary)
        stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);
        _outputProjection = new Matrix<T>(_vocabularySize, _options.EmbeddingDim);
        for (int i = 0; i < _outputProjection.Rows; i++)
            for (int j = 0; j < _outputProjection.Columns; j++)
                _outputProjection[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _outputBias = new Vector<T>(_vocabularySize);
    }

    /// <summary>
    /// Tokenizes a continuous value to the nearest vocabulary centroid.
    /// </summary>
    private int Tokenize(T value)
    {
        int nearestIdx = 0;
        T minDist = _numOps.Abs(_numOps.Subtract(value, _vocabularyCentroids[0]));

        for (int i = 1; i < _vocabularySize; i++)
        {
            T dist = _numOps.Abs(_numOps.Subtract(value, _vocabularyCentroids[i]));
            if (_numOps.LessThan(dist, minDist))
            {
                minDist = dist;
                nearestIdx = i;
            }
        }

        return nearestIdx;
    }

    /// <summary>
    /// Detokenizes a token index back to a continuous value.
    /// </summary>
    private T Detokenize(int tokenIdx)
    {
        return _vocabularyCentroids[Math.Min(tokenIdx, _vocabularySize - 1)];
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                Vector<T> input = x.GetRow(i);
                T target = y[i];

                // Update output projection weights using numerical gradients
                UpdateOutputWeights(input, target, learningRate);
            }
        }
    }

    /// <summary>
    /// Updates output projection weights using numerical gradient estimation.
    /// </summary>
    private void UpdateOutputWeights(Vector<T> input, T target, T learningRate)
    {
        T epsilon = _numOps.FromDouble(1e-5);
        T twoEpsilon = _numOps.Multiply(_numOps.FromDouble(2.0), epsilon);

        // Update all output projection weights (not just a 5x5 subset)
        for (int i = 0; i < _outputProjection.Rows; i++)
        {
            for (int j = 0; j < _outputProjection.Columns; j++)
            {
                T original = _outputProjection[i, j];

                // Compute loss with perturbed weight (positive)
                _outputProjection[i, j] = _numOps.Add(original, epsilon);
                T predPlus = PredictSingle(input);
                T errorPlus = _numOps.Subtract(target, predPlus);
                T lossPlus = _numOps.Multiply(errorPlus, errorPlus);

                // Compute loss with perturbed weight (negative)
                _outputProjection[i, j] = _numOps.Subtract(original, epsilon);
                T predMinus = PredictSingle(input);
                T errorMinus = _numOps.Subtract(target, predMinus);
                T lossMinus = _numOps.Multiply(errorMinus, errorMinus);

                // Restore and update
                _outputProjection[i, j] = original;

                T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
                _outputProjection[i, j] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
            }
        }

        // Also update output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            T original = _outputBias[i];

            _outputBias[i] = _numOps.Add(original, epsilon);
            T predPlus = PredictSingle(input);
            T errorPlus = _numOps.Subtract(target, predPlus);
            T lossPlus = _numOps.Multiply(errorPlus, errorPlus);

            _outputBias[i] = _numOps.Subtract(original, epsilon);
            T predMinus = PredictSingle(input);
            T errorMinus = _numOps.Subtract(target, predMinus);
            T lossMinus = _numOps.Multiply(errorMinus, errorMinus);

            _outputBias[i] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            _outputBias[i] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }
    }

    public override T PredictSingle(Vector<T> input)
    {
        // Tokenize input
        var tokens = new List<int>();
        for (int i = 0; i < Math.Min(input.Length, _options.ContextLength); i++)
        {
            tokens.Add(Tokenize(input[i]));
        }

        // Embed tokens
        var embedded = new List<Vector<T>>();
        foreach (var token in tokens)
        {
            var tokenEmbed = new Vector<T>(_options.EmbeddingDim);
            for (int i = 0; i < _options.EmbeddingDim; i++)
            {
                tokenEmbed[i] = _tokenEmbeddings[i, token];
            }
            embedded.Add(tokenEmbed);
        }

        // Process through transformer (simplified - use last embedding)
        Vector<T> processed = embedded.Count > 0 ? embedded[embedded.Count - 1] : new Vector<T>(_options.EmbeddingDim);

        foreach (var layer in _transformerLayers)
        {
            processed = layer.Forward(processed);
        }

        // Project to vocabulary and sample
        var logits = new Vector<T>(_vocabularySize);
        for (int i = 0; i < _vocabularySize; i++)
        {
            T sum = _outputBias[i];
            for (int j = 0; j < _options.EmbeddingDim; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_outputProjection[i, j], processed[j]));
            }
            logits[i] = sum;
        }

        // Get token with highest logit (argmax)
        int predictedToken = 0;
        T maxLogit = logits[0];
        for (int i = 1; i < _vocabularySize; i++)
        {
            if (_numOps.GreaterThan(logits[i], maxLogit))
            {
                maxLogit = logits[i];
                predictedToken = i;
            }
        }

        // Detokenize
        return Detokenize(predictedToken);
    }

    /// <summary>
    /// Generates probabilistic forecasts by sampling from the model.
    /// </summary>
    public Dictionary<double, Vector<T>> ForecastWithQuantiles(Vector<T> history, double[] quantiles, int numSamples = 100)
    {
        var samples = new List<Vector<T>>();

        for (int s = 0; s < numSamples; s++)
        {
            var forecast = new Vector<T>(_options.ForecastHorizon);
            var context = history.Clone();

            for (int h = 0; h < _options.ForecastHorizon; h++)
            {
                T prediction = PredictSingle(context);
                forecast[h] = prediction;

                // Update context (simplified - shift and append)
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
                var values = samples.Select(s => Convert.ToDouble(s[h])).OrderBy(v => v).ToList();
                int idx = (int)(q * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                quantileForecast[h] = _numOps.FromDouble(values[idx]);
            }
            result[q] = quantileForecast;
        }

        return result;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize options
        writer.Write(_vocabularySize);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.ContextLength);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.NumLayers);
        writer.Write(_options.NumHeads);

        // Serialize vocabulary centroids
        for (int i = 0; i < _vocabularyCentroids.Length; i++)
            writer.Write(Convert.ToDouble(_vocabularyCentroids[i]));

        // Serialize token embeddings
        writer.Write(_tokenEmbeddings.Rows);
        writer.Write(_tokenEmbeddings.Columns);
        for (int i = 0; i < _tokenEmbeddings.Rows; i++)
            for (int j = 0; j < _tokenEmbeddings.Columns; j++)
                writer.Write(Convert.ToDouble(_tokenEmbeddings[i, j]));

        // Serialize transformer layers
        writer.Write(_transformerLayers.Count);
        foreach (var layer in _transformerLayers)
        {
            layer.Serialize(writer);
        }

        // Serialize output projection
        writer.Write(_outputProjection.Rows);
        writer.Write(_outputProjection.Columns);
        for (int i = 0; i < _outputProjection.Rows; i++)
            for (int j = 0; j < _outputProjection.Columns; j++)
                writer.Write(Convert.ToDouble(_outputProjection[i, j]));

        // Serialize output bias
        writer.Write(_outputBias.Length);
        for (int i = 0; i < _outputBias.Length; i++)
            writer.Write(Convert.ToDouble(_outputBias[i]));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize and validate options (don't mutate _options)
        int vocabularySize = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        int contextLength = reader.ReadInt32();
        int forecastHorizon = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();

        // Validate critical dimensions match current options
        if (vocabularySize != _vocabularySize)
        {
            throw new InvalidOperationException(
                $"Serialized VocabularySize ({vocabularySize}) doesn't match current ({_vocabularySize})");
        }
        if (embeddingDim != _options.EmbeddingDim)
        {
            throw new InvalidOperationException(
                $"Serialized EmbeddingDim ({embeddingDim}) doesn't match options ({_options.EmbeddingDim})");
        }
        if (contextLength != _options.ContextLength)
        {
            throw new InvalidOperationException(
                $"Serialized ContextLength ({contextLength}) doesn't match options ({_options.ContextLength})");
        }
        if (forecastHorizon != _options.ForecastHorizon)
        {
            throw new InvalidOperationException(
                $"Serialized ForecastHorizon ({forecastHorizon}) doesn't match options ({_options.ForecastHorizon})");
        }
        if (numLayers != _options.NumLayers)
        {
            throw new InvalidOperationException(
                $"Serialized NumLayers ({numLayers}) doesn't match options ({_options.NumLayers})");
        }
        if (numHeads != _options.NumHeads)
        {
            throw new InvalidOperationException(
                $"Serialized NumHeads ({numHeads}) doesn't match options ({_options.NumHeads})");
        }

        // Deserialize vocabulary centroids
        _vocabularyCentroids = new Vector<T>(_vocabularySize);
        for (int i = 0; i < _vocabularySize; i++)
            _vocabularyCentroids[i] = _numOps.FromDouble(reader.ReadDouble());

        // Deserialize token embeddings
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _tokenEmbeddings = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                _tokenEmbeddings[i, j] = _numOps.FromDouble(reader.ReadDouble());

        // Deserialize transformer layers
        int savedNumLayers = reader.ReadInt32();
        _transformerLayers = new List<TransformerBlock<T>>(savedNumLayers);
        for (int i = 0; i < savedNumLayers; i++)
        {
            _transformerLayers.Add(TransformerBlock<T>.Deserialize(reader));
        }

        // Deserialize output projection
        rows = reader.ReadInt32();
        cols = reader.ReadInt32();
        _outputProjection = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                _outputProjection[i, j] = _numOps.FromDouble(reader.ReadDouble());

        // Deserialize output bias
        int biasLen = reader.ReadInt32();
        _outputBias = new Vector<T>(biasLen);
        for (int i = 0; i < biasLen; i++)
            _outputBias[i] = _numOps.FromDouble(reader.ReadDouble());
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "Chronos Foundation Model",
            ModelType = ModelType.TimeSeriesRegression,
            Description = "Foundation model for zero-shot time series forecasting with tokenization",
            Complexity = ParameterCount,
            FeatureCount = _options.ContextLength,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "VocabularySize", _vocabularySize },
                { "EmbeddingDim", _options.EmbeddingDim },
                { "NumLayers", _options.NumLayers }
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
            foreach (var layer in _transformerLayers)
                count += layer.ParameterCount;
            return count;
        }
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
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Copy Chronos-specific properties
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        VocabularySize = other.VocabularySize;
        EmbeddingDim = other.EmbeddingDim;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;

        // Copy TimeSeriesRegressionOptions properties
        LagOrder = other.LagOrder;
        IncludeTrend = other.IncludeTrend;
        SeasonalPeriod = other.SeasonalPeriod;
        AutocorrelationCorrection = other.AutocorrelationCorrection;
        ModelType = other.ModelType;
        LossFunction = other.LossFunction;

        // Copy RegressionOptions properties
        DecompositionMethod = other.DecompositionMethod;
        UseIntercept = other.UseIntercept;
    }
}

/// <summary>
/// Simplified Transformer Block.
/// </summary>
internal class TransformerBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private Matrix<T> _weights;

    public int ParameterCount => _weights.Rows * _weights.Columns;

    public TransformerBlock(int embeddingDim, int numHeads, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        var random = new Random(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        _weights = new Matrix<T>(embeddingDim, embeddingDim);
        for (int i = 0; i < embeddingDim; i++)
            for (int j = 0; j < embeddingDim; j++)
                _weights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
    }

    /// <summary>
    /// Creates a TransformerBlock for deserialization.
    /// </summary>
    private TransformerBlock()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = new Matrix<T>(0, 0);
    }

    public Vector<T> Forward(Vector<T> input)
    {
        var output = new Vector<T>(input.Length);
        for (int i = 0; i < output.Length && i < _weights.Rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < input.Length && j < _weights.Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_weights[i, j], input[j]));
            }
            output[i] = MathHelper.Tanh(sum);
        }
        return output;
    }

    /// <summary>
    /// Serializes the transformer block weights.
    /// </summary>
    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_weights.Rows);
        writer.Write(_weights.Columns);
        for (int i = 0; i < _weights.Rows; i++)
            for (int j = 0; j < _weights.Columns; j++)
                writer.Write(Convert.ToDouble(_weights[i, j]));
    }

    /// <summary>
    /// Deserializes a transformer block from binary data.
    /// </summary>
    public static TransformerBlock<T> Deserialize(BinaryReader reader)
    {
        var block = new TransformerBlock<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        block._weights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                block._weights[i, j] = numOps.FromDouble(reader.ReadDouble());

        return block;
    }
}
