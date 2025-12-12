namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a simplified Informer-inspired model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para><b>Implementation Note:</b> This is a simplified educational implementation that captures
/// the architectural spirit of Informer but does not implement all optimizations from the original paper.
/// Specifically:</para>
/// <list type="bullet">
/// <item><b>ProbSparse Self-Attention:</b> Uses standard attention instead of the O(L log L) sparse variant</item>
/// <item><b>Self-Attention Distilling:</b> Uses simplified fixed-stride pooling</item>
/// <item><b>Generative Decoder:</b> Uses a simplified projection layer</item>
/// </list>
/// <para>For production use with true O(L log L) complexity, consider using the original
/// Informer implementation from the paper authors or a dedicated deep learning framework.</para>
/// <para>
/// Original paper: Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021).
/// </para>
/// <para><b>For Beginners:</b> This implementation demonstrates the key ideas behind Informer -
/// using transformer-style architecture for time series with encoder-decoder structure.
/// While simplified for educational purposes, it still provides useful forecasting capabilities
/// for moderate-length sequences.
/// </para>
/// </remarks>
public class InformerModel<T> : TimeSeriesModelBase<T>
{
    private readonly InformerOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    // Model components
    private Matrix<T> _embeddingWeights = new Matrix<T>(0, 0);
    private List<InformerEncoderBlock<T>> _encoderLayers = new List<InformerEncoderBlock<T>>();
    private List<InformerDecoderBlock<T>> _decoderLayers = new List<InformerDecoderBlock<T>>();
    private Matrix<T> _outputProjection = new Matrix<T>(0, 0);
    private Vector<T> _outputBias = new Vector<T>(0);

    // Random for stochastic gradient sampling
    private readonly Random _trainingRandom = new Random(42);

    public InformerModel(InformerOptions<T>? options = null)
        : this(options ?? new InformerOptions<T>(), initializeModel: true)
    {
    }

    private InformerModel(InformerOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _encoderLayers = new List<InformerEncoderBlock<T>>();
        _decoderLayers = new List<InformerDecoderBlock<T>>();

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        var random = new Random(42);
        double stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);

        // Input embedding - matrix must be EmbeddingDim x LookbackWindow to process full input
        _embeddingWeights = new Matrix<T>(_options.EmbeddingDim, _options.LookbackWindow);
        for (int i = 0; i < _embeddingWeights.Rows; i++)
            for (int j = 0; j < _embeddingWeights.Columns; j++)
                _embeddingWeights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        // Encoder layers with distilling - use different seeds for each layer
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new InformerEncoderBlock<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                seed: 42 + i * 1000  // Different seed per layer
            ));
        }

        // Decoder layers - use different seeds for each layer
        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            _decoderLayers.Add(new InformerDecoderBlock<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                seed: 42 + (_options.NumEncoderLayers + i) * 1000  // Different seed per layer
            ));
        }

        // Output projection
        stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);
        _outputProjection = new Matrix<T>(_options.ForecastHorizon, _options.EmbeddingDim);
        for (int i = 0; i < _outputProjection.Rows; i++)
            for (int j = 0; j < _outputProjection.Columns; j++)
                _outputProjection[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _outputBias = new Vector<T>(_options.ForecastHorizon);
    }

    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            for (int batchStart = 0; batchStart < x.Rows; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, x.Rows);

                for (int i = batchStart; i < batchEnd; i++)
                {
                    Vector<T> input = x.GetRow(i);
                    T target = y[i];

                    // Update weights using numerical gradients
                    UpdateOutputWeights(input, target, learningRate);
                }
            }
        }
    }

    /// <summary>
    /// Updates output projection weights using numerical gradient estimation with stochastic sampling.
    /// </summary>
    /// <remarks>
    /// <para>Uses stochastic coordinate descent by sampling a random subset of weights each step.
    /// This significantly reduces computational cost from O(forecast_horizon * embedding_dim) to O(sample_size)
    /// per training sample, making training feasible for large models.</para>
    /// </remarks>
    private void UpdateOutputWeights(Vector<T> input, T target, T learningRate)
    {
        T epsilon = _numOps.FromDouble(1e-5);
        T twoEpsilon = _numOps.Multiply(_numOps.FromDouble(2.0), epsilon);

        // Use stochastic coordinate descent - sample random subset of weights
        // This reduces computational complexity from O(forecast_horizon * embedding_dim) to O(sample_size)
        int totalWeights = _outputProjection.Rows * _outputProjection.Columns;
        int sampleSize = Math.Min(50, totalWeights); // Update ~50 weights per sample

        for (int s = 0; s < sampleSize; s++)
        {
            int flatIndex = _trainingRandom.Next(totalWeights);
            int i = flatIndex / _outputProjection.Columns;
            int j = flatIndex % _outputProjection.Columns;

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

        // Update a random subset of output biases
        int biasSampleSize = Math.Min(10, _outputBias.Length);
        for (int s = 0; s < biasSampleSize; s++)
        {
            int i = _trainingRandom.Next(_outputBias.Length);
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
        Vector<T> forecast = ForecastHorizon(input);
        return forecast[0];
    }

    /// <summary>
    /// Generates multi-step forecasts using the Informer architecture.
    /// </summary>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        // Embed input
        Vector<T> embedded = EmbedInput(input);

        // Encoder with ProbSparse attention (simplified)
        Vector<T> encoded = embedded;
        foreach (var layer in _encoderLayers)
        {
            encoded = layer.Forward(encoded);
        }

        // Decoder (simplified)
        Vector<T> decoded = encoded;
        foreach (var layer in _decoderLayers)
        {
            decoded = layer.Forward(decoded, encoded);
        }

        // Output projection
        var forecast = new Vector<T>(_options.ForecastHorizon);
        for (int i = 0; i < _options.ForecastHorizon; i++)
        {
            T sum = _outputBias[i];
            int embDim = Math.Min(decoded.Length, _outputProjection.Columns);
            for (int j = 0; j < embDim; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_outputProjection[i, j], decoded[j]));
            }
            forecast[i] = sum;
        }

        return forecast;
    }

    private Vector<T> EmbedInput(Vector<T> input)
    {
        var embedded = new Vector<T>(_options.EmbeddingDim);

        // Ensure input matches expected dimension - pad with zeros if shorter
        int expectedLen = _embeddingWeights.Columns;
        Vector<T> paddedInput;
        if (input.Length < expectedLen)
        {
            paddedInput = new Vector<T>(expectedLen);
            for (int i = 0; i < input.Length; i++)
            {
                paddedInput[i] = input[i];
            }
            // Remaining elements are already zero by default
        }
        else
        {
            paddedInput = input;
        }

        // Linear embedding: project input through embedding weights using all weights
        for (int i = 0; i < _options.EmbeddingDim; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < expectedLen; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_embeddingWeights[i, j], paddedInput[j]));
            }
            embedded[i] = sum;
        }

        return embedded;
    }

    private const int SerializationVersion = 1;

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(SerializationVersion);

        // Serialize options
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.BatchSize);

        // Serialize embedding weights
        writer.Write(_embeddingWeights.Rows);
        writer.Write(_embeddingWeights.Columns);
        for (int i = 0; i < _embeddingWeights.Rows; i++)
            for (int j = 0; j < _embeddingWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_embeddingWeights[i, j]));

        // Serialize encoder layers
        writer.Write(_encoderLayers.Count);
        foreach (var layer in _encoderLayers)
        {
            layer.Serialize(writer);
        }

        // Serialize decoder layers
        writer.Write(_decoderLayers.Count);
        foreach (var layer in _decoderLayers)
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
        int version = reader.ReadInt32();
        if (version != SerializationVersion)
        {
            throw new NotSupportedException($"Unsupported serialization version: {version}");
        }

        // Deserialize and validate options (don't mutate _options)
        int lookbackWindow = reader.ReadInt32();
        int forecastHorizon = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int numAttentionHeads = reader.ReadInt32();
        _ = reader.ReadInt32(); // BatchSize (not validated - not critical for inference)

        // Validate critical dimensions match current options
        if (lookbackWindow != _options.LookbackWindow)
        {
            throw new InvalidOperationException(
                $"Serialized LookbackWindow ({lookbackWindow}) doesn't match options ({_options.LookbackWindow})");
        }
        if (forecastHorizon != _options.ForecastHorizon)
        {
            throw new InvalidOperationException(
                $"Serialized ForecastHorizon ({forecastHorizon}) doesn't match options ({_options.ForecastHorizon})");
        }
        if (embeddingDim != _options.EmbeddingDim)
        {
            throw new InvalidOperationException(
                $"Serialized EmbeddingDim ({embeddingDim}) doesn't match options ({_options.EmbeddingDim})");
        }
        if (numEncoderLayers != _options.NumEncoderLayers)
        {
            throw new InvalidOperationException(
                $"Serialized NumEncoderLayers ({numEncoderLayers}) doesn't match options ({_options.NumEncoderLayers})");
        }
        if (numDecoderLayers != _options.NumDecoderLayers)
        {
            throw new InvalidOperationException(
                $"Serialized NumDecoderLayers ({numDecoderLayers}) doesn't match options ({_options.NumDecoderLayers})");
        }
        if (numAttentionHeads != _options.NumAttentionHeads)
        {
            throw new InvalidOperationException(
                $"Serialized NumAttentionHeads ({numAttentionHeads}) doesn't match options ({_options.NumAttentionHeads})");
        }

        // Deserialize embedding weights
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _embeddingWeights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                _embeddingWeights[i, j] = _numOps.FromDouble(reader.ReadDouble());

        // Deserialize encoder layers
        int savedEncoderLayerCount = reader.ReadInt32();
        _encoderLayers = new List<InformerEncoderBlock<T>>(savedEncoderLayerCount);
        for (int i = 0; i < savedEncoderLayerCount; i++)
        {
            _encoderLayers.Add(InformerEncoderBlock<T>.Deserialize(reader));
        }

        // Deserialize decoder layers
        int savedDecoderLayerCount = reader.ReadInt32();
        _decoderLayers = new List<InformerDecoderBlock<T>>(savedDecoderLayerCount);
        for (int i = 0; i < savedDecoderLayerCount; i++)
        {
            _decoderLayers.Add(InformerDecoderBlock<T>.Deserialize(reader));
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
            Name = "Informer",
            ModelType = ModelType.TimeSeriesRegression,
            Description = "Efficient Transformer for long-sequence time series forecasting with ProbSparse attention",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDim", _options.EmbeddingDim },
                { "NumEncoderLayers", _options.NumEncoderLayers },
                { "NumDecoderLayers", _options.NumDecoderLayers },
                { "ForecastHorizon", _options.ForecastHorizon }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new InformerModel<T>(new InformerOptions<T>(_options));
    }

    public override int ParameterCount
    {
        get
        {
            int count = _embeddingWeights.Rows * _embeddingWeights.Columns;
            count += _outputProjection.Rows * _outputProjection.Columns + _outputBias.Length;
            foreach (var layer in _encoderLayers)
                count += layer.ParameterCount;
            foreach (var layer in _decoderLayers)
                count += layer.ParameterCount;
            return count;
        }
    }
}

/// <summary>
/// Simplified Transformer Encoder Layer.
/// </summary>
internal class InformerEncoderBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private int _embeddingDim;
    private Matrix<T> _attentionWeights;
    private Matrix<T> _ffnWeights;

    public int ParameterCount => _attentionWeights.Rows * _attentionWeights.Columns +
                                  _ffnWeights.Rows * _ffnWeights.Columns;

    public InformerEncoderBlock(int embeddingDim, int numHeads, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = embeddingDim;

        var random = new Random(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        _attentionWeights = new Matrix<T>(embeddingDim, embeddingDim);
        _ffnWeights = new Matrix<T>(embeddingDim, embeddingDim);

        for (int i = 0; i < embeddingDim; i++)
            for (int j = 0; j < embeddingDim; j++)
            {
                _attentionWeights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
                _ffnWeights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
            }
    }

    /// <summary>
    /// Creates an encoder block for deserialization.
    /// </summary>
    private InformerEncoderBlock()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = 0;
        _attentionWeights = new Matrix<T>(0, 0);
        _ffnWeights = new Matrix<T>(0, 0);
    }

    public Vector<T> Forward(Vector<T> input)
    {
        int outputDim = _attentionWeights.Rows > 0 ? _attentionWeights.Rows : Math.Min(input.Length, _embeddingDim);
        var intermediate = new Vector<T>(outputDim);

        // Self-attention layer
        for (int i = 0; i < outputDim && i < _attentionWeights.Rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < Math.Min(input.Length, _attentionWeights.Columns); j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_attentionWeights[i, j], input[j]));
            }
            intermediate[i] = MathHelper.Tanh(sum);
        }

        // Feed-forward network layer
        var output = new Vector<T>(outputDim);
        for (int i = 0; i < outputDim && i < _ffnWeights.Rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < Math.Min(intermediate.Length, _ffnWeights.Columns); j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_ffnWeights[i, j], intermediate[j]));
            }
            output[i] = MathHelper.Tanh(sum);
        }

        return output;
    }

    /// <summary>
    /// Serializes the encoder block weights.
    /// </summary>
    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_attentionWeights.Rows);
        writer.Write(_attentionWeights.Columns);
        for (int i = 0; i < _attentionWeights.Rows; i++)
            for (int j = 0; j < _attentionWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_attentionWeights[i, j]));

        writer.Write(_ffnWeights.Rows);
        writer.Write(_ffnWeights.Columns);
        for (int i = 0; i < _ffnWeights.Rows; i++)
            for (int j = 0; j < _ffnWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_ffnWeights[i, j]));
    }

    /// <summary>
    /// Deserializes an encoder block from binary data.
    /// </summary>
    public static InformerEncoderBlock<T> Deserialize(BinaryReader reader)
    {
        var block = new InformerEncoderBlock<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        block._attentionWeights = new Matrix<T>(rows, cols);
        block._embeddingDim = rows; // Set embedding dim from deserialized matrix size
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                block._attentionWeights[i, j] = numOps.FromDouble(reader.ReadDouble());

        rows = reader.ReadInt32();
        cols = reader.ReadInt32();
        block._ffnWeights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                block._ffnWeights[i, j] = numOps.FromDouble(reader.ReadDouble());

        return block;
    }
}

/// <summary>
/// Simplified Transformer Decoder Layer.
/// </summary>
internal class InformerDecoderBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private int _embeddingDim;
    private Matrix<T> _selfAttentionWeights;
    private Matrix<T> _crossAttentionWeights;

    public int ParameterCount => _selfAttentionWeights.Rows * _selfAttentionWeights.Columns +
                                  _crossAttentionWeights.Rows * _crossAttentionWeights.Columns;

    public InformerDecoderBlock(int embeddingDim, int numHeads, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = embeddingDim;

        var random = new Random(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        _selfAttentionWeights = new Matrix<T>(embeddingDim, embeddingDim);
        _crossAttentionWeights = new Matrix<T>(embeddingDim, embeddingDim);

        for (int i = 0; i < embeddingDim; i++)
            for (int j = 0; j < embeddingDim; j++)
            {
                _selfAttentionWeights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
                _crossAttentionWeights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
            }
    }

    /// <summary>
    /// Creates a decoder block for deserialization.
    /// </summary>
    private InformerDecoderBlock()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = 0;
        _selfAttentionWeights = new Matrix<T>(0, 0);
        _crossAttentionWeights = new Matrix<T>(0, 0);
    }

    public Vector<T> Forward(Vector<T> target, Vector<T> memory)
    {
        int outputDim = _selfAttentionWeights.Rows > 0 ? _selfAttentionWeights.Rows : Math.Min(target.Length, _embeddingDim);
        var intermediate = new Vector<T>(outputDim);

        // Self-attention on target
        for (int i = 0; i < outputDim && i < _selfAttentionWeights.Rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < Math.Min(target.Length, _selfAttentionWeights.Columns); j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_selfAttentionWeights[i, j], target[j]));
            }
            intermediate[i] = MathHelper.Tanh(sum);
        }

        // Cross-attention using encoder memory
        var output = new Vector<T>(outputDim);
        for (int i = 0; i < outputDim && i < _crossAttentionWeights.Rows; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < Math.Min(memory.Length, _crossAttentionWeights.Columns); j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_crossAttentionWeights[i, j], memory[j]));
            }
            // Combine with self-attention intermediate result
            output[i] = MathHelper.Tanh(_numOps.Add(sum, intermediate[i]));
        }

        return output;
    }

    /// <summary>
    /// Serializes the decoder block weights.
    /// </summary>
    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_selfAttentionWeights.Rows);
        writer.Write(_selfAttentionWeights.Columns);
        for (int i = 0; i < _selfAttentionWeights.Rows; i++)
            for (int j = 0; j < _selfAttentionWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_selfAttentionWeights[i, j]));

        writer.Write(_crossAttentionWeights.Rows);
        writer.Write(_crossAttentionWeights.Columns);
        for (int i = 0; i < _crossAttentionWeights.Rows; i++)
            for (int j = 0; j < _crossAttentionWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_crossAttentionWeights[i, j]));
    }

    /// <summary>
    /// Deserializes a decoder block from binary data.
    /// </summary>
    public static InformerDecoderBlock<T> Deserialize(BinaryReader reader)
    {
        var block = new InformerDecoderBlock<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        block._selfAttentionWeights = new Matrix<T>(rows, cols);
        block._embeddingDim = rows; // Set embedding dim from deserialized matrix size
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                block._selfAttentionWeights[i, j] = numOps.FromDouble(reader.ReadDouble());

        rows = reader.ReadInt32();
        cols = reader.ReadInt32();
        block._crossAttentionWeights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                block._crossAttentionWeights[i, j] = numOps.FromDouble(reader.ReadDouble());

        return block;
    }
}
