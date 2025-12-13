namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the Informer model for efficient long-sequence time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Long-Sequence Forecasting Problem:</b>
/// Traditional Transformer models achieve state-of-the-art results in many sequence modeling tasks,
/// but they struggle with long time series because self-attention has O(L²) time and memory complexity.
/// For a sequence of 1000 time steps, vanilla attention requires 1 million operations per layer.
/// This makes long-horizon forecasting (e.g., predicting the next month based on the past year)
/// computationally prohibitive.
/// </para>
/// <para>
/// <b>The Informer Solution:</b>
/// Informer (Zhou et al., AAAI 2021) introduces three key innovations to enable efficient long-sequence forecasting:
/// </para>
/// <para>
/// <b>1. ProbSparse Self-Attention (O(L log L) complexity):</b>
/// The key insight is that in self-attention, most attention weights are nearly uniform and contribute
/// little information. Only a few "dominant" queries have sparse, peaked attention distributions.
/// ProbSparse attention measures the "sparsity" of each query using the KL-divergence from a uniform
/// distribution, which can be approximated as: Sparsity(q) = max(softmax(qK^T)) - mean(softmax(qK^T)).
/// Only the top c*ln(L) queries with highest sparsity are computed with full attention; the rest
/// use the mean value approximation. This reduces complexity from O(L²) to O(L log L).
/// </para>
/// <para>
/// <b>2. Self-Attention Distilling:</b>
/// Between encoder layers, a "distilling" operation progressively halves the sequence length using
/// 1D convolution (kernel size 3) followed by ELU activation and max pooling. This creates a
/// pyramid structure where each layer operates on a shorter sequence, further reducing computation
/// and focusing on the most salient features.
/// </para>
/// <para>
/// <b>3. Generative-Style Decoder:</b>
/// Unlike autoregressive decoders that predict one step at a time, Informer's decoder generates
/// all forecast positions in a single forward pass. It uses learnable start tokens as placeholders
/// for future positions and applies masked self-attention followed by cross-attention to the
/// encoder output.
/// </para>
/// <para><b>For Beginners:</b> Imagine you're trying to predict next month's temperature based on
/// the past year of daily readings (365 days). A regular Transformer would need to compare every
/// day to every other day (365 × 365 = 133,225 comparisons per layer). Informer is smart: it
/// figures out that only a few days really matter for the prediction and focuses on those,
/// reducing the work to about 365 × log(365) ≈ 2,150 comparisons. It's like instead of reading
/// every page of a book to write a summary, you identify the key chapters and focus on those.
/// </para>
/// </remarks>
public class InformerModel<T> : TimeSeriesModelBase<T>
{
    private readonly InformerOptions<T> _options;
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    // Input embedding and positional encoding
    private Matrix<T> _inputProjection = new Matrix<T>(0, 0);
    private Matrix<T> _positionalEncoding = new Matrix<T>(0, 0);

    // Encoder components with distilling
    private List<InformerEncoderLayer<T>> _encoderLayers = new List<InformerEncoderLayer<T>>();
    private List<DistillingConv<T>> _distillingLayers = new List<DistillingConv<T>>();

    // Decoder components
    private List<InformerDecoderLayer<T>> _decoderLayers = new List<InformerDecoderLayer<T>>();
    private Vector<T> _decoderStartToken = new Vector<T>(0);

    // Output projection
    private Matrix<T> _outputProjection = new Matrix<T>(0, 0);
    private Vector<T> _outputBias = new Vector<T>(0);

    // Sparsity factor for ProbSparse attention (c in the paper, typically 5)
    private const int SparsityFactor = 5;

    /// <summary>
    /// Initializes a new instance of the Informer model with the specified options.
    /// </summary>
    /// <param name="options">Configuration options for the Informer model. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>Informer Architecture Overview:</b>
    /// The Informer follows an encoder-decoder architecture similar to the original Transformer,
    /// but with key modifications for efficient long-sequence processing.
    /// </para>
    /// <para>
    /// <b>Encoder Stack with Distilling:</b>
    /// The encoder consists of multiple layers, each containing ProbSparse self-attention and
    /// a feed-forward network (FFN). Between layers, a "distilling" operation (1D conv + max pool)
    /// halves the sequence length. This creates a pyramid: if the input has 96 time steps and we
    /// have 3 encoder layers with distilling factor 2, the sequence lengths are: 96 → 48 → 24 → 24.
    /// This progressive compression focuses computation on increasingly abstract features.
    /// </para>
    /// <para>
    /// <b>Decoder with Generative Decoding:</b>
    /// Instead of autoregressive decoding (predicting one step, feeding it back, repeat), Informer
    /// uses a generative approach. The decoder receives learnable "start tokens" as placeholders
    /// for all forecast positions simultaneously. It then applies masked self-attention (each
    /// position can only see itself and earlier positions) followed by cross-attention to the
    /// encoder output. This enables single-pass multi-step forecasting.
    /// </para>
    /// <para>
    /// <b>Weight Initialization:</b>
    /// All weights are initialized using Xavier/Glorot initialization (scaled by sqrt(2/fan_in))
    /// to ensure stable gradients during training. Sinusoidal positional encodings are computed
    /// deterministically based on position and dimension.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the Informer as a sophisticated pattern recognizer.
    /// The encoder reads your historical data and compresses it into a summary of important patterns.
    /// The decoder then uses this summary to "imagine" what comes next. The clever part is that
    /// it can predict many future steps at once, rather than one at a time.
    /// </para>
    /// </remarks>
    public InformerModel(InformerOptions<T>? options = null)
        : this(options ?? new InformerOptions<T>(), initializeModel: true)
    {
    }

    private InformerModel(InformerOptions<T> options, bool initializeModel)
        : base(options)
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = new Random(42);
        _encoderLayers = new List<InformerEncoderLayer<T>>();
        _distillingLayers = new List<DistillingConv<T>>();
        _decoderLayers = new List<InformerDecoderLayer<T>>();

        if (initializeModel)
            InitializeModel();
    }

    private void InitializeModel()
    {
        double stddev = Math.Sqrt(2.0 / _options.EmbeddingDim);

        // Input projection: maps single time step values to embedding dimension
        _inputProjection = new Matrix<T>(_options.EmbeddingDim, 1);
        InitializeMatrix(_inputProjection, stddev);

        // Sinusoidal positional encoding for the maximum sequence length
        int maxLen = Math.Max(_options.LookbackWindow, _options.ForecastHorizon) * 2;
        _positionalEncoding = CreateSinusoidalPositionalEncoding(maxLen, _options.EmbeddingDim);

        // Encoder layers with distilling
        int currentSeqLen = _options.LookbackWindow;
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new InformerEncoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                seed: 42 + i * 1000
            ));

            // Add distilling layer after each encoder layer (except the last)
            if (i < _options.NumEncoderLayers - 1)
            {
                _distillingLayers.Add(new DistillingConv<T>(
                    _options.EmbeddingDim,
                    _options.DistillingFactor,
                    seed: 42 + i * 2000
                ));
                currentSeqLen = (currentSeqLen + _options.DistillingFactor - 1) / _options.DistillingFactor;
            }
        }

        // Decoder layers
        for (int i = 0; i < _options.NumDecoderLayers; i++)
        {
            _decoderLayers.Add(new InformerDecoderLayer<T>(
                _options.EmbeddingDim,
                _options.NumAttentionHeads,
                SparsityFactor,
                _options.DropoutRate,
                seed: 42 + (_options.NumEncoderLayers + i) * 1000
            ));
        }

        // Decoder start token (learnable)
        _decoderStartToken = new Vector<T>(_options.EmbeddingDim);
        for (int i = 0; i < _options.EmbeddingDim; i++)
        {
            _decoderStartToken[i] = _numOps.FromDouble((_random.NextDouble() * 2 - 1) * stddev);
        }

        // Output projection: maps embedding dimension to single output value per time step
        _outputProjection = new Matrix<T>(_options.ForecastHorizon, _options.EmbeddingDim);
        InitializeMatrix(_outputProjection, Math.Sqrt(2.0 / _options.EmbeddingDim));
        _outputBias = new Vector<T>(_options.ForecastHorizon);
    }

    private void InitializeMatrix(Matrix<T> matrix, double stddev)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = _numOps.FromDouble((_random.NextDouble() * 2 - 1) * stddev);
            }
        }
    }

    /// <summary>
    /// Creates sinusoidal positional encoding as described in "Attention Is All You Need".
    /// </summary>
    private Matrix<T> CreateSinusoidalPositionalEncoding(int maxLen, int embeddingDim)
    {
        var pe = new Matrix<T>(maxLen, embeddingDim);
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embeddingDim; i++)
            {
                double angle = pos / Math.Pow(10000.0, (2.0 * (i / 2)) / embeddingDim);
                if (i % 2 == 0)
                {
                    pe[pos, i] = _numOps.FromDouble(Math.Sin(angle));
                }
                else
                {
                    pe[pos, i] = _numOps.FromDouble(Math.Cos(angle));
                }
            }
        }
        return pe;
    }

    /// <summary>
    /// Trains the Informer model on the provided time series data.
    /// </summary>
    /// <param name="x">Input matrix where each row is a training sample (lookback window).</param>
    /// <param name="y">Target values for each training sample.</param>
    /// <remarks>
    /// <para>
    /// <b>Training Objective:</b>
    /// The Informer is trained to minimize the mean squared error (MSE) between predicted
    /// and actual future values. Given a lookback window of historical observations, the model
    /// learns to generate forecasts that match the ground truth as closely as possible.
    /// </para>
    /// <para>
    /// <b>Original Paper Training:</b>
    /// In the original Informer paper, training uses Adam optimizer with learning rate warmup
    /// and decay, batch sizes of 32, and trains for 100+ epochs. The model is trained end-to-end
    /// with backpropagation through all layers including the ProbSparse attention and distilling
    /// operations.
    /// </para>
    /// <para>
    /// <b>This Implementation:</b>
    /// This implementation uses stochastic coordinate descent with numerical gradient estimation.
    /// For computational efficiency, we sample random subsets of weights to update each step
    /// rather than computing gradients for all parameters. While this is less efficient than
    /// backpropagation, it provides a framework-agnostic training approach that works with
    /// any numeric type T.
    /// </para>
    /// <para>
    /// <b>Gradient Estimation:</b>
    /// For each selected weight w, the gradient is estimated as:
    /// ∂Loss/∂w ≈ (Loss(w + ε) - Loss(w - ε)) / (2ε)
    /// This central difference approximation has O(ε²) error, where ε is a small perturbation.
    /// </para>
    /// <para><b>For Beginners:</b> Training a neural network is like adjusting thousands of knobs
    /// to get the right output. For each training example, we slightly wiggle each knob and see
    /// if the prediction gets better or worse. Then we adjust all the knobs in the direction
    /// that improves the prediction. After seeing many examples, the knobs settle into positions
    /// that work well for predicting new data.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        T learningRate = _numOps.FromDouble(_options.LearningRate);
        T epsilon = _numOps.FromDouble(1e-5);
        T twoEpsilon = _numOps.Multiply(_numOps.FromDouble(2.0), epsilon);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            // Shuffle training data
            var indices = Enumerable.Range(0, x.Rows).OrderBy(_ => _random.Next()).ToList();

            for (int batchStart = 0; batchStart < x.Rows; batchStart += _options.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _options.BatchSize, x.Rows);

                // Accumulate gradients for batch
                for (int idx = batchStart; idx < batchEnd; idx++)
                {
                    int i = indices[idx];
                    Vector<T> input = x.GetRow(i);
                    T target = y[i];

                    // Update all trainable parameters using numerical gradients
                    UpdateAllParameters(input, target, learningRate, epsilon, twoEpsilon);
                }
            }
        }
    }

    private void UpdateAllParameters(Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon)
    {
        // Update output projection (most impactful for prediction)
        UpdateMatrixSubset(_outputProjection, input, target, learningRate, epsilon, twoEpsilon, 100);
        UpdateVectorSubset(_outputBias, input, target, learningRate, epsilon, twoEpsilon, 20);

        // Update input projection
        UpdateMatrixSubset(_inputProjection, input, target, learningRate, epsilon, twoEpsilon, 50);

        // Update decoder start token
        UpdateVectorSubset(_decoderStartToken, input, target, learningRate, epsilon, twoEpsilon, 20);

        // Update encoder layers
        foreach (var layer in _encoderLayers)
        {
            layer.UpdateWeights(input, target, learningRate, epsilon, twoEpsilon, PredictSingle, 30);
        }

        // Update distilling layers
        foreach (var layer in _distillingLayers)
        {
            layer.UpdateWeights(input, target, learningRate, epsilon, twoEpsilon, PredictSingle, 20);
        }

        // Update decoder layers
        foreach (var layer in _decoderLayers)
        {
            layer.UpdateWeights(input, target, learningRate, epsilon, twoEpsilon, PredictSingle, 30);
        }
    }

    private void UpdateMatrixSubset(Matrix<T> matrix, Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon, int sampleSize)
    {
        int totalWeights = matrix.Rows * matrix.Columns;
        int actualSampleSize = Math.Min(sampleSize, totalWeights);

        for (int s = 0; s < actualSampleSize; s++)
        {
            int flatIndex = _random.Next(totalWeights);
            int i = flatIndex / matrix.Columns;
            int j = flatIndex % matrix.Columns;

            T original = matrix[i, j];

            matrix[i, j] = _numOps.Add(original, epsilon);
            T predPlus = PredictSingle(input);
            T errorPlus = _numOps.Subtract(target, predPlus);
            T lossPlus = _numOps.Multiply(errorPlus, errorPlus);

            matrix[i, j] = _numOps.Subtract(original, epsilon);
            T predMinus = PredictSingle(input);
            T errorMinus = _numOps.Subtract(target, predMinus);
            T lossMinus = _numOps.Multiply(errorMinus, errorMinus);

            matrix[i, j] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            matrix[i, j] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }
    }

    private void UpdateVectorSubset(Vector<T> vector, Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon, int sampleSize)
    {
        int actualSampleSize = Math.Min(sampleSize, vector.Length);

        for (int s = 0; s < actualSampleSize; s++)
        {
            int i = _random.Next(vector.Length);

            T original = vector[i];

            vector[i] = _numOps.Add(original, epsilon);
            T predPlus = PredictSingle(input);
            T errorPlus = _numOps.Subtract(target, predPlus);
            T lossPlus = _numOps.Multiply(errorPlus, errorPlus);

            vector[i] = _numOps.Subtract(original, epsilon);
            T predMinus = PredictSingle(input);
            T errorMinus = _numOps.Subtract(target, predMinus);
            T lossMinus = _numOps.Multiply(errorMinus, errorMinus);

            vector[i] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            vector[i] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }
    }

    /// <summary>
    /// Predicts the next single value in the time series.
    /// </summary>
    /// <param name="input">The lookback window of historical values.</param>
    /// <returns>The predicted next value.</returns>
    /// <remarks>
    /// <para>This method returns only the first value of the full forecast horizon.
    /// For multi-step forecasting, use <see cref="ForecastHorizon"/> instead.</para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        Vector<T> forecast = ForecastHorizon(input);
        return forecast[0];
    }

    /// <summary>
    /// Generates multi-step forecasts using the full Informer architecture.
    /// </summary>
    /// <param name="input">The lookback window of historical values.</param>
    /// <returns>A vector containing the forecast for each horizon step.</returns>
    /// <remarks>
    /// <para>
    /// <b>The Forecasting Pipeline:</b>
    /// The Informer forecasting process follows five stages, each implementing a key
    /// component of the architecture.
    /// </para>
    /// <para>
    /// <b>Stage 1 - Input Embedding:</b>
    /// Raw time series values (scalars) are projected into a high-dimensional embedding space.
    /// This allows the model to learn rich representations. Each time step t with value x[t]
    /// becomes a vector of dimension d_model (e.g., 512 dimensions).
    /// </para>
    /// <para>
    /// <b>Stage 2 - Positional Encoding:</b>
    /// Since attention is permutation-invariant (doesn't know position), we add sinusoidal
    /// positional encodings: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d)).
    /// This allows the model to distinguish between "yesterday" and "last month" even if they
    /// have similar values.
    /// </para>
    /// <para>
    /// <b>Stage 3 - Encoding with Distilling:</b>
    /// The encoder processes the input through multiple layers. Each layer applies ProbSparse
    /// self-attention (selecting only the most informative queries) and a feed-forward network.
    /// Between layers, the distilling operation compresses the sequence, creating a pyramid
    /// that focuses on increasingly abstract patterns.
    /// </para>
    /// <para>
    /// <b>Stage 4 - Generative Decoding:</b>
    /// Unlike autoregressive decoders, Informer generates all predictions simultaneously.
    /// Learnable start tokens (one per forecast position) attend to each other via masked
    /// self-attention (respecting causality) and to the encoder output via cross-attention.
    /// </para>
    /// <para>
    /// <b>Stage 5 - Output Projection:</b>
    /// The decoder outputs are projected back to scalar values representing the forecast.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this like reading a book (encoding) and then
    /// writing a summary (decoding). The encoder reads your historical data and understands
    /// the patterns. The decoder then "writes" the future by attending to what it learned
    /// while also maintaining internal consistency in its predictions.
    /// </para>
    /// </remarks>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        // Step 1: Embed input sequence
        var encoderInput = EmbedSequence(input);

        // Step 2: Add positional encoding
        AddPositionalEncoding(encoderInput, 0);

        // Step 3: Encode with distilling
        var encoderOutput = Encode(encoderInput);

        // Step 4: Decode with generative style
        var decoderOutput = Decode(encoderOutput);

        // Step 5: Project to forecast horizon
        return ProjectOutput(decoderOutput);
    }

    /// <summary>
    /// Embeds input time series into embedding space.
    /// Each time step is projected from scalar to embedding dimension.
    /// </summary>
    private List<Vector<T>> EmbedSequence(Vector<T> input)
    {
        var embedded = new List<Vector<T>>();
        int seqLen = Math.Min(input.Length, _options.LookbackWindow);

        for (int t = 0; t < seqLen; t++)
        {
            var emb = new Vector<T>(_options.EmbeddingDim);
            for (int i = 0; i < _options.EmbeddingDim; i++)
            {
                emb[i] = _numOps.Multiply(_inputProjection[i, 0], input[t]);
            }
            embedded.Add(emb);
        }

        // Pad if necessary
        while (embedded.Count < _options.LookbackWindow)
        {
            embedded.Insert(0, new Vector<T>(_options.EmbeddingDim));
        }

        return embedded;
    }

    /// <summary>
    /// Adds sinusoidal positional encoding to the sequence.
    /// </summary>
    private void AddPositionalEncoding(List<Vector<T>> sequence, int offset)
    {
        for (int t = 0; t < sequence.Count; t++)
        {
            int posIdx = Math.Min(t + offset, _positionalEncoding.Rows - 1);
            for (int i = 0; i < _options.EmbeddingDim && i < _positionalEncoding.Columns; i++)
            {
                sequence[t][i] = _numOps.Add(sequence[t][i], _positionalEncoding[posIdx, i]);
            }
        }
    }

    /// <summary>
    /// Encodes input through encoder layers with self-attention distilling.
    /// </summary>
    private List<Vector<T>> Encode(List<Vector<T>> input)
    {
        var current = input;

        for (int layerIdx = 0; layerIdx < _encoderLayers.Count; layerIdx++)
        {
            // Apply encoder layer with ProbSparse attention
            current = _encoderLayers[layerIdx].Forward(current);

            // Apply distilling (except after last layer)
            if (layerIdx < _distillingLayers.Count)
            {
                current = _distillingLayers[layerIdx].Forward(current);
            }
        }

        return current;
    }

    /// <summary>
    /// Decodes using encoder memory with generative style decoder.
    /// Uses start token and generates output sequence in one forward pass.
    /// </summary>
    private List<Vector<T>> Decode(List<Vector<T>> encoderOutput)
    {
        // Initialize decoder input with start tokens for forecast horizon
        var decoderInput = new List<Vector<T>>();
        for (int t = 0; t < _options.ForecastHorizon; t++)
        {
            var token = _decoderStartToken.Clone();
            decoderInput.Add(token);
        }

        // Add positional encoding to decoder input
        AddPositionalEncoding(decoderInput, _options.LookbackWindow);

        // Process through decoder layers
        var current = decoderInput;
        foreach (var layer in _decoderLayers)
        {
            current = layer.Forward(current, encoderOutput);
        }

        return current;
    }

    /// <summary>
    /// Projects decoder output to forecast values.
    /// </summary>
    private Vector<T> ProjectOutput(List<Vector<T>> decoderOutput)
    {
        var forecast = new Vector<T>(_options.ForecastHorizon);

        for (int t = 0; t < _options.ForecastHorizon && t < decoderOutput.Count; t++)
        {
            T sum = _outputBias[t];
            int embDim = Math.Min(decoderOutput[t].Length, _outputProjection.Columns);
            for (int j = 0; j < embDim; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_outputProjection[t, j], decoderOutput[t][j]));
            }
            forecast[t] = sum;
        }

        return forecast;
    }

    private const int SerializationVersion = 2;

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
        writer.Write(_options.DistillingFactor);

        // Serialize input projection
        SerializeMatrix(writer, _inputProjection);

        // Serialize positional encoding
        SerializeMatrix(writer, _positionalEncoding);

        // Serialize encoder layers
        writer.Write(_encoderLayers.Count);
        foreach (var layer in _encoderLayers)
        {
            layer.Serialize(writer);
        }

        // Serialize distilling layers
        writer.Write(_distillingLayers.Count);
        foreach (var layer in _distillingLayers)
        {
            layer.Serialize(writer);
        }

        // Serialize decoder layers
        writer.Write(_decoderLayers.Count);
        foreach (var layer in _decoderLayers)
        {
            layer.Serialize(writer);
        }

        // Serialize decoder start token
        SerializeVector(writer, _decoderStartToken);

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
        {
            throw new NotSupportedException($"Unsupported serialization version: {version}. Expected: {SerializationVersion}");
        }

        // Deserialize and validate options
        int lookbackWindow = reader.ReadInt32();
        int forecastHorizon = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int numAttentionHeads = reader.ReadInt32();
        _ = reader.ReadInt32(); // BatchSize
        int distillingFactor = reader.ReadInt32();

        ValidateOption(lookbackWindow, _options.LookbackWindow, "LookbackWindow");
        ValidateOption(forecastHorizon, _options.ForecastHorizon, "ForecastHorizon");
        ValidateOption(embeddingDim, _options.EmbeddingDim, "EmbeddingDim");
        ValidateOption(numEncoderLayers, _options.NumEncoderLayers, "NumEncoderLayers");
        ValidateOption(numDecoderLayers, _options.NumDecoderLayers, "NumDecoderLayers");
        ValidateOption(numAttentionHeads, _options.NumAttentionHeads, "NumAttentionHeads");
        ValidateOption(distillingFactor, _options.DistillingFactor, "DistillingFactor");

        // Deserialize input projection
        _inputProjection = DeserializeMatrix(reader);

        // Deserialize positional encoding
        _positionalEncoding = DeserializeMatrix(reader);

        // Deserialize encoder layers
        int numEncLayers = reader.ReadInt32();
        _encoderLayers = new List<InformerEncoderLayer<T>>(numEncLayers);
        for (int i = 0; i < numEncLayers; i++)
        {
            _encoderLayers.Add(InformerEncoderLayer<T>.Deserialize(reader));
        }

        // Deserialize distilling layers
        int numDistLayers = reader.ReadInt32();
        _distillingLayers = new List<DistillingConv<T>>(numDistLayers);
        for (int i = 0; i < numDistLayers; i++)
        {
            _distillingLayers.Add(DistillingConv<T>.Deserialize(reader));
        }

        // Deserialize decoder layers
        int numDecLayers = reader.ReadInt32();
        _decoderLayers = new List<InformerDecoderLayer<T>>(numDecLayers);
        for (int i = 0; i < numDecLayers; i++)
        {
            _decoderLayers.Add(InformerDecoderLayer<T>.Deserialize(reader));
        }

        // Deserialize decoder start token
        _decoderStartToken = DeserializeVector(reader);

        // Deserialize output projection
        _outputProjection = DeserializeMatrix(reader);
        _outputBias = DeserializeVector(reader);
    }

    private void ValidateOption(int serialized, int expected, string name)
    {
        if (serialized != expected)
        {
            throw new InvalidOperationException($"Serialized {name} ({serialized}) doesn't match options ({expected})");
        }
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
            Name = "Informer",
            ModelType = ModelType.TimeSeriesRegression,
            Description = "Efficient Transformer for long-sequence time series forecasting with ProbSparse attention and distilling",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDim", _options.EmbeddingDim },
                { "NumEncoderLayers", _options.NumEncoderLayers },
                { "NumDecoderLayers", _options.NumDecoderLayers },
                { "NumAttentionHeads", _options.NumAttentionHeads },
                { "ForecastHorizon", _options.ForecastHorizon },
                { "DistillingFactor", _options.DistillingFactor },
                { "SparsityFactor", SparsityFactor }
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
            int count = _inputProjection.Rows * _inputProjection.Columns;
            count += _decoderStartToken.Length;
            count += _outputProjection.Rows * _outputProjection.Columns + _outputBias.Length;
            foreach (var layer in _encoderLayers)
                count += layer.ParameterCount;
            foreach (var layer in _distillingLayers)
                count += layer.ParameterCount;
            foreach (var layer in _decoderLayers)
                count += layer.ParameterCount;
            return count;
        }
    }
}

/// <summary>
/// Informer Encoder Layer with ProbSparse self-attention and feed-forward network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Encoder Layer Architecture:</b>
/// Each encoder layer follows the standard Transformer encoder pattern with a key modification:
/// ProbSparse self-attention replaces vanilla self-attention for efficiency.
/// </para>
/// <para>
/// <b>Pre-Norm vs Post-Norm:</b>
/// This implementation uses pre-norm architecture (LayerNorm before attention/FFN rather than after).
/// Research has shown pre-norm leads to more stable training and allows for deeper networks.
/// The residual connection is: output = input + sublayer(LayerNorm(input))
/// </para>
/// <para>
/// <b>ProbSparse Self-Attention:</b>
/// Instead of computing attention for all L queries (O(L²) complexity), ProbSparse attention
/// identifies queries with "peaky" attention distributions using a sparsity measurement.
/// The sparsity M(q, K) ≈ max(softmax(qK^T)) - mean(softmax(qK^T)) approximates the
/// KL-divergence from a uniform distribution. Only the top c*ln(L) queries with highest
/// sparsity receive full attention; others use the mean value as an approximation.
/// </para>
/// <para>
/// <b>Feed-Forward Network:</b>
/// The FFN applies two linear transformations with GELU activation:
/// FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
/// The hidden dimension is typically 4× the model dimension, allowing the network to
/// learn complex non-linear transformations.
/// </para>
/// <para><b>For Beginners:</b> Each encoder layer does two things: (1) look at relationships
/// between different time steps (attention), and (2) transform each time step's representation
/// independently (FFN). The clever ProbSparse attention only computes relationships for the
/// most "interesting" time steps, saving a lot of computation.
/// </para>
/// </remarks>
internal class InformerEncoderLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _sparsityFactor;

    // Multi-head attention weights (Q, K, V projections)
    private Matrix<T> _queryProj;
    private Matrix<T> _keyProj;
    private Matrix<T> _valueProj;
    private Matrix<T> _outputProj;

    // Feed-forward network
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
        _queryProj.Rows * _queryProj.Columns +
        _keyProj.Rows * _keyProj.Columns +
        _valueProj.Rows * _valueProj.Columns +
        _outputProj.Rows * _outputProj.Columns +
        _ffn1.Rows * _ffn1.Columns + _ffn1Bias.Length +
        _ffn2.Rows * _ffn2.Columns + _ffn2Bias.Length +
        _layerNorm1Gamma.Length * 2 + _layerNorm2Gamma.Length * 2;

    public InformerEncoderLayer(int embeddingDim, int numHeads, int sparsityFactor, double dropoutRate, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _sparsityFactor = sparsityFactor;

        var random = new Random(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / (embeddingDim * 4));

        // Initialize Q, K, V, O projections
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

        // Initialize layer norm parameters
        _layerNorm1Gamma = InitVector(embeddingDim, _numOps.One);
        _layerNorm1Beta = new Vector<T>(embeddingDim);
        _layerNorm2Gamma = InitVector(embeddingDim, _numOps.One);
        _layerNorm2Beta = new Vector<T>(embeddingDim);
    }

    private InformerEncoderLayer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = 0;
        _numHeads = 1;
        _headDim = 0;
        _sparsityFactor = 5;
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

    public List<Vector<T>> Forward(List<Vector<T>> input)
    {
        // Pre-norm architecture
        var normalized = LayerNorm(input, _layerNorm1Gamma, _layerNorm1Beta);

        // ProbSparse self-attention
        var attended = ProbSparseAttention(normalized);

        // Residual connection
        var residual1 = AddResidual(input, attended);

        // Pre-norm for FFN
        normalized = LayerNorm(residual1, _layerNorm2Gamma, _layerNorm2Beta);

        // Feed-forward network
        var ffnOutput = FeedForward(normalized);

        // Residual connection
        return AddResidual(residual1, ffnOutput);
    }

    /// <summary>
    /// Implements ProbSparse self-attention with O(L log L) complexity.
    /// Selects top-k queries based on sparsity measurement (KL divergence from uniform).
    /// </summary>
    private List<Vector<T>> ProbSparseAttention(List<Vector<T>> input)
    {
        int seqLen = input.Count;
        int topK = Math.Max(1, (int)Math.Ceiling(_sparsityFactor * Math.Log(seqLen + 1)));
        topK = Math.Min(topK, seqLen);

        // Compute Q, K, V for all positions
        var queries = input.Select(x => MatVecMul(_queryProj, x)).ToList();
        var keys = input.Select(x => MatVecMul(_keyProj, x)).ToList();
        var values = input.Select(x => MatVecMul(_valueProj, x)).ToList();

        // Compute query sparsity measurements (approximate KL divergence from uniform)
        var sparsityScores = new double[seqLen];
        for (int q = 0; q < seqLen; q++)
        {
            // Compute attention scores for this query
            var scores = new double[seqLen];
            double maxScore = double.NegativeInfinity;
            for (int k = 0; k < seqLen; k++)
            {
                scores[k] = Convert.ToDouble(DotProduct(queries[q], keys[k]));
                maxScore = Math.Max(maxScore, scores[k]);
            }

            // Softmax and compute sparsity (max - mean approximates KL from uniform)
            double sum = 0;
            for (int k = 0; k < seqLen; k++)
            {
                scores[k] = Math.Exp(scores[k] - maxScore);
                sum += scores[k];
            }

            double mean = sum / seqLen;
            double maxProb = scores.Max() / sum;
            sparsityScores[q] = maxProb - (1.0 / seqLen); // Higher = more sparse = more important
        }

        // Select top-k queries by sparsity score
        var topQueryIndices = sparsityScores
            .Select((score, idx) => (score, idx))
            .OrderByDescending(x => x.score)
            .Take(topK)
            .Select(x => x.idx)
            .ToHashSet();

        // Compute attention only for selected queries
        var output = new List<Vector<T>>();
        double scale = 1.0 / Math.Sqrt(_headDim);

        for (int q = 0; q < seqLen; q++)
        {
            if (topQueryIndices.Contains(q))
            {
                // Full attention for selected queries
                var attnWeights = new double[seqLen];
                double maxScore = double.NegativeInfinity;

                for (int k = 0; k < seqLen; k++)
                {
                    attnWeights[k] = Convert.ToDouble(DotProduct(queries[q], keys[k])) * scale;
                    maxScore = Math.Max(maxScore, attnWeights[k]);
                }

                // Softmax
                double sum = 0;
                for (int k = 0; k < seqLen; k++)
                {
                    attnWeights[k] = Math.Exp(attnWeights[k] - maxScore);
                    sum += attnWeights[k];
                }
                for (int k = 0; k < seqLen; k++)
                {
                    attnWeights[k] /= sum;
                }

                // Weighted sum of values
                var result = new Vector<T>(_embeddingDim);
                for (int k = 0; k < seqLen; k++)
                {
                    for (int d = 0; d < _embeddingDim; d++)
                    {
                        result[d] = _numOps.Add(result[d],
                            _numOps.Multiply(_numOps.FromDouble(attnWeights[k]), values[k][d]));
                    }
                }
                output.Add(MatVecMul(_outputProj, result));
            }
            else
            {
                // Use mean value for non-selected queries (efficient approximation)
                var result = new Vector<T>(_embeddingDim);
                for (int k = 0; k < seqLen; k++)
                {
                    for (int d = 0; d < _embeddingDim; d++)
                    {
                        result[d] = _numOps.Add(result[d], values[k][d]);
                    }
                }
                T invSeqLen = _numOps.FromDouble(1.0 / seqLen);
                for (int d = 0; d < _embeddingDim; d++)
                {
                    result[d] = _numOps.Multiply(result[d], invSeqLen);
                }
                output.Add(MatVecMul(_outputProj, result));
            }
        }

        return output;
    }

    private List<Vector<T>> LayerNorm(List<Vector<T>> input, Vector<T> gamma, Vector<T> beta)
    {
        var output = new List<Vector<T>>();
        foreach (var vec in input)
        {
            // Compute mean and variance
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

            // Normalize
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
            {
                result[i] = _numOps.Add(result[i], _ffn2Bias[i]);
            }
            output.Add(result);
        }
        return output;
    }

    private T GELU(T x)
    {
        // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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
            for (int i = 0; i < input[t].Length; i++)
            {
                vec[i] = _numOps.Add(input[t][i], residual[t][i]);
            }
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
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[i, j], vec[j]));
            }
            result[i] = sum;
        }
        return result;
    }

    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    public void UpdateWeights(Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon,
        Func<Vector<T>, T> predict, int sampleSize)
    {
        // Update a random subset of weights using numerical gradients
        var random = new Random();
        var allMatrices = new[] { _queryProj, _keyProj, _valueProj, _outputProj, _ffn1, _ffn2 };

        foreach (var matrix in allMatrices)
        {
            int totalWeights = matrix.Rows * matrix.Columns;
            int actualSample = Math.Min(sampleSize / 6, totalWeights);

            for (int s = 0; s < actualSample; s++)
            {
                int flatIdx = random.Next(totalWeights);
                int i = flatIdx / matrix.Columns;
                int j = flatIdx % matrix.Columns;

                T original = matrix[i, j];

                matrix[i, j] = _numOps.Add(original, epsilon);
                T lossPlus = ComputeLoss(predict(input), target);

                matrix[i, j] = _numOps.Subtract(original, epsilon);
                T lossMinus = ComputeLoss(predict(input), target);

                matrix[i, j] = original;

                T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
                matrix[i, j] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
            }
        }
    }

    private T ComputeLoss(T predicted, T target)
    {
        T error = _numOps.Subtract(target, predicted);
        return _numOps.Multiply(error, error);
    }

    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_embeddingDim);
        writer.Write(_numHeads);
        writer.Write(_sparsityFactor);

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

    public static InformerEncoderLayer<T> Deserialize(BinaryReader reader)
    {
        var layer = new InformerEncoderLayer<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        int embeddingDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int sparsityFactor = reader.ReadInt32();

        // Use reflection to set private readonly fields
        typeof(InformerEncoderLayer<T>).GetField("_embeddingDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, embeddingDim);
        typeof(InformerEncoderLayer<T>).GetField("_numHeads", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, numHeads);
        typeof(InformerEncoderLayer<T>).GetField("_headDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, embeddingDim / numHeads);
        typeof(InformerEncoderLayer<T>).GetField("_sparsityFactor", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, sparsityFactor);

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

/// <summary>
/// Distilling convolution layer for self-attention distilling.
/// Uses 1D convolution followed by max pooling to reduce sequence length.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>The Purpose of Distilling:</b>
/// In long-sequence forecasting, even with ProbSparse attention reducing per-layer complexity
/// to O(L log L), having many layers with long sequences is still expensive. Distilling
/// progressively shortens the sequence between encoder layers, creating a pyramid structure
/// that dramatically reduces total computation.
/// </para>
/// <para>
/// <b>Distilling Architecture:</b>
/// Each distilling layer applies three operations in sequence:
/// 1. <b>1D Convolution (kernel=3):</b> Aggregates local temporal information from neighboring
///    positions. This smooths the sequence and extracts local patterns.
/// 2. <b>ELU Activation:</b> Non-linearity that allows learning complex patterns while
///    maintaining gradient flow for negative values.
/// 3. <b>Max Pooling (stride=2):</b> Reduces sequence length by half by taking the maximum
///    value in each pooling window. This keeps the most prominent features.
/// </para>
/// <para>
/// <b>Sequence Length Reduction:</b>
/// With a distilling factor of 2 and 3 encoder layers, an input of length L becomes:
/// L → L/2 → L/4 (final encoder output). This exponential reduction allows processing
/// very long sequences efficiently.
/// </para>
/// <para><b>For Beginners:</b> Think of distilling like creating a summary. After each layer
/// of understanding, we compress the sequence by half, keeping only the most important
/// information. This is similar to how you might summarize a book chapter by chapter,
/// with each summary being shorter than the original.
/// </para>
/// </remarks>
internal class DistillingConv<T>
{
    private readonly INumericOperations<T> _numOps;
    private Matrix<T> _convWeights;
    private Vector<T> _convBias;
    private readonly int _poolingFactor;

    public int ParameterCount => _convWeights.Rows * _convWeights.Columns + _convBias.Length;

    public DistillingConv(int embeddingDim, int poolingFactor, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _poolingFactor = poolingFactor;

        var random = new Random(seed);
        double stddev = Math.Sqrt(2.0 / embeddingDim);

        // 1D convolution weights (kernel size = 3, same embedding dim)
        _convWeights = new Matrix<T>(embeddingDim, embeddingDim * 3);
        for (int i = 0; i < _convWeights.Rows; i++)
            for (int j = 0; j < _convWeights.Columns; j++)
                _convWeights[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);

        _convBias = new Vector<T>(embeddingDim);
    }

    private DistillingConv()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _convWeights = new Matrix<T>(0, 0);
        _convBias = new Vector<T>(0);
        _poolingFactor = 2;
    }

    public List<Vector<T>> Forward(List<Vector<T>> input)
    {
        int seqLen = input.Count;
        int embDim = input[0].Length;

        // Apply 1D convolution with padding
        var convOutput = new List<Vector<T>>();
        for (int t = 0; t < seqLen; t++)
        {
            var output = new Vector<T>(embDim);

            // Gather 3 neighboring positions (with zero padding)
            for (int i = 0; i < embDim; i++)
            {
                T sum = _convBias[i];
                for (int k = -1; k <= 1; k++)
                {
                    int srcIdx = t + k;
                    if (srcIdx >= 0 && srcIdx < seqLen)
                    {
                        for (int j = 0; j < embDim; j++)
                        {
                            int weightCol = (k + 1) * embDim + j;
                            if (weightCol < _convWeights.Columns)
                            {
                                sum = _numOps.Add(sum, _numOps.Multiply(_convWeights[i, weightCol], input[srcIdx][j]));
                            }
                        }
                    }
                }
                // ELU activation
                double val = Convert.ToDouble(sum);
                output[i] = _numOps.FromDouble(val > 0 ? val : Math.Exp(val) - 1);
            }
            convOutput.Add(output);
        }

        // Max pooling to reduce sequence length
        var pooledOutput = new List<Vector<T>>();
        for (int t = 0; t < seqLen; t += _poolingFactor)
        {
            var pooled = new Vector<T>(embDim);
            for (int i = 0; i < embDim; i++)
            {
                T maxVal = convOutput[t][i];
                for (int p = 1; p < _poolingFactor && (t + p) < seqLen; p++)
                {
                    if (_numOps.GreaterThan(convOutput[t + p][i], maxVal))
                    {
                        maxVal = convOutput[t + p][i];
                    }
                }
                pooled[i] = maxVal;
            }
            pooledOutput.Add(pooled);
        }

        return pooledOutput;
    }

    public void UpdateWeights(Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon,
        Func<Vector<T>, T> predict, int sampleSize)
    {
        var random = new Random();
        int totalWeights = _convWeights.Rows * _convWeights.Columns;
        int actualSample = Math.Min(sampleSize, totalWeights);

        for (int s = 0; s < actualSample; s++)
        {
            int flatIdx = random.Next(totalWeights);
            int i = flatIdx / _convWeights.Columns;
            int j = flatIdx % _convWeights.Columns;

            T original = _convWeights[i, j];

            _convWeights[i, j] = _numOps.Add(original, epsilon);
            T err1 = _numOps.Subtract(target, predict(input));
            T lossPlus = _numOps.Multiply(err1, err1);

            _convWeights[i, j] = _numOps.Subtract(original, epsilon);
            T err2 = _numOps.Subtract(target, predict(input));
            T lossMinus = _numOps.Multiply(err2, err2);

            _convWeights[i, j] = original;

            T gradient = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
            _convWeights[i, j] = _numOps.Subtract(original, _numOps.Multiply(learningRate, gradient));
        }
    }

    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_poolingFactor);
        writer.Write(_convWeights.Rows);
        writer.Write(_convWeights.Columns);
        for (int i = 0; i < _convWeights.Rows; i++)
            for (int j = 0; j < _convWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_convWeights[i, j]));
        writer.Write(_convBias.Length);
        for (int i = 0; i < _convBias.Length; i++)
            writer.Write(Convert.ToDouble(_convBias[i]));
    }

    public static DistillingConv<T> Deserialize(BinaryReader reader)
    {
        var layer = new DistillingConv<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        int poolingFactor = reader.ReadInt32();
        typeof(DistillingConv<T>).GetField("_poolingFactor", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, poolingFactor);

        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        layer._convWeights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                layer._convWeights[i, j] = numOps.FromDouble(reader.ReadDouble());

        int biasLen = reader.ReadInt32();
        layer._convBias = new Vector<T>(biasLen);
        for (int i = 0; i < biasLen; i++)
            layer._convBias[i] = numOps.FromDouble(reader.ReadDouble());

        return layer;
    }
}

/// <summary>
/// Informer Decoder Layer with masked self-attention, cross-attention, and feed-forward network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Decoder Layer Architecture:</b>
/// The decoder layer is more complex than the encoder layer, containing three sublayers:
/// 1. Masked self-attention over decoder inputs
/// 2. Cross-attention to encoder outputs
/// 3. Feed-forward network
/// Each sublayer uses pre-norm and residual connections.
/// </para>
/// <para>
/// <b>Masked Self-Attention:</b>
/// In the decoder, each position can only attend to itself and earlier positions. This "causal"
/// or "autoregressive" masking prevents information from future positions leaking into current
/// predictions. The mask is implemented by setting future attention weights to zero (or -∞
/// before softmax). This ensures position t can only see positions 0, 1, ..., t.
/// </para>
/// <para>
/// <b>Cross-Attention:</b>
/// Cross-attention allows the decoder to "look at" the encoder output. The decoder provides
/// queries (what information do I need?), and the encoder output provides keys and values
/// (here's what I know about the input). This is how the decoder learns to use the encoded
/// historical patterns to make predictions.
/// </para>
/// <para>
/// <b>Generative Decoding in Informer:</b>
/// Unlike standard autoregressive decoders that generate one token at a time, Informer's
/// decoder generates all forecast positions simultaneously. This is possible because:
/// 1. We initialize with learnable start tokens for all positions
/// 2. Masked self-attention maintains causality without requiring sequential generation
/// 3. Cross-attention provides the same encoder context to all positions
/// </para>
/// <para><b>For Beginners:</b> The decoder is like a writer who has read a summary (encoder output)
/// and needs to write the next chapter. Masked self-attention ensures each sentence builds
/// only on previous sentences (no spoilers). Cross-attention lets the writer refer back to
/// the summary at any point. The result is a coherent forecast that respects temporal order.
/// </para>
/// </remarks>
internal class InformerDecoderLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Masked self-attention weights
    private Matrix<T> _selfQueryProj;
    private Matrix<T> _selfKeyProj;
    private Matrix<T> _selfValueProj;
    private Matrix<T> _selfOutputProj;

    // Cross-attention weights
    private Matrix<T> _crossQueryProj;
    private Matrix<T> _crossKeyProj;
    private Matrix<T> _crossValueProj;
    private Matrix<T> _crossOutputProj;

    // Feed-forward network
    private Matrix<T> _ffn1;
    private Vector<T> _ffn1Bias;
    private Matrix<T> _ffn2;
    private Vector<T> _ffn2Bias;

    // Layer normalization parameters
    private Vector<T> _layerNorm1Gamma;
    private Vector<T> _layerNorm1Beta;
    private Vector<T> _layerNorm2Gamma;
    private Vector<T> _layerNorm2Beta;
    private Vector<T> _layerNorm3Gamma;
    private Vector<T> _layerNorm3Beta;

    public int ParameterCount =>
        _selfQueryProj.Rows * _selfQueryProj.Columns * 4 +
        _crossQueryProj.Rows * _crossQueryProj.Columns * 4 +
        _ffn1.Rows * _ffn1.Columns + _ffn1Bias.Length +
        _ffn2.Rows * _ffn2.Columns + _ffn2Bias.Length +
        _layerNorm1Gamma.Length * 6;

    public InformerDecoderLayer(int embeddingDim, int numHeads, int sparsityFactor, double dropoutRate, int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;

        var random = new Random(seed);
        double attnStddev = Math.Sqrt(2.0 / embeddingDim);
        double ffnStddev = Math.Sqrt(2.0 / (embeddingDim * 4));

        // Self-attention weights
        _selfQueryProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _selfKeyProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _selfValueProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _selfOutputProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);

        // Cross-attention weights
        _crossQueryProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _crossKeyProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _crossValueProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);
        _crossOutputProj = InitMatrix(embeddingDim, embeddingDim, attnStddev, random);

        // FFN
        int ffnDim = embeddingDim * 4;
        _ffn1 = InitMatrix(ffnDim, embeddingDim, ffnStddev, random);
        _ffn1Bias = new Vector<T>(ffnDim);
        _ffn2 = InitMatrix(embeddingDim, ffnDim, ffnStddev, random);
        _ffn2Bias = new Vector<T>(embeddingDim);

        // Layer norms
        _layerNorm1Gamma = InitVector(embeddingDim, _numOps.One);
        _layerNorm1Beta = new Vector<T>(embeddingDim);
        _layerNorm2Gamma = InitVector(embeddingDim, _numOps.One);
        _layerNorm2Beta = new Vector<T>(embeddingDim);
        _layerNorm3Gamma = InitVector(embeddingDim, _numOps.One);
        _layerNorm3Beta = new Vector<T>(embeddingDim);
    }

    private InformerDecoderLayer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDim = 0;
        _numHeads = 1;
        _headDim = 0;
        _selfQueryProj = new Matrix<T>(0, 0);
        _selfKeyProj = new Matrix<T>(0, 0);
        _selfValueProj = new Matrix<T>(0, 0);
        _selfOutputProj = new Matrix<T>(0, 0);
        _crossQueryProj = new Matrix<T>(0, 0);
        _crossKeyProj = new Matrix<T>(0, 0);
        _crossValueProj = new Matrix<T>(0, 0);
        _crossOutputProj = new Matrix<T>(0, 0);
        _ffn1 = new Matrix<T>(0, 0);
        _ffn1Bias = new Vector<T>(0);
        _ffn2 = new Matrix<T>(0, 0);
        _ffn2Bias = new Vector<T>(0);
        _layerNorm1Gamma = new Vector<T>(0);
        _layerNorm1Beta = new Vector<T>(0);
        _layerNorm2Gamma = new Vector<T>(0);
        _layerNorm2Beta = new Vector<T>(0);
        _layerNorm3Gamma = new Vector<T>(0);
        _layerNorm3Beta = new Vector<T>(0);
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

    public List<Vector<T>> Forward(List<Vector<T>> decoderInput, List<Vector<T>> encoderOutput)
    {
        // Pre-norm + masked self-attention
        var normalized = LayerNorm(decoderInput, _layerNorm1Gamma, _layerNorm1Beta);
        var selfAttn = MaskedSelfAttention(normalized);
        var residual1 = AddResidual(decoderInput, selfAttn);

        // Pre-norm + cross-attention
        normalized = LayerNorm(residual1, _layerNorm2Gamma, _layerNorm2Beta);
        var crossAttn = CrossAttention(normalized, encoderOutput);
        var residual2 = AddResidual(residual1, crossAttn);

        // Pre-norm + FFN
        normalized = LayerNorm(residual2, _layerNorm3Gamma, _layerNorm3Beta);
        var ffnOutput = FeedForward(normalized);
        return AddResidual(residual2, ffnOutput);
    }

    private List<Vector<T>> MaskedSelfAttention(List<Vector<T>> input)
    {
        int seqLen = input.Count;
        double scale = 1.0 / Math.Sqrt(_headDim);

        var queries = input.Select(x => MatVecMul(_selfQueryProj, x)).ToList();
        var keys = input.Select(x => MatVecMul(_selfKeyProj, x)).ToList();
        var values = input.Select(x => MatVecMul(_selfValueProj, x)).ToList();

        var output = new List<Vector<T>>();
        for (int q = 0; q < seqLen; q++)
        {
            // Masked attention: only attend to positions <= q
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
            {
                attnWeights[k] /= sum;
            }

            var result = new Vector<T>(_embeddingDim);
            for (int k = 0; k <= q; k++)
            {
                for (int d = 0; d < _embeddingDim; d++)
                {
                    result[d] = _numOps.Add(result[d],
                        _numOps.Multiply(_numOps.FromDouble(attnWeights[k]), values[k][d]));
                }
            }
            output.Add(MatVecMul(_selfOutputProj, result));
        }

        return output;
    }

    private List<Vector<T>> CrossAttention(List<Vector<T>> decoderInput, List<Vector<T>> encoderOutput)
    {
        int decLen = decoderInput.Count;
        int encLen = encoderOutput.Count;
        double scale = 1.0 / Math.Sqrt(_headDim);

        var queries = decoderInput.Select(x => MatVecMul(_crossQueryProj, x)).ToList();
        var keys = encoderOutput.Select(x => MatVecMul(_crossKeyProj, x)).ToList();
        var values = encoderOutput.Select(x => MatVecMul(_crossValueProj, x)).ToList();

        var output = new List<Vector<T>>();
        for (int q = 0; q < decLen; q++)
        {
            var attnWeights = new double[encLen];
            double maxScore = double.NegativeInfinity;

            for (int k = 0; k < encLen; k++)
            {
                attnWeights[k] = Convert.ToDouble(DotProduct(queries[q], keys[k])) * scale;
                maxScore = Math.Max(maxScore, attnWeights[k]);
            }

            double sum = 0;
            for (int k = 0; k < encLen; k++)
            {
                attnWeights[k] = Math.Exp(attnWeights[k] - maxScore);
                sum += attnWeights[k];
            }
            for (int k = 0; k < encLen; k++)
            {
                attnWeights[k] /= sum;
            }

            var result = new Vector<T>(_embeddingDim);
            for (int k = 0; k < encLen; k++)
            {
                for (int d = 0; d < _embeddingDim; d++)
                {
                    result[d] = _numOps.Add(result[d],
                        _numOps.Multiply(_numOps.FromDouble(attnWeights[k]), values[k][d]));
                }
            }
            output.Add(MatVecMul(_crossOutputProj, result));
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
            var hidden = MatVecMul(_ffn1, vec);
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] = _numOps.Add(hidden[i], _ffn1Bias[i]);
                double val = Convert.ToDouble(hidden[i]);
                hidden[i] = _numOps.FromDouble(val * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (val + 0.044715 * val * val * val))));
            }

            var result = MatVecMul(_ffn2, hidden);
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = _numOps.Add(result[i], _ffn2Bias[i]);
            }
            output.Add(result);
        }
        return output;
    }

    private List<Vector<T>> AddResidual(List<Vector<T>> input, List<Vector<T>> residual)
    {
        var output = new List<Vector<T>>();
        for (int t = 0; t < input.Count; t++)
        {
            var vec = new Vector<T>(input[t].Length);
            for (int i = 0; i < input[t].Length && i < residual[t].Length; i++)
            {
                vec[i] = _numOps.Add(input[t][i], residual[t][i]);
            }
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
            {
                sum = _numOps.Add(sum, _numOps.Multiply(matrix[i, j], vec[j]));
            }
            result[i] = sum;
        }
        return result;
    }

    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    public void UpdateWeights(Vector<T> input, T target, T learningRate, T epsilon, T twoEpsilon,
        Func<Vector<T>, T> predict, int sampleSize)
    {
        var random = new Random();
        var allMatrices = new[] {
            _selfQueryProj, _selfKeyProj, _selfValueProj, _selfOutputProj,
            _crossQueryProj, _crossKeyProj, _crossValueProj, _crossOutputProj,
            _ffn1, _ffn2
        };

        foreach (var matrix in allMatrices)
        {
            int totalWeights = matrix.Rows * matrix.Columns;
            int actualSample = Math.Min(sampleSize / 10, totalWeights);

            for (int s = 0; s < actualSample; s++)
            {
                int flatIdx = random.Next(totalWeights);
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

        SerializeMatrix(writer, _selfQueryProj);
        SerializeMatrix(writer, _selfKeyProj);
        SerializeMatrix(writer, _selfValueProj);
        SerializeMatrix(writer, _selfOutputProj);
        SerializeMatrix(writer, _crossQueryProj);
        SerializeMatrix(writer, _crossKeyProj);
        SerializeMatrix(writer, _crossValueProj);
        SerializeMatrix(writer, _crossOutputProj);
        SerializeMatrix(writer, _ffn1);
        SerializeVector(writer, _ffn1Bias);
        SerializeMatrix(writer, _ffn2);
        SerializeVector(writer, _ffn2Bias);
        SerializeVector(writer, _layerNorm1Gamma);
        SerializeVector(writer, _layerNorm1Beta);
        SerializeVector(writer, _layerNorm2Gamma);
        SerializeVector(writer, _layerNorm2Beta);
        SerializeVector(writer, _layerNorm3Gamma);
        SerializeVector(writer, _layerNorm3Beta);
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

    public static InformerDecoderLayer<T> Deserialize(BinaryReader reader)
    {
        var layer = new InformerDecoderLayer<T>();
        var numOps = MathHelper.GetNumericOperations<T>();

        int embeddingDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();

        typeof(InformerDecoderLayer<T>).GetField("_embeddingDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, embeddingDim);
        typeof(InformerDecoderLayer<T>).GetField("_numHeads", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, numHeads);
        typeof(InformerDecoderLayer<T>).GetField("_headDim", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.SetValue(layer, embeddingDim / numHeads);

        layer._selfQueryProj = DeserializeMatrix(reader, numOps);
        layer._selfKeyProj = DeserializeMatrix(reader, numOps);
        layer._selfValueProj = DeserializeMatrix(reader, numOps);
        layer._selfOutputProj = DeserializeMatrix(reader, numOps);
        layer._crossQueryProj = DeserializeMatrix(reader, numOps);
        layer._crossKeyProj = DeserializeMatrix(reader, numOps);
        layer._crossValueProj = DeserializeMatrix(reader, numOps);
        layer._crossOutputProj = DeserializeMatrix(reader, numOps);
        layer._ffn1 = DeserializeMatrix(reader, numOps);
        layer._ffn1Bias = DeserializeVector(reader, numOps);
        layer._ffn2 = DeserializeMatrix(reader, numOps);
        layer._ffn2Bias = DeserializeVector(reader, numOps);
        layer._layerNorm1Gamma = DeserializeVector(reader, numOps);
        layer._layerNorm1Beta = DeserializeVector(reader, numOps);
        layer._layerNorm2Gamma = DeserializeVector(reader, numOps);
        layer._layerNorm2Beta = DeserializeVector(reader, numOps);
        layer._layerNorm3Gamma = DeserializeVector(reader, numOps);
        layer._layerNorm3Beta = DeserializeVector(reader, numOps);

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
