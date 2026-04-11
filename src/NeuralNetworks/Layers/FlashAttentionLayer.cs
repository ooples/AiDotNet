
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A multi-head attention layer using the Flash Attention algorithm for memory-efficient computation.
/// </summary>
/// <remarks>
/// <para>
/// FlashAttentionLayer provides the same functionality as MultiHeadAttentionLayer but uses the
/// Flash Attention algorithm which is 2-4x faster and uses significantly less memory.
/// It can be used as a drop-in replacement in transformer architectures.
/// </para>
/// <para><b>For Beginners:</b> This is like MultiHeadAttentionLayer but faster and more memory-efficient.
///
/// Flash Attention is a breakthrough algorithm that makes transformers much faster:
/// - Standard attention: O(N^2) memory, slow for long sequences
/// - Flash Attention: O(N) memory, 2-4x faster
///
/// Use this layer when:
/// - Training with long sequences (1024+ tokens)
/// - Training large models with limited GPU memory
/// - You need faster training/inference
///
/// The output is mathematically identical to standard attention - only the computation is different.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations (typically float or double).</typeparam>
public partial class FlashAttentionLayer<T> : LayerBase<T>
{
    private readonly int _headCount;
    private readonly int _headDimension;
    private readonly FlashAttentionConfig _config;

    // Positional encoding support
    private RotaryPositionalEncodingLayer<T>? _ropeLayer;
    private ALiBiPositionalBiasLayer<T>? _alibiLayer;

    /// <summary>
    /// Gets the positional encoding type used by this attention layer.
    /// </summary>
    public PositionalEncodingType PositionalEncoding { get; private set; } = PositionalEncodingType.None;

    /// <summary>
    /// Gets the RoPE base frequency (theta) if RoPE is configured.
    /// </summary>
    public double RoPETheta => _ropeLayer?.Theta ?? 10000.0;

    // Projection weights stored as Tensor<T> so they can participate in the
    // gradient tape. Previously stored as Matrix<T> / Vector<T>, which
    // silently excluded them from the tape graph entirely — every diffusion
    // UNet attention block was training nothing through these weights.
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;
    private Tensor<T> _outputWeights;
    private Tensor<T> _outputBias;

    // Tracks the original input shape so ForwardGpu / Forward can reshape the
    // output back to the caller's rank before returning.
    private int[]? _originalInputShape;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int HeadCount => _headCount;

    /// <summary>
    /// Gets the dimension of each attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the Flash Attention configuration.
    /// </summary>
    public FlashAttentionConfig Config => _config;

    /// <summary>
    /// Creates a new Flash Attention layer with the specified dimensions.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each embedding vector.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="config">Optional Flash Attention configuration.</param>
    /// <param name="activationFunction">Optional activation function (defaults to identity).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a Flash Attention layer.
    ///
    /// Parameters:
    /// - sequenceLength: How many tokens/words in your sequence (e.g., 512, 1024, 4096)
    /// - embeddingDimension: Size of each token's representation (e.g., 768 for BERT, 4096 for GPT-3)
    /// - headCount: Number of attention heads (e.g., 12 for BERT-base, 96 for GPT-3)
    ///
    /// The embeddingDimension must be divisible by headCount.
    /// Each head will have dimension = embeddingDimension / headCount.
    /// </para>
    /// </remarks>
    public FlashAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        FlashAttentionConfig? config = null,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by head count ({headCount}).",
                nameof(headCount));
        }

        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _config = config ?? FlashAttentionConfig.Default;

        // Initialize projection weights as Tensor<T> [embedDim, embedDim].
        _queryWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _keyWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _valueWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputBias = new Tensor<T>([embeddingDimension]);

        InitializeParameters();
        RegisterAttentionParameters();
    }

    /// <summary>
    /// Creates a new Flash Attention layer with vector activation function.
    /// </summary>
    public FlashAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        FlashAttentionConfig? config,
        IVectorActivationFunction<T>? vectorActivationFunction)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            vectorActivationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by head count ({headCount}).",
                nameof(headCount));
        }

        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _config = config ?? FlashAttentionConfig.Default;

        _queryWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _keyWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _valueWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputBias = new Tensor<T>([embeddingDimension]);

        InitializeParameters();
        RegisterAttentionParameters();
    }

    /// <summary>
    /// Registers all trainable projection tensors with the layer base so that
    /// recursive parameter collection and tape-based gradient training pick them up.
    /// </summary>
    private void RegisterAttentionParameters()
    {
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Configures positional encoding for this Flash Attention layer.
    /// </summary>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <param name="ropeTheta">Base frequency for RoPE (default: 10000.0).</param>
    /// <param name="maxSequenceLength">Maximum sequence length for pre-computation (default: 2048).</param>
    public void ConfigurePositionalEncoding(
        PositionalEncodingType encodingType,
        double ropeTheta = 10000.0,
        int maxSequenceLength = 2048)
    {
        PositionalEncoding = encodingType;
        _ropeLayer = null;
        _alibiLayer = null;

        switch (encodingType)
        {
            case PositionalEncodingType.Rotary:
                _ropeLayer = new RotaryPositionalEncodingLayer<T>(
                    maxSequenceLength, _headDimension, ropeTheta);
                break;
            case PositionalEncodingType.ALiBi:
                _alibiLayer = new ALiBiPositionalBiasLayer<T>(_headCount, maxSequenceLength);
                break;
            case PositionalEncodingType.None:
                break;
            default:
                throw new ArgumentException(
                    $"Unsupported positional encoding type for FlashAttentionLayer: {encodingType}.",
                    nameof(encodingType));
        }
    }

    /// <summary>
    /// Initializes projection weights using Xavier/Glorot initialization.
    /// </summary>
    private void InitializeParameters()
    {
        int rows = _queryWeights.Shape[0];
        int cols = _queryWeights.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (rows + cols)));

        InitializeWeightTensor(_queryWeights, scale);
        InitializeWeightTensor(_keyWeights, scale);
        InitializeWeightTensor(_valueWeights, scale);
        InitializeWeightTensor(_outputWeights, scale);

        // Bias starts at zero. Write in-place so any existing tape references
        // (or already-registered parameter entries) keep the same tensor handle.
        var biasSpan = _outputBias.Data.Span;
        for (int i = 0; i < biasSpan.Length; i++) biasSpan[i] = NumOps.Zero;
    }

    private void InitializeWeightTensor(Tensor<T> weights, T scale)
    {
        var span = weights.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <summary>
    /// Performs the forward pass using Flash Attention.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, sequenceLength, embeddingDimension].</param>
    /// <returns>Output tensor of the same shape as input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is where the Flash Attention computation happens.
    ///
    /// The forward pass:
    /// 1. Projects input to Query, Key, Value using learned weights
    /// 2. Reshapes into multiple heads
    /// 3. Applies Flash Attention (the fast, memory-efficient algorithm)
    /// 4. Concatenates heads and projects output
    ///
    /// Flash Attention computes the same result as standard attention but:
    /// - Never materializes the full N x N attention matrix
    /// - Processes in tiles that fit in fast cache memory
    /// - Uses online softmax for numerical stability
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input._shape;
        var input3D = NormalizeTo3D(input, out int batchSize, out int sequenceLength, out int embeddingDimension);

        // Every shape + projection op goes through Engine so the gradient tape
        // records the forward chain. The previous implementation used direct
        // Tensor<T>.Multiply / .Reshape / .Transpose / .Add, which bypassed the
        // tape and left the attention block's weights with zero gradient on
        // every diffusion training step.

        // Project input to Q, K, V via tape-tracked matmul.
        // input3D is [batch, seq, embed]; weights are [embed, embed].
        // Engine.TensorMatMul handles the any-rank generalization.
        var queries = Engine.TensorMatMul(input3D, _queryWeights);
        var keys = Engine.TensorMatMul(input3D, _keyWeights);
        var values = Engine.TensorMatMul(input3D, _valueWeights);

        // Reshape to [batch, heads, seq, headDim] via tape-tracked shape ops.
        queries = Engine.TensorPermute(
            Engine.Reshape(queries, new[] { batchSize, sequenceLength, _headCount, _headDimension }),
            new[] { 0, 2, 1, 3 });
        keys = Engine.TensorPermute(
            Engine.Reshape(keys, new[] { batchSize, sequenceLength, _headCount, _headDimension }),
            new[] { 0, 2, 1, 3 });
        values = Engine.TensorPermute(
            Engine.Reshape(values, new[] { batchSize, sequenceLength, _headCount, _headDimension }),
            new[] { 0, 2, 1, 3 });

        // Apply RoPE to Q and K if configured
        if (_ropeLayer != null)
        {
            (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: 0);
        }

        // Compute scale factor for attention (null lets IEngine use 1/sqrt(headDim))
        double? scale = _config.ScaleFactor.HasValue
            ? (double)_config.ScaleFactor.Value
            : null;

        // Compute ALiBi bias if configured, passing it directly to the engine.
        // The engine natively supports additive attention bias.
        Tensor<T>? aliBiBias = _alibiLayer != null
            ? _alibiLayer.ComputeBias(queries.Shape[2], keys.Shape[2], _config.UseCausalMask)
            : null;

        // Apply Flash Attention — tape records the op so backward flows through
        // FlashAttentionBackward automatically via the gradient tape.
        var attentionOutput = Engine.FlashAttention(
            queries,
            keys,
            values,
            scale,
            _config.UseCausalMask,
            out _,
            attentionBias: aliBiBias);

        // Reshape back to [batch, seq, embedding] via tape-tracked shape ops.
        var permutedOut = Engine.TensorPermute(attentionOutput, new[] { 0, 2, 1, 3 });
        attentionOutput = Engine.Reshape(permutedOut, new[] { batchSize, sequenceLength, embeddingDimension });

        // Output projection with broadcast bias add — all tape-tracked.
        var projected = Engine.TensorMatMul(attentionOutput, _outputWeights);
        var output = Engine.TensorBroadcastAdd(projected, _outputBias);
        var activated = ApplyActivation(output);

        if (_originalInputShape == null || _originalInputShape.Length == 3)
        {
            return activated;
        }

        if (_originalInputShape.Length == 1)
        {
            return Engine.Reshape(activated, new[] { embeddingDimension });
        }

        return Engine.Reshape(activated, _originalInputShape);
    }

    private Tensor<T> NormalizeTo3D(Tensor<T> input, out int batchSize, out int sequenceLength, out int embeddingDimension)
    {
        if (input.Rank == 3)
        {
            batchSize = input.Shape[0];
            sequenceLength = input.Shape[1];
            embeddingDimension = input.Shape[2];
            return input;
        }

        if (input.Rank == 2)
        {
            batchSize = 1;
            sequenceLength = input.Shape[0];
            embeddingDimension = input.Shape[1];
            return Engine.Reshape(input, new[] { 1, sequenceLength, embeddingDimension });
        }

        if (input.Rank > 3)
        {
            int flatBatch = 1;
            for (int d = 0; d < input.Rank - 2; d++)
            {
                flatBatch *= input.Shape[d];
            }
            batchSize = flatBatch;
            sequenceLength = input.Shape[input.Rank - 2];
            embeddingDimension = input.Shape[input.Rank - 1];
            return Engine.Reshape(input, new[] { batchSize, sequenceLength, embeddingDimension });
        }

        batchSize = 1;
        sequenceLength = 1;
        embeddingDimension = input.Shape[0];
        return Engine.Reshape(input, new[] { 1, 1, embeddingDimension });
    }

    /// <summary>
    /// Legacy scalar-learning-rate parameter update. Tape-based training flows through
    /// <see cref="SetParameters"/> after <c>GradientTape&lt;T&gt;</c> computes gradients and
    /// the optimizer applies them, so this override is a no-op. The hand-rolled SPSA /
    /// blame-on-step fallback that used private <c>_*Gradient</c> fields was deleted along
    /// with those fields once the Forward path moved to <c>Engine.FlashAttention</c>.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        // No-op: weights are updated via SetParameters(Vector<T>) after the tape
        // computes gradients through Engine.FlashAttention + FlashAttentionBackward.
    }

    /// <summary>
    /// Gets all layer parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        int weightLen = _queryWeights.Length; // embed × embed
        int totalParams = weightLen * 4 + _outputBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy all weight tensors (flat span)
        foreach (var tensor in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            var span = tensor.AsSpan();
            for (int i = 0; i < span.Length; i++)
            {
                parameters[index++] = span[i];
            }
        }

        // Copy bias
        var biasSpan = _outputBias.AsSpan();
        for (int i = 0; i < biasSpan.Length; i++)
        {
            parameters[index++] = biasSpan[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets all layer parameters from a single vector. Writes in place so the
    /// registered tensor references remain stable — important for tape-based
    /// training where the gradient graph holds direct references to these
    /// tensors. Re-assigning the field would leave the tape pointing at stale
    /// objects.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int weightLen = _queryWeights.Length;
        int expectedParams = weightLen * 4 + _outputBias.Length;
        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");
        }

        int index = 0;

        foreach (var tensor in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            var span = tensor.Data.Span;
            for (int i = 0; i < span.Length; i++)
            {
                span[i] = parameters[index++];
            }
        }

        var biasSpan = _outputBias.Data.Span;
        for (int i = 0; i < biasSpan.Length; i++)
        {
            biasSpan[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the layer's internal state.
    /// </summary>
    public override void ResetState()
    {
        _originalInputShape = null;
    }

    /// <summary>
    /// Gets diagnostic information about the layer.
    /// </summary>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        diagnostics["HeadCount"] = _headCount.ToString();
        diagnostics["HeadDimension"] = _headDimension.ToString();
        diagnostics["UseCausalMask"] = _config.UseCausalMask.ToString();
        diagnostics["BlockSizeQ"] = _config.BlockSizeQ.ToString();
        diagnostics["BlockSizeKV"] = _config.BlockSizeKV.ToString();
        diagnostics["RecomputeInBackward"] = _config.RecomputeInBackward.ToString();
        diagnostics["Precision"] = _config.Precision.ToString();
        diagnostics["PositionalEncoding"] = PositionalEncoding.ToString();

        return diagnostics;
    }

    /// <summary>
    /// Gets the query projection weights (for external access/debugging).
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the key projection weights.
    /// </summary>
    public Tensor<T> GetKeyWeights() => _keyWeights;

    /// <summary>
    /// Gets the value projection weights.
    /// </summary>
    public Tensor<T> GetValueWeights() => _valueWeights;

    /// <summary>
    /// Gets the output projection weights.
    /// </summary>
    public Tensor<T> GetOutputWeights() => _outputWeights;
}
