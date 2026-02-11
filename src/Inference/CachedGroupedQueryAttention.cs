using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Inference;

/// <summary>
/// Grouped-Query Attention with KV-Cache support for efficient autoregressive inference.
/// </summary>
/// <remarks>
/// <para>
/// This layer extends GQA with KV-cache support. The cache stores only numKVHeads sets
/// of keys and values (not numHeads), reducing cache memory proportionally.
/// During generation, K/V heads are expanded to match Q heads only for the attention computation.
/// </para>
/// <para><b>For Beginners:</b> This is a fast version of GQA for text generation.
///
/// Memory savings compared to cached standard MHA:
/// - Llama 2 70B: numHeads=64, numKVHeads=8 → 8x less KV-cache memory
/// - The cache stores only 8 K/V sets instead of 64
/// - During attention, each K/V set is shared by 8 query heads
///
/// This is critical for serving large models with limited GPU memory.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations.</typeparam>
internal class CachedGroupedQueryAttention<T> : LayerBase<T>
{
    private readonly int _numHeads;
    private readonly int _numKVHeads;
    private readonly int _headDimension;
    private readonly int _embeddingDimension;
    private readonly int _headsPerGroup;
    private readonly bool _useFlashAttention;
    private readonly bool _useCausalMask;

    // Projection weights
    private readonly Matrix<T> _queryWeights;
    private readonly Matrix<T> _keyWeights;  // Reduced: [embDim, numKVHeads * headDim]
    private readonly Matrix<T> _valueWeights; // Reduced: [embDim, numKVHeads * headDim]
    private readonly Matrix<T> _outputWeights;
    private Vector<T> _outputBias;

    // KV-Cache reference
    private KVCache<T>? _cache;
    private int _layerIndex;

    // Positional encoding
    private RotaryPositionalEncodingLayer<T>? _ropeLayer;
    private ALiBiPositionalBiasLayer<T>? _alibiLayer;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    /// <inheritdoc />
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets or sets whether the layer is in inference mode (uses cache).
    /// </summary>
    public bool InferenceMode { get; set; } = false;

    /// <summary>
    /// Gets the number of query heads.
    /// </summary>
    public int HeadCount => _numHeads;

    /// <summary>
    /// Gets the number of KV heads (may be less than HeadCount).
    /// </summary>
    public int KVHeadCount => _numKVHeads;

    /// <summary>
    /// Gets the dimension of each attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the positional encoding type.
    /// </summary>
    public PositionalEncodingType PositionalEncoding { get; private set; } = PositionalEncodingType.None;

    /// <summary>
    /// Gets or sets the KV-Cache.
    /// </summary>
    public KVCache<T>? Cache
    {
        get => _cache;
        set => _cache = value;
    }

    /// <summary>
    /// Gets or sets the layer index for cache indexing.
    /// </summary>
    public int LayerIndex
    {
        get => _layerIndex;
        set => _layerIndex = value;
    }

    /// <summary>
    /// Creates a new cached GQA layer.
    /// </summary>
    public CachedGroupedQueryAttention(
        int sequenceLength,
        int embeddingDimension,
        int numHeads,
        int numKVHeads,
        bool useFlashAttention = true,
        int layerIndex = 0,
        bool useCausalMask = true,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % numHeads != 0)
            throw new ArgumentException($"Embedding dimension ({embeddingDimension}) must be divisible by numHeads ({numHeads}).");
        if (numHeads % numKVHeads != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be divisible by numKVHeads ({numKVHeads}).");

        _numHeads = numHeads;
        _numKVHeads = numKVHeads;
        _headDimension = embeddingDimension / numHeads;
        _embeddingDimension = embeddingDimension;
        _headsPerGroup = numHeads / numKVHeads;
        _useFlashAttention = useFlashAttention;
        _layerIndex = layerIndex;
        _useCausalMask = useCausalMask;

        int kvDim = numKVHeads * _headDimension;

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, kvDim);
        _valueWeights = new Matrix<T>(embeddingDimension, kvDim);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeWeights();
    }

    /// <summary>
    /// Configures positional encoding.
    /// </summary>
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
                _alibiLayer = new ALiBiPositionalBiasLayer<T>(_numHeads, maxSequenceLength);
                break;
            case PositionalEncodingType.None:
                break;
            default:
                throw new ArgumentException(
                    $"Unsupported positional encoding type for CachedGroupedQueryAttention: {encodingType}.",
                    nameof(encodingType));
        }
    }

    private void InitializeWeights()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Rows + _queryWeights.Columns)));
        InitializeMatrix(_queryWeights, scale);

        T kvScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_keyWeights.Rows + _keyWeights.Columns)));
        InitializeMatrix(_keyWeights, kvScale);
        InitializeMatrix(_valueWeights, kvScale);
        InitializeMatrix(_outputWeights, scale);

        _outputBias = Vector<T>.CreateDefault(_outputBias.Length, NumOps.Zero);
    }

    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        return InferenceMode && _cache != null
            ? ForwardWithCache(input)
            : ForwardStandard(input);
    }

    private Tensor<T> ForwardWithCache(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];

        // Project Q, K, V
        var queries = input.Multiply(_queryWeights);
        var newKeys = input.Multiply(_keyWeights);
        var newValues = input.Multiply(_valueWeights);

        // Reshape Q: [batch, seq, numHeads, headDim] -> [batch, numHeads, seq, headDim]
        queries = queries.Reshape(batchSize, seqLen, _numHeads, _headDimension).Transpose([0, 2, 1, 3]);

        // Reshape K/V: [batch, seq, numKVHeads, headDim] -> [batch, numKVHeads, seq, headDim]
        newKeys = newKeys.Reshape(batchSize, seqLen, _numKVHeads, _headDimension).Transpose([0, 2, 1, 3]);
        newValues = newValues.Reshape(batchSize, seqLen, _numKVHeads, _headDimension).Transpose([0, 2, 1, 3]);

        // Apply RoPE with position offset
        if (_ropeLayer != null)
        {
            int startPosition = _cache!.CurrentLength;
            (queries, newKeys) = _ropeLayer.ApplyRoPE(queries, newKeys, startPosition);
        }

        // Append to cache (cache stores numKVHeads, not numHeads!)
        var (keys, values) = _cache!.Append(_layerIndex, newKeys, newValues);

        // Expand KV heads to match Q heads
        int cachedSeqLen = keys.Shape[2];
        var expandedKeys = ExpandKVHeads(keys, batchSize, cachedSeqLen);
        var expandedValues = ExpandKVHeads(values, batchSize, cachedSeqLen);

        // Compute attention
        Tensor<T> attentionOutput;
        if (_useFlashAttention || _alibiLayer != null)
        {
            var config = FlashAttentionConfig.Default;
            config.UseCausalMask = _useCausalMask;
            int queryOffset = Math.Max(0, cachedSeqLen - seqLen);

            Tensor<T>? aliBiBias = _alibiLayer?.ComputeBias(seqLen, cachedSeqLen, _useCausalMask);
            var (flashOutput, _) = FlashAttention<T>.Forward(queries, expandedKeys, expandedValues, config, queryOffset: queryOffset, attentionBias: aliBiBias);
            attentionOutput = flashOutput;
        }
        else
        {
            attentionOutput = StandardAttention(queries, expandedKeys, expandedValues, _useCausalMask);
        }

        // Reshape and project output
        attentionOutput = attentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, seqLen, _embeddingDimension);
        var output = attentionOutput.Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    private Tensor<T> ForwardStandard(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];

        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        queries = queries.Reshape(batchSize, seqLen, _numHeads, _headDimension).Transpose([0, 2, 1, 3]);
        keys = keys.Reshape(batchSize, seqLen, _numKVHeads, _headDimension).Transpose([0, 2, 1, 3]);
        values = values.Reshape(batchSize, seqLen, _numKVHeads, _headDimension).Transpose([0, 2, 1, 3]);

        if (_ropeLayer != null)
        {
            (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: 0);
        }

        var expandedKeys = ExpandKVHeads(keys, batchSize, seqLen);
        var expandedValues = ExpandKVHeads(values, batchSize, seqLen);

        Tensor<T> attentionOutput;
        if (_useFlashAttention || _alibiLayer != null)
        {
            var config = FlashAttentionConfig.Default;
            config.UseCausalMask = _useCausalMask;

            Tensor<T>? aliBiBias = _alibiLayer?.ComputeBias(seqLen, seqLen, _useCausalMask);
            var (flashOutput, _) = FlashAttention<T>.Forward(queries, expandedKeys, expandedValues, config, attentionBias: aliBiBias);
            attentionOutput = flashOutput;
        }
        else
        {
            attentionOutput = StandardAttention(queries, expandedKeys, expandedValues, _useCausalMask);
        }

        attentionOutput = attentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, seqLen, _embeddingDimension);
        var output = attentionOutput.Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    private Tensor<T> ExpandKVHeads(Tensor<T> kv, int batchSize, int seqLen)
    {
        if (_numKVHeads == _numHeads)
            return kv;

        var expanded = new Tensor<T>(new[] { batchSize, _numHeads, seqLen, _headDimension });
        for (int b = 0; b < batchSize; b++)
        {
            for (int kvh = 0; kvh < _numKVHeads; kvh++)
            {
                for (int g = 0; g < _headsPerGroup; g++)
                {
                    int qh = kvh * _headsPerGroup + g;
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int d = 0; d < _headDimension; d++)
                        {
                            expanded[new[] { b, qh, s, d }] = kv[new[] { b, kvh, s, d }];
                        }
                    }
                }
            }
        }
        return expanded;
    }

    private Tensor<T> StandardAttention(Tensor<T> query, Tensor<T> key, Tensor<T> value, bool useCausalMask)
    {
        int batchSize = query.Shape[0];
        int numHeads = query.Shape[1];
        int seqLenQ = query.Shape[2];
        int seqLenKV = key.Shape[2];
        int headDim = query.Shape[3];

        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));
        T negInf = NumOps.FromDouble(double.NegativeInfinity);

        var output = new Tensor<T>(new[] { batchSize, numHeads, seqLenQ, headDim });

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < seqLenQ; i++)
                {
                    int queryPos = seqLenKV - seqLenQ + i;

                    var scores = new T[seqLenKV];
                    T maxScore = negInf;
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        if (useCausalMask && j > queryPos)
                        {
                            scores[j] = negInf;
                            continue;
                        }

                        T dot = NumOps.Zero;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot = NumOps.Add(dot, NumOps.Multiply(
                                query[new[] { b, h, i, d }],
                                key[new[] { b, h, j, d }]));
                        }
                        scores[j] = NumOps.Multiply(dot, scale);
                        if (NumOps.GreaterThan(scores[j], maxScore))
                            maxScore = scores[j];
                    }

                    T sumExp = NumOps.Zero;
                    var weights = new T[seqLenKV];
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        weights[j] = NumOps.Exp(NumOps.Subtract(scores[j], maxScore));
                        sumExp = NumOps.Add(sumExp, weights[j]);
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int j = 0; j < seqLenKV; j++)
                        {
                            T w = NumericalStabilityHelper.SafeDiv(weights[j], sumExp);
                            sum = NumOps.Add(sum, NumOps.Multiply(w,
                                value[new[] { b, h, j, d }]));
                        }
                        output[new[] { b, h, i, d }] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        return new Tensor<T>(_lastInput.Shape);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        // Simplified for inference-focused layer
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int qSize = _queryWeights.Rows * _queryWeights.Columns;
        int kvSize = _keyWeights.Rows * _keyWeights.Columns;
        int oSize = _outputWeights.Rows * _outputWeights.Columns;
        int totalParams = qSize + kvSize * 2 + oSize + _outputBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Columns; j++)
                    parameters[index++] = matrix[i, j];
        }
        for (int i = 0; i < _outputBias.Length; i++)
            parameters[index++] = _outputBias[i];

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Columns; j++)
                    matrix[i, j] = parameters[index++];
        }
        for (int i = 0; i < _outputBias.Length; i++)
            _outputBias[i] = parameters[index++];
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);
        return inputNode;
    }

    /// <inheritdoc />
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();
        diagnostics["NumHeads"] = _numHeads.ToString();
        diagnostics["NumKVHeads"] = _numKVHeads.ToString();
        diagnostics["HeadsPerGroup"] = _headsPerGroup.ToString();
        diagnostics["HeadDimension"] = _headDimension.ToString();
        diagnostics["InferenceMode"] = InferenceMode.ToString();
        diagnostics["PositionalEncoding"] = PositionalEncoding.ToString();
        diagnostics["CacheAttached"] = (_cache != null).ToString();
        return diagnostics;
    }
}
