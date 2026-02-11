
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Inference;

/// <summary>
/// Multi-head attention layer with KV-Cache support for efficient autoregressive inference.
/// </summary>
/// <remarks>
/// <para>
/// CachedMultiHeadAttention wraps standard multi-head attention with KV-Cache support.
/// It automatically caches Key and Value projections across inference steps,
/// enabling efficient token-by-token generation.
/// </para>
/// <para><b>For Beginners:</b> This is a fast version of attention for text generation.
///
/// Normal attention recalculates everything for each new token:
/// - Token 1: Process token 1
/// - Token 2: Process tokens 1-2 (redo token 1!)
/// - Token 3: Process tokens 1-3 (redo tokens 1-2!)
/// - ... gets slower and slower
///
/// Cached attention remembers previous computations:
/// - Token 1: Compute and cache K, V for token 1
/// - Token 2: Only compute K, V for token 2, use cache for token 1
/// - Token 3: Only compute K, V for token 3, use cache for tokens 1-2
/// - ... stays fast!
///
/// Use this layer when:
/// - Generating text token by token (autoregressive)
/// - Running inference (not training)
/// - You want fast generation speed
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations.</typeparam>
internal class CachedMultiHeadAttention<T> : LayerBase<T>
{
    private readonly int _headCount;
    private readonly int _headDimension;
    private readonly int _embeddingDimension;
    private readonly bool _useFlashAttention;
    private readonly bool _useCausalMask;

    // Positional encoding
    private RotaryPositionalEncodingLayer<T>? _ropeLayer;
    private ALiBiPositionalBiasLayer<T>? _alibiLayer;

    /// <summary>
    /// Gets the positional encoding type used by this attention layer.
    /// </summary>
    public PositionalEncodingType PositionalEncoding { get; private set; } = PositionalEncodingType.None;

    // Projection weights
    private Matrix<T> _queryWeights;
    private Matrix<T> _keyWeights;
    private Matrix<T> _valueWeights;
    private Matrix<T> _outputWeights;
    private Vector<T> _outputBias;

    // KV-Cache reference (shared across layers)
    private KVCache<T>? _cache;
    private int _layerIndex;

    // Cached values for backward (training mode only)
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    // Gradients
    private Matrix<T>? _queryWeightsGradient;
    private Matrix<T>? _keyWeightsGradient;
    private Matrix<T>? _valueWeightsGradient;
    private Matrix<T>? _outputWeightsGradient;
    private Vector<T>? _outputBiasGradient;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// CachedMultiHeadAttention supports training, but KV-Cache is only used during inference.
    /// During training, it behaves like standard MultiHeadAttention.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets or sets whether the layer is in inference mode (uses cache).
    /// </summary>
    public bool InferenceMode { get; set; } = false;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int HeadCount => _headCount;

    /// <summary>
    /// Gets the dimension of each attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets whether Flash Attention is enabled.
    /// </summary>
    public bool UsesFlashAttention => _useFlashAttention;

    /// <summary>
    /// Gets whether causal masking is enabled for attention.
    /// </summary>
    /// <remarks>
    /// Causal masking is required for autoregressive decoding (GPT-style), where each token may only attend
    /// to itself and previous tokens. Disable for bidirectional attention (BERT-style) and most encoders.
    /// </remarks>
    public bool UsesCausalMask => _useCausalMask;

    /// <summary>
    /// Gets or sets the KV-Cache. Must be set before inference.
    /// </summary>
    public KVCache<T>? Cache
    {
        get => _cache;
        set => _cache = value;
    }

    /// <summary>
    /// Gets or sets the layer index in the transformer (for cache indexing).
    /// </summary>
    public int LayerIndex
    {
        get => _layerIndex;
        set => _layerIndex = value;
    }

    /// <summary>
    /// Creates a new cached multi-head attention layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="embeddingDimension">Embedding dimension (must be divisible by headCount).</param>
    /// <param name="headCount">Number of attention heads.</param>
    /// <param name="useFlashAttention">Whether to use Flash Attention algorithm.</param>
    /// <param name="layerIndex">Index of this layer in the transformer (for cache access).</param>
    /// <param name="useCausalMask">Whether to apply causal masking (required for autoregressive decoding).</param>
    /// <param name="activationFunction">Optional activation function (defaults to identity).</param>
    public CachedMultiHeadAttention(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        bool useFlashAttention = true,
        int layerIndex = 0,
        bool useCausalMask = true,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by head count ({headCount}).");
        }

        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _embeddingDimension = embeddingDimension;
        _useFlashAttention = useFlashAttention;
        _layerIndex = layerIndex;
        _useCausalMask = useCausalMask;

        // Initialize projection weights
        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Configures positional encoding for this cached attention layer.
    /// </summary>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <param name="ropeTheta">Base frequency for RoPE (default: 10000.0).</param>
    /// <param name="maxSequenceLength">Maximum sequence length for pre-computation.</param>
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
                    $"Unsupported positional encoding type for CachedMultiHeadAttention: {encodingType}.",
                    nameof(encodingType));
        }
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Rows + _queryWeights.Columns)));

        InitializeMatrix(_queryWeights, scale);
        InitializeMatrix(_keyWeights, scale);
        InitializeMatrix(_valueWeights, scale);
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

    /// <summary>
    /// Performs the forward pass with optional KV-Cache support.
    /// </summary>
    /// <param name="input">Input tensor [batch, seqLen, embDim].</param>
    /// <returns>Output tensor of same shape.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> How this works in different modes:
    ///
    /// Training mode (InferenceMode = false):
    /// - Computes full attention like standard MultiHeadAttention
    /// - Does NOT use cache (cache is for inference only)
    ///
    /// Inference mode (InferenceMode = true):
    /// - Uses KV-Cache for efficient generation
    /// - For prefill: Processes full prompt, caches all K, V
    /// - For generation: Processes single new token, uses cached K, V
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        if (InferenceMode && _cache != null)
        {
            return ForwardWithCache(input);
        }
        else
        {
            return ForwardStandard(input);
        }
    }

    /// <summary>
    /// Forward pass using KV-Cache for efficient inference.
    /// </summary>
    private Tensor<T> ForwardWithCache(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];

        // Compute Q, K, V projections
        var queries = input.Multiply(_queryWeights);
        var newKeys = input.Multiply(_keyWeights);
        var newValues = input.Multiply(_valueWeights);

        // Reshape to [batch, heads, seq, headDim]
        queries = queries.Reshape(batchSize, seqLen, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        newKeys = newKeys.Reshape(batchSize, seqLen, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        newValues = newValues.Reshape(batchSize, seqLen, _headCount, _headDimension).Transpose([0, 2, 1, 3]);

        // Apply RoPE with position offset for incremental decoding
        if (_ropeLayer != null)
        {
            int startPosition = _cache!.CurrentLength;
            (queries, newKeys) = _ropeLayer.ApplyRoPE(queries, newKeys, startPosition);
        }

        // Append to cache and get full K, V
        var (keys, values) = _cache!.Append(_layerIndex, newKeys, newValues);

        // Compute attention using cached K, V
        Tensor<T> attentionOutput;
        if (_useFlashAttention)
        {
            var config = FlashAttentionConfig.Default;
            config.UseCausalMask = _useCausalMask;

            int seqLenKV = keys.Shape[2];
            int seqLenQ = queries.Shape[2];
            int queryOffset = Math.Max(0, seqLenKV - seqLenQ);
            var (flashOutput, _) = FlashAttention<T>.Forward(queries, keys, values, config, queryOffset: queryOffset);
            attentionOutput = flashOutput;
        }
        else if (_alibiLayer != null)
        {
            int seqLenQ = queries.Shape[2];
            int seqLenKV = keys.Shape[2];
            attentionOutput = StandardAttentionWithALiBi(queries, keys, values, _useCausalMask, seqLenQ, seqLenKV);
        }
        else
        {
            attentionOutput = StandardAttention(queries, keys, values, useCausalMask: _useCausalMask);
        }

        // Reshape back to [batch, seq, embDim]
        attentionOutput = attentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, seqLen, _embeddingDimension);

        // Output projection
        var output = attentionOutput.Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    /// <summary>
    /// Standard forward pass without caching (for training).
    /// </summary>
    private Tensor<T> ForwardStandard(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];

        // Compute Q, K, V projections
        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        // Reshape to [batch, heads, seq, headDim]
        queries = queries.Reshape(batchSize, seqLen, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        keys = keys.Reshape(batchSize, seqLen, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        values = values.Reshape(batchSize, seqLen, _headCount, _headDimension).Transpose([0, 2, 1, 3]);

        // Apply RoPE to Q and K if configured (position starts at 0 for standard forward)
        if (_ropeLayer != null)
        {
            (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: 0);
        }

        // Compute attention
        Tensor<T> attentionOutput;
        if (_useFlashAttention)
        {
            var config = FlashAttentionConfig.Default;
            config.UseCausalMask = _useCausalMask;
            var (flashOutput, _) = FlashAttention<T>.Forward(queries, keys, values, config);
            attentionOutput = flashOutput;
        }
        else if (_alibiLayer != null)
        {
            attentionOutput = StandardAttentionWithALiBi(queries, keys, values, _useCausalMask, seqLen, seqLen);
        }
        else
        {
            attentionOutput = StandardAttention(queries, keys, values, useCausalMask: _useCausalMask);
        }

        // Reshape back
        attentionOutput = attentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, seqLen, _embeddingDimension);

        // Output projection
        var output = attentionOutput.Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    /// <summary>
    /// Standard scaled dot-product attention implementation.
    /// </summary>
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
                // Compute attention scores
                var scores = new T[seqLenQ, seqLenKV];
                for (int i = 0; i < seqLenQ; i++)
                {
                    // Find position in full sequence for causal masking
                    int queryPos = seqLenKV - seqLenQ + i; // Position in full KV sequence

                    for (int j = 0; j < seqLenKV; j++)
                    {
                        if (useCausalMask && j > queryPos)
                        {
                            scores[i, j] = negInf;
                            continue;
                        }

                        T dot = NumOps.Zero;
                        for (int d = 0; d < headDim; d++)
                        {
                            T qVal = query[new[] { b, h, i, d }];
                            T kVal = key[new[] { b, h, j, d }];
                            dot = NumOps.Add(dot, NumOps.Multiply(qVal, kVal));
                        }
                        scores[i, j] = NumOps.Multiply(dot, scale);
                    }
                }

                // Apply softmax row-wise
                for (int i = 0; i < seqLenQ; i++)
                {
                    // Find max
                    T maxScore = negInf;
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        if (NumOps.GreaterThan(scores[i, j], maxScore))
                        {
                            maxScore = scores[i, j];
                        }
                    }

                    // Compute exp and sum
                    T sumExp = NumOps.Zero;
                    var weights = new T[seqLenKV];
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        weights[j] = NumOps.Exp(NumOps.Subtract(scores[i, j], maxScore));
                        sumExp = NumOps.Add(sumExp, weights[j]);
                    }

                    // Normalize and compute output
                    for (int d = 0; d < headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int j = 0; j < seqLenKV; j++)
                        {
                            T weight = NumericalStabilityHelper.SafeDiv(weights[j], sumExp);
                            T vVal = value[new[] { b, h, j, d }];
                            sum = NumOps.Add(sum, NumOps.Multiply(weight, vVal));
                        }
                        output[new[] { b, h, i, d }] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Standard attention with ALiBi position bias injection.
    /// </summary>
    private Tensor<T> StandardAttentionWithALiBi(
        Tensor<T> query, Tensor<T> key, Tensor<T> value,
        bool useCausalMask, int seqLenQ, int seqLenKV)
    {
        int batchSize = query.Shape[0];
        int numHeads = query.Shape[1];
        int headDim = query.Shape[3];

        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));
        T negInf = NumOps.FromDouble(double.NegativeInfinity);
        var bias = _alibiLayer!.ComputeBias(seqLenQ, seqLenKV);

        var output = new Tensor<T>(new[] { batchSize, numHeads, seqLenQ, headDim });

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                var scores = new T[seqLenQ, seqLenKV];
                for (int i = 0; i < seqLenQ; i++)
                {
                    int queryPos = seqLenKV - seqLenQ + i;
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        if (useCausalMask && j > queryPos)
                        {
                            scores[i, j] = negInf;
                            continue;
                        }

                        T dot = NumOps.Zero;
                        for (int d = 0; d < headDim; d++)
                        {
                            T qVal = query[new[] { b, h, i, d }];
                            T kVal = key[new[] { b, h, j, d }];
                            dot = NumOps.Add(dot, NumOps.Multiply(qVal, kVal));
                        }

                        // Add ALiBi bias to scaled dot-product score
                        scores[i, j] = NumOps.Add(
                            NumOps.Multiply(dot, scale),
                            bias[new[] { h, i, j }]);
                    }
                }

                // Softmax row-wise
                for (int i = 0; i < seqLenQ; i++)
                {
                    T maxScore = negInf;
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        if (NumOps.GreaterThan(scores[i, j], maxScore))
                            maxScore = scores[i, j];
                    }

                    T sumExp = NumOps.Zero;
                    var weights = new T[seqLenKV];
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        weights[j] = NumOps.Exp(NumOps.Subtract(scores[i, j], maxScore));
                        sumExp = NumOps.Add(sumExp, weights[j]);
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int j = 0; j < seqLenKV; j++)
                        {
                            T weight = NumericalStabilityHelper.SafeDiv(weights[j], sumExp);
                            T vVal = value[new[] { b, h, j, d }];
                            sum = NumOps.Add(sum, NumOps.Multiply(weight, vVal));
                        }
                        output[new[] { b, h, i, d }] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs backward pass (training mode only, cache not used).
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        // Standard backward pass (no cache during training)
        // Implementation similar to MultiHeadAttentionLayer
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Simplified gradient computation
        // In practice, use autodiff or detailed manual gradient
        _queryWeightsGradient = new Matrix<T>(_queryWeights.Rows, _queryWeights.Columns);
        _keyWeightsGradient = new Matrix<T>(_keyWeights.Rows, _keyWeights.Columns);
        _valueWeightsGradient = new Matrix<T>(_valueWeights.Rows, _valueWeights.Columns);
        _outputWeightsGradient = new Matrix<T>(_outputWeights.Rows, _outputWeights.Columns);
        _outputBiasGradient = activationGradient.Sum([0, 1]).ToVector();

        return inputGradient;
    }

    /// <summary>
    /// Updates parameters using computed gradients.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        _queryWeights = _queryWeights.Subtract(_queryWeightsGradient!.Multiply(learningRate));
        _keyWeights = _keyWeights.Subtract(_keyWeightsGradient!.Multiply(learningRate));
        _valueWeights = _valueWeights.Subtract(_valueWeightsGradient!.Multiply(learningRate));
        _outputWeights = _outputWeights.Subtract(_outputWeightsGradient!.Multiply(learningRate));
        _outputBias = _outputBias.Subtract(_outputBiasGradient!.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all layer parameters.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        int totalParams = _queryWeights.Rows * _queryWeights.Columns * 4 + _outputBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    parameters[index++] = matrix[i, j];
                }
            }
        }

        for (int i = 0; i < _outputBias.Length; i++)
        {
            parameters[index++] = _outputBias[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets all layer parameters.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = _queryWeights.Rows * _queryWeights.Columns * 4 + _outputBias.Length;
        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");
        }

        int index = 0;

        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = parameters[index++];
                }
            }
        }

        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the layer's state.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;

        // Note: Does not clear the cache - use cache.Clear() separately
    }

    /// <summary>
    /// Clears the KV-Cache if attached.
    /// </summary>
    public void ClearCache()
    {
        _cache?.Clear();
    }

    /// <summary>
    /// Gets diagnostic information.
    /// </summary>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        diagnostics["HeadCount"] = _headCount.ToString();
        diagnostics["HeadDimension"] = _headDimension.ToString();
        diagnostics["InferenceMode"] = InferenceMode.ToString();
        diagnostics["UsesFlashAttention"] = _useFlashAttention.ToString();
        diagnostics["UsesCausalMask"] = _useCausalMask.ToString();
        diagnostics["PositionalEncoding"] = PositionalEncoding.ToString();
        diagnostics["LayerIndex"] = _layerIndex.ToString();
        diagnostics["CacheAttached"] = (_cache != null).ToString();

        if (_cache != null)
        {
            diagnostics["CacheLength"] = _cache.CurrentLength.ToString();
            diagnostics["CacheMaxLength"] = _cache.MaxLength.ToString();
            diagnostics["CacheHitRate"] = $"{(_cache.CacheHits + _cache.CacheMisses > 0 ? (double)_cache.CacheHits / (_cache.CacheHits + _cache.CacheMisses) : 0):P2}";
        }

        return diagnostics;
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => _queryWeights != null && _queryWeights.Rows > 0;

    /// <summary>
    /// Exports computation graph for JIT compilation.
    /// </summary>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        // Similar to FlashAttentionLayer
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var seqLen = InputShape[0];
        var embDim = InputShape[1];
        var symbolicInput = new Tensor<T>(new[] { 1, seqLen, embDim });
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        var wqTensor = MatrixToTensor(_queryWeights);
        var wkTensor = MatrixToTensor(_keyWeights);
        var wvTensor = MatrixToTensor(_valueWeights);
        var woTensor = MatrixToTensor(_outputWeights);

        var wqNode = Autodiff.TensorOperations<T>.Constant(wqTensor, "Wq");
        var wkNode = Autodiff.TensorOperations<T>.Constant(wkTensor, "Wk");
        var wvNode = Autodiff.TensorOperations<T>.Constant(wvTensor, "Wv");
        var woNode = Autodiff.TensorOperations<T>.Constant(woTensor, "Wo");

        var output = Autodiff.TensorOperations<T>.MultiHeadAttention(
            query: inputNode,
            key: inputNode,
            value: inputNode,
            numHeads: _headCount,
            wQ: wqNode,
            wK: wkNode,
            wV: wvNode,
            wO: woNode);

        return output;
    }

    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var tensor = new Tensor<T>(new[] { matrix.Rows, matrix.Columns });
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                tensor[i, j] = matrix[i, j];
            }
        }
        return tensor;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        return new Dictionary<string, string>
        {
            ["HeadCount"] = _headCount.ToString(),
            ["UseFlashAttention"] = _useFlashAttention.ToString(),
            ["UseCausalMask"] = _useCausalMask.ToString()
        };
    }
}
