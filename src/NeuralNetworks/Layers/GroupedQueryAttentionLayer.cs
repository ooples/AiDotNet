using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements Grouped-Query Attention (GQA) from Ainslie et al., 2023.
/// </summary>
/// <remarks>
/// <para>
/// GQA generalizes standard Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
/// Instead of having one set of Key/Value projections per head, GQA uses fewer K/V heads
/// that are shared among groups of Query heads. This reduces the KV-cache memory by
/// a factor of numHeads/numKVHeads while preserving most of the model quality.
/// </para>
/// <para>
/// When numKVHeads == numHeads, GQA is equivalent to standard MHA.
/// When numKVHeads == 1, GQA is equivalent to MQA.
/// </para>
/// <para><b>For Beginners:</b> GQA is a memory-efficient attention mechanism used by modern LLMs.
///
/// In standard attention with 64 heads:
/// - 64 Query projections, 64 Key projections, 64 Value projections
/// - KV-cache stores 64 sets of keys and values per layer
///
/// With GQA (64 Q heads, 8 KV heads):
/// - 64 Query projections, but only 8 Key and 8 Value projections
/// - Each K/V head is shared by 8 Query heads (64/8 = 8)
/// - KV-cache stores only 8 sets → 8x less memory!
///
/// Used by Llama 2 70B, Llama 3, Mistral, Gemma 2, and most modern large LLMs.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
internal class GroupedQueryAttentionLayer<T> : LayerBase<T>
{
    private readonly int _numHeads;
    private readonly int _numKVHeads;
    private readonly int _headDimension;
    private readonly int _embeddingDimension;
    private readonly int _headsPerGroup;

    // Q projection: [embDim, numHeads * headDim]
    private Tensor<T> _queryWeights;
    // K projection: [embDim, numKVHeads * headDim] (smaller!)
    private Tensor<T> _keyWeights;
    // V projection: [embDim, numKVHeads * headDim] (smaller!)
    private Tensor<T> _valueWeights;
    // Output projection: [numHeads * headDim, embDim]
    private Tensor<T> _outputWeights;
    private Tensor<T> _outputBias;

    // Positional encoding
    private RotaryPositionalEncodingLayer<T>? _ropeLayer;
    private ALiBiPositionalBiasLayer<T>? _alibiLayer;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedQueries;
    private Tensor<T>? _lastProjectedKeys;
    private Tensor<T>? _lastProjectedValues;
    private Tensor<T>? _lastExpandedKeys;
    private Tensor<T>? _lastExpandedValues;
    private Tensor<T>? _lastAttentionWeights;
    private Tensor<T>? _lastAttentionContext;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _outputWeightsGradient;
    private Tensor<T>? _outputBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the total number of query heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the number of key/value heads (fewer than query heads in GQA).
    /// </summary>
    public int NumKVHeads => _numKVHeads;

    /// <summary>
    /// Gets the dimension of each attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the number of query heads per KV head group.
    /// </summary>
    public int HeadsPerGroup => _headsPerGroup;

    /// <summary>
    /// Gets the attention variant this layer implements.
    /// </summary>
    public AttentionVariant Variant
    {
        get
        {
            if (_numKVHeads == _numHeads) return AttentionVariant.MultiHead;
            if (_numKVHeads == 1) return AttentionVariant.MultiQuery;
            return AttentionVariant.GroupedQuery;
        }
    }

    /// <summary>
    /// Gets the positional encoding type used by this attention layer.
    /// </summary>
    public PositionalEncodingType PositionalEncoding { get; private set; } = PositionalEncodingType.None;

    /// <summary>
    /// Gets the RoPE theta parameter if RoPE is configured, or the default 10000.0.
    /// </summary>
    public double RoPETheta => _ropeLayer?.Theta ?? 10000.0;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _outputWeights.Length + _outputBias.Length;

    /// <summary>
    /// Creates a new Grouped-Query Attention layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="embeddingDimension">Embedding dimension (must be divisible by numHeads).</param>
    /// <param name="numHeads">Total number of query heads.</param>
    /// <param name="numKVHeads">Number of key/value heads (must divide numHeads evenly).</param>
    /// <param name="activationFunction">Optional activation function (defaults to identity).</param>
    public GroupedQueryAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int numHeads,
        int numKVHeads,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % numHeads != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by numHeads ({numHeads}).");
        }

        if (numHeads % numKVHeads != 0)
        {
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKVHeads ({numKVHeads}).");
        }

        _numHeads = numHeads;
        _numKVHeads = numKVHeads;
        _headDimension = embeddingDimension / numHeads;
        _embeddingDimension = embeddingDimension;
        _headsPerGroup = numHeads / numKVHeads;

        // Q projection: full-sized [embDim, numHeads * headDim]
        _queryWeights = new Tensor<T>([embeddingDimension, numHeads * _headDimension]);
        // K/V projections: reduced [embDim, numKVHeads * headDim]
        _keyWeights = new Tensor<T>([embeddingDimension, numKVHeads * _headDimension]);
        _valueWeights = new Tensor<T>([embeddingDimension, numKVHeads * _headDimension]);
        // Output projection: [numHeads * headDim, embDim]
        _outputWeights = new Tensor<T>([numHeads * _headDimension, embeddingDimension]);
        _outputBias = new Tensor<T>([embeddingDimension]);

        InitializeParameters();
    }

    /// <summary>
    /// Configures positional encoding for this GQA layer.
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
        }
    }

    private void InitializeParameters()
    {
        // Xavier initialization
        InitializeTensor(_queryWeights);
        InitializeTensor(_keyWeights);
        InitializeTensor(_valueWeights);
        InitializeTensor(_outputWeights);
        _outputBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(
                NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int embDim = input.Shape[rank - 1];

        // Flatten to 3D [batch, seq, embDim]
        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? input.Reshape(1, seqLen, embDim)
            : input.Reshape(batchSize, seqLen, embDim);

        _lastInput = input3D;

        // Project Q, K, V
        var input2D = input3D.Reshape(batchSize * seqLen, embDim);
        var Q_flat = Engine.TensorMatMul(input2D, _queryWeights);
        var K_flat = Engine.TensorMatMul(input2D, _keyWeights);
        var V_flat = Engine.TensorMatMul(input2D, _valueWeights);

        // Reshape Q: [batch, seq, numHeads, headDim] -> [batch, numHeads, seq, headDim]
        var queries = Q_flat.Reshape(batchSize, seqLen, _numHeads, _headDimension)
            .Transpose(new[] { 0, 2, 1, 3 });

        // Reshape K/V: [batch, seq, numKVHeads, headDim] -> [batch, numKVHeads, seq, headDim]
        var keys = K_flat.Reshape(batchSize, seqLen, _numKVHeads, _headDimension)
            .Transpose(new[] { 0, 2, 1, 3 });
        var values = V_flat.Reshape(batchSize, seqLen, _numKVHeads, _headDimension)
            .Transpose(new[] { 0, 2, 1, 3 });

        // Apply RoPE to Q and K (before KV head expansion)
        if (_ropeLayer != null)
        {
            (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: 0);
        }

        _lastProjectedQueries = queries;
        _lastProjectedKeys = keys;
        _lastProjectedValues = values;

        // Expand K/V heads to match Q heads via repeat
        var expandedKeys = ExpandKVHeads(keys, batchSize, seqLen);
        var expandedValues = ExpandKVHeads(values, batchSize, seqLen);

        _lastExpandedKeys = expandedKeys;
        _lastExpandedValues = expandedValues;

        // Compute attention with weights caching: [batch, numHeads, seqQ, seqKV]
        Tensor<T> attentionWeights;
        Tensor<T> context;
        if (_alibiLayer != null)
        {
            var aliBiBias = _alibiLayer.ComputeBias(seqLen, seqLen);
            var flashConfig = FlashAttentionConfig.Default;
            flashConfig.ReturnAttentionWeights = true;
            var (flashOutput, flashWeights) = FlashAttention<T>.Forward(queries, expandedKeys, expandedValues, flashConfig, attentionBias: aliBiBias);
            context = flashOutput;
            attentionWeights = flashWeights ?? new Tensor<T>(new[] { batchSize, _numHeads, seqLen, seqLen });
        }
        else
        {
            context = ComputeStandardAttention(queries, expandedKeys, expandedValues, out attentionWeights);
        }

        _lastAttentionWeights = attentionWeights;

        // Reshape back: [batch, numHeads, seq, headDim] -> [batch, seq, embDim]
        var contextTransposed = context.Transpose(new[] { 0, 2, 1, 3 })
            .Reshape(batchSize * seqLen, _numHeads * _headDimension);

        // Cache pre-projection context for output weights gradient
        _lastAttentionContext = contextTransposed.Reshape(batchSize, seqLen, _numHeads * _headDimension);

        // Output projection
        var output = Engine.TensorMatMul(contextTransposed, _outputWeights);
        var output3D = output.Reshape(batchSize, seqLen, _embeddingDimension);

        // Add bias
        var biasBroadcast = _outputBias.Reshape(1, 1, _embeddingDimension);
        var outputWithBias = Engine.TensorBroadcastAdd(output3D, biasBroadcast);
        var result = ApplyActivation(outputWithBias);

        _lastOutput = result;

        // Reshape back to original rank
        if (rank == 2)
            return result.Reshape(seqLen, _embeddingDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _embeddingDimension;
        return result.Reshape(outputShape);
    }

    /// <summary>
    /// Expands K/V from [batch, numKVHeads, seq, headDim] to [batch, numHeads, seq, headDim]
    /// by repeating each KV head headsPerGroup times.
    /// </summary>
    private Tensor<T> ExpandKVHeads(Tensor<T> kv, int batchSize, int seqLen)
    {
        if (_numKVHeads == _numHeads)
            return kv; // No expansion needed (standard MHA)

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

    private Tensor<T> ComputeStandardAttention(Tensor<T> queries, Tensor<T> keys, Tensor<T> values, out Tensor<T> attentionWeightsOut)
    {
        int batchSize = queries.Shape[0];
        int numHeads = queries.Shape[1];
        int seqLenQ = queries.Shape[2];
        int seqLenKV = keys.Shape[2];
        int headDim = queries.Shape[3];

        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));
        T negInf = NumOps.FromDouble(double.NegativeInfinity);

        var output = new Tensor<T>(new[] { batchSize, numHeads, seqLenQ, headDim });
        attentionWeightsOut = new Tensor<T>(new[] { batchSize, numHeads, seqLenQ, seqLenKV });

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < seqLenQ; i++)
                {
                    var scores = new T[seqLenKV];
                    T maxScore = negInf;
                    for (int j = 0; j < seqLenKV; j++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot = NumOps.Add(dot, NumOps.Multiply(
                                queries[new[] { b, h, i, d }],
                                keys[new[] { b, h, j, d }]));
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

                    for (int j = 0; j < seqLenKV; j++)
                    {
                        T w = NumericalStabilityHelper.SafeDiv(weights[j], sumExp);
                        attentionWeightsOut[new[] { b, h, i, j }] = w;
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int j = 0; j < seqLenKV; j++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(
                                attentionWeightsOut[new[] { b, h, i, j }],
                                values[new[] { b, h, j, d }]));
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
        if (_lastInput == null || _lastOutput == null ||
            _lastProjectedQueries == null || _lastExpandedKeys == null ||
            _lastExpandedValues == null || _lastAttentionWeights == null ||
            _lastAttentionContext == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int rank = outputGradient.Shape.Length;
        int seqLen = rank >= 2 ? outputGradient.Shape[rank - 2] : 1;
        int embDim = outputGradient.Shape[rank - 1];

        int batchSize = _lastInput.Shape[0];
        int seqLength = _lastInput.Shape[1];

        // Normalize gradient to 3D
        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, seqLen, embDim)
            : outputGradient.Reshape(batchSize, seqLength, _embeddingDimension);

        // Apply activation derivative
        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // 1. Output bias gradient: sum over batch and sequence
        _outputBiasGradient = activationGrad.Sum([0, 1]);

        // 2. Output weights gradient: context^T @ dOut
        // context: [batch, seq, numHeads*headDim], dOut: [batch, seq, embDim]
        _outputWeightsGradient = _lastAttentionContext.Transpose([0, 2, 1])
            .Multiply(activationGrad)
            .Sum([0])
            .Reshape([_numHeads * _headDimension, _embeddingDimension]);

        // 3. Gradient through output projection: dOut @ W_o^T -> [batch, seq, numHeads*headDim]
        var dContext = activationGrad.Multiply(_outputWeights.Transpose([1, 0]));

        // 4. Reshape to [batch, numHeads, seq, headDim]
        var dContext4D = dContext.Reshape(batchSize, seqLength, _numHeads, _headDimension)
            .Transpose(new[] { 0, 2, 1, 3 });

        // 5. Backward through scaled dot-product attention
        // Using Engine.ScaledDotProductAttentionBackward with expanded K/V
        Engine.ScaledDotProductAttentionBackward(
            dContext4D,
            _lastProjectedQueries,
            _lastExpandedKeys,
            _lastExpandedValues,
            _lastAttentionWeights,
            1.0 / Math.Sqrt(_headDimension),
            out var dQ_4D,
            out var dExpandedK_4D,
            out var dExpandedV_4D);

        // 6. GQA-specific: aggregate expanded K/V gradients back to numKVHeads
        // dExpandedK/V: [batch, numHeads, seq, headDim] -> dK/V: [batch, numKVHeads, seq, headDim]
        var dK_4D = AggregateKVGradients(dExpandedK_4D, batchSize, seqLength);
        var dV_4D = AggregateKVGradients(dExpandedV_4D, batchSize, seqLength);

        // 6b. Apply inverse RoPE rotation to Q/K gradients
        // The forward pass cached post-RoPE Q/K, so attention backward gives gradients w.r.t.
        // rotated Q/K. We need gradients w.r.t. pre-rotation Q/K for correct weight updates.
        if (_ropeLayer != null)
        {
            dQ_4D = _ropeLayer.Backward(dQ_4D);
            dK_4D = _ropeLayer.Backward(dK_4D);
        }

        // 7. Reshape gradients from 4D to 2D for weight gradient computation
        var dQ_flat = dQ_4D.Transpose(new[] { 0, 2, 1, 3 })
            .Reshape(batchSize * seqLength, _numHeads * _headDimension);
        var dK_flat = dK_4D.Transpose(new[] { 0, 2, 1, 3 })
            .Reshape(batchSize * seqLength, _numKVHeads * _headDimension);
        var dV_flat = dV_4D.Transpose(new[] { 0, 2, 1, 3 })
            .Reshape(batchSize * seqLength, _numKVHeads * _headDimension);

        // 8. Compute projection weight gradients: input^T @ dProjection
        var input2D = _lastInput.Reshape(batchSize * seqLength, _embeddingDimension);
        var input2DT = input2D.Transpose([1, 0]);

        _queryWeightsGradient = Engine.TensorMatMul(input2DT, dQ_flat);
        _keyWeightsGradient = Engine.TensorMatMul(input2DT, dK_flat);
        _valueWeightsGradient = Engine.TensorMatMul(input2DT, dV_flat);

        // 9. Compute input gradient: dInput = dQ @ W_q^T + dK @ W_k^T + dV @ W_v^T
        var inputGradient = Engine.TensorMatMul(dQ_flat, _queryWeights.Transpose([1, 0]));
        inputGradient = Engine.TensorAdd(inputGradient,
            Engine.TensorMatMul(dK_flat, _keyWeights.Transpose([1, 0])));
        inputGradient = Engine.TensorAdd(inputGradient,
            Engine.TensorMatMul(dV_flat, _valueWeights.Transpose([1, 0])));

        var inputGrad3D = inputGradient.Reshape(batchSize, seqLength, _embeddingDimension);

        // Reshape back to original input rank
        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return inputGrad3D.Reshape(seqLength, _embeddingDimension);

        if (_originalInputShape != null)
            return inputGrad3D.Reshape(_originalInputShape);

        return inputGrad3D;
    }

    /// <summary>
    /// Aggregates expanded K/V gradients from [batch, numHeads, seq, headDim]
    /// back to [batch, numKVHeads, seq, headDim] by summing across head groups.
    /// </summary>
    private Tensor<T> AggregateKVGradients(Tensor<T> expandedGrad, int batchSize, int seqLen)
    {
        if (_numKVHeads == _numHeads)
            return expandedGrad; // No aggregation needed (standard MHA)

        var aggregated = new Tensor<T>(new[] { batchSize, _numKVHeads, seqLen, _headDimension });

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
                            aggregated[new[] { b, kvh, s, d }] = NumOps.Add(
                                aggregated[new[] { b, kvh, s, d }],
                                expandedGrad[new[] { b, qh, s, d }]);
                        }
                    }
                }
            }
        }

        return aggregated;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _outputWeights = Engine.TensorAdd(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGradient!, negLR));
        _outputBias = Engine.TensorAdd(_outputBias, Engine.TensorMultiplyScalar(_outputBiasGradient!, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int totalParams = _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
                          _outputWeights.Length + _outputBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var tensor in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights, _outputBias })
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
                             _outputWeights.Length + _outputBias.Length;
        if (parameters.Length != expectedParams)
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");

        int index = 0;
        foreach (var tensor in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights, _outputBias })
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedQueries = null;
        _lastProjectedKeys = null;
        _lastProjectedValues = null;
        _lastExpandedKeys = null;
        _lastExpandedValues = null;
        _lastAttentionWeights = null;
        _lastAttentionContext = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return inputNode;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["NumKVHeads"] = _numKVHeads.ToString();
        metadata["HeadsPerGroup"] = _headsPerGroup.ToString();
        metadata["Variant"] = Variant.ToString();
        metadata["PositionalEncoding"] = PositionalEncoding.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the query projection weights for external use (e.g., quantization).
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the key projection weights for external use.
    /// </summary>
    public Tensor<T> GetKeyWeights() => _keyWeights;

    /// <summary>
    /// Gets the value projection weights for external use.
    /// </summary>
    public Tensor<T> GetValueWeights() => _valueWeights;

    /// <summary>
    /// Gets the output projection weights for external use.
    /// </summary>
    public Tensor<T> GetOutputWeights() => _outputWeights;
}
