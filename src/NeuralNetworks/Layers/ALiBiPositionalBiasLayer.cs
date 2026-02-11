using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements Attention with Linear Biases (ALiBi) from Press et al., 2022.
/// </summary>
/// <remarks>
/// <para>
/// ALiBi adds a position-dependent linear bias to attention scores instead of modifying
/// input embeddings. Each attention head uses a different slope, and the bias for head h
/// between query position i and key position j is: bias[h, i, j] = -slope_h * |i - j|.
/// </para>
/// <para>
/// Per-head slopes follow a geometric sequence: slope_h = 2^(-8/numHeads * (h+1)).
/// This ensures different heads attend at different distance scales, from very local
/// to broader context.
/// </para>
/// <para><b>For Beginners:</b> ALiBi is a simple way to encode position in attention.
///
/// Instead of adding position embeddings to tokens, ALiBi penalizes attention scores
/// based on how far apart two tokens are:
/// - Nearby tokens: small penalty (easy to attend to)
/// - Far away tokens: large penalty (harder to attend to)
/// - Different heads use different penalty strengths
///
/// Benefits:
/// - No extra parameters to learn
/// - Excellent length extrapolation (works well on longer sequences than training)
/// - Very simple to implement and efficient
///
/// Used by BLOOM, MPT, and some Falcon variants.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
internal class ALiBiPositionalBiasLayer<T> : LayerBase<T>
{
    private readonly int _numHeads;
    private readonly int _maxSequenceLength;

    /// <summary>
    /// Pre-computed per-head slopes: slope_h = 2^(-8/numHeads * (h+1)).
    /// Shape: [numHeads].
    /// </summary>
    private readonly T[] _slopes;

    /// <summary>
    /// Pre-computed bias tensor [numHeads, maxSequenceLength, maxSequenceLength].
    /// Lazily computed on first use and cached.
    /// </summary>
    private Tensor<T>? _biasCache;
    private int _biasCacheQueryLen;
    private int _biasCacheKeyLen;
    private readonly object _biasLock = new();

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Creates a new ALiBi positional bias layer.
    /// </summary>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="maxSequenceLength">Initial maximum sequence length for pre-computation (auto-extends).</param>
    public ALiBiPositionalBiasLayer(int numHeads, int maxSequenceLength = 2048)
        : base([numHeads, maxSequenceLength, maxSequenceLength], [numHeads, maxSequenceLength, maxSequenceLength])
    {
        if (numHeads <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be greater than zero.");
        }

        if (maxSequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "maxSequenceLength must be greater than zero.");
        }

        _numHeads = numHeads;
        _maxSequenceLength = maxSequenceLength;

        // Compute per-head slopes: slope_h = 2^(-8/numHeads * (h+1))
        _slopes = new T[numHeads];
        for (int h = 0; h < numHeads; h++)
        {
            double exponent = -8.0 / numHeads * (h + 1);
            _slopes[h] = NumOps.FromDouble(Math.Pow(2.0, exponent));
        }
    }

    /// <summary>
    /// Computes the ALiBi bias tensor for the given query and key lengths.
    /// </summary>
    /// <param name="queryLen">Number of query positions.</param>
    /// <param name="keyLen">Number of key positions.</param>
    /// <param name="useCausalMask">Whether to apply causal masking (future positions get -inf). Default: true.</param>
    /// <returns>Bias tensor of shape [numHeads, queryLen, keyLen].</returns>
    public Tensor<T> ComputeBias(int queryLen, int keyLen, bool useCausalMask = true)
    {
        if (useCausalMask && queryLen > keyLen)
        {
            throw new ArgumentException(
                $"Causal ALiBi requires queryLen ({queryLen}) <= keyLen ({keyLen}).",
                nameof(queryLen));
        }

        // Only use cache for default causal masking (most common path)
        if (useCausalMask)
        {
            lock (_biasLock)
            {
                if (_biasCache != null && _biasCacheQueryLen == queryLen && _biasCacheKeyLen == keyLen)
                {
                    return _biasCache;
                }
            }
        }

        var bias = new Tensor<T>([_numHeads, queryLen, keyLen]);
        T negInf = NumOps.FromDouble(double.NegativeInfinity);

        for (int h = 0; h < _numHeads; h++)
        {
            T slope = _slopes[h];

            for (int i = 0; i < queryLen; i++)
            {
                // Map local query index to absolute position in the full sequence.
                // For KV-cached decoding (queryLen=1, keyLen=full_seq): effectiveQueryPos = 0 + (full_seq - 1) = last position
                // For full attention (queryLen=keyLen): effectiveQueryPos = i
                int effectiveQueryPos = i + (keyLen - queryLen);

                for (int j = 0; j < keyLen; j++)
                {
                    // Causal masking: mask out key positions beyond the effective query position
                    if (useCausalMask && j > effectiveQueryPos)
                    {
                        bias[new[] { h, i, j }] = negInf;
                    }
                    else
                    {
                        // ALiBi bias: -slope * |effective_query_pos - key_pos|
                        int distance = Math.Abs(effectiveQueryPos - j);
                        bias[new[] { h, i, j }] = NumOps.Negate(
                            NumOps.Multiply(slope, NumOps.FromDouble(distance)));
                    }
                }
            }
        }

        if (useCausalMask)
        {
            lock (_biasLock)
            {
                _biasCache = bias;
                _biasCacheQueryLen = queryLen;
                _biasCacheKeyLen = keyLen;
            }
        }

        return bias;
    }

    /// <summary>
    /// Gets the per-head slope values.
    /// </summary>
    /// <returns>Array of slopes, one per head.</returns>
    public T[] GetSlopes() => (T[])_slopes.Clone();

    /// <summary>
    /// Forward pass adds ALiBi bias to the input attention scores tensor.
    /// </summary>
    /// <param name="input">Attention scores tensor of shape [batch, numHeads, queryLen, keyLen]
    /// or [numHeads, queryLen, keyLen].</param>
    /// <returns>Biased attention scores with the same shape.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        int numHeads, queryLen, keyLen;

        if (rank == 4)
        {
            numHeads = input.Shape[1];
            queryLen = input.Shape[2];
            keyLen = input.Shape[3];
        }
        else if (rank == 3)
        {
            numHeads = input.Shape[0];
            queryLen = input.Shape[1];
            keyLen = input.Shape[2];
        }
        else
        {
            throw new ArgumentException(
                $"ALiBi expects 3D [heads, qLen, kLen] or 4D [batch, heads, qLen, kLen] input. Got rank {rank}.");
        }

        if (numHeads != _numHeads)
        {
            throw new ArgumentException(
                $"Expected {_numHeads} heads, got {numHeads}.");
        }

        var bias = ComputeBias(queryLen, keyLen);

        // Add bias to scores
        var output = new Tensor<T>(input.Shape);

        if (rank == 4)
        {
            int batchSize = input.Shape[0];
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int i = 0; i < queryLen; i++)
                    {
                        for (int j = 0; j < keyLen; j++)
                        {
                            output[new[] { b, h, i, j }] = NumOps.Add(
                                input[new[] { b, h, i, j }],
                                bias[new[] { h, i, j }]);
                        }
                    }
                }
            }
        }
        else
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < queryLen; i++)
                {
                    for (int j = 0; j < keyLen; j++)
                    {
                        output[new[] { h, i, j }] = NumOps.Add(
                            input[new[] { h, i, j }],
                            bias[new[] { h, i, j }]);
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass: gradient flows through unchanged (constant additive bias).
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // ALiBi bias is constant, so gradient with respect to input scores is identity.
        return outputGradient;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        // No trainable parameters
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        // Clear cached bias (it will be recomputed on next use)
        lock (_biasLock)
        {
            _biasCache = null;
            _biasCacheQueryLen = 0;
            _biasCacheKeyLen = 0;
        }
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

        // ALiBi is an additive bias; for graph export, treat as identity placeholder
        return inputNode;
    }
}
