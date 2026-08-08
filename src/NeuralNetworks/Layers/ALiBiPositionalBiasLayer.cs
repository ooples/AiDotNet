using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;

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
[LayerCategory(LayerCategory.Positional)]
[LayerTask(LayerTask.PositionalEncoding)]
[LayerProperty(IsTrainable = false, TestInputShape = "2, 4, 4", TestConstructorArgs = "2, 4", ProducesNonFiniteOutput = true)]
// Ranks 3 and 4 ONLY, and that is the layer's own rule, not a guess: ForwardTraced names
// "3D [heads, qLen, kLen] or 4D [batch, heads, qLen, kLen]" and throws for every other rank.
//
// This operates on an ATTENTION SCORE MATRIX, so it has two sequence axes rather than one, and roles in
// a single layout must be distinct (ADNSHAPE002). Query positions take Time - they are the axis causal
// masking is defined over, and ComputeBias rejects queryLen > keyLen for exactly that reason - and key
// positions take Length. Naming them apart is what lets a relation refer to one without the other; here
// every relation is Same(role) anyway, since the forward rents an output of `input._shape` and only adds
// a bias, so the choice cannot change a resolved size.
//
// Two separate declarations rather than one with BatchOptional, because the derived contract is keyed on
// the declared axis count: a batch-optional layout would leave the unbatched rank resolving to nothing.
[TensorLayout(TensorAxis.Heads, TensorAxis.Time, TensorAxis.Length,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Heads, TensorAxis.Time, TensorAxis.Length,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Heads, TensorAxis.Time, TensorAxis.Length,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Heads, TensorAxis.Time, TensorAxis.Length,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ALiBiPositionalBiasLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _numHeads;
    private readonly int _maxSequenceLength;

    /// <summary>
    /// Pre-computed per-head slopes: slope_h = 2^(-8/numHeads * (h+1)).
    /// Shape: [numHeads].
    /// </summary>
    private readonly Tensor<T> _slopes;

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
        var slopeData = new T[numHeads];
        for (int h = 0; h < numHeads; h++)
        {
            double exponent = -8.0 / numHeads * (h + 1);
            slopeData[h] = NumOps.FromDouble(Math.Pow(2.0, exponent));
        }
        _slopes = new Tensor<T>(slopeData, new[] { numHeads });
        RegisterBuffer(_slopes, nameof(_slopes));
    }

    /// <summary>
    /// Computes the ALiBi bias tensor for the given query and key lengths.
    /// </summary>
    /// <param name="queryLen">Number of query positions.</param>
    /// <param name="keyLen">Number of key positions.</param>
    /// <param name="useCausalMask">Whether to apply causal masking (future positions get -inf). Default: true.</param>
    /// <returns>Bias tensor of shape [numHeads, queryLen, keyLen].</returns>
    /// <remarks>
    /// Masked positions are filled with <see cref="double.NegativeInfinity"/>
    /// (via <c>NumOps.FromDouble(double.NegativeInfinity)</c>), NOT
    /// <c>NumOps.MinValue</c>. A literal −∞ guarantees the downstream softmax
    /// sees <c>exp(−∞) = 0</c> exactly on masked positions; <c>MinValue</c>
    /// is a large finite value (~−3.4e38 for float) whose exp underflows to
    /// 0 most of the time but can leak tiny non-zero attention weight under
    /// FP intermediate ordering, and any code path that ADDS the bias
    /// without going through softmax (e.g. residual-summing) would
    /// accumulate a real finite penalty instead of cleanly masking. The
    /// <c>[LayerProperty(ProducesNonFiniteOutput = true)]</c> annotation
    /// on this class tells the test scaffold generator to skip the
    /// <c>Forward_ShouldProduceFiniteOutput</c> invariant.
    /// </remarks>
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
        // True −Infinity for masked positions — see ComputeBias remarks
        // for the full rationale (softmax exactness vs MinValue, the
        // ProducesNonFiniteOutput annotation contract).
        T negInf = NumOps.FromDouble(double.NegativeInfinity);

        for (int h = 0; h < _numHeads; h++)
        {
            T slope = _slopes[h];

            for (int i = 0; i < queryLen; i++)
            {
                // For KV-cached causal decoding (queryLen=1, keyLen=full_seq):
                //   effectiveQueryPos = 0 + (full_seq - 1) = last position
                // For full attention without causal masking, positions are just i (no offset).
                int effectiveQueryPos = useCausalMask ? i + (keyLen - queryLen) : i;

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
    public T[] GetSlopes() => _slopes.GetDataArray();

    /// <summary>
    /// Forward pass adds ALiBi bias to the input attention scores tensor.
    /// </summary>
    /// <param name="input">Attention scores tensor of shape [batch, numHeads, queryLen, keyLen]
    /// or [numHeads, queryLen, keyLen].</param>
    /// <returns>Biased attention scores with the same shape.</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
        var output = TensorAllocator.Rent<T>(input._shape);

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



    // GetParameters is deliberately NOT overridden. This layer has no trainable weights, but it
    // DOES own a registered buffer -- the ALiBi slope table -- and returning Vector<T>.Empty()
    // excluded it from every checkpoint. LayerBase now folds registered buffers into the flat
    // vector alongside parameters, so the slopes save and restore with the model while staying
    // invisible to the optimizer, which reads GetTrainableParameters() and still sees nothing here.

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
}
