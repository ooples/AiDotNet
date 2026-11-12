using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements Rotary Position Embedding (RoPE) as described by Su et al., 2021.
/// </summary>
/// <remarks>
/// <para>
/// RoPE encodes absolute position by rotating query and key vectors in 2D subspaces
/// using rotation matrices derived from position-dependent frequencies. The dot product
/// of rotated queries and keys naturally captures relative position information.
/// </para>
/// <para>
/// The rotation formula for each pair of dimensions (2i, 2i+1):
/// <code>
/// x'[2i]   = x[2i]   * cos(pos * freq_i) - x[2i+1] * sin(pos * freq_i)
/// x'[2i+1] = x[2i]   * sin(pos * freq_i) + x[2i+1] * cos(pos * freq_i)
/// where freq_i = 1 / theta^(2i / headDim)
/// </code>
/// </para>
/// <para><b>For Beginners:</b> RoPE is the most popular position encoding for modern LLMs.
///
/// Instead of adding position information to embeddings (like sinusoidal encoding),
/// RoPE rotates the query and key vectors based on their position. This means:
/// - The angle of rotation depends on the token's position
/// - When two tokens attend to each other, the rotation encodes their relative distance
/// - This works naturally with KV-caching (just rotate new tokens at their position)
///
/// Used by Llama 2/3, Mistral, Phi-3, Gemma, and most modern LLMs.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Positional)]
[LayerTask(LayerTask.PositionalEncoding)]
[LayerProperty(IsTrainable = false, TestInputShape = "16, 8", TestConstructorArgs = "16, 8")]
// Rotates query/key pairs by position: shape-preserving at rank 2 [Time, Features]. Needs a real
// sequence axis to rotate against, so it is not declared rank-agnostic.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RotaryPositionalEncodingLayer<T> : LayerBase<T>, IShapeContract
{
    private int _maxSequenceLength;
    private readonly int _headDimension;
    private readonly double _theta;

    /// <summary>
    /// Pre-computed cosine values: cos_cache[pos, i] = cos(pos * freq_i).
    /// Shape: [maxSequenceLength, headDimension / 2].
    /// </summary>
    // Memoised cos table for a deterministic function of position; recomputed on demand, never learned.
    [Scratch]
    private Tensor<T> _cosCache;

    /// <summary>
    /// Pre-computed sine values: sin_cache[pos, i] = sin(pos * freq_i).
    /// Shape: [maxSequenceLength, headDimension / 2].
    /// </summary>
    // Memoised sin table for a deterministic function of position; recomputed on demand, never learned.
    [Scratch]
    private Tensor<T> _sinCache;

    private readonly object _cacheLock = new();

    /// <inheritdoc />
    public override bool SupportsTraining => false; // Precomputed frequency caches, no trainable parameters

    /// <summary>
    /// Gets the head dimension this RoPE layer operates on.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the base frequency parameter theta.
    /// </summary>
    public double Theta => _theta;

    /// <summary>
    /// Creates a new Rotary Position Embedding layer.
    /// </summary>
    /// <param name="maxSequenceLength">Initial maximum sequence length for pre-computation (auto-extends).</param>
    /// <param name="headDimension">Dimension of each attention head (must be even).</param>
    /// <param name="theta">Base frequency parameter (default: 10000.0, standard for Llama/Mistral).</param>
    public RotaryPositionalEncodingLayer(int maxSequenceLength, int headDimension, double theta = 10000.0)
        : base([maxSequenceLength, headDimension], [maxSequenceLength, headDimension])
    {
        if (headDimension % 2 != 0)
        {
            throw new ArgumentException(
                $"Head dimension must be even for RoPE rotation pairs. Got: {headDimension}",
                nameof(headDimension));
        }

        _maxSequenceLength = maxSequenceLength;
        _headDimension = headDimension;
        _theta = theta;

        (_cosCache, _sinCache) = BuildCacheRows(0, maxSequenceLength);
    }

    /// <summary>
    /// Builds the cos/sin cache rows for positions [startPos, endPos):
    /// angle[pos, i] = pos * freq_i with freq_i = 1 / theta^(2i / headDim).
    /// </summary>
    private (Tensor<T> Cos, Tensor<T> Sin) BuildCacheRows(int startPos, int endPos)
    {
        int halfDim = _headDimension / 2;
        int count = endPos - startPos;
        if (count <= 0 || halfDim <= 0)
        {
            return (new Tensor<T>([Math.Max(0, count), halfDim]), new Tensor<T>([Math.Max(0, count), halfDim]));
        }

        // The caches are constants built from fresh tensors, so they stay off the tape.
        using var noGrad = new NoGradScope<T>();
        var positions = Engine.TensorArange<T>(startPos, endPos, 1.0);                                // [count]
        var frequencies = Engine.TensorLogspace<T>(0.0, -(2.0 * (halfDim - 1)) / _headDimension, halfDim, _theta);
        var angles = Engine.TensorOuter(positions, frequencies);                                      // [count, halfDim]
        return (Engine.TensorCos(angles), Engine.TensorSin(angles));
    }

    /// <summary>
    /// Ensures the cache covers at least up to the specified sequence length.
    /// </summary>
    /// <summary>
    /// Ensures the cos/sin caches cover <paramref name="requiredLength"/> positions and returns them as
    /// <c>[maxSeq, headDim/2]</c> tensors laid out for the fused interleaved-RoPE engine op
    /// (<see cref="AiDotNet.Tensors.Engines.IEngine.ApplyRoPEInterleaved"/>). Lets a caller apply RoPE as a
    /// single device-agnostic (and, on GPU, graph-recordable) op instead of this layer's managed rotation.
    /// </summary>
    public (Tensor<T> Cos, Tensor<T> Sin) GetInterleavedCaches(int requiredLength)
    {
        EnsureCacheLength(requiredLength);
        return (_cosCache, _sinCache);
    }

    private void EnsureCacheLength(int requiredLength)
    {
        if (requiredLength <= _maxSequenceLength)
            return;

        lock (_cacheLock)
        {
            if (requiredLength <= _maxSequenceLength)
                return;

            int oldLength = _maxSequenceLength;

            // Keep the existing rows and append the new positions; build both tables before
            // swapping so other threads checking _maxSequenceLength never see positions beyond
            // what the current cache tensors actually contain.
            var (extraCos, extraSin) = BuildCacheRows(oldLength, requiredLength);
            Tensor<T> newCos;
            Tensor<T> newSin;
            using (new NoGradScope<T>())
            {
                newCos = Engine.TensorConcatenate(new[] { _cosCache, extraCos }, 0);
                newSin = Engine.TensorConcatenate(new[] { _sinCache, extraSin }, 0);
            }

            _cosCache = newCos;
            _sinCache = newSin;

            // Update length last so concurrent readers never index beyond cache bounds
            _maxSequenceLength = requiredLength;
        }
    }

    /// <summary>
    /// Applies RoPE rotation to query and key tensors.
    /// </summary>
    /// <param name="queries">Query tensor with shape [..., seqLen, headDim].</param>
    /// <param name="keys">Key tensor with shape [..., seqLen, headDim].</param>
    /// <param name="startPosition">Starting position offset (for incremental decoding with KV-cache).</param>
    /// <returns>Tuple of rotated (queries, keys) with the same shapes.</returns>
    public (Tensor<T> RotatedQueries, Tensor<T> RotatedKeys) ApplyRoPE(
        Tensor<T> queries, Tensor<T> keys, int startPosition = 0)
    {
        if (startPosition < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(startPosition),
                $"startPosition must be non-negative. Got: {startPosition}.");
        }

        int rank = queries.Shape.Length;
        int seqLen = queries.Shape[rank - 2];
        int headDim = queries.Shape[rank - 1];

        if (headDim != _headDimension)
        {
            throw new ArgumentException(
                $"Expected head dimension {_headDimension}, got {headDim}.");
        }

        // Validate key tensor shape matches query head dimension
        int keyHeadDim = keys.Shape[keys.Shape.Length - 1];
        if (keyHeadDim != _headDimension)
        {
            throw new ArgumentException(
                $"Key head dimension ({keyHeadDim}) must match query head dimension ({_headDimension}).",
                nameof(keys));
        }

        int endPosition = startPosition + seqLen;
        EnsureCacheLength(endPosition);

        // Route the rotation through the engine primitive. Besides using the same optimized
        // CPU/GPU implementation as decoder inference, ApplyRoPEInterleaved records its inverse
        // rotation on an active gradient tape. The former scalar-copy implementation returned
        // detached tensors and silently discarded every Q/K contribution during tape training.
        var rotatedQ = Engine.ApplyRoPEInterleaved(
            queries, _cosCache, _sinCache, startPosition);
        var rotatedK = Engine.ApplyRoPEInterleaved(
            keys, _cosCache, _sinCache, startPosition);

        return (rotatedQ, rotatedK);
    }

    /// <summary>
    /// Applies inverse RoPE rotation to query and key gradients (for backward pass).
    /// Since RoPE rotation is orthogonal, the inverse is rotation by the negative angle.
    /// </summary>
    /// <param name="gradQuery">Gradient tensor for queries in the rotated space.</param>
    /// <param name="gradKey">Gradient tensor for keys in the rotated space.</param>
    /// <param name="startPosition">Starting position offset (must match the forward pass).</param>
    /// <returns>Tuple of un-rotated (gradQuery, gradKey) with the same shapes.</returns>
    public (Tensor<T> GradQuery, Tensor<T> GradKey) ApplyInverseRoPE(
        Tensor<T> gradQuery, Tensor<T> gradKey, int startPosition = 0)
    {
        int seqLen = gradQuery.Shape[gradQuery.Shape.Length - 2];
        int endPosition = startPosition + seqLen;
        EnsureCacheLength(endPosition);

        var unrotatedGradQ = InverseRotateTensor(gradQuery, startPosition);
        var unrotatedGradK = InverseRotateTensor(gradKey, startPosition);

        return (unrotatedGradQ, unrotatedGradK);
    }

    /// <summary>
    /// Applies inverse RoPE rotation (for backward pass).
    /// Since rotation is orthogonal, the inverse is simply rotating by -angle.
    /// </summary>
    private Tensor<T> InverseRotateTensor(Tensor<T> input, int startPosition)
    {
        // Inverse rotation: use -sin (transpose of rotation matrix)
        // x'[2i]   = x[2i] * cos + x[2i+1] * sin
        // x'[2i+1] = -x[2i] * sin + x[2i+1] * cos
        // i.e. the forward interleaved rotation with the sine table negated. Manual backward
        // into a fresh tensor, as master: not recorded on the tape.
        using var noGrad = new NoGradScope<T>();
        return Engine.ApplyRoPEInterleaved(input, _cosCache, Engine.TensorNegate(_sinCache), startPosition);
    }

    /// <summary>
    /// Forward pass applies RoPE rotation to the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with shape [..., seqLen, headDim].</param>
    /// <returns>Rotated tensor with the same shape.</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 2)
        {
            throw new ArgumentException(
                $"RoPE input rank must be at least 2 ([..., sequence, head dimension]); got rank {rank}.",
                nameof(input));
        }

        int headDim = input.Shape[rank - 1];
        if (headDim != _headDimension)
        {
            throw new ArgumentException(
                $"RoPE expected head dimension {_headDimension}, got {headDim}.",
                nameof(input));
        }

        int seqLen = input.Shape[rank - 2];
        EnsureCacheLength(seqLen);

        return Engine.ApplyRoPEInterleaved(input, _cosCache, _sinCache, startPosition: 0);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        // Cache is stateless across sequences
    }
}
