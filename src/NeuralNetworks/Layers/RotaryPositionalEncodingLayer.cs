using AiDotNet.Autodiff;

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
public class RotaryPositionalEncodingLayer<T> : LayerBase<T>
{
    private int _maxSequenceLength;
    private readonly int _headDimension;
    private readonly double _theta;

    /// <summary>
    /// Pre-computed cosine values: cos_cache[pos, i] = cos(pos * freq_i).
    /// Shape: [maxSequenceLength, headDimension / 2].
    /// </summary>
    private Tensor<T> _cosCache;

    /// <summary>
    /// Pre-computed sine values: sin_cache[pos, i] = sin(pos * freq_i).
    /// Shape: [maxSequenceLength, headDimension / 2].
    /// </summary>
    private Tensor<T> _sinCache;

    private readonly object _cacheLock = new();

    /// <inheritdoc />
    public override bool SupportsTraining => true;

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

        int halfDim = headDimension / 2;
        _cosCache = new Tensor<T>([maxSequenceLength, halfDim]);
        _sinCache = new Tensor<T>([maxSequenceLength, halfDim]);

        InitializeCache(0, maxSequenceLength);
    }

    /// <summary>
    /// Initializes or extends the cos/sin cache for positions [startPos, endPos).
    /// </summary>
    private void InitializeCache(int startPos, int endPos)
    {
        int halfDim = _headDimension / 2;

        for (int pos = startPos; pos < endPos; pos++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                double freq = 1.0 / Math.Pow(_theta, (2.0 * i) / _headDimension);
                double angle = pos * freq;
                _cosCache[pos, i] = NumOps.FromDouble(Math.Cos(angle));
                _sinCache[pos, i] = NumOps.FromDouble(Math.Sin(angle));
            }
        }
    }

    /// <summary>
    /// Ensures the cache covers at least up to the specified sequence length.
    /// </summary>
    private void EnsureCacheLength(int requiredLength)
    {
        if (requiredLength <= _maxSequenceLength)
            return;

        lock (_cacheLock)
        {
            if (requiredLength <= _maxSequenceLength)
                return;

            int oldLength = _maxSequenceLength;
            _maxSequenceLength = requiredLength;

            int halfDim = _headDimension / 2;
            var newCos = new Tensor<T>([_maxSequenceLength, halfDim]);
            var newSin = new Tensor<T>([_maxSequenceLength, halfDim]);

            // Copy existing cache
            for (int pos = 0; pos < oldLength; pos++)
            {
                for (int i = 0; i < halfDim; i++)
                {
                    newCos[pos, i] = _cosCache[pos, i];
                    newSin[pos, i] = _sinCache[pos, i];
                }
            }

            _cosCache = newCos;
            _sinCache = newSin;

            // Compute new entries
            InitializeCache(oldLength, _maxSequenceLength);
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
        int rank = queries.Shape.Length;
        int seqLen = queries.Shape[rank - 2];
        int headDim = queries.Shape[rank - 1];

        if (headDim != _headDimension)
        {
            throw new ArgumentException(
                $"Expected head dimension {_headDimension}, got {headDim}.");
        }

        int endPosition = startPosition + seqLen;
        EnsureCacheLength(endPosition);

        var rotatedQ = RotateTensor(queries, startPosition);
        var rotatedK = RotateTensor(keys, startPosition);

        return (rotatedQ, rotatedK);
    }

    /// <summary>
    /// Applies RoPE rotation to a single tensor.
    /// </summary>
    private Tensor<T> RotateTensor(Tensor<T> input, int startPosition)
    {
        var output = new Tensor<T>(input.Shape);
        int rank = input.Shape.Length;
        int seqLen = input.Shape[rank - 2];
        int headDim = input.Shape[rank - 1];
        int halfDim = headDim / 2;

        // Compute total number of elements in leading dimensions
        int leadingSize = 1;
        for (int d = 0; d < rank - 2; d++)
        {
            leadingSize *= input.Shape[d];
        }

        // Flatten to [leadingSize, seqLen, headDim] for processing
        int seqStride = headDim;
        int batchStride = seqLen * headDim;

        for (int b = 0; b < leadingSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int pos = startPosition + s;
                int baseIdx = b * batchStride + s * seqStride;

                for (int i = 0; i < halfDim; i++)
                {
                    T cos_val = _cosCache[pos, i];
                    T sin_val = _sinCache[pos, i];

                    T x_even = input[baseIdx + 2 * i];
                    T x_odd = input[baseIdx + 2 * i + 1];

                    // x'[2i]   = x[2i] * cos - x[2i+1] * sin
                    // x'[2i+1] = x[2i] * sin + x[2i+1] * cos
                    output[baseIdx + 2 * i] = NumOps.Subtract(
                        NumOps.Multiply(x_even, cos_val),
                        NumOps.Multiply(x_odd, sin_val));
                    output[baseIdx + 2 * i + 1] = NumOps.Add(
                        NumOps.Multiply(x_even, sin_val),
                        NumOps.Multiply(x_odd, cos_val));
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Applies inverse RoPE rotation (for backward pass).
    /// Since rotation is orthogonal, the inverse is simply rotating by -angle.
    /// </summary>
    private Tensor<T> InverseRotateTensor(Tensor<T> input, int startPosition)
    {
        var output = new Tensor<T>(input.Shape);
        int rank = input.Shape.Length;
        int seqLen = input.Shape[rank - 2];
        int headDim = input.Shape[rank - 1];
        int halfDim = headDim / 2;

        int leadingSize = 1;
        for (int d = 0; d < rank - 2; d++)
        {
            leadingSize *= input.Shape[d];
        }

        int seqStride = headDim;
        int batchStride = seqLen * headDim;

        for (int b = 0; b < leadingSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int pos = startPosition + s;
                int baseIdx = b * batchStride + s * seqStride;

                for (int i = 0; i < halfDim; i++)
                {
                    T cos_val = _cosCache[pos, i];
                    T sin_val = _sinCache[pos, i];

                    T x_even = input[baseIdx + 2 * i];
                    T x_odd = input[baseIdx + 2 * i + 1];

                    // Inverse rotation: use -sin (transpose of rotation matrix)
                    // x'[2i]   = x[2i] * cos + x[2i+1] * sin
                    // x'[2i+1] = -x[2i] * sin + x[2i+1] * cos
                    output[baseIdx + 2 * i] = NumOps.Add(
                        NumOps.Multiply(x_even, cos_val),
                        NumOps.Multiply(x_odd, sin_val));
                    output[baseIdx + 2 * i + 1] = NumOps.Add(
                        NumOps.Negate(NumOps.Multiply(x_even, sin_val)),
                        NumOps.Multiply(x_odd, cos_val));
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Forward pass applies RoPE rotation to the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with shape [..., seqLen, headDim].</param>
    /// <returns>Rotated tensor with the same shape.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return RotateTensor(input, 0);
    }

    /// <summary>
    /// Backward pass applies inverse rotation (orthogonal transpose).
    /// </summary>
    /// <param name="outputGradient">Gradient tensor with same shape as forward output.</param>
    /// <returns>Gradient with inverse rotation applied.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return InverseRotateTensor(outputGradient, 0);
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
        // Cache is stateless across sequences
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

        // RoPE is a unary transform; for graph export, treat as identity placeholder
        return inputNode;
    }
}
