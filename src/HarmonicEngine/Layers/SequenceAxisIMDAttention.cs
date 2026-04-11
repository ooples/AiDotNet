using AiDotNet.HarmonicEngine.Core;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Layers;

/// <summary>
/// Sequence-axis IMD attention: a drop-in replacement for transformer self-attention
/// that computes the S×S attention pattern via a single FFT over the sequence axis
/// instead of the O(S²·d_k) Q·K^T dot product.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard self-attention computes a score between every pair of
/// tokens in a sequence by multiplying query and key vectors: Q·K^T. This costs O(S²·d_k)
/// for a sequence of length S and key dimension d_k. Sequence-axis IMD attention replaces
/// this with a single FFT:
///
/// 1. Pool each token's embedding vector down to a single scalar "amplitude" per position
///    (e.g., take the L2 norm across the embedding dimension).
/// 2. Place the S pooled amplitudes onto S orthogonal frequency carriers via OFDM.
/// 3. Square the resulting time-domain signal (creates intermodulation products).
/// 4. Take one FFT. The intermodulation products at frequencies f_i ± f_j encode the
///    pairwise interaction a_i·a_j — exactly the attention score between positions i and j.
/// 5. Read the S² scores from the FFT output and softmax-normalize them per row.
/// 6. Apply the resulting attention matrix to the full [B, S, E] value tensor to produce
///    attended outputs.
///
/// This turns the score computation from O(S²·d_k) into O(S log S + S²), a substantial
/// saving when the embedding dimension is large.
/// </para>
/// <para>
/// The layer has zero learnable parameters — the attention pattern emerges from wave
/// interference of the input itself, not from learned Q/K/V projection matrices. This
/// is a deliberate architectural choice: HRE's training story is that the Hebbian FFN
/// layers are where learning happens, and the attention pattern is a free consequence
/// of the input statistics.
/// </para>
/// </remarks>
public class SequenceAxisIMDAttention<T> : LayerBase<T>
{
    private readonly SpectralBus<T> _bus;
    private readonly IMDExtractor<T> _extractor;
    private readonly int _seqLen;
    private readonly int _embedDim;
    private readonly int _fftSize;

    private Matrix<T>? _lastAttentionWeights;

    /// <inheritdoc/>
    public override string LayerName => $"SeqAxisIMDAttn_{_seqLen}s_{_embedDim}e";

    /// <inheritdoc/>
    public override int ParameterCount => 0;

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets the most recent attention weight matrix for the last-seen batch item 0
    /// (for visualization and debugging).
    /// </summary>
    public Matrix<T>? LastAttentionWeights => _lastAttentionWeights;

    /// <summary>
    /// Creates a new sequence-axis IMD attention layer.
    /// </summary>
    /// <param name="seqLen">Sequence length S.</param>
    /// <param name="embedDim">Embedding dimension E. Must be equal to CarrierCount in the
    /// current "no projections" architecture — the embedding dimensions are used directly
    /// as carrier slots in the FFN, but the attention pattern is computed on the sequence axis.</param>
    /// <param name="fftSize">FFT size used for IMD extraction. Must be a power of 2 and
    /// large enough to allocate S collision-free carriers via the Sidon-set algorithm.
    /// Typically 4S² is a safe choice.</param>
    public SequenceAxisIMDAttention(int seqLen, int embedDim, int fftSize)
        : base([seqLen, embedDim], [seqLen, embedDim])
    {
        if (seqLen < 2)
            throw new ArgumentOutOfRangeException(nameof(seqLen), "Sequence length must be at least 2.");
        if (embedDim < 1)
            throw new ArgumentOutOfRangeException(nameof(embedDim), "Embedding dimension must be at least 1.");

        _seqLen = seqLen;
        _embedDim = embedDim;
        _fftSize = fftSize;

        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(seqLen, fftSize);
        _bus = new SpectralBus<T>(carriers, fftSize);
        _extractor = new IMDExtractor<T>(carriers, fftSize);

        Parameters = Vector<T>.Empty();
    }

    /// <summary>
    /// Forward pass. Computes the S×S attention pattern from a pooled sequence
    /// representation via IMD, then applies it to the full [B, S, E] value tensor.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Accept either [B, S, E] (batched) or [S, E] (unbatched). Normalize to [B, S, E].
        bool was2D = input._shape.Length == 2;
        int batchSize = was2D ? 1 : input._shape[0];
        int actualSeqLen = was2D ? input._shape[0] : input._shape[1];
        int actualEmbedDim = was2D ? input._shape[1] : input._shape[2];

        if (actualSeqLen != _seqLen)
            throw new ArgumentException(
                $"Input sequence length ({actualSeqLen}) must match layer's seqLen ({_seqLen}).");
        if (actualEmbedDim != _embedDim)
            throw new ArgumentException(
                $"Input embedding dimension ({actualEmbedDim}) must match layer's embedDim ({_embedDim}).");

        var output = was2D
            ? new Tensor<T>([_seqLen, _embedDim])
            : new Tensor<T>([batchSize, _seqLen, _embedDim]);

        for (int b = 0; b < batchSize; b++)
        {
            // Step 1: pool the [S, E] slice down to a per-position scalar amplitude.
            // We use the L2 norm across the embedding dimension, which is position-
            // translation-invariant and captures "how much energy does this token carry".
            var poolAmplitudes = new Vector<T>(_seqLen);
            for (int s = 0; s < _seqLen; s++)
            {
                T sumSq = NumOps.Zero;
                for (int e = 0; e < _embedDim; e++)
                {
                    T v = was2D ? input[s, e] : input[b, s, e];
                    sumSq = NumOps.Add(sumSq, NumOps.Multiply(v, v));
                }
                poolAmplitudes[s] = NumOps.Sqrt(sumSq);
            }

            // Step 2: encode the S pooled values onto S orthogonal carriers (OFDM-style).
            var encoded = _bus.Encode(poolAmplitudes);

            // Step 3: apply the squaring nonlinearity that creates IMD products.
            var squared = Engine.Multiply(encoded, encoded);

            // Step 4+5: extract the [S, S] attention matrix via FFT + softmax per row.
            var attentionWeights = _extractor.ExtractAttentionWeights(squared);
            if (b == 0) _lastAttentionWeights = attentionWeights;

            // Step 6: apply the attention pattern to the value tensor.
            // out[i, e] = Σ_j A[i, j] · x[j, e]
            for (int i = 0; i < _seqLen; i++)
            {
                for (int e = 0; e < _embedDim; e++)
                {
                    T accum = NumOps.Zero;
                    for (int j = 0; j < _seqLen; j++)
                    {
                        T v = was2D ? input[j, e] : input[b, j, e];
                        accum = NumOps.Add(accum, NumOps.Multiply(attentionWeights[i, j], v));
                    }
                    if (was2D) output[i, e] = accum;
                    else output[b, i, e] = accum;
                }
            }
        }

        return output;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters) { }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters) { }

    /// <inheritdoc/>
    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_seqLen);
        writer.Write(_embedDim);
        writer.Write(_fftSize);
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // seqLen
        _ = reader.ReadInt32(); // embedDim
        _ = reader.ReadInt32(); // fftSize
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastAttentionWeights = null;
    }
}
