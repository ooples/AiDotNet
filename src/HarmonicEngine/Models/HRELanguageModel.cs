using AiDotNet.HarmonicEngine.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.HarmonicEngine.Models;

/// <summary>
/// Decoder-only language model built from stacked <see cref="HREBlock{T}"/> layers —
/// a transformer-replacement architecture that uses no gradient backpropagation and
/// no Q/K/V projection matrices. The only learnable parameters are the token embeddings,
/// the per-block LayerNorm scales, and the Hebbian spectral filters inside each block's FFN.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A transformer language model has four pieces: (1) a token
/// embedding that maps each input token ID to a dense vector, (2) a positional encoding
/// that adds information about "which position is this token in the sequence", (3) a
/// stack of transformer blocks, each mixing information across the sequence via
/// attention and then transforming each position via an FFN, and (4) an "unembedding"
/// that maps the final vectors back to scores over the vocabulary.
/// </para>
/// <para>
/// HRELanguageModel keeps exactly this structure but replaces the transformer blocks
/// with HREBlocks — where attention comes from IMD and the FFN comes from spectral
/// Hebbian filters. Everything else (token embedding, positional encoding, unembedding)
/// stays standard because those pieces aren't the source of transformer's parameter
/// bloat and replacing them wouldn't change the architecture's story.
/// </para>
/// <para>
/// Input/output shape:
/// <list type="bullet">
/// <item><description><b>Input:</b> token IDs <c>[B, S]</c> as a <c>Tensor&lt;T&gt;</c>
/// with integer-valued entries in <c>[0, vocabSize)</c>.</description></item>
/// <item><description><b>Output:</b> logits <c>[B, S, V]</c> over the vocabulary.</description></item>
/// </list>
/// </para>
/// </remarks>
public class HRELanguageModel<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _vocabSize;
    private readonly int _seqLen;
    private readonly int _embedDim;
    private readonly int _numLayers;

    // Token embedding table: [vocabSize, embedDim]
    private readonly Matrix<T> _tokenEmbedding;

    // Sinusoidal positional encoding: [seqLen, embedDim]. Fixed, not learned.
    private readonly Matrix<T> _positionalEncoding;

    // Stack of HRE blocks
    private readonly List<HREBlock<T>> _blocks;

    /// <summary>
    /// Gets the total learnable parameter count across all components.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            // Token embedding (tied with unembedding, so counted once)
            int count = _vocabSize * _embedDim;
            foreach (var block in _blocks) count += block.ParameterCount;
            return count;
        }
    }

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    public int VocabSize => _vocabSize;

    /// <summary>
    /// Gets the sequence length (context window size).
    /// </summary>
    public int SequenceLength => _seqLen;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDim => _embedDim;

    /// <summary>
    /// Gets the number of HRE blocks stacked in the model.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets the list of HRE blocks (for external Hebbian updates and inspection).
    /// </summary>
    public IReadOnlyList<HREBlock<T>> Blocks => _blocks;

    /// <summary>
    /// Creates a new HRE language model.
    /// </summary>
    /// <param name="vocabSize">Vocabulary size V. Determines the embedding table height.</param>
    /// <param name="seqLen">Context window length S. Must be large enough for your task but
    /// small enough that the per-block FFT over S carriers is feasible.</param>
    /// <param name="embedDim">Embedding dimension E. Must be a power of 2 for the FFN's spectral filters.</param>
    /// <param name="numLayers">Number of stacked HRE blocks.</param>
    /// <param name="fftSize">FFT size used inside each block's sequence-axis IMD attention.
    /// Typical choice: <c>~4·S²</c> to accommodate Sidon-set carrier allocation.</param>
    /// <param name="hebbianLearningRate">Learning rate for the Hebbian filters inside each block's FFN.</param>
    /// <param name="antiHebbianAlpha">Anti-Hebbian decorrelation strength for the Hebbian filters.</param>
    /// <param name="seed">Optional seed for reproducible embedding initialization.</param>
    public HRELanguageModel(
        int vocabSize,
        int seqLen,
        int embedDim,
        int numLayers,
        int fftSize,
        double hebbianLearningRate = 0.01,
        double antiHebbianAlpha = 0.5,
        int? seed = null)
    {
        if (vocabSize < 2) throw new ArgumentOutOfRangeException(nameof(vocabSize), "Vocab size must be >= 2.");
        if (seqLen < 2) throw new ArgumentOutOfRangeException(nameof(seqLen), "Sequence length must be >= 2.");
        if (embedDim < 2 || (embedDim & (embedDim - 1)) != 0)
            throw new ArgumentException($"Embedding dim must be a power of 2, got {embedDim}.", nameof(embedDim));
        if (numLayers < 1) throw new ArgumentOutOfRangeException(nameof(numLayers), "numLayers must be >= 1.");

        _numOps = MathHelper.GetNumericOperations<T>();
        _vocabSize = vocabSize;
        _seqLen = seqLen;
        _embedDim = embedDim;
        _numLayers = numLayers;

        // Initialize token embedding with small Gaussian noise
        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        _tokenEmbedding = new Matrix<T>(vocabSize, embedDim);
        double embedScale = 1.0 / Math.Sqrt(embedDim);
        for (int v = 0; v < vocabSize; v++)
        {
            for (int e = 0; e < embedDim; e++)
            {
                // Box-Muller normal sample
                double u1 = 1.0 - rng.NextDouble();
                double u2 = rng.NextDouble();
                double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                _tokenEmbedding[v, e] = _numOps.FromDouble(z * embedScale);
            }
        }

        // Build fixed sinusoidal positional encoding (Vaswani et al. 2017)
        _positionalEncoding = new Matrix<T>(seqLen, embedDim);
        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < embedDim; i++)
            {
                double angle = pos / Math.Pow(10000.0, 2.0 * (i / 2) / embedDim);
                double val = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                _positionalEncoding[pos, i] = _numOps.FromDouble(val);
            }
        }

        // Stack N HRE blocks
        _blocks = new List<HREBlock<T>>(numLayers);
        for (int l = 0; l < numLayers; l++)
        {
            _blocks.Add(new HREBlock<T>(seqLen, embedDim, fftSize, hebbianLearningRate, antiHebbianAlpha));
        }
    }

    /// <summary>
    /// Forward pass: tokens → embedding + positional encoding → N × HREBlock → logits.
    /// </summary>
    /// <param name="tokens">Tensor of token IDs, shape <c>[B, S]</c> or <c>[S]</c>.</param>
    /// <returns>Logits tensor, shape <c>[B, S, V]</c> or <c>[S, V]</c>.</returns>
    public Tensor<T> Forward(Tensor<T> tokens)
    {
        bool was1D = tokens._shape.Length == 1;
        int batchSize = was1D ? 1 : tokens._shape[0];
        int actualSeqLen = was1D ? tokens._shape[0] : tokens._shape[1];

        if (actualSeqLen > _seqLen)
            throw new ArgumentException(
                $"Input sequence length ({actualSeqLen}) exceeds model's context window ({_seqLen}).");

        // Step 1: Build the [B, S, E] embedding tensor.
        // For each (b, s), look up tokens[b, s] in the embedding table and add positional encoding.
        var embedded = new Tensor<T>([batchSize, _seqLen, _embedDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < actualSeqLen; s++)
            {
                int tokenId = was1D
                    ? (int)_numOps.ToDouble(tokens[s])
                    : (int)_numOps.ToDouble(tokens[b, s]);

                if (tokenId < 0 || tokenId >= _vocabSize)
                    throw new ArgumentException(
                        $"Token ID {tokenId} at position [{b},{s}] is outside vocab range [0, {_vocabSize}).");

                for (int e = 0; e < _embedDim; e++)
                {
                    embedded[b, s, e] = _numOps.Add(_tokenEmbedding[tokenId, e], _positionalEncoding[s, e]);
                }
            }
            // Zero-pad unused positions (if input was shorter than _seqLen)
            for (int s = actualSeqLen; s < _seqLen; s++)
            {
                for (int e = 0; e < _embedDim; e++)
                {
                    embedded[b, s, e] = _numOps.Zero;
                }
            }
        }

        // Step 2: Pass through the stack of HRE blocks.
        Tensor<T> current = embedded;
        foreach (var block in _blocks)
        {
            current = block.Forward(current);
        }

        // Step 3: Unembedding (tied to the token embedding — no extra parameters).
        // For each (b, s), compute logits[b, s, v] = dot(current[b, s, :], tokenEmbedding[v, :])
        int outSeqLen = actualSeqLen;
        var logits = was1D
            ? new Tensor<T>([outSeqLen, _vocabSize])
            : new Tensor<T>([batchSize, outSeqLen, _vocabSize]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < outSeqLen; s++)
            {
                for (int v = 0; v < _vocabSize; v++)
                {
                    T dot = _numOps.Zero;
                    for (int e = 0; e < _embedDim; e++)
                    {
                        dot = _numOps.Add(dot, _numOps.Multiply(current[b, s, e], _tokenEmbedding[v, e]));
                    }
                    if (was1D) logits[s, v] = dot;
                    else logits[b, s, v] = dot;
                }
            }
        }

        return logits;
    }

    /// <summary>
    /// Sets training mode on all internal HRE blocks.
    /// </summary>
    public void SetTrainingMode(bool isTraining)
    {
        foreach (var block in _blocks) block.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Returns the embedding vector (row of the token embedding matrix) for a
    /// given token ID. Used by training strategies that need direct access to
    /// the embedding table (e.g., SpectralTargetPropagation).
    /// </summary>
    /// <param name="tokenId">Token ID in <c>[0, VocabSize)</c>.</param>
    /// <returns>The length-<see cref="EmbeddingDim"/> embedding vector for this token.</returns>
    public Vector<T> GetTokenEmbedding(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _vocabSize)
            throw new ArgumentOutOfRangeException(nameof(tokenId),
                $"Token ID {tokenId} is outside vocab range [0, {_vocabSize}).");

        var row = new Vector<T>(_embedDim);
        for (int e = 0; e < _embedDim; e++) row[e] = _tokenEmbedding[tokenId, e];
        return row;
    }
}
