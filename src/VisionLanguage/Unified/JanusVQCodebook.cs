using System;
using AiDotNet.Helpers;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Vector-Quantized codebook used by Janus / Janus-Pro for image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Janus and Janus-Pro (Wu et al. DeepSeek 2024 / Chen et al. DeepSeek 2025, arXiv:2410.13848 /
/// arXiv:2501.17811) generate images by autoregressively predicting VQ token IDs and then
/// decoding the resulting grid back to pixels through a frozen VQ-VAE decoder. This class
/// owns the codebook (a learnable lookup table from token ID to a continuous code embedding)
/// and provides the two operations the rest of the generation path needs:
/// </para>
/// <list type="number">
///   <item><b>Quantize</b>: map a continuous embedding to its nearest codebook entry (used during VAE training).</item>
///   <item><b>Lookup</b>: map a token ID to its embedding (used at generation time after the LLM picks a token).</item>
/// </list>
/// <para>
/// The codebook is initialized with a deterministic spectral-style spread so that nearby IDs
/// have nearby embeddings; this keeps the un-trained model's generation output structured rather
/// than purely random while preserving the API surface a fully trained codebook would expose.
/// </para>
/// <para><b>References:</b></para>
/// <list type="bullet">
///   <item>van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), 2017, arXiv:1711.00937 — original codebook learning algorithm.</item>
///   <item>Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", 2019, arXiv:1906.00446 — multi-level codebook used by Janus.</item>
///   <item>Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation", DeepSeek 2024, arXiv:2410.13848.</item>
///   <item>Chen et al., "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling", DeepSeek 2025, arXiv:2501.17811 — 16384-entry codebook variant used by Janus-Pro.</item>
/// </list>
/// </remarks>
public sealed class JanusVQCodebook<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double[,] _codebook;

    /// <summary>Number of entries in the codebook (Janus-Pro: 16384; Janus: 8192).</summary>
    public int CodebookSize { get; }

    /// <summary>Dimensionality of each codebook entry's embedding (Janus-Pro: 8).</summary>
    public int EmbeddingDim { get; }

    /// <summary>
    /// Initialises a deterministic, structured codebook.
    /// </summary>
    /// <param name="codebookSize">Number of discrete code entries. Default 16384 (Janus-Pro).</param>
    /// <param name="embeddingDim">Per-code embedding dimensionality. Default 8 (Janus-Pro).</param>
    public JanusVQCodebook(int codebookSize = 16384, int embeddingDim = 8)
    {
        if (codebookSize <= 0) throw new ArgumentOutOfRangeException(nameof(codebookSize), codebookSize, "codebookSize must be positive.");
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim), embeddingDim, "embeddingDim must be positive.");

        CodebookSize = codebookSize;
        EmbeddingDim = embeddingDim;
        _numOps = MathHelper.GetNumericOperations<T>();
        _codebook = InitialiseDeterministic(codebookSize, embeddingDim);
    }

    /// <summary>
    /// Returns the codebook entry for a token ID as a tensor. Token IDs outside the codebook are clamped.
    /// </summary>
    public Tensor<T> Lookup(int tokenId)
    {
        int safeId = Math.Max(0, Math.Min(CodebookSize - 1, tokenId));
        var embed = new Tensor<T>([EmbeddingDim]);
        for (int d = 0; d < EmbeddingDim; d++)
            embed[d] = _numOps.FromDouble(_codebook[safeId, d]);
        return embed;
    }

    /// <summary>
    /// Quantizes a continuous embedding to its nearest codebook entry's token ID under squared-Euclidean distance.
    /// Equivalent to the encoder-side VQ step in van den Oord et al. 2017.
    /// </summary>
    public int Quantize(Tensor<T> continuousEmbedding)
    {
        if (continuousEmbedding is null) throw new ArgumentNullException(nameof(continuousEmbedding));
        if (continuousEmbedding.Length < EmbeddingDim)
            throw new ArgumentException($"continuousEmbedding has length {continuousEmbedding.Length} but codebook expects {EmbeddingDim}.", nameof(continuousEmbedding));

        int bestId = 0;
        double bestDist = double.PositiveInfinity;
        for (int id = 0; id < CodebookSize; id++)
        {
            double dist = 0.0;
            for (int d = 0; d < EmbeddingDim; d++)
            {
                double v = _numOps.ToDouble(continuousEmbedding[d]) - _codebook[id, d];
                dist += v * v;
            }
            if (dist < bestDist) { bestDist = dist; bestId = id; }
        }
        return bestId;
    }

    /// <summary>
    /// Builds a grid of code embeddings from a flat token sequence. Used by the VQ-VAE detokenizer
    /// path: token grid (H/8 × W/8) → embedding grid → deconv stack → pixels.
    /// </summary>
    /// <param name="tokenIds">Flat token sequence of length <paramref name="gridHeight"/> × <paramref name="gridWidth"/>.</param>
    /// <param name="gridHeight">Number of rows in the token grid.</param>
    /// <param name="gridWidth">Number of columns in the token grid.</param>
    /// <returns>Tensor of shape <c>[gridHeight, gridWidth, EmbeddingDim]</c> flattened as <c>[gridHeight * gridWidth * EmbeddingDim]</c>.</returns>
    public Tensor<T> LookupGrid(int[] tokenIds, int gridHeight, int gridWidth)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        if (gridHeight <= 0) throw new ArgumentOutOfRangeException(nameof(gridHeight));
        if (gridWidth <= 0) throw new ArgumentOutOfRangeException(nameof(gridWidth));
        int expected = gridHeight * gridWidth;
        if (tokenIds.Length < expected)
            throw new ArgumentException($"tokenIds has length {tokenIds.Length} but grid expects {expected}.", nameof(tokenIds));

        var grid = new Tensor<T>([gridHeight * gridWidth * EmbeddingDim]);
        for (int y = 0; y < gridHeight; y++)
        {
            for (int x = 0; x < gridWidth; x++)
            {
                int idx = y * gridWidth + x;
                int safeId = Math.Max(0, Math.Min(CodebookSize - 1, tokenIds[idx]));
                for (int d = 0; d < EmbeddingDim; d++)
                    grid[(y * gridWidth + x) * EmbeddingDim + d] = _numOps.FromDouble(_codebook[safeId, d]);
            }
        }
        return grid;
    }

    /// <summary>
    /// Replaces the codebook entries in bulk (used when loading a trained checkpoint).
    /// The supplied array must be shape <c>[CodebookSize, EmbeddingDim]</c>.
    /// </summary>
    public void LoadCodebook(double[,] codebook)
    {
        if (codebook is null) throw new ArgumentNullException(nameof(codebook));
        if (codebook.GetLength(0) != CodebookSize || codebook.GetLength(1) != EmbeddingDim)
            throw new ArgumentException($"codebook shape ({codebook.GetLength(0)}×{codebook.GetLength(1)}) does not match expected ({CodebookSize}×{EmbeddingDim}).", nameof(codebook));
        for (int i = 0; i < CodebookSize; i++)
            for (int d = 0; d < EmbeddingDim; d++)
                _codebook[i, d] = codebook[i, d];
    }

    private static double[,] InitialiseDeterministic(int codebookSize, int embeddingDim)
    {
        var book = new double[codebookSize, embeddingDim];
        double inv = 1.0 / Math.Sqrt(embeddingDim);
        for (int id = 0; id < codebookSize; id++)
        {
            for (int d = 0; d < embeddingDim; d++)
            {
                // Quasi-random low-discrepancy spread so nearby IDs have nearby embeddings.
                double angle = (id + 1) * Math.PI * (d + 1) * 0.0001;
                book[id, d] = inv * (Math.Sin(angle) + 0.5 * Math.Cos(angle * 2.7183));
            }
        }
        return book;
    }
}
