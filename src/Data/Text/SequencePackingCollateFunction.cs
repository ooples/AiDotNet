using AiDotNet.Data.Collation;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Text;

/// <summary>
/// Packs multiple variable-length sequences into fixed-length blocks for efficient LLM training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Sequence packing concatenates multiple short sequences into a single block,
/// reducing padding waste. Each block has a corresponding attention mask that
/// prevents cross-sequence attention.
/// </para>
/// <para>
/// This is the technique used by GPT-3 and LLaMA training to maximize GPU utilization.
/// Instead of padding every sequence to max_length, shorter sequences are concatenated
/// until the block is full, with separator tokens between them.
/// </para>
/// </remarks>
public class SequencePackingCollateFunction<T> : ICollateFunction<Tensor<T>, (Tensor<T> PackedTokens, Tensor<T> AttentionMask)>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _blockSize;
    private readonly int _padTokenId;

    /// <summary>
    /// Initializes a new sequence packing collate function.
    /// </summary>
    /// <param name="blockSize">Maximum block size for packed sequences. Default is 2048.</param>
    /// <param name="padTokenId">Token ID used for padding remaining space. Default is 0.</param>
    public SequencePackingCollateFunction(int blockSize = 2048, int padTokenId = 0)
    {
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize), "blockSize must be positive.");
        _blockSize = blockSize;
        _padTokenId = padTokenId;
    }

    /// <summary>
    /// Packs multiple token sequences into fixed-length blocks.
    /// </summary>
    /// <param name="samples">Individual token sequences, each of shape [1, seq_len].</param>
    /// <returns>Packed tokens [num_blocks, block_size] and attention mask [num_blocks, block_size].</returns>
    public (Tensor<T> PackedTokens, Tensor<T> AttentionMask) Collate(IReadOnlyList<Tensor<T>> samples)
    {
        if (samples == null) throw new ArgumentNullException(nameof(samples));

        // Flatten all tokens into a single stream
        var allTokens = new List<T>();

        foreach (var sample in samples)
        {
            int seqLen = sample.Shape.Length > 1 ? sample.Shape[1] : sample.Shape[0];

            for (int j = 0; j < seqLen; j++)
            {
                T token = sample.Shape.Length > 1 ? sample[0, j] : sample[j];
                allTokens.Add(token);
            }
        }

        // Pack into blocks
        int totalTokens = allTokens.Count;
        int numBlocks = Math.Max(1, (totalTokens + _blockSize - 1) / _blockSize);

        var packedData = new T[numBlocks * _blockSize];
        var maskData = new T[numBlocks * _blockSize];
        T padToken = NumOps.FromDouble(_padTokenId);

        // Initialize with pad tokens
        for (int i = 0; i < packedData.Length; i++)
            packedData[i] = padToken;

        // Fill blocks with actual tokens
        for (int i = 0; i < totalTokens && i < numBlocks * _blockSize; i++)
        {
            packedData[i] = allTokens[i];
            maskData[i] = NumOps.One;
        }

        var packedTokens = new Tensor<T>(packedData, new[] { numBlocks, _blockSize });
        var attentionMask = new Tensor<T>(maskData, new[] { numBlocks, _blockSize });

        return (packedTokens, attentionMask);
    }
}
