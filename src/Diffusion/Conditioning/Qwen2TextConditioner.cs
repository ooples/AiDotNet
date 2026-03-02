using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Qwen2-based text encoder conditioning module for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Alibaba's Qwen2 language model adapted as a text encoder for diffusion models.
/// Qwen2 excels at Chinese-English bilingual understanding and is used in models
/// like Kolors 2 and other Chinese-first diffusion pipelines.
/// </para>
/// <para>
/// <b>For Beginners:</b> Qwen2 is a strong language model from Alibaba that understands
/// both Chinese and English very well. When used in diffusion models, it helps generate
/// images that accurately match prompts in both languages.
///
/// Key characteristics:
/// - Excellent Chinese-English bilingual understanding
/// - 151K vocabulary for broad language coverage
/// - GQA (Grouped Query Attention) for efficient inference
/// - Used in: Kolors 2, HunyuanDiT, Chinese-first diffusion models
/// </para>
/// <para>
/// Reference: Yang et al., "Qwen2 Technical Report", 2024
/// </para>
/// </remarks>
public class Qwen2TextConditioner<T> : TextConditioningBase<T>
{
    /// <inheritdoc />
    public override bool ProducesPooledOutput => false;

    /// <summary>
    /// Initializes a new Qwen2 text encoder.
    /// </summary>
    /// <param name="variant">Qwen2 variant. Default: OnePointFiveB (1536-dim, 28 layers).</param>
    /// <param name="seed">Optional random seed.</param>
    public Qwen2TextConditioner(Qwen2Variant variant = Qwen2Variant.OnePointFiveB, int? seed = null)
        : base(
            vocabSize: 151936,
            embeddingDimension: GetEmbeddingDim(variant),
            hiddenSize: GetHiddenSize(variant),
            numLayers: GetNumLayers(variant),
            numHeads: GetNumHeads(variant),
            maxSequenceLength: 256,
            seed: seed)
    {
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input) => EncodeText(input);

    /// <inheritdoc />
    public override Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        var shape = tokenIds.Shape;
        int batchSize = shape[0];
        int seqLen = shape.Length > 1 ? shape[1] : MaxSequenceLength;
        var outputData = new Vector<T>(batchSize * seqLen * EmbeddingDimension);

        for (int b = 0; b < batchSize; b++)
        {
            var hidden = new Vector<T>(seqLen * HiddenSize);
            for (int s = 0; s < seqLen; s++)
            {
                int flatIdx = b * seqLen + s;
                int tokenId = flatIdx < tokenIds.Shape[0] * (tokenIds.Shape.Length > 1 ? tokenIds.Shape[1] : 1)
                    ? (int)NumOps.ToDouble(tokenIds[flatIdx]) : 0;
                tokenId = Math.Max(0, Math.Min(tokenId, VocabSize - 1));

                for (int d = 0; d < HiddenSize; d++)
                    hidden[s * HiddenSize + d] = NumOps.Add(
                        TokenEmbeddings[tokenId * HiddenSize + d],
                        PositionEmbeddings[s * HiddenSize + d]);
            }

            hidden = LayerNorm(hidden, FinalLayerNormWeights, FinalLayerNormBias, HiddenSize);

            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < EmbeddingDimension; d++)
                {
                    outputData[b * seqLen * EmbeddingDimension + s * EmbeddingDimension + d] =
                        d < HiddenSize ? hidden[s * HiddenSize + d] : NumOps.Zero;
                }
            }
        }

        return new Tensor<T>(new[] { batchSize, seqLen, EmbeddingDimension }, outputData);
    }

    /// <inheritdoc />
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        var shape = sequenceEmbeddings.Shape;
        int batchSize = shape[0];
        int seqLen = shape[1];
        var pooledData = new Vector<T>(batchSize * EmbeddingDimension);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < EmbeddingDimension; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqLen; s++)
                    sum = NumOps.Add(sum, sequenceEmbeddings[b * seqLen * EmbeddingDimension + s * EmbeddingDimension + d]);
                pooledData[b * EmbeddingDimension + d] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
            }
        }

        return new Tensor<T>(new[] { batchSize, EmbeddingDimension }, pooledData);
    }

    /// <inheritdoc />
    public override Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        var tokenIds = new Vector<T>(batchSize * MaxSequenceLength);
        for (int b = 0; b < batchSize; b++)
            tokenIds[b * MaxSequenceLength] = NumOps.FromDouble(1);
        return EncodeText(new Tensor<T>(new[] { batchSize, MaxSequenceLength }, tokenIds));
    }

    /// <inheritdoc />
    public override Tensor<T> Tokenize(string text)
    {
        var tokens = SimpleTokenize(text, MaxSequenceLength);
        var tokenData = new Vector<T>(MaxSequenceLength);
        for (int i = 0; i < MaxSequenceLength; i++) tokenData[i] = NumOps.FromDouble(tokens[i]);
        return new Tensor<T>(new[] { 1, MaxSequenceLength }, tokenData);
    }

    /// <inheritdoc />
    public override Tensor<T> TokenizeBatch(string[] texts)
    {
        var tokenData = new Vector<T>(texts.Length * MaxSequenceLength);
        for (int b = 0; b < texts.Length; b++)
        {
            var tokens = SimpleTokenize(texts[b], MaxSequenceLength);
            for (int i = 0; i < MaxSequenceLength; i++)
                tokenData[b * MaxSequenceLength + i] = NumOps.FromDouble(tokens[i]);
        }
        return new Tensor<T>(new[] { texts.Length, MaxSequenceLength }, tokenData);
    }

    private static int GetEmbeddingDim(Qwen2Variant variant) => variant switch { Qwen2Variant.SevenB => 4096, _ => 1536 };
    private static int GetHiddenSize(Qwen2Variant variant) => variant switch { Qwen2Variant.SevenB => 4096, _ => 1536 };
    private static int GetNumLayers(Qwen2Variant variant) => variant switch { Qwen2Variant.SevenB => 32, _ => 28 };
    private static int GetNumHeads(Qwen2Variant variant) => variant switch { Qwen2Variant.SevenB => 32, _ => 12 };
}
