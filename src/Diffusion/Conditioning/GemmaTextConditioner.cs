using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Gemma-based text encoder conditioning module for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Google's Gemma language model adapted as a text encoder for diffusion conditioning.
/// Gemma provides strong multilingual understanding and long-context support, making it
/// suitable for detailed prompt understanding in text-to-image models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Gemma is Google's lightweight language model used here as a
/// text understanding component for image generation.
///
/// Key characteristics:
/// - Strong multilingual understanding
/// - 256K token vocabulary for broad language coverage
/// - RoPE (Rotary Position Embeddings) for better positional encoding
/// - Used in models like Imagen 3 and other Google diffusion pipelines
/// </para>
/// <para>
/// Reference: Gemma Team, "Gemma: Open Models Based on Gemini Research and Technology", 2024
/// </para>
/// </remarks>
public class GemmaTextConditioner<T> : TextConditioningBase<T>
{
    /// <inheritdoc />
    public override bool ProducesPooledOutput => false;

    /// <summary>
    /// Initializes a new Gemma text encoder.
    /// </summary>
    /// <param name="variant">Gemma variant. Default: TwoB (2048-dim, 18 layers).</param>
    /// <param name="seed">Optional random seed.</param>
    public GemmaTextConditioner(GemmaVariant variant = GemmaVariant.TwoB, int? seed = null)
        : base(
            vocabSize: 256000,
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
            tokenIds[b * MaxSequenceLength] = NumOps.FromDouble(2);
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

    private static int GetEmbeddingDim(GemmaVariant variant) => variant switch { GemmaVariant.SevenB => 3072, _ => 2048 };
    private static int GetHiddenSize(GemmaVariant variant) => variant switch { GemmaVariant.SevenB => 3072, _ => 2048 };
    private static int GetNumLayers(GemmaVariant variant) => variant switch { GemmaVariant.SevenB => 28, _ => 18 };
    private static int GetNumHeads(GemmaVariant variant) => variant switch { GemmaVariant.SevenB => 16, _ => 8 };
}
