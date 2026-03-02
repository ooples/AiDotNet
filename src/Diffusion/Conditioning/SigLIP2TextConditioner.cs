using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// SigLIP 2 text encoder with improved multilingual and compositional understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SigLIP 2 builds on the original SigLIP architecture with improved training recipes,
/// NaFlex token handling for variable-length sequences, and enhanced multilingual support.
/// </para>
/// <para>
/// <b>For Beginners:</b> SigLIP 2 is the next generation of SigLIP with better understanding
/// of complex descriptions and support for more languages. It's used in newer diffusion
/// models that need precise text understanding.
///
/// Key improvements over SigLIP:
/// - NaFlex: handles variable token lengths more efficiently
/// - Better multilingual support across 100+ languages
/// - Improved compositional understanding (e.g., "red car on blue road")
/// - Used in models like Gemini-based diffusion pipelines
/// </para>
/// <para>
/// Reference: Tschannen et al., "SigLIP 2: Scaling Up Multilingual Vision-Language Models", 2025
/// </para>
/// </remarks>
public class SigLIP2TextConditioner<T> : TextConditioningBase<T>
{
    private readonly Vector<T> _textProjection;
    private readonly SigLIP2Variant _variant;

    /// <inheritdoc />
    public override bool ProducesPooledOutput => true;

    /// <summary>
    /// Initializes a new SigLIP 2 text encoder.
    /// </summary>
    /// <param name="variant">SigLIP 2 variant. Default: Large (1024-dim).</param>
    /// <param name="seed">Optional random seed.</param>
    public SigLIP2TextConditioner(SigLIP2Variant variant = SigLIP2Variant.Large, int? seed = null)
        : base(
            vocabSize: 64000,
            embeddingDimension: GetEmbeddingDim(variant),
            hiddenSize: GetHiddenSize(variant),
            numLayers: GetNumLayers(variant),
            numHeads: GetNumHeads(variant),
            maxSequenceLength: 128,
            seed: seed)
    {
        _variant = variant;
        _textProjection = InitializeWeights(HiddenSize * EmbeddingDimension);
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
                {
                    hidden[s * HiddenSize + d] = NumOps.Add(
                        TokenEmbeddings[tokenId * HiddenSize + d],
                        PositionEmbeddings[s * HiddenSize + d]);
                }
            }

            hidden = LayerNorm(hidden, FinalLayerNormWeights, FinalLayerNormBias, HiddenSize);

            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < EmbeddingDimension; d++)
                {
                    T sum = NumOps.Zero;
                    for (int h = 0; h < HiddenSize; h++)
                        sum = NumOps.Add(sum, NumOps.Multiply(hidden[s * HiddenSize + h], _textProjection[h * EmbeddingDimension + d]));
                    outputData[b * seqLen * EmbeddingDimension + s * EmbeddingDimension + d] = sum;
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
        {
            tokenIds[b * MaxSequenceLength] = NumOps.FromDouble(1);
            tokenIds[b * MaxSequenceLength + 1] = NumOps.FromDouble(VocabSize - 1);
        }
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

    private static int GetEmbeddingDim(SigLIP2Variant variant) => variant switch { SigLIP2Variant.Base => 768, SigLIP2Variant.So400M => 1152, _ => 1024 };
    private static int GetHiddenSize(SigLIP2Variant variant) => variant switch { SigLIP2Variant.Base => 768, SigLIP2Variant.So400M => 1152, _ => 1024 };
    private static int GetNumLayers(SigLIP2Variant variant) => variant switch { SigLIP2Variant.Base => 12, SigLIP2Variant.So400M => 27, _ => 24 };
    private static int GetNumHeads(SigLIP2Variant variant) => variant switch { SigLIP2Variant.Base => 12, SigLIP2Variant.So400M => 16, _ => 16 };
}
