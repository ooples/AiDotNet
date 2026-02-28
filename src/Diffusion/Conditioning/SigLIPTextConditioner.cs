using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// SigLIP text encoder conditioning module for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SigLIP (Sigmoid Loss for Language-Image Pre-training) replaces the softmax contrastive
/// loss of CLIP with a simpler sigmoid loss that operates on image-text pairs independently.
/// This results in better zero-shot transfer and more efficient training.
/// </para>
/// <para>
/// <b>For Beginners:</b> SigLIP is an improved version of CLIP that understands text better.
///
/// Key differences from CLIP:
/// - Uses sigmoid loss instead of softmax contrastive loss
/// - Doesn't need global normalization across the batch
/// - Better at understanding nuanced text descriptions
/// - Used in newer models like SD3.5 and some FLUX variants
///
/// SigLIP produces embeddings that tell the diffusion model what to generate,
/// similar to CLIP but with improved text-image alignment.
/// </para>
/// <para>
/// Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training", ICCV 2023
/// </para>
/// </remarks>
public class SigLIPTextConditioner<T> : TextConditioningBase<T>
{
    private readonly Vector<T> _textProjection;
    private readonly SigLIPVariant _variant;

    /// <inheritdoc />
    public override bool ProducesPooledOutput => true;

    /// <summary>
    /// Initializes a new SigLIP text encoder conditioning module.
    /// </summary>
    /// <param name="variant">SigLIP variant. Default: Large (1024-dim).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SigLIPTextConditioner(SigLIPVariant variant = SigLIPVariant.Large, int? seed = null)
        : base(
            vocabSize: 32000,
            embeddingDimension: GetEmbeddingDim(variant),
            hiddenSize: GetHiddenSize(variant),
            numLayers: GetNumLayers(variant),
            numHeads: GetNumHeads(variant),
            maxSequenceLength: 64,
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
                    ? (int)NumOps.ToDouble(tokenIds[flatIdx])
                    : 0;
                tokenId = Math.Max(0, Math.Min(tokenId, VocabSize - 1));

                for (int d = 0; d < HiddenSize; d++)
                {
                    T tokenEmb = TokenEmbeddings[tokenId * HiddenSize + d];
                    T posEmb = PositionEmbeddings[s * HiddenSize + d];
                    hidden[s * HiddenSize + d] = NumOps.Add(tokenEmb, posEmb);
                }
            }

            hidden = ApplyTransformerLayers(hidden, seqLen);
            hidden = LayerNorm(hidden, FinalLayerNormWeights, FinalLayerNormBias, HiddenSize);

            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < EmbeddingDimension; d++)
                {
                    T sum = NumOps.Zero;
                    for (int h = 0; h < HiddenSize; h++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(
                            hidden[s * HiddenSize + h],
                            _textProjection[h * EmbeddingDimension + d]));
                    }
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
            // SigLIP uses mean pooling over non-padding tokens
            for (int d = 0; d < EmbeddingDimension; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqLen; s++)
                {
                    sum = NumOps.Add(sum, sequenceEmbeddings[b * seqLen * EmbeddingDimension + s * EmbeddingDimension + d]);
                }
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

        var input = new Tensor<T>(new[] { batchSize, MaxSequenceLength }, tokenIds);
        return EncodeText(input);
    }

    /// <inheritdoc />
    public override Tensor<T> Tokenize(string text)
    {
        var tokens = SimpleTokenize(text, MaxSequenceLength);
        var tokenData = new Vector<T>(MaxSequenceLength);
        for (int i = 0; i < MaxSequenceLength; i++)
            tokenData[i] = NumOps.FromDouble(tokens[i]);

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

    private Vector<T> ApplyTransformerLayers(Vector<T> hidden, int seqLen)
    {
        int weightsPerLayer = 12 * HiddenSize * HiddenSize + 4 * HiddenSize;

        for (int layer = 0; layer < NumLayers; layer++)
        {
            int layerOffset = layer * weightsPerLayer;
            var residual = CopyVector(hidden);

            var lnGamma = ExtractSubVector(TransformerWeights, layerOffset, HiddenSize);
            var lnBeta = ExtractSubVector(TransformerWeights, layerOffset + HiddenSize, HiddenSize);
            hidden = LayerNorm(hidden, lnGamma, lnBeta, HiddenSize);

            int attnWeightOffset = layerOffset + 2 * HiddenSize;
            hidden = LinearProject(hidden, TransformerWeights, attnWeightOffset, HiddenSize, HiddenSize, seqLen);
            hidden = AddVectors(hidden, residual);

            residual = CopyVector(hidden);
            int ln2Offset = layerOffset + 2 * HiddenSize + HiddenSize * HiddenSize;
            var ln2Gamma = ExtractSubVector(TransformerWeights, ln2Offset, HiddenSize);
            var ln2Beta = ExtractSubVector(TransformerWeights, ln2Offset + HiddenSize, HiddenSize);
            hidden = LayerNorm(hidden, ln2Gamma, ln2Beta, HiddenSize);

            int mlpOffset = ln2Offset + 2 * HiddenSize;
            hidden = LinearProject(hidden, TransformerWeights, mlpOffset, HiddenSize, HiddenSize, seqLen);
            hidden = AddVectors(hidden, residual);
        }

        return hidden;
    }

    private Vector<T> LinearProject(Vector<T> input, Vector<T> weights, int weightOffset, int inDim, int outDim, int seqLen)
    {
        var output = new Vector<T>(seqLen * outDim);
        for (int s = 0; s < seqLen; s++)
        {
            for (int o = 0; o < outDim; o++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < inDim; i++)
                {
                    int wIdx = weightOffset + i * outDim + o;
                    if (wIdx < weights.Length)
                        sum = NumOps.Add(sum, NumOps.Multiply(input[s * inDim + i], weights[wIdx]));
                }
                output[s * outDim + o] = sum;
            }
        }
        return output;
    }

    private static Vector<T> CopyVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) copy[i] = source[i];
        return copy;
    }

    private static Vector<T> ExtractSubVector(Vector<T> source, int offset, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length && offset + i < source.Length; i++) result[i] = source[offset + i];
        return result;
    }

    private static Vector<T> AddVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++) result[i] = NumOps.Add(a[i], b[i]);
        return result;
    }

    private static int GetEmbeddingDim(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 768,
        SigLIPVariant.So400M => 1152,
        _ => 1024
    };

    private static int GetHiddenSize(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 768,
        SigLIPVariant.So400M => 1152,
        _ => 1024
    };

    private static int GetNumLayers(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 12,
        SigLIPVariant.So400M => 27,
        _ => 24
    };

    private static int GetNumHeads(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 12,
        SigLIPVariant.So400M => 16,
        _ => 16
    };
}
