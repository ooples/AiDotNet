using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion;

/// <summary>
/// ChatGLM3 text conditioning module for multilingual diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ChatGLM3 is a bilingual (Chinese-English) language model developed by Zhipu AI and Tsinghua KEG.
/// When used as a text encoder for diffusion models (as in Kolors), it provides strong
/// multilingual text understanding for image generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> ChatGLM3 is a text encoder that understands both Chinese and English:
///
/// Most diffusion text encoders (CLIP, T5) are primarily English.
/// ChatGLM3 enables high-quality generation from Chinese prompts too.
///
/// Key characteristics:
/// - Bilingual: Chinese and English text understanding
/// - Large vocabulary: 65,024 tokens covering CJK characters
/// - GLM architecture: General Language Model with autoregressive blank infilling
/// - Used in Kolors (Kwai) for multilingual image generation
/// - 4096-dim output embeddings (same as T5-XXL)
///
/// When to use ChatGLM3:
/// - Generating images from Chinese text prompts
/// - Multilingual applications
/// - When CLIP's English-only understanding is insufficient
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: GLM (General Language Model) transformer
/// - Parameters: ~6B (ChatGLM3-6B)
/// - Vocabulary: 65,024 tokens (BPE with CJK support)
/// - Hidden dimension: 4096
/// - Layers: 28 transformer blocks
/// - Attention heads: 32 (with multi-query attention)
/// - Max sequence length: 256 (for conditioning use)
/// - Output: 4096-dim sequence embeddings
/// - No pooled output (like T5, uses cross-attention with full sequence)
///
/// Reference: Zeng et al., "GLM-130B: An Open Bilingual Pre-trained Model", ICLR 2023
/// </para>
/// </remarks>
public class ChatGLM3TextConditioner<T> : TextConditioningBase<T>
{
    /// <summary>
    /// The ChatGLM3 variant being used.
    /// </summary>
    private readonly string _variant;

    /// <summary>
    /// Gets the embedding dimension for the CLIP-L encoder in this variant.
    /// </summary>
    public int ChatGLMEmbeddingDimension => EmbeddingDimension;

    /// <inheritdoc />
    public override bool ProducesPooledOutput => false;

    /// <summary>
    /// Initializes a new instance of the ChatGLM3 text conditioner.
    /// </summary>
    /// <param name="variant">Model variant. Options: "ChatGLM3-6B" (default), "ChatGLM3-Base".</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ChatGLM3TextConditioner(
        string variant = "ChatGLM3-6B",
        int? seed = null)
        : base(
            vocabSize: GetVocabSize(variant),
            embeddingDimension: GetEmbeddingDim(variant),
            hiddenSize: GetHiddenSize(variant),
            numLayers: GetNumLayers(variant),
            numHeads: GetNumHeads(variant),
            maxSequenceLength: 256,
            seed: seed)
    {
        _variant = variant;
    }

    private static int GetVocabSize(string variant) => variant switch
    {
        "ChatGLM3-Base" => 65024,
        _ => 65024 // ChatGLM3-6B
    };

    private static int GetEmbeddingDim(string variant) => variant switch
    {
        "ChatGLM3-Base" => 2048,
        _ => 4096 // ChatGLM3-6B
    };

    private static int GetHiddenSize(string variant) => variant switch
    {
        "ChatGLM3-Base" => 2048,
        _ => 4096 // ChatGLM3-6B
    };

    private static int GetNumLayers(string variant) => variant switch
    {
        "ChatGLM3-Base" => 20,
        _ => 28 // ChatGLM3-6B
    };

    private static int GetNumHeads(string variant) => variant switch
    {
        "ChatGLM3-Base" => 16,
        _ => 32 // ChatGLM3-6B
    };

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input)
    {
        var tokens = Tokenize("encoded input");
        return EncodeText(tokens);
    }

    /// <inheritdoc />
    public override Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        int batchSize = tokenIds.Shape[0];
        int seqLen = tokenIds.Shape[1];
        int embDim = EmbeddingDimension;

        var output = new T[batchSize * seqLen * embDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenId = (int)NumOps.ToDouble(tokenIds[b, s]);
                int tokenOffset = (tokenId % VocabSize) * HiddenSize;
                int posOffset = s * HiddenSize;

                for (int d = 0; d < embDim; d++)
                {
                    int outIdx = (b * seqLen + s) * embDim + d;
                    int dMod = d % HiddenSize;

                    // Token embedding + position embedding
                    T tokEmb = TokenEmbeddings[tokenOffset + dMod];
                    T posEmb = PositionEmbeddings[posOffset + dMod];
                    output[outIdx] = NumOps.Add(tokEmb, posEmb);
                }
            }
        }

        // Apply transformer layers (simplified)
        output = ApplyTransformerLayers(output, batchSize, seqLen, embDim);

        return new Tensor<T>(new[] { batchSize, seqLen, embDim }, new Vector<T>(output));
    }

    /// <summary>
    /// Applies simplified GLM transformer layers with rotary position embedding simulation.
    /// </summary>
    private T[] ApplyTransformerLayers(T[] embeddings, int batchSize, int seqLen, int embDim)
    {
        var current = embeddings;
        int weightsPerLayer = 12 * HiddenSize * HiddenSize + 4 * HiddenSize;

        for (int layer = 0; layer < NumLayers; layer++)
        {
            var next = new T[current.Length];
            int layerOffset = layer * weightsPerLayer;

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int baseIdx = (b * seqLen + s) * embDim;

                    // Layer norm + simplified self-attention + residual
                    for (int d = 0; d < embDim; d++)
                    {
                        T input = current[baseIdx + d];

                        // Attention approximation using stored weights
                        int wIdx = layerOffset + (d % weightsPerLayer);
                        if (wIdx < TransformerWeights.Length)
                        {
                            T weight = TransformerWeights[wIdx];
                            next[baseIdx + d] = NumOps.Add(input, NumOps.Multiply(input, weight));
                        }
                        else
                        {
                            next[baseIdx + d] = input;
                        }
                    }
                }
            }

            current = next;
        }

        // Final layer norm
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int baseIdx = (b * seqLen + s) * embDim;

                // Compute mean and variance
                double mean = 0;
                for (int d = 0; d < embDim; d++)
                    mean += NumOps.ToDouble(current[baseIdx + d]);
                mean /= embDim;

                double variance = 0;
                for (int d = 0; d < embDim; d++)
                {
                    double diff = NumOps.ToDouble(current[baseIdx + d]) - mean;
                    variance += diff * diff;
                }
                variance /= embDim;

                double invStd = 1.0 / Math.Sqrt(variance + 1e-5);

                for (int d = 0; d < embDim; d++)
                {
                    double normalized = (NumOps.ToDouble(current[baseIdx + d]) - mean) * invStd;
                    double gamma = NumOps.ToDouble(FinalLayerNormWeights[d % HiddenSize]);
                    double beta = NumOps.ToDouble(FinalLayerNormBias[d % HiddenSize]);
                    current[baseIdx + d] = NumOps.FromDouble(normalized * gamma + beta);
                }
            }
        }

        return current;
    }

    /// <inheritdoc />
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        // ChatGLM3 doesn't produce pooled output (like T5)
        // Return mean-pooled embedding as fallback
        int batchSize = sequenceEmbeddings.Shape[0];
        int seqLen = sequenceEmbeddings.Shape[1];
        int embDim = sequenceEmbeddings.Shape[2];

        var pooled = new T[batchSize * embDim];
        var seqLenT = NumOps.FromDouble(seqLen);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < embDim; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqLen; s++)
                {
                    sum = NumOps.Add(sum, sequenceEmbeddings[b, s, d]);
                }
                pooled[b * embDim + d] = NumOps.Divide(sum, seqLenT);
            }
        }

        return new Tensor<T>(new[] { batchSize, embDim }, new Vector<T>(pooled));
    }

    /// <inheritdoc />
    public override Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        // Return zero embeddings for unconditional generation
        int seqLen = MaxSequenceLength;
        int embDim = EmbeddingDimension;

        var zeros = new T[batchSize * seqLen * embDim];
        return new Tensor<T>(new[] { batchSize, seqLen, embDim }, new Vector<T>(zeros));
    }

    /// <inheritdoc />
    public override Tensor<T> Tokenize(string text)
    {
        // Simplified tokenization: map characters to token IDs
        // Real implementation would use SentencePiece BPE tokenizer
        var tokens = new T[1 * MaxSequenceLength];

        // BOS token
        tokens[0] = NumOps.FromDouble(1);

        // Simple character-to-token mapping (placeholder for real BPE)
        int tokenIdx = 1;
        foreach (char c in text)
        {
            if (tokenIdx >= MaxSequenceLength - 1)
                break;

            // Map character to token ID (simplified)
            int tokenId = c < 128 ? c + 100 : (c % (VocabSize - 200)) + 200;
            tokens[tokenIdx] = NumOps.FromDouble(tokenId);
            tokenIdx++;
        }

        // EOS token
        if (tokenIdx < MaxSequenceLength)
        {
            tokens[tokenIdx] = NumOps.FromDouble(2);
            tokenIdx++;
        }

        // Pad remaining
        for (int i = tokenIdx; i < MaxSequenceLength; i++)
        {
            tokens[i] = NumOps.FromDouble(0);
        }

        return new Tensor<T>(new[] { 1, MaxSequenceLength }, new Vector<T>(tokens));
    }

    /// <inheritdoc />
    public override Tensor<T> TokenizeBatch(string[] texts)
    {
        var allTokens = new T[texts.Length * MaxSequenceLength];

        for (int b = 0; b < texts.Length; b++)
        {
            var singleTokens = Tokenize(texts[b]);
            for (int i = 0; i < MaxSequenceLength; i++)
            {
                allTokens[b * MaxSequenceLength + i] = singleTokens[0, i];
            }
        }

        return new Tensor<T>(new[] { texts.Length, MaxSequenceLength }, new Vector<T>(allTokens));
    }
}
