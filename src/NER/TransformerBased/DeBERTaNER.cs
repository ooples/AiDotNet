using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// DeBERTa-NER: Decoding-enhanced BERT with disentangled Attention for NER.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DeBERTa-NER (He et al., ICLR 2021 - "DeBERTa: Decoding-enhanced BERT with Disentangled Attention")
/// is a state-of-the-art transformer model that introduces two key architectural innovations:
///
/// <b>1. Disentangled Attention:</b>
/// Instead of combining content and position into a single embedding (as in BERT), DeBERTa
/// represents each token with two separate vectors: one for content and one for position.
/// The attention score between two tokens is computed as the sum of four components:
/// - Content-to-content: semantic similarity
/// - Content-to-position: how important a token's meaning is relative to another's position
/// - Position-to-content: how important a token's position is relative to another's meaning
/// - Position-to-position: relative position bias
///
/// This disentangled approach captures richer token relationships, particularly beneficial for
/// NER where both semantic meaning and positional context matter.
///
/// <b>2. Enhanced Mask Decoder:</b>
/// Incorporates absolute position information in the final decoding layers, combining the
/// benefits of relative position (used throughout the model) with absolute position (needed
/// for tasks sensitive to word order).
///
/// <b>Performance (CoNLL-2003):</b>
/// - DeBERTa-base: ~93.1% F1
/// - DeBERTa-large: ~93.5% F1 (state-of-the-art for single models)
/// - DeBERTa-xlarge: ~93.8% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> DeBERTa is one of the most accurate transformer models for NER.
/// It improves on BERT by being smarter about how it handles word positions. In normal BERT,
/// the word's meaning and its position in the sentence are mixed together. DeBERTa keeps them
/// separate, which helps it better understand relationships between words.
///
/// Use DeBERTa-NER when you want the highest possible accuracy and can afford the compute cost.
/// </para>
/// </remarks>
public class DeBERTaNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a DeBERTa-NER model in ONNX inference mode.
    /// </summary>
    public DeBERTaNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "DeBERTa-NER", "He et al., ICLR 2021")
    {
    }

    /// <summary>
    /// Creates a DeBERTa-NER model in native training mode.
    /// </summary>
    public DeBERTaNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "DeBERTa-NER", "He et al., ICLR 2021", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new DeBERTaNER<T>(Architecture, p, optionsCopy);
        return new DeBERTaNER<T>(Architecture, optionsCopy);
    }
}
