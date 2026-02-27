using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// SpanBERT-NER: Span-level BERT pre-training with token classification for NER.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SpanBERT-NER (Joshi et al., TACL 2020 - "SpanBERT: Improving Pre-training by Representing
/// and Predicting Spans") uses a BERT variant specifically designed for span-level tasks like NER.
///
/// <b>Key Innovations:</b>
/// - <b>Span masking:</b> Instead of masking random individual tokens, SpanBERT masks contiguous
///   spans of tokens (geometric distribution, mean length 3.8). This forces the model to learn
///   better span-level representations, directly beneficial for NER where entities are spans.
/// - <b>Span boundary objective (SBO):</b> The model predicts masked tokens using the span
///   boundary tokens (positions just before and after the span), encouraging the model to encode
///   span information at boundary positions. This is particularly useful for detecting entity
///   boundaries (B- tags) in NER.
/// - <b>No NSP:</b> Removes next sentence prediction, following RoBERTa's findings.
///
/// <b>Why SpanBERT excels at NER:</b>
/// NER is fundamentally a span-level task: entities are contiguous spans of tokens. SpanBERT's
/// span masking pre-training teaches the model to understand multi-token entities as units,
/// rather than treating each token independently. The span boundary objective ensures that
/// B-tag positions (entity starts) contain strong span representations.
///
/// <b>Performance (CoNLL-2003):</b>
/// - SpanBERT-base: ~92.8% F1
/// - SpanBERT-large: ~93.4% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> SpanBERT is a BERT variant designed for tasks involving spans of text,
/// like NER. While regular BERT masks individual words during training, SpanBERT masks groups
/// of words together. This teaches it to understand multi-word entities like "New York City" or
/// "Goldman Sachs" as single units, which directly helps with NER accuracy.
/// </para>
/// </remarks>
public class SpanBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a SpanBERT-NER model in ONNX inference mode.
    /// </summary>
    public SpanBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "SpanBERT-NER", "Joshi et al., TACL 2020")
    {
    }

    /// <summary>
    /// Creates a SpanBERT-NER model in native training mode.
    /// </summary>
    public SpanBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "SpanBERT-NER", "Joshi et al., TACL 2020", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new SpanBERTNER<T>(Architecture, p, optionsCopy);
        return new SpanBERTNER<T>(Architecture, optionsCopy);
    }
}
