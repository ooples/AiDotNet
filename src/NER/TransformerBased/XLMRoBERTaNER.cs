using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// XLM-RoBERTa-NER: Cross-lingual RoBERTa for multilingual Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// XLM-RoBERTa-NER (Conneau et al., ACL 2020 - "Unsupervised Cross-lingual Representation
/// Learning at Scale") fine-tunes a multilingual transformer for NER across 100+ languages.
///
/// <b>Key Features:</b>
/// - <b>100+ languages:</b> Pre-trained on 2.5TB of CommonCrawl data in 100 languages
/// - <b>Cross-lingual transfer:</b> Fine-tune on English NER data, apply to any supported language
/// - <b>SentencePiece tokenization:</b> Shared vocabulary across all languages (250K tokens)
/// - <b>RoBERTa architecture:</b> Same architecture as RoBERTa but with multilingual pre-training
///
/// <b>Zero-shot Cross-lingual NER:</b>
/// XLM-RoBERTa enables zero-shot cross-lingual NER: train on English CoNLL-2003, then
/// predict entities in German, Spanish, Dutch, etc. without any target-language training data.
/// This is possible because the multilingual pre-training creates a shared representation space
/// where similar concepts in different languages have similar embeddings.
///
/// <b>Performance:</b>
/// - English CoNLL-2003: ~92.5% F1
/// - German (zero-shot from English): ~79% F1
/// - Spanish (zero-shot from English): ~81% F1
/// - Cross-lingual average: ~85% F1 (significantly better than mBERT)
/// </para>
/// <para>
/// <b>For Beginners:</b> XLM-RoBERTa can do NER in over 100 languages. The remarkable thing
/// is you can train it on English data and it will recognize entities in other languages too.
/// Use this when you need NER for non-English text or multilingual applications.
/// </para>
/// </remarks>
public class XLMRoBERTaNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates an XLM-RoBERTa-NER model in ONNX inference mode.
    /// </summary>
    public XLMRoBERTaNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "XLM-RoBERTa-NER", "Conneau et al., ACL 2020")
    {
    }

    /// <summary>
    /// Creates an XLM-RoBERTa-NER model in native training mode.
    /// </summary>
    public XLMRoBERTaNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "XLM-RoBERTa-NER", "Conneau et al., ACL 2020", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new XLMRoBERTaNER<T>(Architecture, p, optionsCopy);
        return new XLMRoBERTaNER<T>(Architecture, optionsCopy);
    }
}
