using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// RoBERTa-NER: Robustly Optimized BERT Approach with token classification for NER.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RoBERTa-NER (Liu et al., 2019 - "RoBERTa: A Robustly Optimized BERT Pretraining Approach")
/// fine-tunes a RoBERTa model for NER. RoBERTa improves upon BERT through better pre-training:
///
/// <b>Key Improvements over BERT:</b>
/// - <b>No Next Sentence Prediction (NSP):</b> Removes the NSP objective which was found unhelpful
/// - <b>Dynamic masking:</b> Different tokens are masked in each epoch (vs BERT's static masking)
/// - <b>Longer training:</b> Trained on 160GB of text (10x more than BERT)
/// - <b>Larger batch sizes:</b> 8K sequences per batch for more stable training
/// - <b>Byte-Pair Encoding (BPE):</b> Uses BPE tokenization instead of WordPiece
///
/// <b>Performance (CoNLL-2003):</b>
/// - RoBERTa-base: ~92.8% F1 (vs BERT-base ~92.4%)
/// - RoBERTa-large: ~93.2% F1 (vs BERT-large ~92.8%)
/// </para>
/// <para>
/// <b>For Beginners:</b> RoBERTa is a "better-trained BERT." The architecture is identical,
/// but RoBERTa was trained more carefully on more data, resulting in consistently better performance.
/// Use RoBERTa-NER when you want the best accuracy from a standard transformer model.
/// </para>
/// </remarks>
public class RoBERTaNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a RoBERTa-NER model in ONNX inference mode.
    /// </summary>
    public RoBERTaNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "RoBERTa-NER", "Liu et al., 2019")
    {
    }

    /// <summary>
    /// Creates a RoBERTa-NER model in native training mode.
    /// </summary>
    public RoBERTaNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "RoBERTa-NER", "Liu et al., 2019", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new RoBERTaNER<T>(Architecture, p, optionsCopy);
        return new RoBERTaNER<T>(Architecture, optionsCopy);
    }
}
