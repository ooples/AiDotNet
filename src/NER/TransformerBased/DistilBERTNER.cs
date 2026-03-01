using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// DistilBERT-NER: Knowledge-distilled BERT for efficient Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DistilBERT-NER (Sanh et al., NeurIPS 2019 Workshop - "DistilBERT, a distilled version of BERT:
/// smaller, faster, cheaper and lighter") uses knowledge distillation to create a compact BERT
/// variant that retains 97% of BERT's language understanding while being 60% faster.
///
/// <b>Knowledge Distillation Process:</b>
/// - <b>Teacher:</b> Full BERT-base model (12 layers, 110M parameters)
/// - <b>Student:</b> DistilBERT (6 layers, 66M parameters - 40% fewer)
/// - <b>Training signal:</b> Combination of distillation loss (soft targets from teacher),
///   masked language modeling loss, and cosine embedding loss (align hidden states)
/// - <b>Architecture:</b> Removes token-type embeddings and the pooler layer, keeps every other
///   transformer layer from BERT
///
/// <b>Performance vs Efficiency:</b>
/// - 40% smaller than BERT-base (66M vs 110M parameters)
/// - 60% faster inference (fewer transformer layers)
/// - Retains 97% of BERT's performance on GLUE benchmark
/// - NER (CoNLL-2003): ~91.2% F1 (vs BERT-base ~92.4%)
///
/// <b>Trade-offs:</b>
/// DistilBERT sacrifices ~1.2% F1 on NER compared to BERT-base, but gains significant speed
/// and memory improvements. This makes it ideal for production deployments where latency
/// and resource constraints matter more than the last percentage of accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> DistilBERT is a compressed version of BERT that runs 60% faster
/// while keeping 97% of the accuracy. Think of it as a "student" model that learned from
/// the "teacher" (BERT). Use DistilBERT-NER when you need fast NER with good (but not
/// maximum) accuracy, especially for production deployments with latency requirements.
/// </para>
/// </remarks>
public class DistilBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a DistilBERT-NER model in ONNX inference mode.
    /// </summary>
    public DistilBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? CreateDistilBERTDefaults(),
            "DistilBERT-NER", "Sanh et al., NeurIPS 2019 Workshop")
    {
    }

    /// <summary>
    /// Creates a DistilBERT-NER model in native training mode.
    /// </summary>
    public DistilBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? CreateDistilBERTDefaults(),
            "DistilBERT-NER", "Sanh et al., NeurIPS 2019 Workshop", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new DistilBERTNER<T>(Architecture, p, optionsCopy);
        return new DistilBERTNER<T>(Architecture, optionsCopy);
    }

    private static TransformerNEROptions CreateDistilBERTDefaults()
    {
        return new TransformerNEROptions
        {
            HiddenDimension = 768,
            NumAttentionHeads = 12,
            NumTransformerLayers = 6,   // Half of BERT-base (12 -> 6)
            IntermediateDimension = 3072,
            LearningRate = 5e-5,
            DropoutRate = 0.1
        };
    }
}
