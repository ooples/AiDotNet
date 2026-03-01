using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// TinyBERT-NER: Two-stage distilled BERT for ultra-efficient Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TinyBERT-NER (Jiao et al., EMNLP 2020 Findings - "TinyBERT: Distilling BERT for Natural
/// Language Understanding") uses a novel two-stage knowledge distillation framework to create
/// an extremely compact BERT model (~7.5x smaller and ~9.4x faster than BERT-base).
///
/// <b>Two-Stage Distillation:</b>
/// - <b>Stage 1 - General Distillation:</b> Distill BERT's general language knowledge using
///   a large unlabeled corpus. Transfers attention matrices, hidden states, and embedding
///   layer knowledge from teacher to student.
/// - <b>Stage 2 - Task-Specific Distillation:</b> Fine-tune the student on task-specific data
///   (e.g., NER) using both the labeled data and the teacher's task-specific knowledge.
///
/// <b>Distillation Losses (Layer-by-Layer):</b>
/// - <b>Embedding loss:</b> MSE between teacher and student embedding layers
/// - <b>Attention loss:</b> MSE between teacher and student attention matrices
/// - <b>Hidden state loss:</b> MSE between teacher and student hidden representations
/// - <b>Prediction loss:</b> Cross-entropy between teacher and student output distributions
///
/// <b>TinyBERT-4L Architecture:</b>
/// - 4 transformer layers (vs 12 in BERT-base)
/// - 312 hidden dimension (vs 768 in BERT-base)
/// - 12 attention heads (maintained for knowledge transfer)
/// - ~14.5M parameters (vs 110M in BERT-base)
///
/// <b>Performance:</b>
/// - NER (CoNLL-2003): ~89.5% F1 (vs BERT-base ~92.4%)
/// - 7.5x smaller than BERT-base
/// - 9.4x faster inference
/// </para>
/// <para>
/// <b>For Beginners:</b> TinyBERT is one of the smallest BERT models available. It uses a
/// sophisticated two-step learning process to compress BERT down to a fraction of its size
/// while retaining useful accuracy. Use TinyBERT-NER for edge deployment, mobile applications,
/// or when you need the fastest possible NER with reasonable accuracy.
/// </para>
/// </remarks>
public class TinyBERTNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a TinyBERT-NER model in ONNX inference mode.
    /// </summary>
    public TinyBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? CreateTinyBERTDefaults(),
            "TinyBERT-NER", "Jiao et al., EMNLP 2020 Findings")
    {
    }

    /// <summary>
    /// Creates a TinyBERT-NER model in native training mode.
    /// </summary>
    public TinyBERTNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? CreateTinyBERTDefaults(),
            "TinyBERT-NER", "Jiao et al., EMNLP 2020 Findings", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new TinyBERTNER<T>(Architecture, p, optionsCopy);
        return new TinyBERTNER<T>(Architecture, optionsCopy);
    }

    private static TransformerNEROptions CreateTinyBERTDefaults()
    {
        return new TransformerNEROptions
        {
            HiddenDimension = 312,      // Much smaller than BERT-base (768)
            NumAttentionHeads = 12,
            NumTransformerLayers = 4,    // Only 4 layers (vs 12 in BERT-base)
            IntermediateDimension = 1200, // Proportionally smaller FF dimension
            LearningRate = 5e-5,
            DropoutRate = 0.1
        };
    }
}
