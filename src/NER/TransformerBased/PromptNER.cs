using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// PromptNER: Prompt-based learning for few-shot Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PromptNER (Shen et al., EMNLP 2023 - "PromptNER: Prompt Locating and Typing for Named
/// Entity Recognition") uses prompt-based learning with entity type descriptions to enable
/// effective few-shot NER. It combines the advantages of prompt tuning with span-level
/// entity recognition.
///
/// <b>Key Innovation - Entity Type Descriptions as Prompts:</b>
/// Instead of using class labels like "PER", "ORG", "LOC", PromptNER uses rich natural
/// language descriptions of entity types as prompts:
/// - PER: "a person's name, including first names, last names, and full names"
/// - ORG: "the name of an organization, company, institution, or team"
/// - LOC: "a geographical location, city, country, landmark, or region"
///
/// These descriptions are encoded by the model and used to compute similarity scores
/// between candidate entity spans and entity type descriptions.
///
/// <b>Architecture:</b>
/// 1. <b>Text Encoder:</b> Encode the input sentence with a pre-trained transformer
/// 2. <b>Prompt Encoder:</b> Encode each entity type description with the same transformer
/// 3. <b>Span Locating:</b> Identify candidate entity spans using learned span boundaries
/// 4. <b>Span Typing:</b> Compute similarity between each candidate span representation
///    and each entity type prompt representation
/// 5. <b>Label Assignment:</b> Assign each span the entity type with the highest similarity
///
/// <b>Few-Shot Learning Mechanism:</b>
/// - The prompt descriptions act as "soft labels" that carry semantic information
/// - Adding a few labeled examples refines the span boundary detection
/// - The similarity-based typing naturally generalizes to new entity types
///
/// <b>Performance:</b>
/// - 5-shot CoNLL-2003: ~68-73% F1
/// - 20-shot CoNLL-2003: ~78-83% F1
/// - Full training CoNLL-2003: ~93.2% F1
/// - Zero-shot transfer to new domains: ~55-65% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> PromptNER uses descriptions of entity types (like "a person's name")
/// instead of simple labels (like "PER") to help the model understand what it's looking for.
/// This makes it much easier to adapt to new entity types - just write a description of what
/// the new type looks like, and the model can start recognizing it without extensive retraining.
/// </para>
/// </remarks>
public class PromptNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a PromptNER model in ONNX inference mode.
    /// </summary>
    public PromptNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "PromptNER", "Shen et al., EMNLP 2023")
    {
    }

    /// <summary>
    /// Creates a PromptNER model in native training mode.
    /// </summary>
    public PromptNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "PromptNER", "Shen et al., EMNLP 2023", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new PromptNER<T>(Architecture, p, optionsCopy);
        return new PromptNER<T>(Architecture, optionsCopy);
    }
}
