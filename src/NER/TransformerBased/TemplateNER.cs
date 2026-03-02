using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// Template-NER: Template-based prompt approach for few-shot and zero-shot NER.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Template-NER (Cui et al., ACL 2021 - "Template-Based Named Entity Recognition Using BART")
/// reformulates NER as a sequence-to-sequence generation problem using natural language templates.
/// Instead of predicting labels, the model generates text that fills in template slots.
///
/// <b>Key Innovation - Template-Based Generation:</b>
/// Given an input sentence, Template-NER constructs a template that describes the NER task
/// in natural language, then uses a seq2seq model (BART) to generate the filled template.
///
/// <b>Example:</b>
/// Input: "Barack Obama visited New York City"
/// Template: "[Person] visited [Location]"
/// Model generates: "Barack Obama visited New York City"
/// Extracted: Person="Barack Obama", Location="New York City"
///
/// <b>How Templates Work:</b>
/// 1. Define templates for each entity type: "[Person] is a person", "[Location] is a location"
/// 2. For each entity type, prompt the model: "In the sentence X, [Person] is a person"
/// 3. The model generates the entity that fills [Person] based on the sentence context
/// 4. Repeat for all entity types to extract all entities
///
/// <b>Few-Shot Capability:</b>
/// Because the templates are expressed in natural language, the model can leverage its
/// pre-trained knowledge to perform NER with very few labeled examples:
/// - Zero-shot: No labeled data, just template definitions
/// - Few-shot (10 examples): ~70-75% F1 on CoNLL-2003
/// - Few-shot (100 examples): ~80-85% F1 on CoNLL-2003
/// - Full training: ~93.0% F1 on CoNLL-2003
///
/// <b>Template Design Patterns:</b>
/// - Entity extraction: "In the sentence, [TYPE] refers to ___"
/// - Type classification: "___ is a [PERSON/LOCATION/ORGANIZATION]"
/// - Cloze-style: "Obama is a [MASK] entity" -> "person"
/// </para>
/// <para>
/// <b>For Beginners:</b> Template-NER treats entity recognition like a fill-in-the-blank exercise.
/// Instead of labeling each word, it asks the model: "What person is mentioned in this sentence?"
/// and the model answers: "Barack Obama." This approach works even with very few training examples
/// because the model already understands language from pre-training.
/// </para>
/// </remarks>
public class TemplateNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates a Template-NER model in ONNX inference mode.
    /// </summary>
    public TemplateNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "Template-NER", "Cui et al., ACL 2021")
    {
    }

    /// <summary>
    /// Creates a Template-NER model in native training mode.
    /// </summary>
    public TemplateNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "Template-NER", "Cui et al., ACL 2021", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new TemplateNER<T>(Architecture, p, optionsCopy);
        return new TemplateNER<T>(Architecture, optionsCopy);
    }
}
