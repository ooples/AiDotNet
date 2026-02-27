using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// InstructionNER: Instruction-tuned transformer for few-shot and zero-shot NER via natural
/// language instructions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InstructionNER (Wang et al., ACL 2022 - "InstructionNER: A Multi-Task Instruction-Based
/// Generative Framework for Few-shot NER") uses instruction tuning to enable NER through
/// natural language task descriptions, enabling strong few-shot and zero-shot performance.
///
/// <b>Key Innovation - Instruction Tuning for NER:</b>
/// Instead of training a model to predict BIO labels, InstructionNER trains the model to
/// follow natural language instructions that describe the NER task. This leverages the
/// instruction-following capabilities of large language models.
///
/// <b>Instruction Format:</b>
/// <code>
/// Instruction: "Extract all person names, organization names, and locations from the
///               following text. Output them in the format: [TYPE: entity]"
/// Input: "Barack Obama, the president of the United States, visited Tokyo."
/// Output: "[PER: Barack Obama] [ORG: United States] [LOC: Tokyo]"
/// </code>
///
/// <b>Multi-Task Training:</b>
/// InstructionNER is trained on multiple NER datasets simultaneously with different
/// instruction prompts, teaching the model to:
/// 1. Extract entities of specified types from text
/// 2. Classify given spans into entity types
/// 3. Verify whether a span is a valid entity of a given type
/// 4. Generate all entities of a specific type
///
/// <b>Few-Shot and Zero-Shot Performance:</b>
/// - Zero-shot (new entity types): ~50-60% F1 depending on similarity to training types
/// - 5-shot: ~65-75% F1
/// - 20-shot: ~75-85% F1
/// - Full training: ~93.0% F1 on CoNLL-2003
///
/// <b>Cross-Domain Transfer:</b>
/// Because instructions describe the task in natural language, InstructionNER can transfer
/// to new domains and entity types that were not seen during training. For example, a model
/// trained on general NER (PER, ORG, LOC) can be prompted to extract domain-specific entities
/// (drug names, gene names) by simply changing the instruction.
/// </para>
/// <para>
/// <b>For Beginners:</b> InstructionNER works like giving a smart assistant a task description:
/// "Find all the company names in this text." The model reads the instruction and the text,
/// then extracts the relevant entities. Because it understands instructions in plain English,
/// it can adapt to new entity types just by changing the instruction, without retraining.
/// </para>
/// </remarks>
public class InstructionNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates an InstructionNER model in ONNX inference mode.
    /// </summary>
    public InstructionNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "InstructionNER", "Wang et al., ACL 2022")
    {
    }

    /// <summary>
    /// Creates an InstructionNER model in native training mode.
    /// </summary>
    public InstructionNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "InstructionNER", "Wang et al., ACL 2022", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new InstructionNER<T>(Architecture, p, optionsCopy);
        return new InstructionNER<T>(Architecture, optionsCopy);
    }
}
