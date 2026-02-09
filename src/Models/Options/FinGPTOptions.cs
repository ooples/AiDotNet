using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinGPT (Financial GPT) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinGPT is an open-source financial large language model designed for various
/// financial NLP tasks including sentiment analysis, question answering, and text generation.
/// </para>
/// <para><b>For Beginners:</b> FinGPT brings GPT capabilities to finance:
///
/// <b>The Key Insight:</b>
/// While BERT models excel at understanding tasks (classification, NER), GPT models
/// are better at generation tasks and can handle more open-ended financial queries.
/// FinGPT adapts the GPT architecture for financial applications.
///
/// <b>What Problems Does FinGPT Solve?</b>
/// - Financial sentiment analysis with explanations
/// - Question answering about financial documents
/// - Generating financial summaries
/// - Extracting insights from earnings calls
/// - Financial text completion and generation
///
/// <b>Key Benefits:</b>
/// - Open-source and fine-tunable
/// - Supports both understanding and generation
/// - Can provide reasoning for predictions
/// - Handles longer context than BERT models
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "FinGPT: Open-Source Financial Large Language Models", 2023.
/// </para>
/// </remarks>
public class FinGPTOptions<T> : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default FinGPT configuration.
    /// </summary>
    public FinGPTOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FinGPTOptions(FinGPTOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        MaxSequenceLength = other.MaxSequenceLength;
        VocabularySize = other.VocabularySize;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        IntermediateDimension = other.IntermediateDimension;
        NumLayers = other.NumLayers;
        NumClasses = other.NumClasses;
        DropoutRate = other.DropoutRate;
        TaskType = other.TaskType;
    }

    /// <summary>Maximum sequence length in tokens (default: 2048 for GPT-style).</summary>
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Vocabulary size (default: 50257 for GPT-2 tokenizer).</summary>
    public int VocabularySize { get; set; } = 50257;

    /// <summary>Hidden dimension (default: 768 for GPT-2 small).</summary>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>Number of attention heads (default: 12).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Intermediate feed-forward dimension (default: 3072).</summary>
    public int IntermediateDimension { get; set; } = 3072;

    /// <summary>Number of transformer layers (default: 12).</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Number of output classes for classification tasks (default: 3).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Task type: "classification", "generation", "qa" (default: "classification").
    /// </summary>
    public string TaskType { get; set; } = "classification";

    /// <summary>
    /// Validates the FinGPT options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This makes sure sizes and settings are valid
    /// so the model can be built safely.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (MaxSequenceLength < 1)
            throw new ArgumentException("MaxSequenceLength must be at least 1.", nameof(MaxSequenceLength));
        if (VocabularySize < 1)
            throw new ArgumentException("VocabularySize must be at least 1.", nameof(VocabularySize));
        if (HiddenDimension < 1)
            throw new ArgumentException("HiddenDimension must be at least 1.", nameof(HiddenDimension));
        if (NumAttentionHeads < 1)
            throw new ArgumentException("NumAttentionHeads must be at least 1.", nameof(NumAttentionHeads));
        if (IntermediateDimension < 1)
            throw new ArgumentException("IntermediateDimension must be at least 1.", nameof(IntermediateDimension));
        if (NumLayers < 1)
            throw new ArgumentException("NumLayers must be at least 1.", nameof(NumLayers));
        if (NumClasses < 1)
            throw new ArgumentException("NumClasses must be at least 1.", nameof(NumClasses));
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be between 0 (inclusive) and 1 (exclusive).", nameof(DropoutRate));
        if (string.IsNullOrWhiteSpace(TaskType))
            throw new ArgumentException("TaskType cannot be null or empty.", nameof(TaskType));
    }
}
