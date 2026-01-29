using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for BloombergGPT-style financial language model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BloombergGPT-style models are large language models trained on extensive financial
/// data including Bloomberg's proprietary financial text corpus along with general text.
/// </para>
/// <para><b>For Beginners:</b> BloombergGPT represents finance-specialized LLMs:
///
/// <b>The Key Insight:</b>
/// General LLMs (GPT-3, GPT-4) lack deep financial knowledge. BloombergGPT-style models
/// are trained on massive financial corpora to understand terminology, concepts, and
/// relationships specific to finance.
///
/// <b>What Problems Does BloombergGPT Solve?</b>
/// - Financial sentiment analysis with high accuracy
/// - Named entity recognition for financial entities
/// - Financial question answering
/// - News classification and headline generation
/// - Understanding complex financial documents
///
/// <b>Architecture Highlights:</b>
/// - Large-scale decoder-only transformer
/// - Trained on mixed financial and general text
/// - Supports various financial NLP benchmarks
///
/// <b>Key Benefits:</b>
/// - Deep financial domain knowledge
/// - State-of-the-art on financial NLP tasks
/// - Understands complex financial terminology
/// - Can handle nuanced financial queries
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "BloombergGPT: A Large Language Model for Finance", 2023.
/// </para>
/// </remarks>
public class BloombergGPTOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default configuration.
    /// </summary>
    public BloombergGPTOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public BloombergGPTOptions(BloombergGPTOptions<T> other)
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

    /// <summary>Maximum sequence length in tokens (default: 2048).</summary>
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Vocabulary size (default: 50257).</summary>
    public int VocabularySize { get; set; } = 50257;

    /// <summary>Hidden dimension (default: 1024 for medium model).</summary>
    public int HiddenDimension { get; set; } = 1024;

    /// <summary>Number of attention heads (default: 16).</summary>
    public int NumAttentionHeads { get; set; } = 16;

    /// <summary>Intermediate feed-forward dimension (default: 4096).</summary>
    public int IntermediateDimension { get; set; } = 4096;

    /// <summary>Number of transformer layers (default: 24).</summary>
    public int NumLayers { get; set; } = 24;

    /// <summary>Number of output classes (default: 3).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Task type: "classification", "ner", "qa", "generation" (default: "classification").
    /// </summary>
    public string TaskType { get; set; } = "classification";

    /// <summary>
    /// Validates the BloombergGPT options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks that model dimensions and settings are valid
    /// before creating the network.
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
